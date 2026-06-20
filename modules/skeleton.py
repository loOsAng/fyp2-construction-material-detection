"""Skeleton extraction from instance segmentation masks."""

import heapq
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import medial_axis, skeletonize

_MASK_SMOOTH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_DCE_EPSILON = 2.0  # contour simplification tolerance (px) before skeletonization
_DENSE_BRICK_FAST_PATH_MIN = 80
_DENSE_BRICK_FAST_PATH_RATIO = 0.8


def _smooth_mask(mask: np.ndarray) -> np.ndarray:
    """Light morphological open — removes noise without bridging adjacent objects."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MASK_SMOOTH_KERNEL)


def _dce_simplify_mask(binary: np.ndarray, epsilon: float = _DCE_EPSILON) -> np.ndarray:
    """Simplify mask contour with approxPolyDP before skeletonization.

    Discrete Curve Evolution (DCE) removes small boundary protrusions that
    would otherwise create spur branches in the morphological skeleton.
    cv2.approxPolyDP is a fast approximation of the DCE algorithm.

    Args:
        binary: uint8 binary mask, 0/255 or 0/1.
        epsilon: Douglas-Peucker tolerance in pixels (default 2.0).

    Returns:
        Simplified uint8 binary mask of same shape.
    """
    mask_u8 = binary.astype(np.uint8)
    # Ensure 0/255 format for findContours
    if mask_u8.max() <= 1:
        mask_u8 = mask_u8 * 255

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary

    simplified = np.zeros_like(mask_u8, dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < 20.0:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        cv2.drawContours(simplified, [approx], -1, 255, -1)
    return simplified


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    """Light erosion — shrinks masks slightly to prevent edge contact between neighbours."""
    return cv2.erode(mask, _ERODE_KERNEL, iterations=1)


def _isolate_by_erosion(masks_list: list[np.ndarray]) -> list[np.ndarray]:
    """Soft isolation: erodes other masks slightly before subtraction.

    Shrinking the *other* mask by 1 px before subtracting protects the
    current mask's true physical edge from being eaten by a neighbour
    whose mask happens to touch it.
    """
    n = len(masks_list)
    if n <= 1:
        return masks_list

    shrink_kernel = np.ones((3, 3), np.uint8)
    eroded_masks: list[np.ndarray] = []
    coverage = np.zeros_like(masks_list[0], dtype=np.uint16)

    for mask in masks_list:
        eroded = cv2.erode(mask.astype(np.uint8), shrink_kernel, iterations=1)
        eroded_masks.append(eroded)
        coverage += (eroded > 0).astype(np.uint16)

    isolated: list[np.ndarray] = []
    for mask, eroded in zip(masks_list, eroded_masks):
        clean = mask.copy()
        other_coverage = coverage - (eroded > 0).astype(np.uint16)
        clean[other_coverage > 0] = 0
        isolated.append(clean)
    return isolated


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a binary mask."""
    mask_u8 = mask.astype(np.uint8)
    if np.count_nonzero(mask_u8) == 0:
        return mask_u8

    padded = np.pad((mask_u8 > 0).astype(np.uint8) * 255, 1, mode="constant", constant_values=0)
    h, w = padded.shape[:2]
    flood = padded.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    exterior = flood[1:-1, 1:-1] > 0
    holes = (~exterior) & (mask_u8 == 0)
    filled = mask_u8.copy()
    filled[holes] = 1
    return filled.astype(np.uint8)


def _mask_threshold_for_class(class_key: str) -> float:
    """Per-class mask binarisation threshold.

    I-beams use 0.50 (same as other classes) because the stricter 0.55
    trimmed flange edges and caused incomplete masks, degrading the
    hybrid I-profile geometry overlay quality.
    """
    return 0.50


def _refine_instance_mask(binary: np.ndarray, class_key: str = "") -> np.ndarray:
    """Clean one predicted instance mask before skeleton/geometry extraction.

    Uses a cross-shaped kernel for the opening pass instead of an ellipse.
    An elliptical kernel rounds off sharp flange corners; a MORPH_CROSS
    only erodes in the 4 cardinal directions, preserving diagonal edges and
    thin structural tips.  The closing pass keeps an ellipse for smooth fill.
    All operations stay in OpenCV C++ — no per-instance Python connected-
    component analysis (area_opening/area_closing would add 2 extra CC passes).
    """
    mask = (binary > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return mask

    if class_key == "i-beam":
        # I-beams have long straight edges — cross open preserves flange tips.
        # Slightly larger close kernel to fill internal voids (bolt-holes etc.).
        open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        min_area = 30
    else:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        min_area = 20

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = _fill_binary_holes(mask)
    mask = _largest_component(mask, min_area=min_area)
    return (mask > 0).astype(np.uint8)


def _refine_instance_mask_cropped(binary: np.ndarray, class_key: str = "", pad: int = 3) -> np.ndarray:
    """Refine a mask inside its tight bounding box, then paste it back."""
    binary_u8 = (binary > 0).astype(np.uint8)
    if np.count_nonzero(binary_u8) == 0:
        return binary_u8

    ys, xs = np.nonzero(binary_u8)
    h, w = binary_u8.shape[:2]
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(h, int(ys.max()) + pad + 1)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)

    refined_crop = _refine_instance_mask(binary_u8[y1:y2, x1:x2], class_key)
    refined = np.zeros_like(binary_u8, dtype=np.uint8)
    refined[y1:y2, x1:x2] = refined_crop
    return refined


def _prepare_instance_binaries(result, h: int, w: int) -> List[np.ndarray]:
    """Convert model masks to refined per-instance binaries at image resolution."""
    if result.masks is None:
        return []

    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else None

    binaries: List[np.ndarray] = []
    class_keys: List[str] = []
    for idx, m in enumerate(masks):
        class_key = ""
        if cls_ids is not None and idx < len(cls_ids):
            class_key = _class_name(result, cls_ids[idx]).casefold()
        class_keys.append(class_key)
        resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (resized > _mask_threshold_for_class(class_key)).astype(np.uint8)
        binaries.append(_refine_instance_mask_cropped(binary, class_key))

    brick_count = sum(1 for key in class_keys if key == "brick")
    if (
        len(binaries) >= _DENSE_BRICK_FAST_PATH_MIN
        and brick_count / max(1, len(binaries)) >= _DENSE_BRICK_FAST_PATH_RATIO
    ):
        return binaries

    return _isolate_by_erosion(binaries)


def apply_mask_postprocessing(image_rgb: np.ndarray, result):
    """Write refined binary masks back into a YOLO Results object for rendering."""
    if result.masks is None or result.boxes is None:
        return result

    h, w = image_rgb.shape[:2]
    binaries = _prepare_instance_binaries(result, h, w)
    if not binaries:
        return result

    import torch

    old_data = result.masks.data
    device = old_data.device
    dtype = old_data.dtype
    new_masks = torch.zeros((len(binaries), h, w), device=device, dtype=dtype)
    for idx, binary in enumerate(binaries):
        new_masks[idx] = torch.from_numpy(binary.astype(np.float32)).to(device=device, dtype=dtype)

    result.masks.data = new_masks
    return result


_BRICK_BOX_COLORS = [
    (255, 214, 0),
    (255, 96, 96),
    (96, 172, 255),
    (207, 125, 255),
    (64, 220, 180),
]


def _class_name(result, cls_id: float) -> str:
    """Resolve a YOLO class id to its display name."""
    cls_idx = int(cls_id)
    if isinstance(result.names, dict):
        return str(result.names.get(cls_idx, cls_idx))
    return str(result.names[cls_idx])


def _component_centroids(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return connected-component centroids as (x, y) points."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    points: List[Tuple[int, int]] = []
    for label_idx in range(1, num_labels):
        x, y = centroids[label_idx]
        points.append((int(round(x)), int(round(y))))
    return points


def _detect_endpoints_and_branches(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Detect skeleton endpoints and branch points using 8-neighbour counts."""
    skel_u8 = skel.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0

    neighbour_count = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = (skel_u8 == 1) & (neighbour_count == 1)
    branches = (skel_u8 == 1) & (neighbour_count >= 3)

    return _component_centroids(endpoints), _component_centroids(branches)


def _skeleton_length_px(skel: np.ndarray) -> float:
    """Approximate skeleton graph length in pixels with 8-connected edge weights."""
    skel_bool = skel.astype(bool)
    diag = float(np.sqrt(2.0))

    horizontal = np.count_nonzero(skel_bool[:, :-1] & skel_bool[:, 1:])
    vertical = np.count_nonzero(skel_bool[:-1, :] & skel_bool[1:, :])
    diag_down_right = np.count_nonzero(skel_bool[:-1, :-1] & skel_bool[1:, 1:])
    diag_down_left = np.count_nonzero(skel_bool[:-1, 1:] & skel_bool[1:, :-1])

    return float(horizontal + vertical + diag * (diag_down_right + diag_down_left))


def _neighbor_points(
    point: Tuple[int, int],
    point_set: set[Tuple[int, int]],
    connectivity: int = 8,
) -> List[Tuple[int, int]]:
    """Return neighbouring skeleton pixels as (y, x) points."""
    y, x = point
    if connectivity == 4:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    return [(y + dy, x + dx) for dy, dx in offsets if (y + dy, x + dx) in point_set]


def _edge_weight(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Return 8-neighbour graph edge weight between two (y, x) pixels."""
    return float(np.sqrt(2.0)) if a[0] != b[0] and a[1] != b[1] else 1.0


def _prune_skeleton_branches(skel: np.ndarray, max_branch_length: float = 4.0) -> np.ndarray:
    """Remove short leaf branches while preserving the main skeleton component.

    Pruning uses 4-neighbour tracing to avoid diagonal shortcuts around junctions.
    The measured geometry still uses an 8-neighbour graph.
    """
    pruned = skel.astype(bool).copy()

    while True:
        points = set(zip(*np.nonzero(pruned)))
        if len(points) < 3:
            return pruned

        remove: set[Tuple[int, int]] = set()
        endpoints = [p for p in points if len(_neighbor_points(p, points, connectivity=4)) == 1]

        for endpoint in endpoints:
            neighbours = _neighbor_points(endpoint, points, connectivity=4)
            if not neighbours:
                continue

            branch_nodes = [endpoint]
            prev = endpoint
            current = neighbours[0]
            length_px = _edge_weight(prev, current)
            reaches_junction = False

            while True:
                current_neighbours = _neighbor_points(current, points, connectivity=4)
                degree = len(current_neighbours)
                if degree >= 3:
                    reaches_junction = True
                    break
                if degree <= 1:
                    branch_nodes.append(current)
                    break

                next_candidates = [p for p in current_neighbours if p != prev]
                if not next_candidates:
                    branch_nodes.append(current)
                    break

                branch_nodes.append(current)
                prev, current = current, next_candidates[0]
                length_px += _edge_weight(prev, current)

            if reaches_junction and length_px <= max_branch_length:
                remove.update(branch_nodes)

        if not remove:
            return pruned

        for y, x in remove:
            pruned[y, x] = False


def _build_skeleton_graph(skel: np.ndarray) -> Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]:
    """Build an 8-neighbour weighted graph from a skeleton mask."""
    points = set(zip(*np.nonzero(skel.astype(bool))))
    graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]] = {}
    for point in points:
        graph[point] = [(nb, _edge_weight(point, nb)) for nb in _neighbor_points(point, points)]
    return graph


def _dijkstra_skeleton(
    graph: Dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]],
    start: Tuple[int, int],
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Tuple[int, int]]]:
    """Shortest paths from a skeleton node."""
    distances: Dict[Tuple[int, int], float] = {start: 0.0}
    parents: Dict[Tuple[int, int], Tuple[int, int]] = {}
    queue: list[Tuple[float, Tuple[int, int]]] = [(0.0, start)]

    while queue:
        dist, point = heapq.heappop(queue)
        if dist > distances.get(point, float("inf")):
            continue
        for neighbour, weight in graph.get(point, []):
            new_dist = dist + weight
            if new_dist < distances.get(neighbour, float("inf")):
                distances[neighbour] = new_dist
                parents[neighbour] = point
                heapq.heappush(queue, (new_dist, neighbour))

    return distances, parents


def _reconstruct_path(
    parents: Dict[Tuple[int, int], Tuple[int, int]],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Reconstruct a path as (y, x) points."""
    path = [end]
    current = end
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def _longest_skeleton_path(skel: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
    """Return the longest endpoint-to-endpoint skeleton path as (x, y) points."""
    graph = _build_skeleton_graph(skel)
    if len(graph) < 2:
        return [], 0.0

    endpoints = [point for point, neighbours in graph.items() if len(neighbours) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(graph.keys())

    best_start: Optional[Tuple[int, int]] = None
    best_end: Optional[Tuple[int, int]] = None
    best_dist = -1.0
    best_parents: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for start in candidates:
        distances, parents = _dijkstra_skeleton(graph, start)
        for end in candidates:
            if end == start or end not in distances:
                continue
            if distances[end] > best_dist:
                best_start = start
                best_end = end
                best_dist = distances[end]
                best_parents = parents

    if best_start is None or best_end is None:
        return [], 0.0

    path_yx = _reconstruct_path(best_parents, best_start, best_end)
    path_xy = [(int(x), int(y)) for y, x in path_yx]
    return path_xy, float(best_dist)


def _largest_component(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or int(np.max(areas)) < min_area:
        return np.zeros_like(mask_u8)

    largest_label = int(np.argmax(areas)) + 1
    return (labels == largest_label).astype(np.uint8)


def _pca_axis(points_xy: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Estimate the dominant axis and orientation angle from (x, y) points."""
    if len(points_xy) < 2:
        return None, None

    centered = points_xy.astype(np.float32) - np.mean(points_xy, axis=0)
    if np.allclose(centered, 0):
        return None, None

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]

    # Keep the orientation stable while treating opposite directions as the same line.
    if axis[0] < 0:
        axis = -axis

    angle = float(np.degrees(np.arctan2(axis[1], axis[0])))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0

    return axis.astype(np.float32), angle


def _axis_extreme_points(
    axis: Optional[np.ndarray],
    skeleton_points_xy: np.ndarray,
    endpoints: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Choose the two points that best represent the instance's main polyline ends."""
    if axis is None or len(skeleton_points_xy) == 0:
        return []

    candidates = np.asarray(endpoints if len(endpoints) >= 2 else skeleton_points_xy, dtype=np.float32)
    projections = candidates @ axis
    start = candidates[int(np.argmin(projections))]
    end = candidates[int(np.argmax(projections))]

    return [
        (int(round(float(start[0]))), int(round(float(start[1])))),
        (int(round(float(end[0]))), int(round(float(end[1])))),
    ]


# ---------------------------------------------------------------------------
# PCA computed on SKELETON pixels so the axis is naturally centred,
# unlike the original which used all mask pixels (flanges pull the axis).
# ---------------------------------------------------------------------------

def _skeleton_pca_axis(
    binary: np.ndarray,
    dce_epsilon: Optional[float] = _DCE_EPSILON,
) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray, np.ndarray]:
    """Run PCA on the *medial axis* of the mask, not on all mask pixels.

    Uses medial_axis (distance-transform ridge detection) instead of
    Zhang-Suen thinning — produces cleaner skeletons with fewer spurs.
    Optionally applies DCE contour simplification before skeletonization
    to remove boundary noise that causes spurious branches.

    Returns (axis, angle_deg, center, skeleton_points_xy).
    """
    smoothed = _smooth_mask(binary)
    if dce_epsilon is not None and dce_epsilon > 0:
        smoothed = _dce_simplify_mask(smoothed, epsilon=dce_epsilon)
    skel, distance = medial_axis(smoothed.astype(bool), return_distance=True)
    # Adaptive distance threshold: scale with the mask's actual extent so
    # narrow I-beams (e.g. 20 px wide) don't lose their entire skeleton.
    # Uses 2% of the mask's smaller dimension, clamped to [1.0, 2.5] px.
    ys, xs = np.nonzero(binary)
    if len(ys) > 0:
        mask_h = int(ys.max() - ys.min() + 1)
        mask_w = int(xs.max() - xs.min() + 1)
        dist_thresh = max(1.0, min(2.5, min(mask_h, mask_w) * 0.02))
    else:
        dist_thresh = 1.5
    skel[distance < dist_thresh] = False
    points_yx = np.column_stack(np.nonzero(skel))
    if len(points_yx) < 4:
        return None, None, np.zeros(2, dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    points_xy = points_yx[:, ::-1].astype(np.float32)
    axis, angle_deg = _pca_axis(points_xy)
    if axis is None:
        return None, None, np.zeros(2, dtype=np.float32), points_xy

    center = np.mean(points_xy, axis=0)
    return axis, angle_deg, center.astype(np.float32), points_xy


def _mask_axis_geometry(binary: np.ndarray) -> Tuple[List[Tuple[int, int]], Optional[float], float]:
    """Estimate a clean centreline from mask pixels, not from the full raw skeleton.

    This is intended for presentation/demo views. Morphological skeletonization is
    useful for topology diagnostics, but I/H-shaped steel profiles naturally create
    many branches. The mask PCA axis produces a cleaner geometry overlay.
    """
    component = _largest_component(binary, min_area=50)
    points_yx = np.column_stack(np.nonzero(component))
    if len(points_yx) < 2:
        return [], None, 0.0

    points_xy = points_yx[:, ::-1].astype(np.float32)
    axis, angle_deg = _pca_axis(points_xy)
    if axis is None:
        return [], None, 0.0

    center = np.mean(points_xy, axis=0)
    projections = (points_xy - center) @ axis
    start = center + axis * float(np.min(projections))
    end = center + axis * float(np.max(projections))
    p1 = (int(round(float(start[0]))), int(round(float(start[1]))))
    p2 = (int(round(float(end[0]))), int(round(float(end[1]))))
    length_px = float(np.linalg.norm(np.asarray(p2, dtype=np.float32) - np.asarray(p1, dtype=np.float32)))
    return [p1, p2], angle_deg, length_px


def _normalize_line_angle(angle_deg: float) -> float:
    """Normalize an undirected line angle to (-90, 90]."""
    angle = float(angle_deg)
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def _axis_from_angle(angle_deg: float) -> np.ndarray:
    """Return a unit axis vector from an image-coordinate angle."""
    rad = np.radians(float(angle_deg))
    axis = np.asarray([np.cos(rad), np.sin(rad)], dtype=np.float32)
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    axis = axis / norm
    if axis[0] < 0:
        axis = -axis
    return axis.astype(np.float32)


def _rect_long_axis(component: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Estimate orientation from the longer side of the rotated bounding box."""
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 8.0:
        return None, None

    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    edges = [
        (box[(idx + 1) % 4] - box[idx], float(np.linalg.norm(box[(idx + 1) % 4] - box[idx])))
        for idx in range(4)
    ]
    vector, length = max(edges, key=lambda item: item[1])
    if length <= 0.0:
        return None, None

    axis = vector / length
    if axis[0] < 0:
        axis = -axis
    angle_deg = _normalize_line_angle(float(np.degrees(np.arctan2(axis[1], axis[0]))))
    return axis.astype(np.float32), angle_deg


def _axis_quality(points_xy: np.ndarray, axis: np.ndarray) -> Tuple[float, float, float]:
    """Score an axis using robust projected length and cross-axis width."""
    perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    axis_proj = points_xy @ axis
    perp_proj = points_xy @ perp
    length_px = float(np.percentile(axis_proj, 98) - np.percentile(axis_proj, 2))
    width_px = float(np.percentile(perp_proj, 95) - np.percentile(perp_proj, 5))
    quality = length_px / max(width_px, 1.0)
    return quality, length_px, width_px


def _clip_point_to_shape(point_xy: np.ndarray, mask_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Clip an (x, y) point to image bounds."""
    h, w = mask_shape[:2]
    clipped = np.clip(np.round(point_xy).astype(int), [0, 0], [w - 1, h - 1])
    return int(clipped[0]), int(clipped[1])


def _polyline_length(points: List[Tuple[int, int]]) -> float:
    """Return total Euclidean length of a polyline."""
    if len(points) < 2:
        return 0.0
    arr = np.asarray(points, dtype=np.float32)
    return float(np.sum(np.linalg.norm(arr[1:] - arr[:-1], axis=1)))


def _mask_slice_midpoint_polyline_geometry(
    binary: np.ndarray,
    force_angle_deg: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], Optional[float], float]:
    """Estimate FYP1-style centre polyline using PCA slice midpoints.

    This avoids drawing every raw skeleton branch. It follows the FYP1
    approach: clean the instance mask, estimate the main direction with PCA,
    split the object along that direction, then connect each slice's median
    centre point into a stable centre polyline.
    """
    mask_u8 = binary.astype(np.uint8)
    # Use small symmetric kernels to minimise geometric shift between the
    # original segmentation mask and the polyline overlay drawn on top.
    close_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    open_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_element)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_element)
    component = _largest_component(cleaned, min_area=30)

    points_yx = np.column_stack(np.nonzero(component))
    if len(points_yx) < 20:
        # --- Small-mask fallback ---
        # Too few pixels for slicing → estimate a simple centreline from
        # the rotated bounding box, then fall back to raw extrema.
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) >= 4:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect).astype(np.int32)
                # Use the longer edge of the bounding box as the centreline
                edges = [
                    (tuple(box[i]), tuple(box[(i + 1) % 4]))
                    for i in range(4)
                ]
                edge_lengths = [
                    (e, float(np.linalg.norm(np.asarray(e[1]) - np.asarray(e[0]))))
                    for e in edges
                ]
                best_edge, best_len = max(edge_lengths, key=lambda x: x[1])
                if best_len >= 4.0:
                    p1 = _clip_point_to_shape(
                        np.asarray(best_edge[0], dtype=np.float32), component.shape)
                    p2 = _clip_point_to_shape(
                        np.asarray(best_edge[1], dtype=np.float32), component.shape)
                    angle = _normalize_line_angle(
                        float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))))
                    return [p1, p2], angle, best_len
        # Last resort: raw extrema
        if len(points_yx) >= 2:
            ys, xs = points_yx[:, 0], points_yx[:, 1]
            p1 = (int(xs.min()), int(ys.min()))
            p2 = (int(xs.max()), int(ys.max()))
            length = float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))
            if length >= 2.0:
                angle = _normalize_line_angle(
                    float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))))
                return [p1, p2], angle, length
        return [], None, 0.0

    points_xy = points_yx[:, ::-1].astype(np.float32)
    mean = np.mean(points_xy, axis=0)

    if force_angle_deg is None:
        # Prefer the rotated bounding-box axis — it is based on the overall
        # shape contour and is immune to mask-edge noise that skews PCA.
        # Fall back to PCA on skeleton pixels, then PCA on all mask pixels.
        rect_axis, rect_angle = _rect_long_axis(component)
        if rect_axis is not None and rect_angle is not None:
            axis, angle_deg = rect_axis, rect_angle
        else:
            axis, angle_deg, _center, _skel_xy = _skeleton_pca_axis(binary, dce_epsilon=0)
            if axis is None:
                axis, angle_deg = _pca_axis(points_xy)
        if axis is None or angle_deg is None:
            return [], None, 0.0
    else:
        angle_deg = _normalize_line_angle(force_angle_deg)
        axis = _axis_from_angle(angle_deg)

    minor_axis = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    centered = points_xy - mean
    major_projection = centered @ axis
    minor_projection = centered @ minor_axis

    lower = float(np.percentile(major_projection, 2))
    upper = float(np.percentile(major_projection, 98))
    projection_span = upper - lower
    if projection_span < 10.0:
        start = mean + axis * lower
        end = mean + axis * upper
        points = [
            _clip_point_to_shape(start, component.shape),
            _clip_point_to_shape(end, component.shape),
        ]
        return points, angle_deg, _polyline_length(points)

    slice_count = int(np.clip(projection_span / 14.0, 8, 24))
    bin_edges = np.linspace(lower, upper, slice_count + 1)

    polyline_points: List[Tuple[int, int]] = []
    for slice_idx, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if slice_idx == slice_count - 1:
            in_slice = (major_projection >= lo) & (major_projection <= hi)
        else:
            in_slice = (major_projection >= lo) & (major_projection < hi)
        if np.count_nonzero(in_slice) < 10:
            continue

        major_mid = float(np.median(major_projection[in_slice]))
        minor_mid = float(np.median(minor_projection[in_slice]))
        point = mean + axis * major_mid + minor_axis * minor_mid
        integer_point = _clip_point_to_shape(point, component.shape)
        if not polyline_points or polyline_points[-1] != integer_point:
            polyline_points.append(integer_point)

    if len(polyline_points) < 2:
        start = mean + axis * lower
        end = mean + axis * upper
        polyline_points = [
            _clip_point_to_shape(start, component.shape),
            _clip_point_to_shape(end, component.shape),
        ]

    return polyline_points, angle_deg, _polyline_length(polyline_points)


def _mask_hybrid_i_profile_geometry(
    binary: np.ndarray,
    force_angle_deg: Optional[float] = None,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    Optional[float],
    float,
]:
    """Estimate an I-beam overlay from FYP1 centreline plus local flange caps."""
    polyline_points, angle_deg, length_px = _mask_slice_midpoint_polyline_geometry(
        binary,
        force_angle_deg=force_angle_deg,
    )
    if len(polyline_points) < 2 or angle_deg is None:
        return polyline_points, [], angle_deg, length_px

    mask_u8 = binary.astype(np.uint8)
    # Small kernels to match the polyline geometry pass — avoids offset.
    close_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    open_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_element)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_element)
    component = _largest_component(cleaned, min_area=30)

    points_yx = np.column_stack(np.nonzero(component))
    if len(points_yx) < 20:
        return polyline_points, [], angle_deg, length_px

    points_xy = points_yx[:, ::-1].astype(np.float32)
    axis = _axis_from_angle(angle_deg)
    perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)

    axis_proj = points_xy @ axis
    perp_proj = points_xy @ perp
    projection_span = float(np.percentile(axis_proj, 98) - np.percentile(axis_proj, 2))
    if projection_span < 10.0:
        return polyline_points, [], angle_deg, length_px

    band = max(6.0, min(22.0, projection_span * 0.14))
    caps: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    endpoint_vectors = [np.asarray(polyline_points[0], dtype=np.float32), np.asarray(polyline_points[-1], dtype=np.float32)]
    for endpoint in endpoint_vectors:
        endpoint_proj = float(endpoint @ axis)
        in_band = np.abs(axis_proj - endpoint_proj) <= band
        if np.count_nonzero(in_band) < 3:  # relaxed for thin side-view masks
            continue

        local_axis_mid = float(np.median(axis_proj[in_band]))
        local_perp = perp_proj[in_band]
        width = float(np.percentile(local_perp, 95) - np.percentile(local_perp, 5))
        if width < 4.0:
            continue

        lo = float(np.percentile(local_perp, 5))
        hi = float(np.percentile(local_perp, 95))
        cap_center_on_axis = axis * local_axis_mid
        cap_start = cap_center_on_axis + perp * lo
        cap_end = cap_center_on_axis + perp * hi
        caps.append((
            _clip_point_to_shape(cap_start, component.shape),
            _clip_point_to_shape(cap_end, component.shape),
        ))

    return polyline_points, caps, angle_deg, length_px


def _draw_clean_axis(
    viz: np.ndarray,
    endpoints: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> None:
    """Draw a presentation-friendly centreline and endpoints."""
    if len(endpoints) != 2:
        return
    cv2.line(viz, endpoints[0], endpoints[1], color, thickness, cv2.LINE_AA)
    for point in endpoints:
        cv2.circle(viz, point, max(4, thickness + 1), color, -1, cv2.LINE_AA)
        cv2.circle(viz, point, max(6, thickness + 3), (255, 255, 255), 1, cv2.LINE_AA)


def _draw_i_profile(
    viz: np.ndarray,
    endpoints: List[Tuple[int, int]],
    caps: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
    draw_points: bool = False,
) -> None:
    """Draw a clean I-shaped I-beam skeleton."""
    if len(endpoints) != 2:
        return
    cv2.line(viz, endpoints[0], endpoints[1], color, thickness, cv2.LINE_AA)
    for start, end in caps:
        cv2.line(viz, start, end, color, thickness, cv2.LINE_AA)
    if draw_points:
        for point in endpoints:
            cv2.circle(viz, point, max(3, thickness), color, -1, cv2.LINE_AA)


def _draw_slice_midpoint_polyline(
    viz: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    draw_points: bool = False,
) -> None:
    """Draw the FYP1-style centre polyline estimated from mask slices."""
    if len(points) < 2:
        return
    pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(viz, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    if draw_points:
        radius = max(2, thickness)
        for point in points:
            cv2.circle(viz, point, radius, color, -1, cv2.LINE_AA)


def _mask_t_profile_geometry(
    binary: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[Tuple[int, int], Tuple[int, int]]], Optional[float], float]:
    """Estimate a T-shaped nail skeleton from mask pixels.

    PCA is computed on the *skeleton* (not all mask pixels) so the stem axis
    follows the geometric center of the nail body.

    The stem follows the nail/screw main axis. The transverse cap is placed on
    the wider endpoint band, which usually corresponds to the nail head.
    """
    component = _largest_component(binary, min_area=20)
    # --- PCA on skeleton pixels for centred stem axis ---
    axis, angle_deg, center, _skel_xy = _skeleton_pca_axis(component)
    if axis is None:
        return [], [], None, 0.0

    # Still use all mask pixels for extent
    points_yx = np.column_stack(np.nonzero(component))
    points_xy = points_yx[:, ::-1].astype(np.float32)

    perp = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    axis_proj = (points_xy - center) @ axis
    perp_proj = (points_xy - center) @ perp

    min_axis = float(np.min(axis_proj))
    max_axis = float(np.max(axis_proj))
    length = max_axis - min_axis
    if length <= 1.0:
        return [], [], angle_deg, 0.0

    band = max(6.0, min(28.0, length * 0.18))
    side_data = []
    for side, edge_value in (("min", min_axis), ("max", max_axis)):
        if side == "min":
            edge_mask = axis_proj <= min_axis + band
        else:
            edge_mask = axis_proj >= max_axis - band

        local_axis = axis_proj[edge_mask]
        local_perp = perp_proj[edge_mask]
        if len(local_perp) >= 3:
            lo = float(np.percentile(local_perp, 8))
            hi = float(np.percentile(local_perp, 92))
            mid = float(np.median(local_perp))
            cap_axis_value = float(np.median(local_axis))
            width = hi - lo
        else:
            mid = 0.0
            cap_axis_value = edge_value
            width = 0.0
        side_data.append((side, cap_axis_value, mid, width))

    head_side, cap_axis_value, cap_perp_mid, cap_width = max(side_data, key=lambda item: item[3])
    cap_half_width = max(5.0, min(18.0, max(cap_width * 0.45, length * 0.045)))

    cap_center = center + axis * cap_axis_value + perp * cap_perp_mid
    cap_start = cap_center - perp * cap_half_width
    cap_end = cap_center + perp * cap_half_width

    if head_side == "min":
        tip = center + axis * max_axis
    else:
        tip = center + axis * min_axis

    endpoints = [
        (int(round(float(tip[0]))), int(round(float(tip[1])))),
        (int(round(float(cap_center[0]))), int(round(float(cap_center[1])))),
    ]
    caps = [
        (
            (int(round(float(cap_start[0]))), int(round(float(cap_start[1])))),
            (int(round(float(cap_end[0]))), int(round(float(cap_end[1])))),
        )
    ]
    length_px = float(np.linalg.norm(np.asarray(endpoints[1], dtype=np.float32) - np.asarray(endpoints[0], dtype=np.float32)))
    return endpoints, caps, angle_deg, length_px


def _draw_t_profile(
    viz: np.ndarray,
    endpoints: List[Tuple[int, int]],
    caps: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
    draw_points: bool = False,
) -> None:
    """Draw a clean T-shaped nail skeleton."""
    if len(endpoints) != 2:
        return
    cv2.line(viz, endpoints[0], endpoints[1], color, thickness, cv2.LINE_AA)
    for start, end in caps:
        cv2.line(viz, start, end, color, thickness, cv2.LINE_AA)
    if draw_points:
        for point in endpoints:
            cv2.circle(viz, point, max(3, thickness), color, -1, cv2.LINE_AA)


def _mask_bbox(binary: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return a tight bounding box as x1, y1, x2, y2."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _draw_brick_segmentation(
    viz: np.ndarray,
    binary: np.ndarray,
    color: Tuple[int, int, int] = (0, 96, 255),
    box_color: Tuple[int, int, int] = (255, 214, 0),
    outline_thickness: int = 2,
) -> None:
    """Draw a brick segmentation mask and a contrasting per-instance box."""
    mask = (binary > 0).astype(np.uint8)
    if np.count_nonzero(mask) == 0:
        return

    bbox = _mask_bbox(mask)
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    crop = mask[y1:y2 + 1, x1:x2 + 1]

    viz_crop = viz[y1:y2 + 1, x1:x2 + 1]
    viz_crop[crop > 0] = color
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        shifted = [cnt + np.array([[[x1, y1]]], dtype=cnt.dtype) for cnt in contours]
        cv2.drawContours(viz, shifted, -1, color, outline_thickness, cv2.LINE_AA)
    cv2.rectangle(viz, (x1, y1), (x2, y2), box_color, outline_thickness, cv2.LINE_AA)


def _ensure_binary_shape(binary: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return a binary mask at image resolution."""
    mask = (binary > 0).astype(np.uint8)
    if mask.shape[:2] == (h, w):
        return mask
    return (cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)


def _clean_overlay_thickness(class_key: str, image_shape: Tuple[int, int], class_count: int) -> int:
    """Use thinner geometry in dense scenes so overlays do not hide the material."""
    h, w = image_shape
    base = 2 if max(h, w) <= 900 else 3
    if class_key == "nail":
        return 1 if class_count > 16 else base
    if class_key == "i-beam":
        return 2 if class_count > 20 else base
    return base


def extract_clean_geometry_overlay(
    image_rgb: np.ndarray,
    result,
    target_classes: Tuple[str, ...] = ("Brick", "I-beam", "nail"),
    draw_labels: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Draw class-specific clean overlays.

    Brick: segmentation mask only.
    I-beam: semantic I-shaped skeleton.
    nail: semantic T-shaped skeleton.
    """
    viz = image_rgb.copy()
    geometry_viz = image_rgb.copy()
    h, w = image_rgb.shape[:2]
    analyses: List[Dict[str, object]] = []
    brick_masks: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []
    drew_geometry = False

    if result.masks is None or result.boxes is None:
        return viz, analyses

    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.full(len(cls_ids), np.nan)
    target_keys = {name.casefold() for name in target_classes}

    class_counts: Dict[str, int] = {}
    for cls_id in cls_ids:
        class_key = _class_name(result, cls_id).casefold()
        class_counts[class_key] = class_counts.get(class_key, 0) + 1

    # --- Use already-refined masks from apply_mask_postprocessing ---
    # The full morph open/close/hole-fill/largest-component pipeline already
    # ran before this function is called. Read directly without re-isolation.
    binaries_isolated = [
        (result.masks.data[idx].cpu().numpy() > 0).astype(np.uint8)
        for idx in range(result.masks.data.shape[0])
    ]

    if "i-beam" in target_keys:
        ibeam_viz, ibeam_analyses = analyze_skeleton_polylines(
            image_rgb,
            result,
            target_class="I-beam",
            draw_labels=False,
            draw_axis=False,
        )
        changed = np.any(ibeam_viz != image_rgb, axis=2)
        if np.any(changed):
            geometry_viz[changed] = ibeam_viz[changed]
            drew_geometry = True
        analyses.extend(ibeam_analyses)

    # --- Angle voting: compute median I-beam main-axis angle for outlier correction ---
    # Only vote when ≥3 I-beams exist. With exactly 2 beams the median is their
    # average, which incorrectly forces both to a wrong angle when they have
    # genuinely different orientations (e.g. 0° and 80° → median 40°).
    # With 3+ beams the median is robust to a single outlier.
    order = list(range(min(len(masks), len(cls_ids))))

    for idx in order:
        if idx >= len(cls_ids):
            continue

        class_name = _class_name(result, cls_ids[idx])
        class_key = class_name.casefold()
        if class_key not in target_keys:
            continue

        # Use isolated binary mask; class-specific geometry functions handle morphology.
        binary = _ensure_binary_shape(binaries_isolated[idx], h, w)
        bbox = _mask_bbox(binary)

        if class_key == "brick":
            if bbox is not None and np.count_nonzero(binary) >= 20:
                box_color = _BRICK_BOX_COLORS[len(brick_masks) % len(_BRICK_BOX_COLORS)]
                brick_masks.append((binary, box_color))
            continue

        if class_key == "i-beam":
            continue

        caps: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        if class_key == "nail":
            endpoints, caps, angle_deg, length_px = _mask_t_profile_geometry(binary)
        else:
            endpoints, angle_deg, length_px = _mask_axis_geometry(binary)

        instance_id = len(analyses) + 1
        if class_key == "nail":
            status = "ok_t_profile_skeleton" if len(endpoints) == 2 and len(caps) == 1 else "needs_review_no_clear_t_profile"
        else:
            status = "ok_clean_axis" if len(endpoints) == 2 else "needs_review_no_clear_axis"
        analysis: Dict[str, object] = {
            "instance_id": instance_id,
            "detection_index": idx,
            "class": class_name,
            "confidence": None if np.isnan(float(confs[idx])) else round(float(confs[idx]), 3),
            "length_px": round(length_px, 2),
            "angle_deg": None if angle_deg is None else round(angle_deg, 2),
            "geometry_method": "t_profile_skeleton" if class_key == "nail" else "clean_shape_overlay",
            "endpoints": endpoints,
            "polyline_points": [],
            "branch_points": [],
            "axis_endpoints": endpoints,
            "flange_caps": caps if class_key in ("i-beam", "nail") else [],
            "status": status,
        }
        analyses.append(analysis)

        if len(endpoints) == 2:
            color = (0, 255, 0)
            thickness = _clean_overlay_thickness(class_key, (h, w), class_counts.get(class_key, 0))
            if class_key == "i-beam":
                # Draw a straight PCA axis line instead of the wobbly
                # slice-midpoint polyline — I-beams are rigid straight
                # objects, the polyline just adds visual noise.
                _draw_clean_axis(geometry_viz, endpoints, color=color, thickness=thickness)
                for cap_start, cap_end in caps:
                    cv2.line(geometry_viz, cap_start, cap_end, color, thickness, cv2.LINE_AA)
                drew_geometry = True
            elif class_key == "nail":
                _draw_t_profile(geometry_viz, endpoints, caps, color=color, thickness=thickness)
                drew_geometry = True
            else:
                _draw_clean_axis(geometry_viz, endpoints, color=color, thickness=thickness)
                drew_geometry = True
            if draw_labels:
                label = f"#{instance_id} {class_name} {analysis['length_px']}px"
                anchor = endpoints[0]
                cv2.putText(geometry_viz, label, (anchor[0] + 6, anchor[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(geometry_viz, label, (anchor[0] + 6, anchor[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if drew_geometry or brick_masks:
        for brick_mask, box_color in brick_masks:
            _draw_brick_segmentation(geometry_viz, brick_mask, box_color=box_color)
        viz = cv2.addWeighted(geometry_viz, 0.72, image_rgb, 0.28, 0)
    return viz, analyses


def analyze_skeleton_polylines(
    image_rgb: np.ndarray,
    result,
    target_class: str = "I-beam",
    draw_labels: bool = True,
    draw_axis: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Analyze per-instance I-beam skeletons as approximate polylines.

    Returns:
        viz: RGB image with target-instance skeletons, endpoints, branch points, and axis overlays.
        analyses: One row per target instance with length_px, angle_deg, endpoints,
            branch_points, and status.
    """
    viz = image_rgb.copy()
    h, w = image_rgb.shape[:2]
    analyses: List[Dict[str, object]] = []

    if result.masks is None or result.boxes is None:
        return viz, analyses

    masks = result.masks.data.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [None] * len(cls_ids)
    kernel = np.ones((3, 3), dtype=np.uint8)
    target_key = target_class.casefold()

    # --- Use already-refined masks from apply_mask_postprocessing ---
    binaries_isolated = [
        (result.masks.data[idx].cpu().numpy() > 0).astype(np.uint8)
        for idx in range(result.masks.data.shape[0])
    ]

    for idx, mask in enumerate(masks):
        if idx >= len(cls_ids):
            continue

        class_name = _class_name(result, cls_ids[idx])
        if class_name.casefold() != target_key:
            continue

        binary = binaries_isolated[idx]

        instance_id = len(analyses) + 1
        analysis: Dict[str, object] = {
            "instance_id": instance_id,
            "detection_index": idx,
            "class": class_name,
            "confidence": None if confs[idx] is None else round(float(confs[idx]), 3),
            "length_px": 0.0,
            "angle_deg": None,
            "endpoints": [],
            "branch_points": [],
            "axis_endpoints": [],
            "status": "skipped_small_mask",
        }

        if np.sum(binary) < 50:
            analyses.append(analysis)
            continue

        # Masks are already refined by apply_mask_postprocessing;
        # only apply light smoothing before skeletonization.
        binary = _smooth_mask(binary)
        skel = skeletonize(binary.astype(bool))
        raw_endpoints, raw_branch_points = _detect_endpoints_and_branches(skel)
        pruned_skel = _prune_skeleton_branches(skel, max_branch_length=4.0)
        if np.count_nonzero(pruned_skel) >= 2:
            analysis_skel = pruned_skel
        else:
            analysis_skel = skel

        skeleton_points_yx = np.column_stack(np.nonzero(skel))
        analysis_points_yx = np.column_stack(np.nonzero(analysis_skel))

        if len(analysis_points_yx) < 2:
            analysis["status"] = "skeleton_too_short"
            analyses.append(analysis)
            continue

        skeleton_points_xy = analysis_points_yx[:, ::-1]
        endpoints, branch_points = _detect_endpoints_and_branches(analysis_skel)
        main_path, main_path_length = _longest_skeleton_path(analysis_skel)
        length_px = main_path_length if main_path else _skeleton_length_px(analysis_skel)
        axis_points = np.asarray(main_path if len(main_path) >= 2 else skeleton_points_xy, dtype=np.float32)
        axis, angle_deg = _pca_axis(axis_points)
        axis_endpoints = [main_path[0], main_path[-1]] if len(main_path) >= 2 else _axis_extreme_points(axis, skeleton_points_xy, endpoints)
        display_endpoints = axis_endpoints if len(axis_endpoints) == 2 else endpoints

        if len(axis_endpoints) < 2:
            status = "needs_review_no_clear_endpoints"
        elif raw_branch_points and np.count_nonzero(pruned_skel) < np.count_nonzero(skel):
            status = "ok_pruned_main_path"
        elif raw_branch_points:
            status = "branched_needs_review"
        else:
            status = "ok"

        analysis.update(
            {
                "length_px": round(length_px, 2),
                "angle_deg": None if angle_deg is None else round(angle_deg, 2),
                "endpoints": [(int(round(float(x))), int(round(float(y)))) for x, y in display_endpoints],
                "branch_points": [(int(round(float(x))), int(round(float(y)))) for x, y in raw_branch_points],
                "axis_endpoints": [
                    (int(round(float(x))), int(round(float(y)))) for x, y in axis_endpoints
                ],
                "geometry_method": "pruned_longest_path_pca" if status == "ok_pruned_main_path" else "longest_path_pca",
                "status": status,
            }
        )
        analyses.append(analysis)

        # Draw the true morphological skeleton first. Keep it visible because
        # this mode is diagnostic evidence, not the clean geometry overlay.
        raw_skel_u8 = skel.astype(np.uint8) * 255
        raw_dilated = cv2.dilate(raw_skel_u8, kernel, iterations=1)
        raw_pixels = raw_dilated > 0
        viz[raw_pixels] = (viz[raw_pixels] * 0.35 + np.array([20, 180, 60]) * 0.65).astype(np.uint8)

        if np.count_nonzero(analysis_skel) >= 2:
            pruned_skel_u8 = analysis_skel.astype(np.uint8) * 255
            pruned_dilated = cv2.dilate(pruned_skel_u8, kernel, iterations=1)
            pruned_pixels = pruned_dilated > 0
            viz[pruned_pixels] = (viz[pruned_pixels] * 0.20 + np.array([0, 255, 0]) * 0.80).astype(np.uint8)

        # Draw the PCA axis as a thin reference line, separate from the skeleton.
        if draw_axis and len(axis_endpoints) == 2:
            cv2.line(viz, axis_endpoints[0], axis_endpoints[1], (30, 120, 255), 1, cv2.LINE_AA)

        # Draw the longest skeleton path as the measured centreline.
        if len(main_path) >= 2:
            path_points = np.asarray(main_path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(viz, [path_points], False, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(viz, [path_points], False, (0, 255, 255), 3, cv2.LINE_AA)

        # Endpoints are green, branch points are red.
        for ep in analysis["endpoints"]:
            cv2.circle(viz, ep, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(viz, ep, 6, (255, 255, 255), 1, cv2.LINE_AA)

        for bp in analysis["branch_points"]:
            cv2.circle(viz, bp, 4, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(viz, bp, 6, (255, 255, 255), 1, cv2.LINE_AA)

    if draw_labels:
        for a in analyses:
            if len(a["axis_endpoints"]) == 2:
                p1, p2 = a["axis_endpoints"]
                label = f"#{a['instance_id']} L={a['length_px']}px"
                cv2.putText(viz, label, (p1[0] + 6, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(viz, label, (p1[0] + 6, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return viz, analyses


def get_confidence_stats(result) -> Dict[str, Dict[str, float]]:
    """Compute per-class confidence statistics.

    Returns:
        {class_name: {'mean': float, 'min': float, 'max': float}}
    """
    if result.boxes is None or result.boxes.conf is None:
        return {}

    confs_all = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy()

    stats: Dict[str, List[float]] = {}
    for cls_id, conf in zip(cls_ids, confs_all):
        name = result.names[int(cls_id)]
        stats.setdefault(name, []).append(float(conf))

    return {
        name: {"mean": round(np.mean(vals), 3), "min": round(np.min(vals), 3), "max": round(np.max(vals), 3)}
        for name, vals in stats.items()
    }
