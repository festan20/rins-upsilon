"""Shared perception utilities for face and ring detectors."""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers transform methods


class MapBoundsTracker:
    """Subscribes to /map and checks whether an (x, y) is within the map bounds.

    Until a map message arrives, every point is considered in-bounds so the
    detectors don't drop everything at startup.
    """

    def __init__(self, node: Node, topic: str = '/map'):
        self._x_min = None
        self._x_max = None
        self._y_min = None
        self._y_max = None
        # The map server publishes with TRANSIENT_LOCAL durability so late
        # subscribers still get the map; match it.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(OccupancyGrid, topic, self._map_cb, qos)

    def _map_cb(self, msg: OccupancyGrid) -> None:
        info = msg.info
        self._x_min = info.origin.position.x
        self._y_min = info.origin.position.y
        self._x_max = self._x_min + info.width * info.resolution
        self._y_max = self._y_min + info.height * info.resolution

    def is_in_bounds(self, x: float, y: float) -> bool:
        if self._x_min is None:
            return True
        return (self._x_min <= x <= self._x_max and
                self._y_min <= y <= self._y_max)


class OccupancyGridMap:
    """Subscribes to /map and lets you raycast against the static walls.

    Faces and rings live on walls, so a detection's true position is the first
    occupied cell along the camera bearing. This removes the dependence on noisy
    depth for range — we only need the (reliable) bearing from the pixel, then
    snap to the wall.
    """

    def __init__(self, node: Node, topic: str = '/map'):
        self._grid: np.ndarray | None = None  # (H, W) int8, -1 unknown / 0 free / 100 occ
        self._res = 0.0
        self._ox = 0.0
        self._oy = 0.0
        self._h = 0
        self._w = 0
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        node.create_subscription(OccupancyGrid, topic, self._map_cb, qos)

    def _map_cb(self, msg: OccupancyGrid) -> None:
        info = msg.info
        self._res = info.resolution
        self._ox = info.origin.position.x
        self._oy = info.origin.position.y
        self._h = info.height
        self._w = info.width
        self._grid = np.asarray(msg.data, dtype=np.int8).reshape(self._h, self._w)

    @property
    def ready(self) -> bool:
        return self._grid is not None

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self._ox) / self._res)
        row = int((y - self._oy) / self._res)
        return col, row

    def is_occupied(self, x: float, y: float, occ_thresh: int = 50) -> bool:
        if self._grid is None:
            return False
        col, row = self._cell(x, y)
        if 0 <= row < self._h and 0 <= col < self._w:
            return self._grid[row, col] >= occ_thresh
        return False

    def raycast(self, x0: float, y0: float, dx: float, dy: float,
                max_range: float = 10.0, occ_thresh: int = 50,
                skip_near: float = 0.15):
        """March from (x0, y0) along unit direction (dx, dy) in map frame.

        Returns (x, y) of the first occupied cell hit, or None if the ray leaves
        the map or reaches `max_range` without hitting a wall. `skip_near` skips
        the first few centimetres so the robot's own start cell can't self-hit.
        """
        if self._grid is None or self._res <= 0.0:
            return None
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        dx, dy = dx / norm, dy / norm
        step = self._res * 0.5
        n = int(max_range / step)
        dist = skip_near
        for _ in range(n):
            dist += step
            if dist > max_range:
                break
            x = x0 + dx * dist
            y = y0 + dy * dist
            col, row = self._cell(x, y)
            if not (0 <= row < self._h and 0 <= col < self._w):
                return None
            if self._grid[row, col] >= occ_thresh:
                return (x, y)
        return None

    def is_in_bounds(self, x: float, y: float) -> bool:
        if self._grid is None:
            return True
        return (self._ox <= x <= self._ox + self._w * self._res and
                self._oy <= y <= self._oy + self._h * self._res)


def decode_compressed_depth(msg) -> np.ndarray | None:
    """Decode a compressedDepth message to a float32 depth image in metres.

    The Gemini compressedDepth format is: 12-byte header + PNG data (16UC1, mm).
    Returns (H, W) float32 array in metres, or None on failure.
    """
    data = bytes(msg.data)
    png_magic = b'\x89PNG'
    idx = data.find(png_magic)
    if idx < 0:
        return None
    png_buf = np.frombuffer(data[idx:], dtype=np.uint8)
    img = cv2.imdecode(png_buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img.astype(np.float32) / 1000.0  # mm → metres


def decode_depth_message(msg, bridge) -> np.ndarray | None:
    """Decode compressedDepth or raw Image depth into a float32 depth map in metres."""
    if isinstance(msg, CompressedImage):
        return decode_compressed_depth(msg)

    if not isinstance(msg, Image):
        return None

    try:
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    except CvBridgeError:
        return None

    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32)
    return depth.astype(np.float32)


class DepthCameraGeometry:
    """Extract 3D points at a pixel (u, v) from either:
      * a depth image + camera intrinsics  (update_depth + update_intrinsics), or
      * an organized PointCloud2           (update) — preferred when available.

    When an organized cloud has been provided via ``update``, ``get_point`` reads
    the XYZ directly from the cloud (no intrinsics needed). Otherwise it falls
    back to deprojecting the depth image with the cached intrinsics.
    """

    def __init__(self, patch_radius: int = 3):
        self.patch_radius = patch_radius
        self._depth: np.ndarray | None = None
        self._xyz: np.ndarray | None = None   # (H, W, 3) organized cloud, camera frame
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._has_intrinsics = False

    def update(self, cloud_msg) -> None:
        """Cache an organized PointCloud2 as an (H, W, 3) XYZ array (camera frame).

        Also exposes the Z channel as a depth image for masking. No-op (clears
        cloud state) if the cloud is unorganized (height <= 1).
        """
        h, w = cloud_msg.height, cloud_msg.width
        if h <= 1 or w <= 1:
            self._xyz = None
            return
        arr = pc2.read_points_numpy(
            cloud_msg, field_names=('x', 'y', 'z'), reshape_organized_cloud=True)
        self._xyz = arr.astype(np.float32).reshape(h, w, 3)
        depth = self._xyz[:, :, 2].copy()
        depth[~np.isfinite(depth)] = 0.0
        self._depth = depth

    def update_intrinsics(self, camera_info_msg) -> None:
        """Cache camera intrinsics from a CameraInfo message."""
        K = camera_info_msg.k
        self._fx = K[0]
        self._fy = K[4]
        self._cx = K[2]
        self._cy = K[5]
        self._has_intrinsics = True

    def update_depth(self, depth_img: np.ndarray) -> None:
        """Cache a depth image (float32, metres)."""
        self._depth = depth_img

    def get_depth_image(self) -> np.ndarray | None:
        """Return the cached depth image (H, W) float32 in metres, or None."""
        return self._depth

    def get_point(self, u: int, v: int):
        """Return (x, y, z) in camera frame at pixel (u, v), or None if invalid.

        Cloud mode (organized PointCloud2 via ``update``): reads XYZ directly.
        Image mode: deprojects the depth image with the cached intrinsics.
        Both sample a small patch around (u, v) and return the median valid point.
        """
        # --- cloud mode ---
        if self._xyz is not None:
            h, w = self._xyz.shape[:2]
            r = self.patch_radius
            v_lo, v_hi = max(0, v - r), min(h, v + r + 1)
            u_lo, u_hi = max(0, u - r), min(w, u + r + 1)
            patch = self._xyz[v_lo:v_hi, u_lo:u_hi].reshape(-1, 3)
            valid = np.isfinite(patch).all(axis=1) & (patch[:, 2] > 0.0)
            if not np.any(valid):
                return None
            p = patch[valid]
            return (float(np.median(p[:, 0])),
                    float(np.median(p[:, 1])),
                    float(np.median(p[:, 2])))

        # --- image mode ---
        if self._depth is None or not self._has_intrinsics:
            return None

        h, w = self._depth.shape[:2]
        r = self.patch_radius

        v_lo = max(0, v - r)
        v_hi = min(h, v + r + 1)
        u_lo = max(0, u - r)
        u_hi = min(w, u + r + 1)

        patch = self._depth[v_lo:v_hi, u_lo:u_hi]
        valid_mask = (patch > 0.0) & np.isfinite(patch)
        if not np.any(valid_mask):
            return None

        # Get pixel coordinates for valid points
        vs, us = np.where(valid_mask)
        vs += v_lo
        us += u_lo
        depths = self._depth[vs, us]

        # Deproject using intrinsics
        xs = (us - self._cx) * depths / self._fx
        ys = (vs - self._cy) * depths / self._fy

        return (float(np.median(xs)), float(np.median(ys)), float(np.median(depths)))

    @property
    def has_intrinsics(self) -> bool:
        return self._has_intrinsics

    def pixel_to_ray(self, u: float, v: float):
        """Return a unit ray direction (x, y, z) in the camera optical frame
        pointing through pixel (u, v). Independent of depth. None if no intrinsics.
        """
        if not self._has_intrinsics:
            return None
        x = (u - self._cx) / self._fx
        y = (v - self._cy) / self._fy
        z = 1.0
        n = math.sqrt(x * x + y * y + z * z)
        return (x / n, y / n, z / n)


class TF2Helper:
    """Thin wrapper around tf2_ros for point transforms.

    The listener runs on its OWN spin thread (``spin_thread=True``) so the TF
    buffer keeps filling even while a detector's callback is busy with heavy CV
    work on a single-threaded executor. Without this, ``/tf`` gets starved and
    lookups at a message's (often slightly stale) stamp fail with "no TF".
    """

    def __init__(self, node: Node):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, node, spin_thread=True)

    def transform_point(self, point_stamped: PointStamped, target_frame: str):
        """Transform a PointStamped to target_frame, or None.

        Tries the point's own stamp first; if that exact time isn't available
        (e.g. a laggy camera cloud whose stamp predates the buffered TF), retries
        against the latest available transform (zero stamp). Fine for a slow /
        teleop robot where the latest transform ~ the capture-time transform.
        """
        try:
            return self.buffer.transform(
                point_stamped, target_frame, timeout=Duration(seconds=0.2))
        except Exception:
            pass
        # Fallback: latest available transform (stamp = 0).
        try:
            latest = PointStamped()
            latest.header.frame_id = point_stamped.header.frame_id
            latest.header.stamp = Time().to_msg()
            latest.point = point_stamped.point
            return self.buffer.transform(
                latest, target_frame, timeout=Duration(seconds=0.2))
        except Exception:
            return None


class IncrementalTrackManager:
    """Deduplicates detections by clustering nearby poses.

    A new detection is merged with an existing track if it falls within
    `merge_distance` metres; otherwise a new track is created.
    Returns a unique integer ID for each track.
    """

    def __init__(self, merge_distance: float = 0.8):
        self.merge_distance = merge_distance
        self._tracks: list[dict] = []  # [{'id': int, 'x': float, 'y': float, 'count': int}]
        self._next_id = 0

    def update(self, x: float, y: float, colour: str = 'unknown') -> tuple[int, bool]:
        """Register a detection at (x, y) in map frame.

        Returns (track_id, is_new) where is_new=True means first time this
        track has been seen.
        """
        for track in self._tracks:
            dist = math.sqrt((track['x'] - x) ** 2 + (track['y'] - y) ** 2)
            if dist < self.merge_distance:
                # Merge: update running average position
                n = track['count']
                track['x'] = (track['x'] * n + x) / (n + 1)
                track['y'] = (track['y'] * n + y) / (n + 1)
                track['count'] += 1
                if colour != 'unknown':
                    track['colour'] = colour
                return track['id'], False

        # New track
        tid = self._next_id
        self._next_id += 1
        self._tracks.append({'id': tid, 'x': x, 'y': y, 'count': 1, 'colour': colour})
        return tid, True

    def get_position(self, track_id: int) -> tuple[float, float] | None:
        for t in self._tracks:
            if t['id'] == track_id:
                return t['x'], t['y']
        return None

    def get_count(self, track_id: int) -> int:
        """Return the number of detections for a given track."""
        for t in self._tracks:
            if t['id'] == track_id:
                return t['count']
        return 0

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    @property
    def tracks(self) -> list[dict]:
        return self._tracks
