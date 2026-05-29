"""Shared perception utilities for face and ring detectors."""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
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


class DepthCameraGeometry:
    """Extract 3D world points from a depth image + camera intrinsics at given pixel (u, v)."""

    def __init__(self, patch_radius: int = 3):
        self.patch_radius = patch_radius
        self._depth: np.ndarray | None = None
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0
        self._has_intrinsics = False

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

        Uses depth image + camera intrinsics to deproject.
        Samples a small patch around (u, v) and returns the median of valid points.
        """
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


class TF2Helper:
    """Thin wrapper around tf2_ros for point transforms."""

    def __init__(self, node: Node):
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer, node)

    def transform_point(self, point_stamped: PointStamped, target_frame: str):
        """Transform a PointStamped to target_frame. Returns transformed PointStamped or None."""
        try:
            return self.buffer.transform(point_stamped, target_frame, timeout=rclpy.duration.Duration(seconds=0.5))
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
