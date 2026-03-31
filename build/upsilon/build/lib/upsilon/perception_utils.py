"""Shared perception utilities for face and ring detectors."""

import struct
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # noqa: F401 — registers transform methods


class DepthCameraGeometry:
    """Extract 3D world points from a PointCloud2 message at given pixel (u, v)."""

    def __init__(self, patch_radius: int = 3):
        self.patch_radius = patch_radius
        self._cloud_data: np.ndarray | None = None
        self._cloud_width: int = 0
        self._cloud_height: int = 0

    def update(self, cloud_msg) -> None:
        """Cache the latest PointCloud2 message for subsequent lookups."""
        self._cloud_msg = cloud_msg
        self._cloud_width = cloud_msg.width
        self._cloud_height = cloud_msg.height

        # Read all point data as a flat numpy array of bytes, then reshape
        data = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        self._raw = data
        self._point_step = cloud_msg.point_step
        self._row_step = cloud_msg.row_step

    def get_point(self, u: int, v: int):
        """Return (x, y, z) in camera frame at pixel (u, v), or None if invalid.

        Samples a small patch around (u, v) and returns the median of valid points
        to reduce sensitivity to noise and missing depth values.
        """
        if not hasattr(self, '_raw'):
            return None

        r = self.patch_radius
        xs, ys, zs = [], [], []

        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                pu = u + du
                pv = v + dv
                if pu < 0 or pu >= self._cloud_width or pv < 0 or pv >= self._cloud_height:
                    continue
                offset = pv * self._row_step + pu * self._point_step
                xb = self._raw[offset:offset + 4].tobytes()
                yb = self._raw[offset + 4:offset + 8].tobytes()
                zb = self._raw[offset + 8:offset + 12].tobytes()
                x = struct.unpack('f', xb)[0]
                y = struct.unpack('f', yb)[0]
                z = struct.unpack('f', zb)[0]
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and z > 0.0:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

        if not xs:
            return None
        return (float(np.median(xs)), float(np.median(ys)), float(np.median(zs)))


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

    def update(self, x: float, y: float) -> tuple[int, bool]:
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
                return track['id'], False

        # New track
        tid = self._next_id
        self._next_id += 1
        self._tracks.append({'id': tid, 'x': x, 'y': y, 'count': 1})
        return tid, True

    def get_position(self, track_id: int) -> tuple[float, float] | None:
        for t in self._tracks:
            if t['id'] == track_id:
                return t['x'], t['y']
        return None

    @property
    def track_count(self) -> int:
        return len(self._tracks)
