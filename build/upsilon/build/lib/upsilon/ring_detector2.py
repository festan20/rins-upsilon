"""Ring detector v2 — topology-based approach.

Different strategy from ring_detector.py:
  Per-colour binary masks + cv2.findContours(RETR_CCOMP) to find
  outer contours that have an inner hole. A real ring is, topologically,
  a coloured annulus with a hole in the middle — the hierarchy mode gives
  us this directly without separate hole/annulus checks.

The colour is known up-front (each mask is single-colour), so no
post-classification annulus sampling is needed either.

Pipeline
--------
  RGB → CLAHE
      → for each ring colour:
          inRange  → small CLOSE  → findContours(RETR_CCOMP)
                        outer + child hole = candidate
                        validate:
                          - outer area in range
                          - hole area > min
                          - outer/hole circularity
                          - aspect ratio of fitted ellipse
                          - hole/outer area ratio
                          - centre alignment between outer & hole
      → in-frame dedup across colours (pixel distance)
      → depth at 12 perimeter points → 3D pose
      → TF to map → track + publish

Published topics
----------------
/detected_rings2           (geometry_msgs/PointStamped)
/ring_markers2             (visualization_msgs/MarkerArray)
/ring_detector2/debug      (sensor_msgs/Image) — annotated BGR frame
/ring_detector2/threshold  (sensor_msgs/Image) — combined per-colour masks
/ring_detector2/contour    (sensor_msgs/Image) — hierarchy debug overlay
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import cv2
import numpy as np

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

from upsilon.perception_utils import (
    decode_compressed_depth, DepthCameraGeometry, TF2Helper, IncrementalTrackManager,
    MapBoundsTracker,
)


# ---------------------------------------------------------------------------
# Shape validation thresholds
# ---------------------------------------------------------------------------
OUTER_AREA_MIN = 200           # px²
OUTER_AREA_MAX = 60000
HOLE_AREA_MIN = 40             # holes smaller than this are noise
OUTER_CIRCULARITY_MIN = 0.55   # 4πA/P² — 1.0 = perfect circle
HOLE_CIRCULARITY_MIN = 0.35    # holes can be a bit more ragged
ASPECT_RATIO_MAX = 2.5
HOLE_OUTER_AREA_RATIO_MIN = 0.08
HOLE_OUTER_AREA_RATIO_MAX = 0.80
CENTER_ALIGN_MAX_FRAC = 0.30   # |c_out - c_hole| / r_outer max

# Depth-range filter (post-detection)
DEPTH_MIN = 0.4
DEPTH_MAX = 3.5

# In-frame dedup (pixels)
SAME_RING_PX = 30

# Only the top fraction of the frame is searched
TOP_FRACTION = 0.35

USE_CLAHE = True

# ---------------------------------------------------------------------------
# HSV ranges (hue in [0,179])
# ---------------------------------------------------------------------------
RED_RANGES = [
    (np.array([0, 100, 80]),   np.array([5, 255, 255])),
    (np.array([170, 100, 80]), np.array([179, 255, 255])),
]
# Each entry: (colour_name, list_of_(lo,hi)_ranges)
COLOUR_BUCKETS = [
    ('red',    RED_RANGES),
    ('blue',   [(np.array([100, 80, 50]),  np.array([130, 255, 255]))]),
    ('green',  [(np.array([40, 60, 50]),   np.array([80, 255, 255]))]),
    ('yellow', [(np.array([20, 100, 100]), np.array([35, 255, 255]))]),
    ('orange', [(np.array([5, 150, 100]),  np.array([20, 255, 255]))]),
    ('purple', [(np.array([130, 50, 50]),  np.array([160, 255, 255]))]),
    ('black',  [(np.array([0, 0, 0]),      np.array([179, 255, 80]))]),
]

COLOUR_RGBA = {
    'red':     ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'orange':  ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    'purple':  ColorRGBA(r=0.7, g=0.0, b=0.9, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
}

_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Black rings often have broken outlines (reflections, uneven dark) — bridge bigger gaps.
# Stays small enough that it won't fill the actual ring hole.
_BLACK_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _bgr_for_colour(name: str):
    rgba = COLOUR_RGBA.get(name, ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))
    return (int(rgba.b * 255), int(rgba.g * 255), int(rgba.r * 255))


def _circularity(area: float, perimeter: float) -> float:
    if perimeter < 1e-3:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def _build_colour_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else (mask | m)
    return mask


class RingDetector2Node(Node):
    def __init__(self):
        super().__init__('ring_detector2')

        self.bridge = CvBridge()
        self.depth_cam = DepthCameraGeometry(patch_radius=4)
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.35)
        self.map_bounds = MapBoundsTracker(self)

        self._latest_bgr: np.ndarray | None = None
        self._depth_frame_id = 'camera_depth_optical_frame'
        self._last_process_time = 0.0
        self._process_interval = 1.0 / 5.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(CompressedImage, '/gemini/color/image_raw/compressed',
                                 self._rgb_cb, qos)
        self.create_subscription(CompressedImage, '/gemini/depth/image_raw/compressedDepth',
                                 self._depth_cb, qos)
        self.create_subscription(CameraInfo, '/gemini/depth/camera_info',
                                 self._caminfo_cb, qos)

        self._ring_pub = self.create_publisher(PointStamped, '/detected_rings2', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/ring_markers2', 10)
        self._debug_pub = self.create_publisher(Image, '/ring_detector2/debug', 10)
        self._threshold_pub = self.create_publisher(Image, '/ring_detector2/threshold', 10)
        self._contour_pub = self.create_publisher(Image, '/ring_detector2/contour', 10)

        self.get_logger().info('Ring detector v2 ready — topology approach.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: CompressedImage) -> None:
        try:
            self._latest_bgr = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def _caminfo_cb(self, msg: CameraInfo) -> None:
        self.depth_cam.update_intrinsics(msg)
        self._depth_frame_id = msg.header.frame_id

    def _depth_cb(self, msg: CompressedImage) -> None:
        if self._latest_bgr is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now
        try:
            self._depth_cb_inner(msg)
        except Exception as e:
            self.get_logger().error(f'Ring detection error (recovering): {e}')

    # ------------------------------------------------------------------
    def _depth_cb_inner(self, msg: CompressedImage) -> None:
        bgr_raw = self._latest_bgr

        depth_m = decode_compressed_depth(msg)
        if depth_m is None:
            self.get_logger().error('Failed to decode compressedDepth')
            return
        self.depth_cam.update_depth(depth_m)

        # Lighting normalisation on a working copy; debug is drawn on bgr_raw
        if USE_CLAHE:
            lab = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
            bgr_work = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            bgr_work = bgr_raw

        hsv = cv2.cvtColor(bgr_work, cv2.COLOR_BGR2HSV)
        # Black is the one colour where CLAHE hurts (it brightens dark pixels and
        # bumps V above the threshold). Use the raw BGR's HSV for black only.
        hsv_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2HSV) if USE_CLAHE else hsv
        h, w = bgr_work.shape[:2]
        crop_y = int(h * TOP_FRACTION)

        debug = bgr_raw.copy()
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        contour_vis = np.zeros((h, w, 3), dtype=np.uint8)

        candidates = []  # list of dicts
        for colour, ranges in COLOUR_BUCKETS:
            hsv_src = hsv_raw if colour == 'black' else hsv
            close_kernel = _BLACK_CLOSE_KERNEL if colour == 'black' else _CLOSE_KERNEL
            mask = _build_colour_mask(hsv_src, ranges)
            mask[crop_y:, :] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
            combined_mask |= mask

            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            if hierarchy is None:
                continue
            # hierarchy: shape (1, N, 4) — [next, prev, first_child, parent]
            for i, (_next, _prev, child_i, parent_i) in enumerate(hierarchy[0]):
                if parent_i != -1:
                    continue                       # we only want outer contours
                if child_i == -1:
                    continue                       # outer without a hole → not a ring

                outer = contours[i]
                # Walk through ALL child holes; pick the largest one
                largest_hole = None
                largest_hole_area = 0.0
                ci = child_i
                while ci != -1:
                    ca = cv2.contourArea(contours[ci])
                    if ca > largest_hole_area:
                        largest_hole_area = ca
                        largest_hole = contours[ci]
                    ci = hierarchy[0][ci][0]   # next sibling

                if largest_hole is None or largest_hole_area < HOLE_AREA_MIN:
                    continue

                outer_area = cv2.contourArea(outer)
                if outer_area < OUTER_AREA_MIN or outer_area > OUTER_AREA_MAX:
                    continue

                area_ratio = largest_hole_area / outer_area
                if not (HOLE_OUTER_AREA_RATIO_MIN < area_ratio < HOLE_OUTER_AREA_RATIO_MAX):
                    continue

                # Circularity
                outer_perim = cv2.arcLength(outer, True)
                outer_circ = _circularity(outer_area, outer_perim)
                if outer_circ < OUTER_CIRCULARITY_MIN:
                    continue
                hole_perim = cv2.arcLength(largest_hole, True)
                hole_circ = _circularity(largest_hole_area, hole_perim)
                if hole_circ < HOLE_CIRCULARITY_MIN:
                    continue

                # Aspect ratio via fitted ellipse
                if len(outer) < 5:
                    continue
                try:
                    ellipse = cv2.fitEllipse(outer)
                except cv2.error:
                    continue
                a, b = ellipse[1]
                if a < 1e-3 or b < 1e-3:
                    continue
                if max(a, b) / min(a, b) > ASPECT_RATIO_MAX:
                    continue

                # Centre alignment between outer-contour centroid and hole centroid
                Mo = cv2.moments(outer)
                Mh = cv2.moments(largest_hole)
                if Mo['m00'] < 1e-3 or Mh['m00'] < 1e-3:
                    continue
                ocx = Mo['m10'] / Mo['m00']
                ocy = Mo['m01'] / Mo['m00']
                hcx = Mh['m10'] / Mh['m00']
                hcy = Mh['m01'] / Mh['m00']
                outer_radius = math.sqrt(outer_area / math.pi)
                center_dist = math.hypot(ocx - hcx, ocy - hcy)
                if center_dist > CENTER_ALIGN_MAX_FRAC * outer_radius * 2:
                    continue

                cx = int(round(ocx))
                cy = int(round(ocy))

                # Cross-colour in-frame dedup
                if any(abs(cx - c['cx']) < SAME_RING_PX and abs(cy - c['cy']) < SAME_RING_PX
                       for c in candidates):
                    continue

                candidates.append({
                    'cx': cx, 'cy': cy,
                    'colour': colour,
                    'outer': outer, 'hole': largest_hole,
                    'ellipse': ellipse,
                    'outer_circ': outer_circ,
                    'hole_circ': hole_circ,
                    'area_ratio': area_ratio,
                })

                # Contour debug overlay
                cv2.drawContours(contour_vis, [outer], -1, (0, 255, 0), 2)
                cv2.drawContours(contour_vis, [largest_hole], -1, (0, 0, 255), 2)

        # ---- debug publishes (gated) ----
        if self._threshold_pub.get_subscription_count() > 0:
            try:
                self._threshold_pub.publish(self.bridge.cv2_to_imgmsg(combined_mask, 'mono8'))
            except CvBridgeError:
                pass
        if self._contour_pub.get_subscription_count() > 0:
            try:
                self._contour_pub.publish(self.bridge.cv2_to_imgmsg(contour_vis, 'bgr8'))
            except CvBridgeError:
                pass

        cv2.putText(debug, f'candidates: {len(candidates)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ---- 3D pose + tracking ----
        any_published = False
        for c in candidates:
            cx, cy = c['cx'], c['cy']
            colour = c['colour']
            ellipse = c['ellipse']
            colour_bgr = _bgr_for_colour(colour)

            # 12-point depth sample at 1.05x outer axis
            a_axis = ellipse[1][0] / 2 * 1.05
            b_axis = ellipse[1][1] / 2 * 1.05
            angle_rad = math.radians(ellipse[2])
            ecx = int(ellipse[0][0])
            ecy = int(ellipse[0][1])

            valid_pts = []
            for i in range(12):
                theta = 2 * math.pi * i / 12
                lx = a_axis * math.cos(theta)
                ly = b_axis * math.sin(theta)
                sx = int(ecx + lx * math.cos(angle_rad) - ly * math.sin(angle_rad))
                sy = int(ecy + lx * math.sin(angle_rad) + ly * math.cos(angle_rad))
                pt = self.depth_cam.get_point(sx, sy)
                if pt is not None:
                    valid_pts.append(pt)
                    cv2.circle(debug, (sx, sy), 2, (0, 255, 0), -1)
                else:
                    cv2.circle(debug, (sx, sy), 2, (0, 0, 255), -1)

            if len(valid_pts) < 3:
                cv2.ellipse(debug, ellipse, colour_bgr, 1)
                cv2.putText(debug, f'{colour} low depth', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            arr = np.array(valid_pts)
            med_x, med_y, med_z = (float(np.median(arr[:, 0])),
                                   float(np.median(arr[:, 1])),
                                   float(np.median(arr[:, 2])))

            if med_z < DEPTH_MIN or med_z > DEPTH_MAX:
                cv2.ellipse(debug, ellipse, colour_bgr, 1)
                cv2.putText(debug, f'{colour} bad z={med_z:.1f}', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            ps = PointStamped()
            ps.header.frame_id = self._depth_frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = med_x, med_y, med_z
            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                cv2.ellipse(debug, ellipse, colour_bgr, 1)
                cv2.putText(debug, f'{colour} no TF', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            if not self.map_bounds.is_in_bounds(mx, my):
                cv2.ellipse(debug, ellipse, _bgr_for_colour(colour), 1)
                cv2.putText(debug, f'{colour} off-map', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue
            track_id, is_new = self.tracker.update(mx, my, colour)
            count = self.tracker.get_count(track_id)

            # CONFIRMED ring — draw fully
            cv2.ellipse(debug, ellipse, colour_bgr, 2)
            cv2.drawContours(debug, [c['hole']], -1, (255, 255, 0), 1)
            label = (f'{colour} c{c["outer_circ"]:.2f} '
                     f'r{c["area_ratio"]:.2f} #{track_id} n={count}')
            cv2.putText(debug, label, (cx, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour_bgr, 2)

            ps_map.header.frame_id = f'map/{colour}/{track_id}/{count}'
            self._ring_pub.publish(ps_map)
            any_published = True

            if is_new:
                self.get_logger().info(
                    f'NEW ring #{track_id} {colour} at ({mx:.2f},{my:.2f}) '
                    f'circ={c["outer_circ"]:.2f} ratio={c["area_ratio"]:.2f}'
                )
            else:
                self.get_logger().info(
                    f'Ring #{track_id} {colour} count={count}',
                    throttle_duration_sec=2.0)

        if any_published:
            self._publish_markers()

        if self._debug_pub.get_subscription_count() > 0:
            try:
                self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
            except CvBridgeError:
                pass

    # ------------------------------------------------------------------
    def _publish_markers(self) -> None:
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for track in self.tracker.tracks:
            colour = track.get('colour', 'unknown')
            count = track['count']

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'rings'
            m.id = track['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = track['x']
            m.pose.position.y = track['y']
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.3
            m.scale.z = 0.05
            m.color = COLOUR_RGBA.get(colour, ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))
            arr.markers.append(m)

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = now
            t.ns = 'ring_labels'
            t.id = track['id']
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = track['x']
            t.pose.position.y = track['y']
            t.pose.position.z = 0.7
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f'{colour} (n={count})'
            arr.markers.append(t)

        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = RingDetector2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
