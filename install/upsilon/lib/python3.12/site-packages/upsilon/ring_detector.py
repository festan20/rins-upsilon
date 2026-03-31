"""Ring detector node.

Detects coloured ring posters using ellipse-pair detection (adapted from
dis_tutorial5) combined with HSV colour classification and OAK-D depth
for 3D localisation in the map frame.

Published topics
----------------
/detected_rings       (geometry_msgs/PointStamped)  — one per NEW unique ring;
                      frame_id encodes colour: "map/<color>"
/ring_markers         (visualization_msgs/MarkerArray) — RViz visualisation
/ring_detector/debug  (sensor_msgs/Image) — annotated BGR frame with detected ellipses
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

from upsilon.perception_utils import DepthCameraGeometry, TF2Helper, IncrementalTrackManager

# ---------------------------------------------------------------------------
# Ellipse filter thresholds
# ---------------------------------------------------------------------------
ECC_THR = 120           # max axis length in pixels
RATIO_THR = 2.5         # max aspect ratio — relaxed from 1.5 to handle perspective
CENTER_THR = 15         # max pixel distance between ellipse centres
MIN_CONTOUR_PTS = 15    # min contour points for ellipse fitting

# ---------------------------------------------------------------------------
# HSV colour ranges  (hue in [0,179] OpenCV convention)
# ---------------------------------------------------------------------------
COLOUR_RANGES = [
    ('blue',   np.array([100, 80, 50]),  np.array([130, 255, 255])),
    ('green',  np.array([40, 60, 50]),   np.array([80, 255, 255])),
    ('yellow', np.array([20, 100, 100]), np.array([35, 255, 255])),
    ('orange', np.array([5, 150, 100]),  np.array([20, 255, 255])),
    ('purple', np.array([130, 50, 50]),  np.array([160, 255, 255])),
    # black: low saturation AND low value — handled separately
]

COLOUR_MIN_FRAC = 0.10


def classify_colour(bgr_patch: np.ndarray) -> str:
    """Return the dominant ring colour name for a BGR image patch."""
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 'unknown'

    best_colour = 'unknown'
    best_count = 0

    for name, lo, hi in COLOUR_RANGES:
        mask = cv2.inRange(hsv, lo, hi)
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_colour = name

    # Check for black (low V regardless of H/S)
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 50]))
    black_count = int(np.count_nonzero(black_mask))
    if black_count > best_count:
        best_count = black_count
        best_colour = 'black'

    if best_count / total < COLOUR_MIN_FRAC:
        return 'unknown'
    return best_colour


# Map colour names to RGBA for RViz markers
COLOUR_RGBA = {
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'orange':  ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    'purple':  ColorRGBA(r=0.7, g=0.0, b=0.9, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
    'unknown': ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
}


class RingDetectorNode(Node):
    def __init__(self):
        super().__init__('ring_detector')

        self.bridge = CvBridge()
        self.depth_cam = DepthCameraGeometry(patch_radius=4)
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.8)

        self._latest_bgr: np.ndarray | None = None
        self._latest_stamp = None

        qos = qos_profile_sensor_data

        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)

        self._ring_pub = self.create_publisher(PointStamped, '/detected_rings', 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/ring_markers', QoSReliabilityPolicy.BEST_EFFORT
        )
        self._debug_pub = self.create_publisher(Image, '/ring_detector/debug', 10)

        self.get_logger().info('Ring detector ready.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image) -> None:
        try:
            self._latest_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._latest_stamp = msg.header.stamp
            self._latest_frame = msg.header.frame_id
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    # ------------------------------------------------------------------
    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._latest_bgr is None:
            return

        bgr = self._latest_bgr
        candidates = self._detect_ring_candidates(bgr)

        debug = bgr.copy()

        if not candidates:
            try:
                self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
            except CvBridgeError:
                pass
            return

        self.depth_cam.update(msg)

        for (cx, cy), patch, outer_ellipse in candidates:
            colour = classify_colour(patch)

            pt = self.depth_cam.get_point(cx, cy)
            if pt is None:
                continue

            ps = PointStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = pt

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                self.get_logger().warn('TF transform to map failed; skipping ring.')
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            track_id, is_new = self.tracker.update(mx, my)

            # Draw ellipse and label on debug image
            rgba = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
            bgr_colour = (int(rgba.b * 255), int(rgba.g * 255), int(rgba.r * 255))
            cv2.ellipse(debug, outer_ellipse, bgr_colour, 2)
            cv2.putText(debug, colour, (cx, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_colour, 1)

            if is_new:
                self.get_logger().info(
                    f'New ring #{track_id} colour={colour} at map ({mx:.2f}, {my:.2f})'
                )
                # Encode colour in frame_id for the controller to read
                ps_map.header.frame_id = f'map/{colour}'
                self._ring_pub.publish(ps_map)

            self._publish_markers()

        try:
            self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except CvBridgeError:
            pass

    # ------------------------------------------------------------------
    def _detect_ring_candidates(self, bgr: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray, tuple]]:
        """Return list of ((cx, cy), colour_patch, outer_ellipse) for each detected ring."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        elps = []
        for cnt in contours:
            if cnt.shape[0] < MIN_CONTOUR_PTS:
                continue
            ellipse = cv2.fitEllipse(cnt)
            e = ellipse[1]
            a, b = e[0], e[1]
            ratio = a / b if a > b else b / a
            if ratio <= RATIO_THR and a < ECC_THR and b < ECC_THR:
                elps.append(ellipse)

        results = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                e1, e2 = elps[n], elps[m]
                dist = np.sqrt((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2)
                if dist >= CENTER_THR:
                    continue

                # Determine which ellipse is the outer (larger) one.
                # Use area (pi * a * b) instead of requiring both axes individually,
                # because perspective warping can make one axis of the "outer"
                # slightly smaller than the inner's corresponding axis.
                area1 = e1[1][0] * e1[1][1]
                area2 = e2[1][0] * e2[1][1]

                if area1 >= area2:
                    outer, inner = e1, e2
                else:
                    outer, inner = e2, e1

                # The outer must be meaningfully larger (at least 10% more area)
                if area1 == 0 or area2 == 0:
                    continue
                bigger = max(area1, area2)
                smaller = min(area1, area2)
                if bigger / smaller < 1.1:
                    continue

                cx = int(outer[0][0])
                cy = int(outer[0][1])
                size = int((outer[1][0] + outer[1][1]) / 2)
                half = max(size // 2, 5)

                h, w = bgr.shape[:2]
                x1 = max(cy - half, 0)
                x2 = min(cy + half, h)
                y1 = max(cx - half, 0)
                y2 = min(cx + half, w)
                patch = bgr[x1:x2, y1:y2]

                results.append(((cx, cy), patch, outer))

        return results

    # ------------------------------------------------------------------
    def _publish_markers(self) -> None:
        arr = MarkerArray()
        for track in self.tracker._tracks:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
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
            colour = track.get('colour', 'unknown')
            m.color = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = RingDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
