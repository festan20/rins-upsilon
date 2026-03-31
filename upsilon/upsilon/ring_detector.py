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
ECC_MIN = 10            # min axis length — reject tiny noise ellipses
RATIO_THR = 2.5         # max aspect ratio — relaxed from 1.5 to handle perspective
CENTER_THR = 15         # max pixel distance between ellipse centres
MIN_CONTOUR_PTS = 15    # min contour points for ellipse fitting
DEPTH_MAX = 3.0         # max depth in metres for foreground mask

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


def classify_colour(bgr_patch: np.ndarray) -> tuple[str, float]:
    """Return (colour_name, fraction) for the dominant ring colour in a BGR patch."""
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 'unknown', 0.0

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

    frac = best_count / total
    if frac < COLOUR_MIN_FRAC:
        return 'unknown', frac
    return best_colour, frac


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
        self._treshold = self.create_publisher(Image,  '/ring_detector/threshold', 10)
        self._contour = self.create_publisher(Image,  '/ring_detector/contour', 10)

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

        try:
            self._cloud_cb_inner(msg)
        except Exception as e:
            self.get_logger().error(f'Ring detection error (recovering): {e}')

    def _cloud_cb_inner(self, msg: PointCloud2) -> None:
        bgr = self._latest_bgr
        self.depth_cam.update(msg)

        depth_img = self.depth_cam.get_depth_image()
        if depth_img is None:
            self.get_logger().warn(
                'No depth image available', throttle_duration_sec=5.0)
            return

        # Binary foreground mask: everything closer than DEPTH_MAX
        mask = ((depth_img > 0.0) & (depth_img < DEPTH_MAX)).astype(np.uint8) * 255

        # Only keep top half — rings are suspended in the air
        h = mask.shape[0]
        mask[h // 2:, :] = 0

        # Publish binary mask as threshold debug
        try:
            self._treshold.publish(self.bridge.cv2_to_imgmsg(mask, 'mono8'))
        except CvBridgeError:
            pass

        debug = bgr.copy()
        candidates, n_ellipses = self._detect_ring_candidates(mask, bgr, debug)

        # Show status on debug image
        status = f'ellipses:{n_ellipses} rings:{len(candidates)}'
        cv2.putText(debug, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for (cx, cy), patch, outer_ellipse in candidates:
            colour, _ = classify_colour(patch)

            pt = self.depth_cam.get_point(cx, cy)
            if pt is None:
                cv2.putText(debug, f'{colour} no depth', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            ps = PointStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = pt

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                cv2.putText(debug, f'{colour} no TF', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            track_id, is_new = self.tracker.update(mx, my)

            # Draw colour label on debug image
            rgba = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
            bgr_colour = (int(rgba.b * 255), int(rgba.g * 255), int(rgba.r * 255))
            cv2.putText(debug, f'{colour} ({mx:.1f},{my:.1f})', (cx, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_colour, 2)

            if is_new:
                self.get_logger().info(
                    f'New ring #{track_id} colour={colour} at map ({mx:.2f}, {my:.2f})'
                )
                ps_map.header.frame_id = f'map/{colour}'
                self._ring_pub.publish(ps_map)

            self._publish_markers()

        try:
            self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except CvBridgeError:
            pass

    # ------------------------------------------------------------------
    def _detect_ring_candidates(self, mask: np.ndarray, bgr_original: np.ndarray, debug: np.ndarray) -> tuple[list[tuple[tuple[int, int], np.ndarray, tuple]], int]:
        """
        Detect rings via ellipse fitting on a binary depth mask.
        Draws all fitted ellipses and paired rings onto debug image.

        Returns (results, n_ellipses) where results is list of ((cx, cy), colour_patch_bgr, outer_ellipse)
        """
        h, w = mask.shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Publish contour debug
        try:
            contour_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
            self._contour.publish(self.bridge.cv2_to_imgmsg(contour_vis, 'bgr8'))
        except CvBridgeError:
            pass

        # Cap contour count to keep processing tractable
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:500]

        # Fit ellipses
        elps = []
        for cnt in contours:
            if cnt.shape[0] < MIN_CONTOUR_PTS:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            a, b = ellipse[1]
            if a < 1e-3 or b < 1e-3:
                continue
            ratio = a / b if a > b else b / a
            if ratio <= RATIO_THR and ECC_MIN < a < ECC_THR and ECC_MIN < b < ECC_THR:
                elps.append(ellipse)

        # Draw ALL fitted ellipses in thin gray on debug
        for e in elps:
            cv2.ellipse(debug, e, (128, 128, 128), 1)

        # Validate each ellipse with hole check + colour
        results = []
        for ellipse in elps:
            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])

            # --- Hole check: a ring has an empty interior in the depth mask ---
            inner_ell = (ellipse[0], (ellipse[1][0] * 0.6, ellipse[1][1] * 0.6), ellipse[2])
            inner_mask = np.zeros_like(mask)
            cv2.ellipse(inner_mask, inner_ell, 255, -1)
            hole_pixels = mask[inner_mask > 0]
            if len(hole_pixels) == 0:
                continue
            hole_fill_ratio = np.count_nonzero(hole_pixels) / len(hole_pixels)

            if hole_fill_ratio > 0.4:
                cv2.putText(debug, f'solid{hole_fill_ratio:.0%}', (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                continue

            # --- Colour check on BGR patch ---
            size = int((ellipse[1][0] + ellipse[1][1]) / 2)
            half = max(size // 2, 5)
            x1 = max(cx - half, 0)
            x2 = min(cx + half, w)
            y1 = max(cy - half, 0)
            y2 = min(cy + half, h)
            patch = bgr_original[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            colour, frac = classify_colour(patch)
            if colour == 'unknown':
                cv2.putText(debug, f'hole ok,no clr {frac:.0%}', (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
                continue

            # Draw confirmed ring candidate in green
            cv2.ellipse(debug, ellipse, (0, 255, 0), 2)
            cv2.putText(debug, f'RING:{colour} h{hole_fill_ratio:.0%}', (cx - 20, cy - int(ellipse[1][1] / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            results.append(((cx, cy), patch, ellipse))

        return results, len(elps)

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
