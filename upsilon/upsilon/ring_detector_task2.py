"""Ring detector (Task 2).

Detects coloured ring posters using ellipse fitting (adapted from
dis_tutorial5) on a depth-foreground mask, combined with HSV colour
classification and the OAK-D organized point cloud for 3D localisation in the
map frame.

Only the top part of the frame is searched and the ring centre (hole) must be
empty in the depth-foreground mask — a ring suspended in the air shows the far
background through its hole, whereas a flat disc (or a ring painted on a near
wall) fills the hole and is rejected as "solid".

Published topics
----------------
/detected_rings_task2        (geometry_msgs/PointStamped)  — frame_id encodes
                             colour/track: "map/<colour>/<id>/<count>"
/ring_markers_task2          (visualization_msgs/MarkerArray) — RViz visualisation
/ring_detector_task2/debug   (sensor_msgs/Image) — annotated BGR frame
/ring_detector_task2/threshold (sensor_msgs/Image) — depth-foreground mask
/ring_detector_task2/contour (sensor_msgs/Image) — contour overlay
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
)

import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

from upsilon.perception_utils import DepthCameraGeometry, TF2Helper, IncrementalTrackManager

_LATCHED_QOS = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# ---------------------------------------------------------------------------
# Ellipse filter thresholds
# ---------------------------------------------------------------------------
ECC_THR = 200           # max axis length in pixels
ECC_MIN = 8             # min axis length — reject tiny noise ellipses
RATIO_THR = 2.5         # max aspect ratio — relaxed to handle perspective
CENTER_THR = 15         # max pixel distance between ellipse centres
MIN_CONTOUR_PTS = 10    # min contour points for ellipse fitting
DEPTH_MAX = 3.0         # max depth in metres for foreground mask

# ---------------------------------------------------------------------------
# HSV colour ranges  (hue in [0,179] OpenCV convention)
# ---------------------------------------------------------------------------
COLOUR_RANGES = [
    ('red',    np.array([0, 100, 80]),   np.array([5, 255, 255])),
    ('red',    np.array([170, 100, 80]), np.array([179, 255, 255])),
    ('blue',   np.array([100, 80, 50]),  np.array([130, 255, 255])),
    ('green',  np.array([40, 60, 50]),   np.array([80, 255, 255])),
    ('yellow', np.array([20, 100, 100]), np.array([35, 255, 255])),
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
    'red':     ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
    'unknown': ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
}


class RingDetectorTask2Node(Node):
    def __init__(self):
        super().__init__('ring_detector_task2')

        self.bridge = CvBridge()
        self.depth_cam = DepthCameraGeometry(patch_radius=4)
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.8)

        self._latest_bgr: np.ndarray | None = None
        self._latest_stamp = None
        self._last_process_time = 0.0
        self._process_interval = 1.0 / 5.0  # 5 Hz rate limit

        qos = qos_profile_sensor_data

        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)

        self._ring_pub = self.create_publisher(PointStamped, '/detected_rings_task2', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/ring_markers_task2', 10)
        self._debug_pub = self.create_publisher(Image, '/ring_detector_task2/debug', 10)
        self._treshold = self.create_publisher(Image, '/ring_detector_task2/threshold', 10)
        self._contour = self.create_publisher(Image, '/ring_detector_task2/contour', 10)

        self._markers_enabled = True
        self.create_subscription(Bool, '/markers_enabled', self._markers_enabled_cb, _LATCHED_QOS)

        self.get_logger().info('Ring detector (Task 2) ready.')
        self._watchdog = self.create_timer(20.0, self._startup_watchdog)

    def _startup_watchdog(self) -> None:
        self._watchdog.cancel()
        if self._latest_bgr is None:
            self.get_logger().error(
                'STARTUP FAILURE: no RGB image received after 20 s. '
                'Camera topics are not connecting — restart the node.')
        else:
            self.get_logger().info('Startup watchdog OK — camera data is flowing.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image) -> None:
        try:
            self._latest_bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._latest_stamp = msg.header.stamp
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    # ------------------------------------------------------------------
    def _cloud_cb(self, msg: PointCloud2) -> None:
        if self._latest_bgr is None:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

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

        for (cx, cy), colour, outer_ellipse in candidates:
            # Sample depth at 12 points on the ring body (1.05x = midpoint of 1.0-1.1x annulus)
            a_axis = outer_ellipse[1][0] / 2 * 1.05
            b_axis = outer_ellipse[1][1] / 2 * 1.05
            angle_rad = np.radians(outer_ellipse[2])
            ecx = int(outer_ellipse[0][0])
            ecy = int(outer_ellipse[0][1])

            n_samples = 12
            valid_pts = []

            for i in range(n_samples):
                theta = 2 * np.pi * i / n_samples
                local_x = a_axis * np.cos(theta)
                local_y = b_axis * np.sin(theta)
                sx = int(ecx + local_x * np.cos(angle_rad) - local_y * np.sin(angle_rad))
                sy = int(ecy + local_x * np.sin(angle_rad) + local_y * np.cos(angle_rad))

                pt = self.depth_cam.get_point(sx, sy)
                if pt is not None:
                    valid_pts.append(pt)
                    cv2.circle(debug, (sx, sy), 2, (0, 255, 0), -1)
                else:
                    cv2.circle(debug, (sx, sy), 2, (0, 0, 255), -1)

            if len(valid_pts) < 3:
                cv2.putText(debug, f'{colour} low depth ({len(valid_pts)})',
                            (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            # Median to reject outliers
            pts_array = np.array(valid_pts)
            med_x = float(np.median(pts_array[:, 0]))
            med_y = float(np.median(pts_array[:, 1]))
            med_z = float(np.median(pts_array[:, 2]))

            ps = PointStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = med_x, med_y, med_z

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                cv2.putText(debug, f'{colour} no TF', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            if colour == 'unknown':
                continue
            track_id, is_new = self.tracker.update(mx, my, colour)

            # Draw colour label on debug image
            rgba = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
            bgr_colour = (int(rgba.b * 255), int(rgba.g * 255), int(rgba.r * 255))
            cv2.putText(debug, f'{colour} ({mx:.1f},{my:.1f})', (cx, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_colour, 2)

            count = self.tracker.get_count(track_id)

            # frame_id format: map/<colour>/<track_id>/<count>
            ps_map.header.frame_id = f'map/{colour}/{track_id}/{count}'
            self._ring_pub.publish(ps_map)

            if is_new:
                self.get_logger().info(
                    f'New ring #{track_id} colour={colour} at map ({mx:.2f}, {my:.2f})'
                )
            else:
                self.get_logger().info(
                    f'Ring #{track_id} colour={colour} count={count}',
                    throttle_duration_sec=2.0)

            self._publish_markers()

        try:
            self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except CvBridgeError:
            pass

    # ------------------------------------------------------------------
    def _detect_ring_candidates(self, mask: np.ndarray, bgr_original: np.ndarray, debug: np.ndarray):
        """
        Detect rings via ellipse fitting + hole check + annulus colour.

        Returns (results, n_ellipses) where results is a list of
            ((cx, cy), colour, ellipse)
        """
        h, w = mask.shape[:2]

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Publish contour debug
        try:
            contour_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
            self._contour.publish(self.bridge.cv2_to_imgmsg(contour_vis, 'bgr8'))
        except CvBridgeError:
            pass

        # Cap contour count
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:500]

        # Fit ellipses
        elps = []
        n_too_few = n_too_big = n_too_small = n_bad_ratio = 0
        for cnt in contours:
            if cnt.shape[0] < MIN_CONTOUR_PTS:
                n_too_few += 1
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            a, b = ellipse[1]
            if a < 1e-3 or b < 1e-3:
                continue
            ratio = a / b if a > b else b / a
            if a >= ECC_THR or b >= ECC_THR:
                n_too_big += 1
            elif a <= ECC_MIN or b <= ECC_MIN:
                n_too_small += 1
            elif ratio > RATIO_THR:
                n_bad_ratio += 1
            else:
                elps.append(ellipse)

        self.get_logger().info(
            f'Contours:{len(contours)} too_few_pts:{n_too_few} '
            f'too_big:{n_too_big} too_small:{n_too_small} bad_ratio:{n_bad_ratio} '
            f'valid_ellipses:{len(elps)}',
            throttle_duration_sec=2.0)

        # Draw ALL fitted ellipses in thin gray
        for e in elps:
            cv2.ellipse(debug, e, (128, 128, 128), 1)

        # Validate each ellipse with hole check + annulus colour
        results = []
        for ellipse in elps:
            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])
            r_max = int(max(ellipse[1][0], ellipse[1][1]) / 2)

            # ROI bounding box (padded for 1.1x outer)
            pad = int(r_max * 1.2) + 2
            roi_x1 = max(cx - pad, 0)
            roi_x2 = min(cx + pad, w)
            roi_y1 = max(cy - pad, 0)
            roi_y2 = min(cy + pad, h)
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            if roi_w < 5 or roi_h < 5:
                continue

            # Ellipse center in ROI coordinates
            rc = (ellipse[0][0] - roi_x1, ellipse[0][1] - roi_y1)

            # --- Hole check on ROI ---
            inner_ell_roi = (rc, (ellipse[1][0] * 0.6, ellipse[1][1] * 0.6), ellipse[2])
            inner_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.ellipse(inner_roi, inner_ell_roi, 255, -1)
            mask_roi = mask[roi_y1:roi_y2, roi_x1:roi_x2]
            hole_pixels = mask_roi[inner_roi > 0]
            if len(hole_pixels) == 0:
                continue
            hole_fill_ratio = np.count_nonzero(hole_pixels) / len(hole_pixels)

            if hole_fill_ratio > 0.4:
                cv2.putText(debug, f'solid{hole_fill_ratio:.0%}', (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                continue

            # --- Annulus colour on ROI (0.65x to 0.95x) — ring body between hole and edge ---
            inner_colour_roi = (rc, (ellipse[1][0] * 0.65, ellipse[1][1] * 0.65), ellipse[2])
            outer_colour_roi = (rc, (ellipse[1][0] * 0.95, ellipse[1][1] * 0.95), ellipse[2])
            annulus_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.ellipse(annulus_roi, outer_colour_roi, 255, -1)
            cv2.ellipse(annulus_roi, inner_colour_roi, 0, -1)

            bgr_roi = bgr_original[roi_y1:roi_y2, roi_x1:roi_x2]
            ring_pixels = bgr_roi[annulus_roi > 0]
            if len(ring_pixels) < 20:
                self.get_logger().info(
                    f'Ellipse at ({cx},{cy}) axes=({ellipse[1][0]:.0f},{ellipse[1][1]:.0f}) '
                    f'hole={hole_fill_ratio:.2f} annulus_px={len(ring_pixels)} — too few annulus pixels',
                    throttle_duration_sec=2.0)
                continue

            # Classify colour from ring body pixels
            patch = ring_pixels.reshape(-1, 1, 3)
            colour, frac = classify_colour(patch)

            # Draw fitted ellipse in green, colour sampling annulus in yellow
            inner_col_ell = (ellipse[0], (ellipse[1][0] * 0.65, ellipse[1][1] * 0.65), ellipse[2])
            outer_col_ell = (ellipse[0], (ellipse[1][0] * 0.95, ellipse[1][1] * 0.95), ellipse[2])
            cv2.ellipse(debug, ellipse, (0, 255, 0), 2)
            cv2.ellipse(debug, inner_col_ell, (255, 255, 0), 1)
            cv2.ellipse(debug, outer_col_ell, (255, 255, 0), 1)
            cv2.putText(debug, f'{colour} h{hole_fill_ratio:.0%} c{frac:.0%}',
                        (cx - 20, cy - int(ellipse[1][1] / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            results.append(((cx, cy), colour, ellipse))

        return results, len(elps)

    # ------------------------------------------------------------------
    def _markers_enabled_cb(self, msg: Bool) -> None:
        self._markers_enabled = msg.data
        self.get_logger().info(f'Marker publishing {"enabled" if msg.data else "disabled"}.')

    def _publish_markers(self) -> None:
        if not self._markers_enabled:
            return
        arr = MarkerArray()
        for track in self.tracker.tracks:
            colour = track.get('colour', 'unknown')
            count = track['count']

            # Cylinder marker
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'rings_task2'
            m.id = track['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = track['x']
            m.pose.position.y = track['y']
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.3
            m.scale.z = 0.05
            m.color = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
            arr.markers.append(m)

            # Text marker showing colour and count (black for white-map readability)
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = m.header.stamp
            t.ns = 'ring_labels_task2'
            t.id = track['id']
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = track['x']
            t.pose.position.y = track['y']
            t.pose.position.z = 0.7
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
            t.text = f'{colour} (n={count})'
            arr.markers.append(t)

        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = RingDetectorTask2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
