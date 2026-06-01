"""Ring detector node.

Detects coloured ring posters using RGB edge detection + HSV colour
filtering, then uses depth only for 3D localisation in the map frame.

Pipeline
--------
  RGB → CLAHE (optional)
      → HSV colour mask  ──┐
      → Canny edges        ├─► bitwise_and → contours → fitEllipse
                           │                              │
                           │           hole check (colour mask)
                           ▼
                    annulus colour
                           │
                           ▼
                  depth used ONLY for 3D pose

Published topics
----------------
/detected_rings           (geometry_msgs/PointStamped)
/ring_markers             (visualization_msgs/MarkerArray)
/ring_detector/debug      (sensor_msgs/Image) — annotated BGR frame
/ring_detector/threshold  (sensor_msgs/Image) — combined edge+colour mask
/ring_detector/contour    (sensor_msgs/Image) — contour overlay
"""

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
    decode_compressed_depth, DepthCameraGeometry, TF2Helper, IncrementalTrackManager
)

# ---------------------------------------------------------------------------
# Ellipse filter thresholds
# ---------------------------------------------------------------------------
ECC_THR = 200           # max axis length in pixels
ECC_MIN = 8             # min axis length
RATIO_THR = 3.5         # max aspect ratio — relaxed for perspective distortion
MIN_CONTOUR_PTS = 10
MAX_CONTOURS = 80

# Hole check (uses colour mask — solid coloured blob inside ellipse → not a ring)
HOLE_FILL_THR = 0.40    # max fraction of ring-coloured pixels inside the ring hole
HOLE_MIN_PIXELS = 30

# Depth range filter (post-detection — reject rings too close or too far)
DEPTH_MIN = 0.4
DEPTH_MAX = 3.5

# Pre-detection proximity gate — RGB pixels beyond this depth get blacked out
# before edge/colour detection, so far-background contours never get fitted.
PROXIMITY_MAX = 0.75

# In-frame dedup distance (pixels)
SAME_RING_PX = 20

# Lighting normalisation
USE_CLAHE = True

# ---------------------------------------------------------------------------
# HSV colour ranges  (hue in [0,179] OpenCV convention)
# ---------------------------------------------------------------------------
RED_RANGES = [
    (np.array([0, 100, 80]),   np.array([5, 255, 255])),
    (np.array([170, 100, 80]), np.array([179, 255, 255])),
]
COLOUR_RANGES = [
    ('blue',   np.array([100, 80, 50]),  np.array([130, 255, 255])),
    ('green',  np.array([40, 60, 50]),   np.array([80, 255, 255])),
    ('yellow', np.array([20, 100, 100]), np.array([35, 255, 255])),
    ('orange', np.array([5, 150, 100]),  np.array([20, 255, 255])),
    ('purple', np.array([130, 50, 50]),  np.array([160, 255, 255])),
]
BLACK_LO = np.array([0, 0, 0])
BLACK_HI = np.array([179, 255, 50])

COLOUR_MIN_FRAC = 0.10


def classify_colour(bgr_patch: np.ndarray) -> tuple[str, float]:
    """Return (colour_name, fraction) for the dominant ring colour in a BGR patch."""
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return 'unknown', 0.0

    # Combine both red wrap-around ranges before counting
    red_mask = None
    for lo, hi in RED_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        red_mask = m if red_mask is None else (red_mask | m)
    best_count = int(np.count_nonzero(red_mask))
    best_colour = 'red'

    for name, lo, hi in COLOUR_RANGES:
        mask = cv2.inRange(hsv, lo, hi)
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_colour = name

    black_count = int(np.count_nonzero(cv2.inRange(hsv, BLACK_LO, BLACK_HI)))
    if black_count > best_count:
        best_count = black_count
        best_colour = 'black'

    frac = best_count / total
    if frac < COLOUR_MIN_FRAC:
        return 'unknown', frac
    return best_colour, frac


COLOUR_RGBA = {
    'red':     ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'orange':  ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    'purple':  ColorRGBA(r=0.7, g=0.0, b=0.9, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
    'unknown': ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
}

# Morphological kernels
_EDGE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_COLOUR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_PROXIMITY_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class RingDetectorNode(Node):
    def __init__(self):
        super().__init__('ring_detector')

        self.bridge = CvBridge()
        self.depth_cam = DepthCameraGeometry(patch_radius=4)
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.35)

        self._latest_bgr: np.ndarray | None = None
        self._depth_frame_id = 'camera_depth_optical_frame'
        self._last_process_time = 0.0
        self._process_interval = 1.0 / 5.0  # 5 Hz rate limit

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Compressed topics to minimise WiFi bandwidth
        self.create_subscription(CompressedImage, '/gemini/color/image_raw/compressed', self._rgb_cb, qos)
        self.create_subscription(CompressedImage, '/gemini/depth/image_raw/compressedDepth', self._depth_cb, qos)
        self.create_subscription(CameraInfo, '/gemini/depth/camera_info', self._caminfo_cb, qos)

        self._ring_pub = self.create_publisher(PointStamped, '/detected_rings', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/ring_markers', 10)
        self._debug_pub = self.create_publisher(Image, '/ring_detector/debug', 10)
        self._threshold_pub = self.create_publisher(Image, '/ring_detector/threshold', 10)
        self._contour_pub = self.create_publisher(Image, '/ring_detector/contour', 10)

        self.get_logger().info('Ring detector ready.')

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
    @staticmethod
    def _clahe_equalise(bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel of LAB to normalise lighting."""
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _get_edge_mask(bgr: np.ndarray) -> np.ndarray:
        """Canny edges with auto-thresholding from image median."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
        median = float(np.median(blurred))
        lo = max(0, int(median * 0.5))
        hi = min(255, int(median * 1.5))
        edges = cv2.Canny(blurred, lo, hi)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, _EDGE_KERNEL)
        return edges

    @staticmethod
    def _get_colour_mask(bgr: np.ndarray) -> np.ndarray:
        """HSV mask of all known ring colours (red, blue, green, yellow, orange, purple, black)."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_RANGES:
            combined |= cv2.inRange(hsv, lo, hi)
        for _, lo, hi in COLOUR_RANGES:
            combined |= cv2.inRange(hsv, lo, hi)
        combined |= cv2.inRange(hsv, BLACK_LO, BLACK_HI)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _COLOUR_KERNEL)
        combined = cv2.morphologyEx(combined, cv2.MORPH_DILATE, _COLOUR_KERNEL)
        return combined

    # ------------------------------------------------------------------
    def _depth_cb_inner(self, msg: CompressedImage) -> None:
        bgr_raw = self._latest_bgr

        # Decode depth — used for 3D point extraction AND the proximity gate below
        depth_m = decode_compressed_depth(msg)
        if depth_m is None:
            self.get_logger().error('Failed to decode compressedDepth')
            return
        self.depth_cam.update_depth(depth_m)

        # Pre-process RGB for lighting
        bgr = self._clahe_equalise(bgr_raw) if USE_CLAHE else bgr_raw

        # Proximity gate: black out RGB pixels whose depth says they're farther
        # than PROXIMITY_MAX (or have no valid depth). This kills far-background
        # contours before they ever reach fitEllipse. Dilate so ring edges
        # (which sit on a depth boundary) aren't clipped.
        proximity_mask = ((depth_m > DEPTH_MIN) & (depth_m < PROXIMITY_MAX)).astype(np.uint8) * 255
        proximity_mask = cv2.dilate(proximity_mask, _PROXIMITY_KERNEL)
        if proximity_mask.shape != bgr.shape[:2]:
            proximity_mask = cv2.resize(
                proximity_mask, (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        bgr = cv2.bitwise_and(bgr, bgr, mask=proximity_mask)

        # Build RGB-based detection mask (now operating only on near pixels)
        colour_mask = self._get_colour_mask(bgr)
        edges = self._get_edge_mask(bgr)
        combined = cv2.bitwise_and(edges, colour_mask)

        # Rings are always near the top of the frame — zero the bottom 65%
        # so we never even consider contours down there.
        h = combined.shape[0]
        combined[int(h * 0.35):, :] = 0

        if self._threshold_pub.get_subscription_count() > 0:
            try:
                self._threshold_pub.publish(self.bridge.cv2_to_imgmsg(combined, 'mono8'))
            except CvBridgeError:
                pass

        debug = bgr_raw.copy()  # draw on the un-equalised image so colours look natural
        candidates, n_ellipses = self._detect_ring_candidates(
            combined, colour_mask, bgr, debug)

        # Status text
        status = f'ellipses:{n_ellipses} rings:{len(candidates)}'
        cv2.putText(debug, status, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        any_published = False
        for (cx, cy), colour, frac, outer_ellipse in candidates:
            # Sample depth at 12 points just outside the ring (1.05x outer axis)
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

            # Skip unknown-colour candidates only AFTER drawing depth samples
            # so you can see they were considered.
            if colour == 'unknown':
                cv2.putText(debug, 'unknown', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            if len(valid_pts) < 3:
                cv2.putText(debug, f'{colour} low depth ({len(valid_pts)})',
                            (cx, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            pts_array = np.array(valid_pts)
            med_x = float(np.median(pts_array[:, 0]))
            med_y = float(np.median(pts_array[:, 1]))
            med_z = float(np.median(pts_array[:, 2]))

            if med_z < DEPTH_MIN or med_z > DEPTH_MAX:
                cv2.putText(debug, f'{colour} bad z={med_z:.1f}', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            ps = PointStamped()
            ps.header.frame_id = self._depth_frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = med_x, med_y, med_z

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                cv2.putText(debug, f'{colour} no TF', (cx, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            track_id, is_new = self.tracker.update(mx, my, colour)
            count = self.tracker.get_count(track_id)

            # This is a confirmed ring — draw the outer ellipse in its colour
            rgba = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
            bgr_colour = (int(rgba.b * 255), int(rgba.g * 255), int(rgba.r * 255))
            cv2.ellipse(debug, outer_ellipse, bgr_colour, 2)

            label = f'{colour} {frac:.0%} #{track_id} n={count}'
            cv2.putText(debug, label, (cx, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_colour, 2)
            ps_map.header.frame_id = f'map/{colour}/{track_id}/{count}'
            self._ring_pub.publish(ps_map)
            any_published = True

            if is_new:
                self.get_logger().info(
                    f'New ring #{track_id} colour={colour} at map ({mx:.2f}, {my:.2f})'
                )
            else:
                self.get_logger().info(
                    f'Ring #{track_id} colour={colour} count={count}',
                    throttle_duration_sec=2.0)

        if any_published:
            self._publish_markers()

        if self._debug_pub.get_subscription_count() > 0:
            try:
                self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
            except CvBridgeError:
                pass

    # ------------------------------------------------------------------
    def _detect_ring_candidates(self, edge_mask: np.ndarray, colour_mask: np.ndarray,
                                bgr: np.ndarray, debug: np.ndarray):
        """Detect rings via ellipse fitting + colour-mask hole check + annulus colour.

        Draws every step to ``debug`` for visual debugging:
          gray ellipse  = fitted from a contour
          red 'solidX%' = rejected by hole check
          cyan circles  = annulus colour-sampling boundaries
          green ellipse = passed hole + has a colour label

        Returns (results, n_ellipses) where results is list of
            ((cx, cy), colour, frac, ellipse).
        """
        h, w = edge_mask.shape[:2]

        contours, _ = cv2.findContours(edge_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if self._contour_pub.get_subscription_count() > 0:
            try:
                contour_vis = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 1)
                self._contour_pub.publish(self.bridge.cv2_to_imgmsg(contour_vis, 'bgr8'))
            except CvBridgeError:
                pass

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_CONTOURS]

        # Fit ellipses
        elps = []
        n_too_few = 0
        n_too_big = 0
        n_too_small = 0
        n_bad_ratio = 0
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

        # Draw every fitted ellipse in gray so you can see what was considered
        for e in elps:
            cv2.ellipse(debug, e, (128, 128, 128), 1)

        results = []
        for ellipse in elps:
            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])

            # In-frame dedup
            if any(abs(cx - ex) < SAME_RING_PX and abs(cy - ey) < SAME_RING_PX
                   for (ex, ey), _, _, _ in results):
                continue

            r_max = int(max(ellipse[1][0], ellipse[1][1]) / 2)

            pad = int(r_max * 1.2) + 2
            roi_x1 = max(cx - pad, 0)
            roi_x2 = min(cx + pad, w)
            roi_y1 = max(cy - pad, 0)
            roi_y2 = min(cy + pad, h)
            roi_w = roi_x2 - roi_x1
            roi_h = roi_y2 - roi_y1
            if roi_w < 5 or roi_h < 5:
                continue

            rc = (ellipse[0][0] - roi_x1, ellipse[0][1] - roi_y1)

            # Hole check using the colour mask: a real ring has an empty (non-coloured) centre.
            # If the inside is mostly ring-coloured, this is a solid blob — reject.
            inner_ell_roi = (rc, (ellipse[1][0] * 0.6, ellipse[1][1] * 0.6), ellipse[2])
            inner_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.ellipse(inner_roi, inner_ell_roi, 255, -1)
            colour_roi = colour_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            hole_pixels = colour_roi[inner_roi > 0]
            if len(hole_pixels) < HOLE_MIN_PIXELS:
                continue
            hole_fill_ratio = np.count_nonzero(hole_pixels) / len(hole_pixels)
            if hole_fill_ratio > HOLE_FILL_THR:
                cv2.putText(debug, f'solid{hole_fill_ratio:.0%}', (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            # Annulus colour (0.65x → 0.95x of the outer axes) — sample ring body in RGB
            inner_colour_roi = (rc, (ellipse[1][0] * 0.65, ellipse[1][1] * 0.65), ellipse[2])
            outer_colour_roi = (rc, (ellipse[1][0] * 0.95, ellipse[1][1] * 0.95), ellipse[2])
            annulus_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)
            cv2.ellipse(annulus_roi, outer_colour_roi, 255, -1)
            cv2.ellipse(annulus_roi, inner_colour_roi, 0, -1)

            bgr_roi = bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            ring_pixels = bgr_roi[annulus_roi > 0]
            if len(ring_pixels) < 20:
                continue

            patch = ring_pixels.reshape(1, -1, 3)
            colour, frac = classify_colour(patch)

            # Visualise the candidate (passes hole check) regardless of colour outcome
            inner_col_ell = (ellipse[0], (ellipse[1][0] * 0.65, ellipse[1][1] * 0.65), ellipse[2])
            outer_col_ell = (ellipse[0], (ellipse[1][0] * 0.95, ellipse[1][1] * 0.95), ellipse[2])
            cv2.ellipse(debug, ellipse, (0, 255, 0), 2)
            cv2.ellipse(debug, inner_col_ell, (255, 255, 0), 1)
            cv2.ellipse(debug, outer_col_ell, (255, 255, 0), 1)
            cv2.putText(debug, f'{colour} h{hole_fill_ratio:.0%} c{frac:.0%}',
                        (cx - 20, cy - int(ellipse[1][1] / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            results.append(((cx, cy), colour, frac, ellipse))

        return results, len(elps)

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
            m.color = COLOUR_RGBA.get(colour, COLOUR_RGBA['unknown'])
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
    node = RingDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
