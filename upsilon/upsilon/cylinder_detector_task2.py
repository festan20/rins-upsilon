"""Cylinder (barrel) detector (Task 2) — colour + upright/fallen orientation.

Barrels are solid coloured cylinders standing on the floor; they may be UPRIGHT
or FALLEN OVER. open3d / PCL are not available, so this is an image + organized
point-cloud method:

  RGB -> per barrel-colour HSV mask -> external contours -> largest SOLID blob
      (reject ring-like / hollow blobs via solidity + fill)
      -> cv2.minAreaRect: long-axis angle vs vertical decides upright vs fallen
      -> IN-FRAME DEDUP: one barrel can match several colour masks (esp. a shaded
         side matching 'black'); merge blobs by pixel proximity, chromatic wins
      -> 3D position = MEDIAN of the organized cloud points INSIDE the blob mask
         (robust — a single centre pixel can land on the curved edge / background)
         pushed back by the barrel radius to the central axis
      -> TF to map -> distance dedup -> CYLINDER marker (laid down if fallen)

Published topics
----------------
/detected_cylinders_task2     (geometry_msgs/PointStamped)
/cylinder_markers_task2       (visualization_msgs/MarkerArray)
/cylinder_detector_task2/debug (sensor_msgs/Image)
/cylinder_detector_task2/threshold (sensor_msgs/Image)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

from upsilon.perception_utils import TF2Helper, IncrementalTrackManager, MapBoundsTracker

# --- blob gates ---
BLOB_AREA_MIN = 600            # px²
BLOB_AREA_MAX = 120000
BLACK_AREA_MIN = 1500          # black is shadow-prone — demand a bigger blob
SOLIDITY_MIN = 0.80            # area / convex-hull area; barrels are solid (rings aren't)
FILL_MIN = 0.55               # area / minAreaRect area
# --- orientation ---
ELONGATION_SQUARE = 1.25       # below this the blob is too square to read a long axis
TILT_VERTICAL_DEG = 45.0       # long-axis tilt from vertical below which = upright
# --- in-frame dedup ---
SAME_BARREL_PX = 45            # blobs whose centres are this close = same barrel
# --- depth / geometry ---
DEPTH_MIN = 0.3
DEPTH_MAX = 4.0
MIN_BLOB_POINTS = 12           # need this many valid cloud points in the blob
BARREL_RADIUS = 0.11           # m — push the front-surface point back to the axis
BARREL_MIN_HEIGHT_M = 0.004    # map-frame z of blob centroid; floor lines are ~0
BARREL_MAX_HEIGHT_M = 0.30     # centroid above this = wall feature / not a barrel
SPILL_MIN_AREA = 50            # min connected px outside barrel rect to call it a spill


COLOUR_BUCKETS = [
    ('red',    [(np.array([0, 100, 60]),   np.array([8, 255, 255])),
                (np.array([170, 100, 60]), np.array([179, 255, 255]))]),
    ('blue',   [(np.array([100, 80, 40]),  np.array([130, 255, 255]))]),
    ('green',  [(np.array([40, 50, 40]),   np.array([85, 255, 255]))]),
    ('yellow', [(np.array([20, 100, 80]),  np.array([35, 255, 255]))]),
    ('black',  [(np.array([0, 0, 0]),      np.array([179, 255, 50]))]),
]
COLOUR_RGBA = {
    'red':     ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
}
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _bgr_for_colour(name: str):
    rgba = COLOUR_RGBA.get(name, ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))
    return (int(rgba.b * 255), int(rgba.g * 255), int(rgba.r * 255))


def _build_colour_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else (mask | m)
    return mask


def _orientation_from_rect(rect):
    """Return ('upright'|'fallen', elongation, tilt_deg) for a minAreaRect.

    tilt_deg = angle of the blob's long axis away from image-vertical.
    """
    (w, h) = rect[1]
    if w < 1e-3 or h < 1e-3:
        return 'upright', 1.0, 0.0
    long_side = max(w, h)
    short_side = min(w, h)
    elong = long_side / short_side

    box = cv2.boxPoints(rect)
    e1 = box[1] - box[0]
    e2 = box[2] - box[1]
    long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    # tilt from vertical: 0deg = perfectly vertical long axis, 90deg = horizontal
    tilt = math.degrees(math.atan2(abs(long_edge[0]), abs(long_edge[1]) + 1e-6))

    if elong < ELONGATION_SQUARE:
        # too square to be sure (end-on view / short barrel) -> assume upright
        return 'upright', elong, tilt
    return ('upright' if tilt < TILT_VERTICAL_DEG else 'fallen'), elong, tilt


class CylinderDetectorTask2Node(Node):
    def __init__(self):
        super().__init__('cylinder_detector_task2')

        self.bridge = CvBridge()
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.4)
        self.map_bounds = MapBoundsTracker(self)
        self._orientations: dict[int, str] = {}  # track_id -> 'upright'|'fallen'
        self._leaking: dict[int, bool] = {}       # track_id -> True if spill ever seen

        self._latest_bgr: np.ndarray | None = None
        self._last_process_time = 0.0
        self._process_interval = 1.0 / 5.0

        qos = qos_profile_sensor_data
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)

        self._cyl_pub = self.create_publisher(PointStamped, '/detected_cylinders_task2', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/cylinder_markers_task2', 10)
        self._debug_pub = self.create_publisher(Image, '/cylinder_detector_task2/debug', 10)
        self._threshold_pub = self.create_publisher(Image, '/cylinder_detector_task2/threshold', 10)
        self._spill_debug_pub = self.create_publisher(Image, '/cylinder_detector_task2/spill_debug', 10)

        self.get_logger().info('Cylinder detector (Task 2) ready — cloud + dedup.')
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
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

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
            self.get_logger().error(f'Cylinder detection error (recovering): {e}')

    # ------------------------------------------------------------------
    def _blob_camera_point(self, xyz: np.ndarray, cnt) -> np.ndarray | None:
        """Median 3D point (camera frame) of cloud points inside the contour.

        Pushed back by BARREL_RADIUS along the ray so it lands on the barrel's
        central axis rather than the front surface. None if too few valid points.
        """
        h, w = xyz.shape[:2]
        blob = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        blob = cv2.erode(blob, _OPEN_KERNEL, iterations=1)  # avoid edge bleed
        pts = xyz[blob == 255]
        if pts.size == 0:
            return None
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < MIN_BLOB_POINTS:
            return None
        # Filter by Euclidean range, not a single axis: the OAK-D cloud is in
        # body convention (x forward, y left, z up), so a barrel on the floor
        # has a NEGATIVE z. Range is convention-independent.
        rng = np.linalg.norm(pts, axis=1)
        pts = pts[(rng > DEPTH_MIN) & (rng < DEPTH_MAX)]
        if len(pts) < MIN_BLOB_POINTS:
            return None
        med = np.median(pts, axis=0)
        d = float(np.linalg.norm(med))
        if d > 1e-3:
            med = med * ((d + BARREL_RADIUS) / d)
        return med

    def _find_spill(self, colour_mask: np.ndarray, barrel_cnt, colour: str):
        """Return (is_spill, debug_bgr).

        The contour may include a merged spill blob, so we can't fit a rect to it
        directly. Instead we erode the filled contour to break the thin barrel-spill
        connection, isolate the largest component (= barrel body), dilate back, then
        fit a rect to that barrel-only region. Pixels outside that rect = spill.
        """
        img_h, img_w = colour_mask.shape[:2]

        # fill the full detected contour (barrel + possibly merged spill)
        filled = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(filled, [barrel_cnt], -1, 255, -1)

        # erode to disconnect spill from barrel body via thin bridges
        kernel = np.ones((7, 7), dtype=np.uint8)
        eroded = cv2.erode(filled, kernel, iterations=2)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
        if n > 1:
            # largest component after erosion = barrel body
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            barrel_body = cv2.dilate(
                (labels == largest).astype(np.uint8) * 255, kernel, iterations=2)
            pts = cv2.findNonZero(barrel_body)
            barrel_rect = cv2.minAreaRect(pts) if pts is not None else cv2.minAreaRect(barrel_cnt)
        else:
            barrel_rect = cv2.minAreaRect(barrel_cnt)

        # erase the barrel-only rect from the colour mask
        rect_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(rect_mask, [np.intp(cv2.boxPoints(barrel_rect))], 255)
        without_barrel = colour_mask.copy()
        without_barrel[rect_mask > 0] = 0

        # search window: one barrel-size margin around bounding box
        bx, by, bw, bh = cv2.boundingRect(barrel_cnt)
        margin = max(bw, bh)
        y1 = max(0, by - margin);  y2 = min(img_h, by + bh + margin)
        x1 = max(0, bx - margin);  x2 = min(img_w, bx + bw + margin)

        threshold_y = by + (bh * 2 // 3)
        above_count = int(np.count_nonzero(without_barrel[y1:threshold_y, x1:x2]))
        below_count = int(np.count_nonzero(without_barrel[threshold_y:y2, x1:x2]))
        total = above_count + below_count
        spill = total >= SPILL_MIN_AREA and (below_count / total) >= 0.75

        # debug image
        below_mask = without_barrel.copy(); below_mask[:threshold_y, :] = 0
        above_mask = without_barrel.copy(); above_mask[threshold_y:, :] = 0
        dbg = cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR)
        dbg[rect_mask > 0]   = (0, 0, 180)    # red    = erased barrel rect
        dbg[above_mask > 0]  = (0, 165, 255)  # orange = surviving above threshold (disqualifies)
        dbg[below_mask > 0]  = (0, 255, 0)    # green  = surviving below threshold (valid spill)
        cv2.line(dbg, (0, threshold_y), (img_w, threshold_y), (0, 255, 255), 1)  # cyan threshold line
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.drawContours(dbg, [barrel_cnt], -1, (255, 255, 0), 1)
        label = f'{colour} SPILL' if spill else f'{colour} no spill'
        cv2.putText(dbg, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return spill, dbg

    def _cloud_cb_inner(self, msg: PointCloud2) -> None:
        bgr = self._latest_bgr
        height, width = msg.height, msg.width
        xyz = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z')).reshape((height, width, 3))

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]
        debug = bgr.copy()
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        # ---- 1) collect raw blob candidates across all colour buckets ----
        colour_masks: dict[str, np.ndarray] = {}
        candidates = []
        for colour, ranges in COLOUR_BUCKETS:
            mask = _build_colour_mask(hsv, ranges)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _OPEN_KERNEL)
            colour_masks[colour] = mask
            combined_mask |= mask

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_min = BLACK_AREA_MIN if colour == 'black' else BLOB_AREA_MIN
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < area_min or area > BLOB_AREA_MAX:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(cnt))
                if hull_area < 1e-3 or area / hull_area < SOLIDITY_MIN:
                    continue
                rect = cv2.minAreaRect(cnt)
                (rcx, rcy), (rw, rh), _ = rect
                rect_area = rw * rh
                if rect_area < 1e-3 or area / rect_area < FILL_MIN:
                    continue
                orientation, elong, tilt = _orientation_from_rect(rect)
                candidates.append({
                    'cx': rcx, 'cy': rcy, 'area': area, 'colour': colour,
                    'is_chromatic': colour != 'black',
                    'cnt': cnt, 'rect': rect, 'orientation': orientation,
                    'elong': elong, 'tilt': tilt,
                })

        # ---- 2) in-frame dedup: chromatic + larger blobs win ----
        candidates.sort(key=lambda c: (c['is_chromatic'], c['area']), reverse=True)
        accepted = []
        for c in candidates:
            if any(math.hypot(c['cx'] - a['cx'], c['cy'] - a['cy']) < SAME_BARREL_PX
                   for a in accepted):
                continue
            accepted.append(c)

        # ---- 3) 3D position (cloud blob median) -> map -> track ----
        any_published = False
        for c in accepted:
            colour, rect = c['colour'], c['rect']
            orientation = c['orientation']
            cbox = np.intp(cv2.boxPoints(rect))
            rcx, rcy = int(c['cx']), int(c['cy'])

            cam_pt = self._blob_camera_point(xyz, c['cnt'])
            if cam_pt is None:
                cv2.drawContours(debug, [cbox], 0, (0, 0, 255), 1)
                cv2.putText(debug, f'{colour} no depth', (rcx, rcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            ps = PointStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = float(cam_pt[0]), float(cam_pt[1]), float(cam_pt[2])
            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                cv2.drawContours(debug, [cbox], 0, (0, 0, 255), 1)
                cv2.putText(debug, f'{colour} no TF', (rcx, rcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            if not (BARREL_MIN_HEIGHT_M <= ps_map.point.z <= BARREL_MAX_HEIGHT_M):
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            if not self.map_bounds.is_in_bounds(mx, my):
                continue

            track_id, is_new = self.tracker.update(mx, my, colour)
            if self._orientations.get(track_id) != 'fallen':
                self._orientations[track_id] = orientation
            effective_orientation = self._orientations[track_id]

            # spill only makes sense for fallen barrels
            spill_now = False
            if effective_orientation == 'fallen':
                spill_now, spill_dbg = self._find_spill(
                    colour_masks[colour], c['cnt'], colour)
                cv2.imshow('Spill Debug', spill_dbg)
                if self._spill_debug_pub.get_subscription_count() > 0:
                    try:
                        self._spill_debug_pub.publish(
                            self.bridge.cv2_to_imgmsg(spill_dbg, 'bgr8'))
                    except CvBridgeError:
                        pass
            was_leaking = self._leaking.get(track_id, False)
            self._leaking[track_id] = spill_now or was_leaking
            is_leaking = self._leaking[track_id]
            count = self.tracker.get_count(track_id)

            leak_tag = '/leaking' if is_leaking else ''
            ps_map.header.frame_id = f'map/{colour}/{orientation}{leak_tag}/{track_id}/{count}'
            self._cyl_pub.publish(ps_map)
            any_published = True

            cv2.drawContours(debug, [cbox], 0, _bgr_for_colour(colour), 2)
            cv2.putText(debug, f'{colour} {effective_orientation} #{track_id} n={count}',
                        (rcx - 20, rcy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        _bgr_for_colour(colour), 2)
            spill_label = 'SPILL' if is_leaking else 'no spill'
            spill_colour = (0, 255, 255) if is_leaking else (180, 180, 180)
            cv2.putText(debug, spill_label, (rcx - 20, rcy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, spill_colour, 2)

            if is_new:
                self.get_logger().info(
                    f'NEW barrel #{track_id} {colour} {orientation} '
                    f"at ({mx:.2f},{my:.2f}) elong={c['elong']:.2f} tilt={c['tilt']:.0f}deg")
            if spill_now and not was_leaking:
                self.get_logger().info(f'SPILL detected near barrel #{track_id} ({colour})')

        cv2.putText(debug, f'barrels: {len(accepted)}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Cylinder detector', debug)
        cv2.imshow('Colour mask', combined_mask)
        cv2.waitKey(1)

        if self._threshold_pub.get_subscription_count() > 0:
            try:
                self._threshold_pub.publish(self.bridge.cv2_to_imgmsg(combined_mask, 'mono8'))
            except CvBridgeError:
                pass
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
            orientation = self._orientations.get(track['id'], 'upright')
            leaking = self._leaking.get(track['id'], False)
            count = track['count']
            fallen = orientation == 'fallen'

            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'cylinders_task2'
            m.id = track['id']
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = track['x']
            m.pose.position.y = track['y']
            m.pose.position.z = 0.15
            if fallen:
                # tip the cylinder 90deg about Y so its axis is horizontal
                m.pose.orientation.x = 0.0
                m.pose.orientation.y = 0.70710678
                m.pose.orientation.z = 0.0
                m.pose.orientation.w = 0.70710678
            else:
                m.pose.orientation.w = 1.0
            m.scale.x = 0.22       # diameter
            m.scale.y = 0.22
            m.scale.z = 0.35       # height
            m.color = COLOUR_RGBA.get(colour, ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0))
            arr.markers.append(m)

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = now
            t.ns = 'cylinder_labels_task2'
            t.id = track['id']
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = track['x']
            t.pose.position.y = track['y']
            t.pose.position.z = 0.5
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color = ColorRGBA(r=0.0, g=0.0, b=0.0, a=1.0)
            leak_str = ' LEAKING' if leaking else ''
            t.text = f'{colour} {orientation}{leak_str} (n={count})'
            arr.markers.append(t)
        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = CylinderDetectorTask2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
