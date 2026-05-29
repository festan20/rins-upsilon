"""Face detector node.

Detects face posters using YOLOv8 person detection on OAK-D RGB + depth,
transforms to map frame and deduplicates.

Published topics
----------------
/detected_faces       (geometry_msgs/PointStamped)  — one per NEW unique face
/face_markers         (visualization_msgs/MarkerArray) — RViz visualisation
/face_detector/debug  (sensor_msgs/Image) — annotated BGR frame
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import cv2

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO

from upsilon.perception_utils import (
    decode_compressed_depth, DepthCameraGeometry, TF2Helper, IncrementalTrackManager,
    MapBoundsTracker,
)


class FaceDetectorNode(Node):

    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameters(namespace='', parameters=[('device', '')])
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.tf2 = TF2Helper(self)
        self.depth_cam = DepthCameraGeometry(patch_radius=10)
        self.tracker = IncrementalTrackManager(merge_distance=0.8)
        self.map_bounds = MapBoundsTracker(self)

        self.detection_color = (0, 0, 255)
        self._latest_bgr = None
        self._faces = []  # list of (cx, cy) from latest RGB frame
        self._depth_frame_id = 'camera_depth_optical_frame'

        self._last_process_time = 0.0
        self._process_interval = 1.0 / 5.0  # 5 Hz rate limit

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers — compressed topics to minimise WiFi bandwidth
        self.create_subscription(CompressedImage, '/gemini/color/image_raw/compressed', self._rgb_cb, qos)
        self.create_subscription(CompressedImage, '/gemini/depth/image_raw/compressedDepth', self._depth_cb, qos)
        self.create_subscription(CameraInfo, '/gemini/depth/camera_info', self._caminfo_cb, qos)

        # Publishers
        self._face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/face_markers', ReliabilityPolicy.BEST_EFFORT)
        self._debug_pub = self.create_publisher(Image, '/face_detector/debug', 10)

        self.model = YOLO("yolov8n.pt")

        self.get_logger().info('Face detector ready.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: CompressedImage) -> None:
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
            return

        self._faces = []
        img_h = cv_image.shape[0]
        # Faces are only in the bottom 65% of the frame — ignore any detection
        # whose centre is above this row (i.e. in the top 35%).
        bottom_half_y = int(img_h * 0.35)

        # Run YOLO inference
        res = self.model.predict(
            cv_image, imgsz=(256, 320), show=False, verbose=False,
            classes=[0], device=self.device
        )

        for x in res:
            bbox = x.boxes.xyxy
            if bbox.nelement() == 0:
                continue

            bbox = bbox[0]

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            if cy < bottom_half_y:
                continue  # top half — not a face

            # Draw bounding box
            cv_image = cv2.rectangle(
                cv_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                self.detection_color, 3
            )
            # Draw center
            cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)
            self._faces.append((cx, cy))

        # Status text
        cv2.putText(
            cv_image,
            f'Faces: {len(self._faces)} | Tracked: {self.tracker.track_count}',
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Store for later use and publish debug image
        self._latest_bgr = cv_image

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            self._debug_pub.publish(debug_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Debug publish error: {e}')

    # ------------------------------------------------------------------
    def _caminfo_cb(self, msg: CameraInfo) -> None:
        self.depth_cam.update_intrinsics(msg)
        self._depth_frame_id = msg.header.frame_id

    def _depth_cb(self, msg: CompressedImage) -> None:
        if not self._faces:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

        try:
            self._depth_cb_inner(msg)
        except Exception as e:
            self.get_logger().error(f'Face detection error (recovering): {e}')

    def _depth_cb_inner(self, msg: CompressedImage) -> None:
        depth_m = decode_compressed_depth(msg)
        if depth_m is None:
            self.get_logger().error('Failed to decode compressedDepth')
            return

        self.depth_cam.update_depth(depth_m)

        for cx, cy in self._faces:
            pt_3d = self.depth_cam.get_point(cx, cy)
            if pt_3d is None:
                self.get_logger().info('Face detected but no valid depth in patch.')
                continue

            # Build PointStamped in camera frame
            pt = PointStamped()
            pt.header.frame_id = self._depth_frame_id
            pt.header.stamp = msg.header.stamp
            pt.point.x = pt_3d[0]
            pt.point.y = pt_3d[1]
            pt.point.z = pt_3d[2]

            # Transform to map frame
            map_pt = self.tf2.transform_point(pt, 'map')
            if map_pt is None:
                self.get_logger().warn(f'TF failed: {self._depth_frame_id} -> map')
                continue

            mx = map_pt.point.x
            my = map_pt.point.y
            mz = map_pt.point.z

            if not self.map_bounds.is_in_bounds(mx, my):
                self.get_logger().info(
                    f'Face off-map ({mx:.2f}, {my:.2f}) — skipped',
                    throttle_duration_sec=2.0)
                continue

            # Deduplicate
            track_id, is_new = self.tracker.update(mx, my)
            count = self.tracker.get_count(track_id)

            # Publish on every detection: frame_id = map/<track_id>/<count>
            det = PointStamped()
            det.header.frame_id = f'map/{track_id}/{count}'
            det.header.stamp = msg.header.stamp
            det.point.x = mx
            det.point.y = my
            det.point.z = mz
            self._face_pub.publish(det)

            if is_new:
                self.get_logger().info(
                    f'NEW FACE #{track_id} at map ({mx:.2f}, {my:.2f}, {mz:.2f})'
                )
            else:
                self.get_logger().info(
                    f'Face #{track_id} count={count}',
                    throttle_duration_sec=2.0)

        self._publish_markers()

    def _publish_markers(self) -> None:
        """Republish markers for ALL tracked faces."""
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for track in self.tracker._tracks:
            # Sphere marker
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'faces'
            m.id = track['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = track['x']
            m.pose.position.y = track['y']
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            arr.markers.append(m)

            # Text label with count
            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = now
            t.ns = 'face_labels'
            t.id = track['id']
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = track['x']
            t.pose.position.y = track['y']
            t.pose.position.z = 0.7
            t.pose.orientation.w = 1.0
            t.scale.z = 0.15
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f'face (n={track["count"]})'
            arr.markers.append(t)
        self._marker_pub.publish(arr)


def main():
    print('Face detection node starting.')
    rclpy.init(args=None)
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
