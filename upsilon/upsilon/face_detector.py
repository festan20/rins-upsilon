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
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO

from upsilon.perception_utils import TF2Helper, IncrementalTrackManager


class FaceDetectorNode(Node):

    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameters(namespace='', parameters=[('device', '')])
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.8)

        self.detection_color = (0, 0, 255)
        self._latest_bgr = None
        self._faces = []  # list of (cx, cy) from latest RGB frame

        self._last_process_time = 0.0
        self._process_interval = 1.0 / 5.0  # 5 Hz rate limit

        qos = qos_profile_sensor_data

        # Subscribers
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)

        # Publishers
        self._face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/face_markers', 10)
        self._debug_pub = self.create_publisher(Image, '/face_detector/debug', 10)

        self.model = YOLO("yolov8n.pt")

        self.get_logger().info('Face detector ready.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
            return

        self._faces = []

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

            # Draw bounding box
            cv_image = cv2.rectangle(
                cv_image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                self.detection_color, 3
            )

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

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
    def _cloud_cb(self, msg: PointCloud2) -> None:
        if not self._faces:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

        try:
            self._cloud_cb_inner(msg)
        except Exception as e:
            self.get_logger().error(f'Face detection error (recovering): {e}')

    def _cloud_cb_inner(self, msg: PointCloud2) -> None:
        height = msg.height
        width = msg.width

        # Get 3D points from pointcloud
        a = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        for cx, cy in self._faces:
            # Bounds check
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                continue

            d = a[cy, cx, :]

            # Check for valid depth
            if not (np.isfinite(d[0]) and np.isfinite(d[1]) and np.isfinite(d[2]) and d[2] > 0):
                self.get_logger().info('Face detected but no valid depth.')
                continue

            # Build PointStamped in camera frame
            pt = PointStamped()
            pt.header.frame_id = msg.header.frame_id
            pt.header.stamp = msg.header.stamp
            pt.point.x = float(d[0])
            pt.point.y = float(d[1])
            pt.point.z = float(d[2])

            # Transform to map frame
            map_pt = self.tf2.transform_point(pt, 'map')
            if map_pt is None:
                self.get_logger().info('Face detected but TF to map failed.')
                continue

            mx = map_pt.point.x
            my = map_pt.point.y
            mz = map_pt.point.z

            # Deduplicate
            track_id, is_new = self.tracker.update(mx, my)

            if is_new:
                self.get_logger().info(
                    f'NEW FACE #{track_id} at map ({mx:.2f}, {my:.2f}, {mz:.2f})'
                )
                # Publish detection
                det = PointStamped()
                det.header.frame_id = 'map'
                det.header.stamp = msg.header.stamp
                det.point.x = mx
                det.point.y = my
                det.point.z = mz
                self._face_pub.publish(det)

        self._publish_markers()

    def _publish_markers(self) -> None:
        """Republish markers for ALL tracked faces."""
        arr = MarkerArray()
        for track in self.tracker._tracks:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
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
