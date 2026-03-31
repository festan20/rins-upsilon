"""Face detector node.

Detects people (face posters) using YOLOv8n, lifts detections to map-frame
3D poses using the OAK-D depth point cloud, deduplicates with
IncrementalTrackManager, and publishes unique detections.

Published topics
----------------
/detected_faces  (geometry_msgs/PointStamped)  — one message per NEW unique face
/face_markers    (visualization_msgs/MarkerArray) — RViz visualisation
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import cv2

from ultralytics import YOLO

from upsilon.perception_utils import DepthCameraGeometry, TF2Helper, IncrementalTrackManager

# Minimum YOLO confidence to consider a detection
CONFIDENCE_THRESHOLD = 0.5
# Marker lifetime (seconds); 0 = forever
MARKER_LIFETIME_S = 0.0


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameter('device', '')
        device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.depth_cam = DepthCameraGeometry(patch_radius=4)
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.8)

        # Latest pixel detections waiting for depth callback
        self._pending_pixels: list[tuple[int, int]] = []
        self._latest_image_stamp = None

        self.model = YOLO('yolov8n.pt')
        self._device = device

        qos = qos_profile_sensor_data

        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)

        self._face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/face_markers', QoSReliabilityPolicy.BEST_EFFORT
        )

        self.get_logger().info('Face detector ready.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
            return

        results = self.model.predict(
            cv_image,
            imgsz=(256, 320),
            show=False,
            verbose=False,
            classes=[0],
            device=self._device,
            conf=CONFIDENCE_THRESHOLD,
        )

        self._pending_pixels = []
        self._latest_image_stamp = msg.header.stamp

        for r in results:
            for box in r.boxes.xyxy:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                self._pending_pixels.append((cx, cy))

    # ------------------------------------------------------------------
    def _cloud_cb(self, msg: PointCloud2) -> None:
        if not self._pending_pixels:
            return

        self.depth_cam.update(msg)

        for cx, cy in self._pending_pixels:
            pt = self.depth_cam.get_point(cx, cy)
            if pt is None:
                continue

            # Build a PointStamped in camera frame and transform to map
            ps = PointStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x, ps.point.y, ps.point.z = pt

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                self.get_logger().warn('TF transform to map failed; skipping.')
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            track_id, is_new = self.tracker.update(mx, my)

            if is_new:
                self.get_logger().info(
                    f'New face #{track_id} at map ({mx:.2f}, {my:.2f})'
                )
                self._face_pub.publish(ps_map)

            self._publish_markers()

        self._pending_pixels = []

    # ------------------------------------------------------------------
    def _publish_markers(self) -> None:
        arr = MarkerArray()
        for i, track in enumerate(self.tracker._tracks):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'faces'
            m.id = track['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = track['x']
            m.pose.position.y = track['y']
            m.pose.position.z = 1.5  # approximate face height
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
