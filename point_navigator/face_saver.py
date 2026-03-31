import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.points = []
        self.subscription = self.create_subscription(
            Marker,
            'people_marker',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def transform_point(self, marker):
        try:
            point_camera = PointStamped()
            point_camera.header.frame_id = marker.header.frame_id  # Usually 'camera'
            point_camera.header.stamp = self.get_clock().now().to_msg()
            point_camera.point = marker.pose.position  # Copy position
            # Wait for transform from camera frame to map frame
            transform = self.tf_buffer.lookup_transform(
                'map',  # target frame
                'turtlebot4/oakd_rgb_camera_frame/rgbd_camera',  # source frame
                rclpy.time.Time()
            )

            # Transform the point
            point_map = do_transform_point(point_camera, transform)
            return point_map.point
        except Exception as e:
            self.get_logger().error(f"Transformation failed: {str(e)}")
            return None

    def listener_callback(self, msg):
        #self.get_logger().info(f"I heard: {msg.pose.position}")
        newPoint = self.transform_point(msg)
        if newPoint is None:
            return
        isNew = True
        for p in self.points:
            dist = (p[0] - newPoint.x)**2 + (p[1] - newPoint.y)**2
            if dist < 2:
                isNew = False
                break
        if isNew:
            self.points.append((newPoint.x, newPoint.y, newPoint.z))
            self.get_logger().info(f"added new point")
            self.get_logger().info(f"{len(self.points)}")
            f = open("face_points.txt", "a")
            f.write(f"{(newPoint.x, newPoint.y, newPoint.z)}\n")

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
