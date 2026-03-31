"""Non-blocking speech synthesis via espeak."""

import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SpeechNode(Node):
    def __init__(self):
        super().__init__('speech')
        self._sub = self.create_subscription(String, '/speech', self._cb, 10)
        self.get_logger().info('Speech node ready.')

    def _cb(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return
        self.get_logger().info(f'Speaking: "{text}"')
        subprocess.Popen(
            ['espeak', text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main():
    rclpy.init()
    node = SpeechNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
