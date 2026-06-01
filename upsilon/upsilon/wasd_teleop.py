"""WASD keyboard teleop — publishes Twist on /cmd_vel.

Keys:
  w / s  — forward / backward
  a / d  — turn left / right
  space  — stop
  q      — quit

Hold key for continuous motion; release to stop after timeout.
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


HELP = """
WASD Teleop
-----------
  w/s : forward / backward
  a/d : turn left / right
  space : stop
  q : quit
"""

LIN_SPEED = 0.25   # m/s
ANG_SPEED = 1.0    # rad/s
STOP_AFTER = 0.5   # seconds without key press → stop


def get_key(timeout: float) -> str:
    """Non-blocking single-char read from stdin."""
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout)
    if rlist:
        return sys.stdin.read(1)
    return ''


class WasdTeleop(Node):
    def __init__(self):
        super().__init__('wasd_teleop')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_key_time = 0.0
        self.linear = 0.0
        self.angular = 0.0
        # 20 Hz publish — keeps cmd_vel fresh for the robot
        self.create_timer(0.05, self._tick)
        self.get_logger().info(HELP)

    def apply_key(self, key: str) -> bool:
        """Return False to quit."""
        if key == 'w':
            self.linear, self.angular = LIN_SPEED, 0.0
        elif key == 's':
            self.linear, self.angular = -LIN_SPEED, 0.0
        elif key == 'a':
            self.linear, self.angular = 0.0, ANG_SPEED
        elif key == 'd':
            self.linear, self.angular = 0.0, -ANG_SPEED
        elif key == ' ':
            self.linear, self.angular = 0.0, 0.0
        elif key == 'q' or key == '\x03':  # q or Ctrl-C
            return False
        else:
            return True
        self.last_key_time = self.get_clock().now().nanoseconds / 1e9
        return True

    def _tick(self) -> None:
        # Auto-stop if no key pressed recently
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_key_time > STOP_AFTER:
            self.linear = 0.0
            self.angular = 0.0
        msg = Twist()
        msg.linear.x = self.linear
        msg.angular.z = self.angular
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WasdTeleop()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = get_key(0.05)
            if key:
                if not node.apply_key(key):
                    break
    finally:
        # Send a final stop
        stop = Twist()
        node.pub.publish(stop)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
