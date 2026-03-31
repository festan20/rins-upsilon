from point_request.srv import GetNewPoint
import random
import rclpy
from rclpy.node import Node

class FaceLocator(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(GetNewPoint, 'get_new_point', self.new_point_callback)
        self.currnet_point = 0
        self.points = []
        f = open("face_points.txt", "r")
        conntent = f.readlines()
        f.close()
        for l in conntent:
            l = l.replace("(", "").replace(")", "").replace("\n", "")
            sp = l.split(", ")
            self.points.append((float(sp[0]), float(sp[1])))
        print(self.points)

    def new_point_callback(self, request, response):
        response.x = self.points[self.currnet_point][0]
        response.y = self.points[self.currnet_point][1]
        response.rotation = float(random.randint(0,100))
        self.get_logger().info("new point requested")
        self.currnet_point = (self.currnet_point + 1) % len(self.points)
        return response


def main():
    rclpy.init()
    face_service = FaceLocator()
    rclpy.spin(face_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
