"""
A stand-in camera for desk tests: publishes synthetic rgb8 Images at a fixed rate.

Same message type, encoding, stamping and rate a RealSense colour stream produces, so the
recorder and the policy client can be exercised end to end on a machine with no camera.  The
picture is a slow colour gradient with a moving square and the frame number, which is enough to
see that stored frames differ and are in order.

    ros2 run hsa_ros2 fake_camera
    ros2 run hsa_ros2 fake_camera --ros-args -p topic:=/camera/color/image_raw -p hz:=30.0 -p width:=640 -p height:=480
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class FakeCamera(Node):
    """Publish a synthetic rgb8 image stream."""

    def __init__(self) -> None:
        """Set up the node."""
        super().__init__('fake_camera')
        self.declare_parameter('topic', '/camera/color/image_raw')
        self.declare_parameter('hz', 30.0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self._w = int(self.get_parameter('width').value)
        self._h = int(self.get_parameter('height').value)
        hz = float(self.get_parameter('hz').value)
        self._pub = self.create_publisher(Image, str(self.get_parameter('topic').value), 1)
        self._n = 0
        yy, xx = np.mgrid[0 : self._h, 0 : self._w]
        self._base = np.stack(
            [xx * 255 // max(1, self._w - 1), yy * 255 // max(1, self._h - 1), np.zeros_like(xx)], axis=-1
        ).astype(np.uint8)
        self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f'fake_camera: {self._w}x{self._h} rgb8 @ {hz:g} Hz on {self._pub.topic_name}')

    def _tick(self) -> None:
        img = self._base.copy()
        s = 40
        x = (self._n * 7) % max(1, self._w - s)
        y = (self._n * 3) % max(1, self._h - s)
        img[y : y + s, x : x + s] = (255, 255, 255)
        img[:, :, 2] = self._n % 256
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fake_camera'
        msg.height, msg.width = self._h, self._w
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = self._w * 3
        msg.data = img.tobytes()
        self._pub.publish(msg)
        self._n += 1


def main() -> None:
    """Run the node."""
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = FakeCamera()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
