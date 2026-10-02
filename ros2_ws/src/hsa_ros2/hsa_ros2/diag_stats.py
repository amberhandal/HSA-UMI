"""
Summarise policy_client_node's diagnostic topics over a fixed window.

The client publishes one Float32 per scalar under /polyumi/diag/<name>. This subscribes to the
ones that describe the inference round trip, collects for --seconds, and prints mean / p50 / p95
per topic, so a bringup or an A/B (lamb vs laptop server, WiFi vs wired) ends in numbers.

    ros2 run hsa_ros2 diag_stats --ros-args -p seconds:=40
"""

from __future__ import annotations

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from std_msgs.msg import Float32

TOPICS = (
    'inference_latency_s',  # whole request, client side
    'inference_model_s',  # forward pass alone, reported by the server
    'inference_overhead_s',  # request minus forward pass: serialization + link + HTTP
    'obs_age_s',  # capture instant -> response landed
    'image_age_s',
    'n_stale_arm',
    'n_published_arm',
)


class DiagStats(Node):
    """Collect the diagnostic scalars and print their statistics."""

    def __init__(self) -> None:
        """Set up the node."""
        super().__init__('diag_stats')
        # dynamic_typing so `-p seconds:=40` (an int on the command line) is accepted as well as 40.0.
        self.declare_parameter('seconds', 30.0, ParameterDescriptor(dynamic_typing=True))
        self._seconds = float(self.get_parameter('seconds').value)
        self._data: dict[str, list[float]] = {t: [] for t in TOPICS}
        for t in TOPICS:
            self.create_subscription(Float32, f'/polyumi/diag/{t}', lambda m, t=t: self._data[t].append(m.data), 50)
        self._t0 = self.get_clock().now()
        self.create_timer(1.0, self._tick)
        self.get_logger().info(f'collecting /polyumi/diag/* for {self._seconds:.0f} s')

    def _tick(self) -> None:
        if (self.get_clock().now() - self._t0).nanoseconds * 1e-9 < self._seconds:
            return
        print(f'\n{"topic":22s} {"n":>4s} {"mean":>9s} {"p50":>9s} {"p95":>9s} {"max":>9s}')
        for t in TOPICS:
            v = np.asarray(self._data[t])
            if v.size == 0:
                print(f'{t:22s} {0:4d}   (nothing received)')
                continue
            unit = 1e3 if t.endswith('_s') else 1.0
            print(
                f'{t:22s} {v.size:4d} {v.mean() * unit:9.1f} {np.median(v) * unit:9.1f} '
                f'{np.percentile(v, 95) * unit:9.1f} {v.max() * unit:9.1f}' + ('  ms' if unit == 1e3 else '')
            )
        raise SystemExit(0)


def main() -> None:
    """Run the node."""
    rclpy.init()
    node = DiagStats()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
