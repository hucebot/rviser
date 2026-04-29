import math
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class TrajectoryConfig:
    frame_id: str = "world"
    topic_name: str = "trajectory"
    publish_rate_hz: float = 30.0
    resample_points: int = 200
    control_points: int = 6
    control_spread: float = 0.8
    xy_radius: float = 1.5
    z_center: float = 0.8
    z_variation: float = 0.4
    regenerate_seconds: float = 8.0


def generate_random_spline_trajectory(
    rng: np.random.Generator,
    config: TrajectoryConfig,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, config.control_points, endpoint=False)
    angular_jitter = rng.uniform(-0.25, 0.25, size=config.control_points)
    radial_jitter = rng.uniform(-config.control_spread, config.control_spread, size=config.control_points)
    x = (config.xy_radius + radial_jitter) * np.cos(angles + angular_jitter)
    y = (config.xy_radius + radial_jitter) * np.sin(angles + angular_jitter)
    z = config.z_center + rng.uniform(-config.z_variation, config.z_variation, size=config.control_points)

    control_points = np.column_stack((x, y, z))
    # duplicate the first control point to close the loop
    control_points = np.vstack((control_points, control_points[0]))

    extended = np.vstack((control_points[-2], control_points, control_points[1]))
    # number of curve segments equals number of control points minus the duplicated one
    segment_count = max(1, len(control_points) - 1)
    samples_per_segment = max(2, config.resample_points // segment_count)
    trajectory = []

    for i in range(segment_count):
        p0, p1, p2, p3 = extended[i : i + 4]
        for j in range(samples_per_segment):
            t = j / float(samples_per_segment)
            t2 = t * t
            t3 = t2 * t
            trajectory.append(
                0.5
                * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
            )

    return np.asarray(trajectory, dtype=np.float64)


def trajectory_to_msg(points: np.ndarray, frame_id: str, stamp) -> MarkerArray:
    line_marker = Marker()
    line_marker.header.frame_id = frame_id
    line_marker.header.stamp = stamp
    line_marker.ns = "trajectory"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.pose.orientation.w = 1.0
    line_marker.scale.x = 0.02
    line_marker.color.a = 1.0
    line_marker.color.r = 0.1
    line_marker.color.g = 0.6
    line_marker.color.b = 1.0
    line_marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]

    marker_array = MarkerArray()
    marker_array.markers.append(line_marker)
    return marker_array


class DummyTrajectoryNode(Node):
    def __init__(self) -> None:
        super().__init__("dummy_trajectory_node")

        self.declare_parameter("frame_id", "world")
        self.declare_parameter("topic_name", "trajectory")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("resample_points", 200)
        self.declare_parameter("control_points", 6)
        self.declare_parameter("control_spread", 0.8)
        self.declare_parameter("xy_radius", 1.5)
        self.declare_parameter("z_center", 0.8)
        self.declare_parameter("z_variation", 0.4)
        self.declare_parameter("regenerate_seconds", 8.0)

        self.config = TrajectoryConfig(
            frame_id=self.get_parameter("frame_id").value,
            topic_name=self.get_parameter("topic_name").value,
            publish_rate_hz=float(self.get_parameter("publish_rate_hz").value),
            resample_points=int(self.get_parameter("resample_points").value),
            control_points=int(self.get_parameter("control_points").value),
            control_spread=float(self.get_parameter("control_spread").value),
            xy_radius=float(self.get_parameter("xy_radius").value),
            z_center=float(self.get_parameter("z_center").value),
            z_variation=float(self.get_parameter("z_variation").value),
            regenerate_seconds=float(self.get_parameter("regenerate_seconds").value),
        )

        self.publisher = self.create_publisher(MarkerArray, self.config.topic_name, 10)
        self.rng = np.random.default_rng()
        self.trajectory_start_time = self.get_clock().now()
        self.trajectory = generate_random_spline_trajectory(self.rng, self.config)

        period = 1.0 / max(self.config.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._timer_cb)

    def _timer_cb(self) -> None:
        elapsed = (self.get_clock().now() - self.trajectory_start_time).nanoseconds * 1e-9
        if elapsed >= self.config.regenerate_seconds:
            self.trajectory = generate_random_spline_trajectory(self.rng, self.config)
            self.trajectory_start_time = self.get_clock().now()

        self.publisher.publish(
            trajectory_to_msg(self.trajectory, self.config.frame_id, self.get_clock().now().to_msg())
        )


def main() -> None:
    rclpy.init()
    node = DummyTrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
