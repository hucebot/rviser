from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    safe_teleop_node = Node(
        package="rviser",
        executable="safe_teleop_node",
        name="safe_teleop_node",
        output="screen",
    )

    dummy_trajectory_node = Node(
        package="rviser",
        executable="dummy_trajectory_node",
        name="dummy_trajectory_node",
        output="screen",
    )

    return LaunchDescription([
        safe_teleop_node,
        dummy_trajectory_node,
    ])
