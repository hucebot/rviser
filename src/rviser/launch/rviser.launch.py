from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # Republish compressed RealSense color images as raw images for subscribers.
    image_decompressor = Node(
        package="image_transport",
        executable="republish",
        name="head_front_color_decompressor",
        output="screen",
        arguments=[
            "compressed",
            "raw",
            "--ros-args",
            "-r",
            "in/compressed:=/rgbd/realsense_head_front/color/image_raw/compressed",
            "-r",
            "out:=/camera/image_raw",
        ],
    )

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
        image_decompressor,
        safe_teleop_node,
        dummy_trajectory_node,
    ])
