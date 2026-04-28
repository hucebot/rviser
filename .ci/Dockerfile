FROM ros:humble-ros-base

# 1. Install system and ROS dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-cv-bridge \
    ros-humble-rmw-cyclonedds-cpp \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Viser and URDF/Mesh parsing requirements
RUN pip3 install viser "numpy<2.0.0" yourdfpy pycollada opencv-python open3d

# 3. Setup the workspace and copy your custom node
WORKDIR /workspace
COPY src/ /workspace/src/
COPY . /workspace/src/rviser/

# 4. Clone all the TIAGo / PAL Robotics description packages
WORKDIR /workspace/src
RUN git clone -b humble-devel https://github.com/pal-robotics/tiago_robot.git && \
    git clone -b humble-devel https://github.com/pal-robotics/tiago_dual_robot.git && \
    git clone -b humble-devel https://github.com/pal-robotics/omni_base_robot.git && \
    git clone -b humble-devel https://github.com/pal-robotics/pal_gripper.git && \
    git clone -b ros2 https://github.com/hucebot/tiago_dual_cartesio_config.git && \
    git clone -b humble-devel https://github.com/pal-robotics/tiago_pro_robot.git && \
    git clone -b humble-devel https://github.com/pal-robotics/pal_urdf_utils.git && \
    git clone -b humble-devel https://github.com/pal-robotics/pal_pro_gripper.git && \
    git clone -b humble-devel https://github.com/pal-robotics/pal_sea_arm.git && \
    git clone -b humble-devel https://github.com/pal-robotics/tiago_pro_head_robot.git

# Install specific Realsense packages
RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-realsense2-camera \
    ros-$ROS_DISTRO-realsense2-description \
    && rm -rf /var/lib/apt/lists/*

# 5. Resolve all dependencies and BUILD the workspace
WORKDIR /workspace
RUN apt-get update && \
    rosdep update && \
    rosdep install -y -r -q --from-paths src --ignore-src --rosdistro humble && \
    rm -rf /var/lib/apt/lists/*

# Compile everything (your node + TIAGo descriptions) so ROS indexes the mesh paths
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"

# 6. Setup Entrypoint
COPY .ci/entrypoint.sh /
RUN chmod +x /entrypoint.sh

# Ensure the workspace is sourced automatically in bash
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]