from http import server

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import cv_bridge
import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
import io, threading, time, cv2, os
from ament_index_python.packages import get_package_share_directory

class RviserEmergencyNode(Node):
    def __init__(self):
        super().__init__('rviser_emergency_node')

        self.server = viser.ViserServer()
        self.bridge = cv_bridge.CvBridge()

        # Add a static floor (thin box) to the 3D scene
        floor_urdf = '''<?xml version="1.0"?>
        <robot name="floor">
            <link name="floor_link">
                <visual>
                    <origin xyz="0 0 -0.01" rpy="0 0 0"/>
                    <geometry>
                        <box size="100 100 0.02"/>
                    </geometry>
                    <material name="gray">
                        <color rgba="0.5 0.5 0.5 1.0"/>
                    </material>
                </visual>
            </link>
        </robot>
        '''
        
        self.floor_urdf = ViserUrdf(self.server, yourdfpy.URDF.load(io.StringIO(floor_urdf)), root_node_name="/floor")
        
        # Set the initial camera pose. This is the pose that new clients will start at,
        # and also what "Reset View" will return to.
        self.server.initial_camera.position = (2.0, 1.0, 1.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        
        # UI Handles
        self.img_handle = self.server.gui.add_image(np.zeros((360, 640, 3), dtype=np.uint8), label="Camera")
        self.viser_urdf = None

        # Internal State Buffers
        self.latest_img = None
        self.latest_joints = None

        self.latest_pc = None
        self.pc_lock = False
        self.latest_traj = None
        self.traj_handle = None
        
        # Camera orbit parameters
        # self.orbit_start_time = time.time()
        # self.orbit_radius = 5.0  # Distance from robot center
        # self.orbit_height = 3.0  # Height of camera above ground
        # self.orbit_speed = 0.5   # Radians per second


        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(Image, '/realsense_head_front/color/image_raw', self.img_cb, sensor_qos)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(MarkerArray, '/trajectory', self.traj_cb, 10)
        self.create_subscription(String, 'robot_description', self.urdf_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # Separate Math and Render Threads
        threading.Thread(target=self.viser_sync_loop, daemon=True).start()

    def img_cb(self, msg):
        self.latest_img = msg

    def joint_cb(self, msg):
        if self.viser_urdf:
            joint_map = dict(zip(msg.name, msg.position))
            self.latest_joints = np.array([joint_map.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits().keys()])

    def pc_cb(self, msg):
        if not self.pc_lock:
            self.latest_pc = msg

    def traj_cb(self, msg):
        if not msg.markers:
            return

        line_marker = msg.markers[0]
        if not line_marker.points:
            return

        self.latest_traj = np.array(
            [[p.x, p.y, p.z] for p in line_marker.points],
            dtype=np.float32,
        )

    def viser_sync_loop(self):
        """Single UI thread. Packages everything into one atomic network payload at 30Hz."""
        while rclpy.ok():
            # Open the atomic grouping context
            with self.server.atomic():

                # 1. Update all connected client cameras to orbit around the robot.
                # elapsed_time = time.time() - self.orbit_start_time
                # angle = elapsed_time * self.orbit_speed
                
                # # Calculate camera position in circular orbit
                # cam_x = self.orbit_radius * np.cos(angle)
                # cam_y = self.orbit_radius * np.sin(angle)
                # cam_z = self.orbit_height
                
                # for client in self.server.get_clients().values():
                #     with client.atomic():
                #         client.camera.position = (cam_x, cam_y, cam_z)
                #         client.camera.look_at = (0.0, 0.0, 1.0)

                # 2. Update Camera Image
                if self.latest_img:
                    try:
                        img = self.bridge.imgmsg_to_cv2(self.latest_img, 'rgb8')
                        self.img_handle.image = cv2.resize(img, (640, 360))
                    except:
                        img = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(img, "Camera Unavailable", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        self.img_handle.image = img

                    self.latest_img = None

                # 3. Update Robot Joints
                if self.latest_joints is not None and self.viser_urdf is not None:
                    self.viser_urdf.update_cfg(self.latest_joints)
                    self.latest_joints = None

                # 4. Update trajectory visualization from MarkerArray.
                if self.latest_traj is not None and self.latest_traj.shape[0] >= 2:
                    points = self.latest_traj
                    segments = np.stack((points[:-1], points[1:]), axis=1)
                    colors = np.array([30, 170, 255], dtype=np.uint8)

                    if self.traj_handle is not None:
                        self.traj_handle.remove()

                    self.traj_handle = self.server.scene.add_line_segments(
                        "/trajectory/path",
                        points=segments,
                        colors=colors,
                        line_width=4.0,
                    )
                    self.latest_traj = None


            # Instantly push the grouped update to the browser
            self.server.flush()
            time.sleep(0.033) # Lock to ~30 FPS

    def urdf_cb(self, msg):
        if self.viser_urdf: return
        def resolver(fname):
            if fname.startswith('package://'):
                p = fname.split('/')
                return os.path.join(get_package_share_directory(p[2]), '/'.join(p[3:]))
            return fname
        self.viser_urdf = ViserUrdf(self.server, yourdfpy.URDF.load(io.StringIO(msg.data), filename_handler=resolver), root_node_name="/opensot")

def main():
    rclpy.init()
    node = RviserEmergencyNode()
    try:
        rclpy.spin(node)
    except:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()