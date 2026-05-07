from http import server

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
import tf2_ros
from geometry_msgs.msg import TransformStamped
import sensor_msgs_py.point_cloud2 as pc2
import cv_bridge
import numpy as np
import viser
import trimesh
from viser import transforms as tf
from viser.extras import ViserUrdf
import yourdfpy
import io, threading, time, cv2, os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

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
        
        # --- FIX 1: Create a distinct Root Frame for the Robot ---
        # Everything attached to "/robot" will move when this frame moves.
        self.robot_frame = self.server.scene.add_frame("/robot")
        # ---------------------------------------------------------

        # Set the initial camera pose. This is the pose that new clients will start at,
        # and also what "Reset View" will return to.
        self.server.initial_camera.position = (2.0, 1.0, 1.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        
        # UI Handles
        self.img_handle = self.server.gui.add_image(np.zeros((360, 640, 3), dtype=np.uint8), label="Camera")
        self.viser_urdf = None
        
        # TF2 buffer and listener for fetching transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Mesh loading - cache for dynamic loading
        self.file_path = Path(__file__).resolve()
        self.mesh_cache = {}  # Dictionary to cache loaded meshes
        self.mesh_scale = 10.  # Default scale for meshes

        # Internal State Buffers
        self.latest_img = None
        self.latest_joints = None

        self.latest_traj = {"left": None, "right": None}
        self.latest_traj_frame = {"left": "gripper_left_grasping_link", "right": "gripper_right_grasping_link"}
        self.traj_handles = {"left": None, "right": None}
        self.latest_object_marker = None
        self.latest_mesh_resource = None
        self.latest_robot_pose = None


        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(MarkerArray, '/rviser/left/trajectory', self.left_traj_cb, 10)
        self.create_subscription(MarkerArray, '/rviser/right/trajectory', self.right_traj_cb, 10)
        self.create_subscription(MarkerArray, '/object_marker', self.object_marker_cb, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self.robot_pose_cb, 10)
        self.create_subscription(String, 'robot_description', self.urdf_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.declare_parameter('first_person', False)
        self.declare_parameter('show_opensot', False) 

        self.first_person = self.get_parameter('first_person').value
        self.first_person_link = 'head_front_camera_link'
        self._first_person_active_last_frame = self.first_person
        self.show_opensot = self.get_parameter('show_opensot').value

        # Subscriptions
        self.create_subscription(Image, '/rgbd/realsense_head_front/color/image_raw', self.img_cb, sensor_qos)
        self.create_subscription(JointState, '/opensot/joint_states', self.opensot_joint_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        

        threading.Thread(target=self.viser_sync_loop, daemon=True).start()

    def img_cb(self, msg):
        self.latest_img = msg

    def joint_cb(self, msg):
        if self.viser_urdf and not self.show_opensot: 
            joint_map = dict(zip(msg.name, msg.position))
            self.latest_joints = np.array([joint_map.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits().keys()])

    def opensot_joint_cb(self, msg):
        if self.viser_urdf and self.show_opensot:
            joint_map = dict(zip(msg.name, msg.position))
            self.latest_joints = np.array([joint_map.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits().keys()])

    def left_traj_cb(self, msg):
        self.traj_cb(msg, "left")

    def right_traj_cb(self, msg):
        self.traj_cb(msg, "right")

    def traj_cb(self, msg, side):
        if not msg.markers:
            self.latest_traj[side] = None
            self.latest_traj_frame[side] = None
            return

        line_marker = msg.markers[0]
        if not line_marker.points:
            self.latest_traj[side] = None
            self.latest_traj_frame[side] = None
            return

        self.latest_traj[side] = np.array(
            [[p.x, p.y, p.z] for p in line_marker.points],
            dtype=np.float32,
        )
        self.latest_traj_frame[side] = line_marker.header.frame_id

    def object_marker_cb(self, msg):
        if msg.markers:
            self.latest_object_marker = msg.markers[0]
            self.latest_mesh_resource = msg.markers[0].mesh_resource

    def robot_pose_cb(self, msg):
        """Handle robot pose updates and move the robot in viser."""
        self.latest_robot_pose = msg

    def _load_mesh_from_resource(self, mesh_resource):
        if mesh_resource in self.mesh_cache:
            return self.mesh_cache[mesh_resource]
             
        relative_path = mesh_resource[7:]  # Remove 'file://'
        mesh_path = self.file_path.parents[1] / relative_path
        
        if not mesh_path.exists():
            self.get_logger().warn(f"Mesh file not found: {mesh_path}")
            return None
        
        mesh = trimesh.load_mesh(str(mesh_path))
        mesh.apply_scale(self.mesh_scale)
        self.mesh_cache[mesh_resource] = mesh
        return mesh

    def _get_base_pose_link(self, link_name):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', link_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pos = np.array([trans.x, trans.y, trans.z])
            quat = np.array([rot.x, rot.y, rot.z, rot.w])
            
            x, y, z, w = quat
            rot_matrix = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ], dtype=np.float64)
            
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos
            return T
        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            print(e)
            return None

    def _pose_matrix(self, position, wxyz):
        w, x, y, z = wxyz
        rotation_matrix = np.array([
            [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
            [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
            [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = np.array(position, dtype=np.float64)
        return transform

    def viser_sync_loop(self):
        while rclpy.ok():
            with self.server.atomic():

                self.first_person = bool(self.get_parameter('first_person').value)

                # 1. First-person view
                if self.first_person:
                    link_pose = self._get_base_pose_link(self.first_person_link)
                    if link_pose is not None:
                        robot_pose = self._pose_matrix(self.robot_frame.position, self.robot_frame.wxyz)
                        world_pose = robot_pose @ link_pose
                        world_position = world_pose[:3, 3]
                        world_rotation = world_pose[:3, :3]
                        camera_forward = world_rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                        for client in self.server.get_clients().values():
                            try:
                                with client.atomic():
                                    client.camera.position = (float(world_position[0]), float(world_position[1]), float(world_position[2]))
                                    camera_look_at = world_position + world_rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64) 
                                    client.camera.look_at = (float(camera_look_at[0]), float(camera_look_at[1]), float(camera_look_at[2]))
                            except Exception as e:
                                print(e)
                                pass
                elif self._first_person_active_last_frame:
                    default_position = tuple(float(v) for v in self.server.initial_camera.position)
                    default_look_at = tuple(float(v) for v in self.server.initial_camera.look_at)

                    for client in self.server.get_clients().values():
                        try:
                            with client.atomic():
                                client.camera.position = default_position
                                client.camera.look_at = default_look_at
                        except Exception as e:
                            print(e)
                            pass

                self._first_person_active_last_frame = self.first_person
                    
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

                # --- FIX 2: Apply pose to the 'robot_frame', not the ViserUrdf helper! ---
                if self.latest_robot_pose is not None and self.viser_urdf is not None:
                    pose = self.latest_robot_pose.pose
                    pos = pose.position
                    quat = pose.orientation
                    
                    self.robot_frame.position = (float(pos.x), float(pos.y), float(pos.z))
                    self.robot_frame.wxyz = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))
                # -------------------------------------------------------------------------

                # 4. Update trajectory visualization
                trajectory_specs = {
                    "left": (np.array([30, 170, 255], dtype=np.uint8), "/trajectory/left"),
                    "right": (np.array([255, 120, 30], dtype=np.uint8), "/trajectory/right"),
                }

                for side, (colors, scene_path) in trajectory_specs.items():
                    points = self.latest_traj[side]
                    header_frame = self.latest_traj_frame[side]
                    handle = self.traj_handles[side]

                    if points is None or header_frame is None:
                        if handle is not None:
                            handle.remove()
                            self.traj_handles[side] = None
                        continue

                    if points.shape[0] < 2:
                        continue

                    transform_matrix = np.eye(4)
                    points_transformed = points
                    if header_frame != 'base_link':
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                'base_link', header_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
                            )
                            trans = transform.transform.translation
                            rot = transform.transform.rotation
                            pos = np.array([trans.x, trans.y, trans.z])     
                            quat = np.array([rot.x, rot.y, rot.z, rot.w])
                            x, y, z, w = quat
                            rot_matrix = np.array([
                                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
                            ], dtype=np.float64)
                            transform_matrix[:3, :3] = rot_matrix
                            transform_matrix[:3, 3] = pos
                            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
                            points_transformed = (transform_matrix @ points_homogeneous.T).T[:, :3]
                        except Exception as e:
                            self.get_logger().warn(f"TF lookup failed for {side}: {e}")
                            points_transformed = points

                    if handle is not None:
                        handle.remove()

                    segment_points = np.stack([points_transformed[:-1], points_transformed[1:]], axis=1)

                    self.traj_handles[side] = self.server.scene.add_line_segments(
                        scene_path,
                        points=segment_points,
                        colors=colors,
                        line_width=4.0,
                    )
                    self.latest_traj[side] = None
                    self.latest_traj_frame[side] = None

                # 5. Add meshes from object_marker topic
                if self.latest_object_marker is not None and self.latest_mesh_resource is not None:
                    marker = self.latest_object_marker
                    mesh = self._load_mesh_from_resource(self.latest_mesh_resource)
                    if mesh is not None:
                        pos = (float(marker.pose.position.x), float(marker.pose.position.y), float(marker.pose.position.z))
                        quat = marker.pose.orientation
                        wxyz = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))
                        self.server.scene.add_mesh_trimesh(
                            "/trimesh",
                            mesh=mesh,
                            wxyz=wxyz,
                            position=pos,
                        )

            self.server.flush()
            time.sleep(0.033) 

    def urdf_cb(self, msg):
        if self.viser_urdf: return
        def resolver(fname):
            if fname.startswith('package://'):
                p = fname.split('/')
                return os.path.join(get_package_share_directory(p[2]), '/'.join(p[3:]))
            return fname
        
        # --- FIX 3: Load the URDF directly beneath the movable robot_frame ---
        self.viser_urdf = ViserUrdf(
            self.server, 
            yourdfpy.URDF.load(io.StringIO(msg.data), filename_handler=resolver), 
            root_node_name="/robot/urdf" 
        )
        # ---------------------------------------------------------------------

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