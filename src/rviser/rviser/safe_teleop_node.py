from http import server
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
import sensor_msgs_py.point_cloud2 as pc2
import cv_bridge
import numpy as np
import viser
import trimesh
from viser import transforms as tf
from viser.extras import ViserUrdf
import yourdfpy

# Suppress the annoying mimic joint warnings from yourdfpy
import logging
logging.getLogger('yourdfpy.urdf').setLevel(logging.ERROR)

import io, threading, time, cv2, os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class RviserEmergencyNode(Node):
    def __init__(self):
        super().__init__('rviser_emergency_node')

        self.server = viser.ViserServer()
        self.bridge = cv_bridge.CvBridge()
        self.state_lock = threading.Lock()

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
        
        self.robot_frame = self.server.scene.add_frame("/robot")

        self.server.initial_camera.position = (2.0, 1.0, 1.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)
        
        self.img_handle = self.server.gui.add_image(np.zeros((360, 640, 3), dtype=np.uint8), label="Camera")
        self.viser_urdf = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.file_path = Path(__file__).resolve()
        self.mesh_cache = {}  
        self.mesh_scale = 10.

        # Internal State Buffers
        self.latest_img = None
        self.latest_joints = None
        self.latest_traj = {"left": None, "right": None}
        self.latest_traj_colors = {"left": None, "right": None}
        self.latest_traj_frame = {"left": "gripper_left_grasping_link", "right": "gripper_right_grasping_link"}
        self.traj_dirty = {"left": False, "right": False}
        self.traj_handles = {"left": None, "right": None}
        
        self.latest_object_marker = None
        self.latest_robot_pose = None

        # Parameters
        self.declare_parameter('first_person', False)
        self.declare_parameter('show_opensot', False)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('opensot_topic', '/opensot/joint_states')

        self.first_person = self.get_parameter('first_person').value
        self.show_opensot = self.get_parameter('show_opensot').value
        self.first_person_link = 'head_front_camera_link'
        self._first_person_active_last_frame = self.first_person

        self.add_on_set_parameters_callback(self.parameter_callback)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(MarkerArray, '/rviser/left/trajectory', self.left_traj_cb, 10)
        self.create_subscription(MarkerArray, '/rviser/right/trajectory', self.right_traj_cb, 10)
        self.create_subscription(Marker, '/object_marker', self.object_marker_cb, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self.robot_pose_cb, 10)
        self.create_subscription(String, 'robot_description', self.urdf_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Image, self.get_parameter('camera_topic').value, self.img_cb, sensor_qos)
        self.create_subscription(JointState, self.get_parameter('opensot_topic').value, self.opensot_joint_cb, 10)
        self.create_subscription(JointState, self.get_parameter('joint_topic').value, self.joint_cb, 10)

        threading.Thread(target=self.viser_sync_loop, daemon=True).start()

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'first_person':
                self.first_person = param.value
            elif param.name == 'show_opensot':
                self.show_opensot = param.value
        return SetParametersResult(successful=True)

    def img_cb(self, msg):
        with self.state_lock:
            self.latest_img = msg

    def joint_cb(self, msg):
        if self.viser_urdf and not self.show_opensot: 
            joint_map = dict(zip(msg.name, msg.position))
            with self.state_lock:
                self.latest_joints = np.array([joint_map.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits().keys()])

    def opensot_joint_cb(self, msg):
        if self.viser_urdf and self.show_opensot:
            joint_map = dict(zip(msg.name, msg.position))
            with self.state_lock:
                self.latest_joints = np.array([joint_map.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits().keys()])

    def left_traj_cb(self, msg):
        self.traj_cb(msg, "left")

    def right_traj_cb(self, msg):
        self.traj_cb(msg, "right")

    def traj_cb(self, msg, side):
        with self.state_lock:
            if not msg.markers or not msg.markers[0].points:
                self.latest_traj[side] = None
                self.latest_traj_colors[side] = None
                self.latest_traj_frame[side] = None
                self.traj_dirty[side] = True
                return

            line_marker = msg.markers[0]
            self.latest_traj[side] = np.array(
                [[p.x, p.y, p.z] for p in line_marker.points], dtype=np.float32
            )
            
            # Extract colors for the gradient mapping
            if line_marker.colors:
                self.latest_traj_colors[side] = np.array(
                    [[c.r, c.g, c.b] for c in line_marker.colors], dtype=np.float32
                )
            else:
                self.latest_traj_colors[side] = None
                
            self.latest_traj_frame[side] = line_marker.header.frame_id
            self.traj_dirty[side] = True

    def object_marker_cb(self, msg):
        with self.state_lock:
            self.latest_object_marker = msg

        # Trigger async load to prevent UI freezing
        if msg.mesh_resource and msg.mesh_resource not in self.mesh_cache:
            self.mesh_cache[msg.mesh_resource] = 'loading'
            threading.Thread(target=self._async_load_mesh, args=(msg.mesh_resource,), daemon=True).start()

    def _async_load_mesh(self, mesh_resource):
        try:
            relative_path = mesh_resource[7:].lstrip('/')  
            mesh_path = Path("/workspace/src/rviser") / relative_path
            
            if not mesh_path.exists():
                self.get_logger().warn(f"Mesh file not found: {mesh_path}")
                self.mesh_cache.pop(mesh_resource, None)
                return
            
            # process=False prevents trimesh from freezing on massive OBJs
            mesh = trimesh.load(str(mesh_path), force='mesh', process=False) 
            mesh.apply_scale(self.mesh_scale)
            
            with self.state_lock:
                self.mesh_cache[mesh_resource] = mesh
        except Exception as e:
            self.get_logger().error(f"Failed to load mesh {mesh_resource}: {e}")
            self.mesh_cache.pop(mesh_resource, None)

    def robot_pose_cb(self, msg):
        with self.state_lock:
            self.latest_robot_pose = msg

    def _get_base_pose_link(self, link_name):
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', link_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.01)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pos = np.array([trans.x, trans.y, trans.z])
            r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
            
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = r.as_matrix()
            T[:3, 3] = pos
            return T
        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f"TF Lookup failed for {link_name}: {e}")
            return None

    def _pose_matrix(self, position, wxyz):
        w, x, y, z = wxyz
        r = R.from_quat([x, y, z, w])
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = r.as_matrix()
        transform[:3, 3] = np.array(position, dtype=np.float64)
        return transform

    def viser_sync_loop(self):
        while rclpy.ok():
            with self.server.atomic():
                
                with self.state_lock:
                    current_img = self.latest_img
                    current_joints = self.latest_joints
                    current_pose = self.latest_robot_pose
                    current_marker = self.latest_object_marker
                    
                    # Copy trajectory data but DO NOT reset self.latest_traj to None
                    current_traj = self.latest_traj.copy()
                    current_traj_colors = self.latest_traj_colors.copy()
                    current_traj_frame = self.latest_traj_frame.copy()
                    dirty_flags = self.traj_dirty.copy()
                    
                    self.latest_img = None
                    self.latest_joints = None
                    self.latest_object_marker = None
                    self.traj_dirty = {"left": False, "right": False}

                # 1. First-person view
                if self.first_person:
                    link_pose = self._get_base_pose_link(self.first_person_link)
                    if link_pose is not None:
                        robot_pose = self._pose_matrix(self.robot_frame.position, self.robot_frame.wxyz)
                        world_pose = robot_pose @ link_pose
                        world_position = world_pose[:3, 3]
                        world_rotation = world_pose[:3, :3]
                        
                        for client in self.server.get_clients().values():
                            try:
                                with client.atomic():
                                    client.camera.position = (float(world_position[0]), float(world_position[1]), float(world_position[2]))
                                    camera_look_at = world_position + world_rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64) 
                                    client.camera.look_at = (float(camera_look_at[0]), float(camera_look_at[1]), float(camera_look_at[2]))
                            except Exception:
                                pass
                elif self._first_person_active_last_frame:
                    default_position = tuple(float(v) for v in self.server.initial_camera.position)
                    default_look_at = tuple(float(v) for v in self.server.initial_camera.look_at)

                    for client in self.server.get_clients().values():
                        try:
                            with client.atomic():
                                client.camera.position = default_position
                                client.camera.look_at = default_look_at
                        except Exception:
                            pass

                self._first_person_active_last_frame = self.first_person
                    
                # 2. Update Camera Image
                if current_img:
                    try:
                        img = self.bridge.imgmsg_to_cv2(current_img, 'rgb8')
                        self.img_handle.image = cv2.resize(img, (640, 360))
                    except Exception as e:
                        self.get_logger().debug(f"CV Bridge Error: {e}")
                        img = np.zeros((360, 640, 3), dtype=np.uint8)
                        cv2.putText(img, "Camera Unavailable", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        self.img_handle.image = img

                # 3. Update Robot Joints
                if current_joints is not None and self.viser_urdf is not None:
                    self.viser_urdf.update_cfg(current_joints)

                if current_pose is not None and self.viser_urdf is not None:
                    pose = current_pose.pose
                    pos = pose.position
                    quat = pose.orientation
                    self.robot_frame.position = (float(pos.x), float(pos.y), float(pos.z))
                    self.robot_frame.wxyz = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))

                # 4. Update trajectory visualization
                trajectory_specs = {
                    "left": (np.array([30, 170, 255], dtype=np.uint8), "/trajectory/left"),
                    "right": (np.array([255, 120, 30], dtype=np.uint8), "/trajectory/right"),
                }

                for side, (default_color, scene_path) in trajectory_specs.items():
                    if not dirty_flags[side]:
                        continue  # Skip re-rendering if the trajectory hasn't changed

                    points = current_traj[side]
                    raw_colors = current_traj_colors[side]
                    header_frame = current_traj_frame[side]
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
                                'base_link', header_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.01)
                            )
                            trans = transform.transform.translation
                            rot = transform.transform.rotation
                            pos = np.array([trans.x, trans.y, trans.z])     
                            r = R.from_quat([rot.x, rot.y, rot.z, rot.w])
                            
                            transform_matrix[:3, :3] = r.as_matrix()
                            transform_matrix[:3, 3] = pos
                            points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
                            points_transformed = (transform_matrix @ points_homogeneous.T).T[:, :3]
                        except Exception as e:
                            self.get_logger().debug(f"TF lookup failed for {side}: {e}")
                            points_transformed = points

                    if handle is not None:
                        handle.remove()

                    segment_points = np.stack([points_transformed[:-1], points_transformed[1:]], axis=1)

                    # Apply gradient colors if provided, otherwise fallback to default
                    if raw_colors is not None and len(raw_colors) == len(points):
                        c_uint8 = (raw_colors * 255).astype(np.uint8)
                        segment_colors = np.stack([c_uint8[:-1], c_uint8[1:]], axis=1)
                    else:
                        segment_colors = default_color

                    self.traj_handles[side] = self.server.scene.add_line_segments(
                        scene_path,
                        points=segment_points,
                        colors=segment_colors,
                        line_width=4.0,
                    )

                # 5. Add meshes from object_marker topic (Now Async)
                if current_marker is not None and current_marker.mesh_resource:
                    mesh = self.mesh_cache.get(current_marker.mesh_resource)
                    
                    if mesh is not None and mesh != 'loading':
                        pos = np.array([current_marker.pose.position.x, current_marker.pose.position.y, current_marker.pose.position.z])
                        quat = current_marker.pose.orientation
                        wxyz = np.array([quat.w, quat.x, quat.y, quat.z])
                        
                        frame_id = current_marker.header.frame_id
                        
                        if frame_id and frame_id != 'base_link':
                            link_tf = self._get_base_pose_link(frame_id)
                            if link_tf is not None:
                                marker_local_tf = self._pose_matrix(pos, wxyz)
                                marker_base_tf = link_tf @ marker_local_tf
                                
                                pos = marker_base_tf[:3, 3]
                                x, y, z, w = R.from_matrix(marker_base_tf[:3, :3]).as_quat()
                                wxyz = np.array([w, x, y, z])
                            else:
                                continue  # Skip rendering if TF fails for this frame
                        
                        self.server.scene.add_mesh_trimesh(
                            f"/robot/markers/{current_marker.ns}_{current_marker.id}",
                            mesh=mesh,
                            wxyz=tuple(wxyz),
                            position=tuple(pos),
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
        
        self.viser_urdf = ViserUrdf(
            self.server, 
            yourdfpy.URDF.load(io.StringIO(msg.data), filename_handler=resolver), 
            root_node_name="/robot/urdf" 
        )

def main():
    rclpy.init()
    node = RviserEmergencyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Node exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()