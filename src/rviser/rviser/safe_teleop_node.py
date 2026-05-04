from http import server

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
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

        # Mesh loading
        file_path = Path(__file__).resolve()
        mesh_path = file_path.parents[3] / 'src' / 'rviser' / 'assets' / 'mustard' / 'mustard.obj'
        
        self.mesh = trimesh.load_mesh(str(mesh_path))
        assert isinstance(self.mesh, trimesh.Trimesh), "Expected a single mesh"
        self.mesh.apply_scale(10.)

        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces

        # Internal State Buffers
        self.latest_img = None
        self.latest_joints = None

        self.latest_traj = None
        self.traj_handle = None
        self.latest_object_marker = None


        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(MarkerArray, '/rviser/left/trajectory', self.traj_cb, 10)
        self.create_subscription(MarkerArray, '/object_marker', self.object_marker_cb, 10)
        self.create_subscription(String, 'robot_description', self.urdf_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.declare_parameter('first_person', True)
        self.declare_parameter('first_person_link', 'head_front_camera_link')
        self.declare_parameter('show_opensot', False) 

        self.first_person = self.get_parameter('first_person').value
        self.first_person_link = self.get_parameter('first_person_link').value
        self._first_person_active_last_frame = self.first_person
        self.show_opensot = self.get_parameter('show_opensot').value

        # Subscriptions
        self.create_subscription(Image, '/realsense_head_front/camera/color/image_raw', self.img_cb, sensor_qos)

        if self.show_opensot:
            self.create_subscription(JointState, '/opensot/joint_states', self.joint_cb, 10)
        else:
            self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        

        threading.Thread(target=self.viser_sync_loop, daemon=True).start()

    def img_cb(self, msg):
        self.latest_img = msg

    def joint_cb(self, msg):
        if self.viser_urdf:
            joint_map = dict(zip(msg.name, msg.position))
            self.latest_joints = np.array([joint_map.get(name, 0.0) for name in self.viser_urdf.get_actuated_joint_limits().keys()])

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

    def object_marker_cb(self, msg):
        if msg.markers:
            self.latest_object_marker = msg.markers[0]

    def _rotation_matrix_x(self, angle):
        """Rotation matrix around x-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c],
        ], dtype=np.float64)

    def _rotation_matrix_y(self, angle):
        """Rotation matrix around y-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c],
        ], dtype=np.float64)

    def _rotation_matrix_z(self, angle):
        """Rotation matrix around z-axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s,  0],
            [s,  c,  0],
            [0,  0,  1],
        ], dtype=np.float64)

    def _get_link_world_pose(self, link_name):
        """Get a link pose from the ROS /tf topic.
        Returns a 4x4 homogeneous transform or None if unavailable.
        """
        try:
            # Fetch the transform from base_link to the link
            transform = self.tf_buffer.lookup_transform(
                'base_link', link_name, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
            # Convert ROS transform to 4x4 matrix
            trans = transform.transform.translation
            rot = transform.transform.rotation
            pos = np.array([trans.x, trans.y, trans.z])
            quat = np.array([rot.x, rot.y, rot.z, rot.w])
            
            # Convert quaternion (x, y, z, w) to rotation matrix
            x, y, z, w = quat
            rot_matrix = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
                [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
            ], dtype=np.float64)
            
            # Build 4x4 homogeneous transform
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = rot_matrix
            T[:3, 3] = pos
            return T
        except (tf2_ros.TransformException, tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            # Silently return None if transform is not available yet
            print(e)
            return None


    def viser_sync_loop(self):
        """Single UI thread. Packages everything into one atomic network payload at 30Hz."""
        while rclpy.ok():
            # Open the atomic grouping context
            with self.server.atomic():

                self.first_person = bool(self.get_parameter('first_person').value)
                self.first_person_link = str(self.get_parameter('first_person_link').value)

                # 1. If first-person view is enabled, set each client's camera to follow
                if self.first_person:
                    link_pose = self._get_link_world_pose(self.first_person_link)
                    if link_pose is not None:
                        link_rot = link_pose[:3, :3]
                    
                        for client in self.server.get_clients().values():
                            try:
                                with client.atomic():
                                    client.camera.position = (float(link_pose[0, 3]), float(link_pose[1, 3]), float(link_pose[2, 3]))
                                    camera_look_at = link_rot @ np.array(client.camera.position) 
                                    client.camera.look_at = (float(camera_look_at[0]), float(camera_look_at[1]), float(camera_look_at[2]))

                            except Exception as e:
                                print(e)
                                # ignore per-client failures so rendering continues
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

                # 5. Add meshes from object_marker topic
                if self.latest_object_marker is not None:
                    marker = self.latest_object_marker
                    pos = (float(marker.pose.position.x), float(marker.pose.position.y), float(marker.pose.position.z))
                    quat = marker.pose.orientation
                    wxyz = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))
                    self.server.scene.add_mesh_trimesh(
                        "/trimesh",
                        mesh=self.mesh,
                        wxyz=wxyz,
                        position=pos,
                    )
                
                # For debugging
                self.server.scene.add_mesh_trimesh(
                    "/trimesh",
                    mesh=self.mesh,
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(5.0, 0.0, 1.),
                )


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