import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, JointState, PointCloud2
from std_msgs.msg import String
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

        # UI Handles
        self.img_handle = self.server.gui.add_image(np.zeros((360, 640, 3), dtype=np.uint8), label="Camera")
        self.viser_urdf = None

        # Internal State Buffers
        self.latest_img = None
        self.latest_joints = None

        self.latest_pc = None
        self.pc_lock = False

        # Finished math ready to be rendered
        self.ready_voxels = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions
        self.create_subscription(Image, '/orbbec_external/color/image_raw', self.img_cb, sensor_qos)
        self.create_subscription(PointCloud2, '/orbbec_external/depth_registered/points', self.pc_cb, sensor_qos)
        self.create_subscription(JointState, 'opensot/joint_states', self.joint_cb, 10)
        self.create_subscription(String, 'opensot_robot_description', self.urdf_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # Separate Math and Render Threads
        threading.Thread(target=self.voxel_worker, daemon=True).start()
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

    def voxel_worker(self):
        """Background thread strictly for heavy NumPy crunching."""
        v_size = 0.08
        while rclpy.ok():
            if self.latest_pc:
                self.pc_lock = True
                try:
                    data = pc2.read_points_numpy(self.latest_pc, field_names=("x", "y", "z", "rgb"), skip_nans=True)
                    if data.shape[0] > 0:
                        data = data[::5] # Slice downsample
                        xyz = data[:, :3]

                        keys = np.floor(xyz / v_size).astype(np.int32)
                        _, idx = np.unique(keys, axis=0, return_index=True)

                        rgb_packed = data[idx, 3].view(np.uint32)
                        colors = np.stack([
                            (rgb_packed >> 16) & 0xFF,
                            (rgb_packed >> 8) & 0xFF,
                            (rgb_packed) & 0xFF
                        ], axis=-1).astype(np.uint8)

                        # Store the finished arrays to be picked up by the render loop
                        self.ready_voxels = (xyz[idx].astype(np.float32), colors)
                finally:
                    self.pc_lock = False
                    self.latest_pc = None
            time.sleep(0.05)

    def viser_sync_loop(self):
        """Single UI thread. Packages everything into one atomic network payload at 30Hz."""
        v_size = 0.08
        while rclpy.ok():
            # Open the atomic grouping context
            with self.server.atomic():

                # 1. Update Camera
                if self.latest_img:
                    try:
                        img = self.bridge.imgmsg_to_cv2(self.latest_img, 'rgb8')
                        self.img_handle.image = cv2.resize(img, (640, 360))
                    except: pass
                    self.latest_img = None

                # 2. Update Robot Joints
                if self.latest_joints is not None and self.viser_urdf is not None:
                    self.viser_urdf.update_cfg(self.latest_joints)
                    self.latest_joints = None

                # 3. Update Voxels
                if self.ready_voxels is not None:
                    pts, colors = self.ready_voxels
                    self.server.scene.add_point_cloud(
                        "/voxels", points=pts, colors=colors,
                        point_size=v_size, precision="float16"
                    )
                    self.ready_voxels = None

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