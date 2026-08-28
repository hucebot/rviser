from http import server
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image, JointState, BatteryState
from std_msgs.msg import String, Float32
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path as PathMsg
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

import base64, io, json, threading, time, cv2, os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class RviserEmergencyNode(Node):
    # Category -> display color for the orchestrator status widget.
    ORCHESTRATOR_CATEGORY_COLORS = {
        "navigating": "#3b82f6",   # blue
        "docking": "#8b5cf6",      # purple
        "locating": "#eab308",   # yellow
        "picking": "#f97316",      # orange
        "placing": "#02501f",      # dark green
        "opening": "#06b6d4",      # cyan
        "closing": "#0ea5e9",      # sky blue
        "done": "#06E02A",         # light green
        "idle": "#6b7280",         # gray
    }

    # Orchestrator state class name -> category, for states whose category doesn't
    # depend on which task is active (nav/dock/localize phases are the same for every task).
    ORCHESTRATOR_STATE_CATEGORIES = {
        "StartPreNavAndMeshState": "navigating",
        "WaitPreNavState": "navigating",
        "StartHomeAndNavState": "navigating",
        "WaitHomeAndNavState": "navigating",
        "WaitDockPoseState": "docking",
        "ExecuteDockState": "docking",
        "WaitDockState": "docking",
        "VerifyMeshState": "locating",
        "WaitTargetPoseState": "locating",
        "DoneState": "done",
    }

    # Two scene_assets mesh variants for the same physical fridge - the orchestrator's
    # "open_fridge"/"close_fridge" tasks (pick_place_orchestrator.py: PickPlaceOrchestrator.tasks)
    # toggle which one is visible once the corresponding skill actually finishes.
    FRIDGE_SCENE_ASSET_KEYS = {
        "closed": "inria_fridge_closed_and_full_kitchen_centered_apriltag",
        "open": "inria_fridge_open_and_full_kitchen_centered_apriltag",
    }

    def __init__(self):
        super().__init__('rviser_emergency_node')

        self.server = viser.ViserServer()
        self.bridge = cv_bridge.CvBridge()
        self.state_lock = threading.Lock()
        
        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self.server.scene.add_grid("/ground_grid", width=20.0, height=20.0, cell_size=0.5)

        self.server.initial_camera.position = (2.0, 1.0, 1.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)

        # GUI is organized into tabs (like browser tabs): switching tabs swaps the whole panel.
        self.gui_tab_group = self.server.gui.add_tab_group()

        self.latest_orchestrator_task = ""
        self.latest_orchestrator_state = ""
        with self.gui_tab_group.add_tab("Main"):
            self.status_html = self.server.gui.add_html(
                self._render_orchestrator_status_html("", "")
            )
            self.img_handle = self.server.gui.add_image(np.zeros((360, 640, 3), dtype=np.uint8), label="Camera")

        self.viser_urdf = None
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.file_path = Path(__file__).resolve()
        self.mesh_cache = {}
        self.mesh_scale = 1.

        self.battery_level_topic = self.declare_parameter('battery_level_topic', '/power/battery_level').value
        self.latest_battery_pct = None
        self.last_rendered_battery = -1

        self.laptop_battery_topic = self.declare_parameter('laptop_battery_topic', 'laptop_battery').value
        self.latest_laptop_battery_pct = None
        self.last_rendered_laptop_battery = -1

        # Internal State Buffers
        self.latest_img = None
        self.latest_joints = None
        self.latest_traj = {"left": None, "right": None}
        self.latest_traj_colors = {"left": None, "right": None}
        self.latest_traj_frame = {"left": "gripper_left_grasping_link", "right": "gripper_right_grasping_link"}
        self.traj_dirty = {"left": False, "right": False}
        self.traj_handles = {"left": None, "right": None}
        self.marker_mesh_handles = {}  # scene_path -> (handle, mesh_resource)
        self.marker_last_seen = {}  # scene_path -> monotonic timestamp of last update
        self.marker_timeout_sec = 3.0  # remove a marker mesh if not refreshed within this long

        self.latest_object_marker = None
        self.latest_robot_pose = None

        self.latest_local_plan = None
        self.local_plan_frame = None
        self.local_plan_dirty = False
        self.local_plan_handle = None

        self.fridge_mesh_handles = {}  # "open"/"closed" -> mesh handle, filled in by _load_scene_assets

        self.scene_assets_dir = self.file_path.parent.parent / "scene_assets"
        self.exported_poses_dir = self.file_path.parent.parent / "exported_poses"
        self._load_scene_assets()

        self.assets_dir = self.file_path.parent.parent / "assets"

        # Parameters
        self.declare_parameter('first_person', False)
        self.declare_parameter('show_opensot', False)
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('opensot_topic', '/opensot/joint_states')

        self.first_person = self.get_parameter('first_person').value
        self.show_opensot = self.get_parameter('show_opensot').value
        self.first_person_link = 'head_front_camera_link'
        self.first_person_fov_deg = float(np.rad2deg(self.server.initial_camera.fov))  # zoom: smaller = zoomed in
        self._first_person_active_last_frame = self.first_person

        self.add_on_set_parameters_callback(self.parameter_callback)

        # Extra GUI tab for manual controls (buttons/sliders go here)
        with self.gui_tab_group.add_tab("Controls"):
            self.first_person_button = self.server.gui.add_button("Enable First Person", color="green")
            self.first_person_button.on_click(lambda _: self._toggle_first_person())

            self.first_person_fov_slider = self.server.gui.add_slider(
                "Zoom (FOV °)", min=20.0, max=120.0, step=1.0,
                initial_value=self.first_person_fov_deg,
            )
            self.first_person_fov_slider.on_update(
                lambda event: setattr(self, "first_person_fov_deg", event.target.value)
            )

        # Absolute-positioned HTML overlays for the battery indicators (bottom right
        # corner, laptop stacked above the robot so they don't overlap).
        self.battery_html = self.server.gui.add_html(
            self._render_battery_html(None, self.ROBOT_BATTERY_THRESHOLDS, "🤖", bottom_px=15)
        )
        self.laptop_battery_html = self.server.gui.add_html(
            self._render_battery_html(None, self.LAPTOP_BATTERY_THRESHOLDS, "💻", bottom_px=60)
        )

        # Absolute-positioned HTML overlay for the Inria logo (bottom left corner)
        logo_html = self._render_logo_html()
        if logo_html is not None:
            self.server.gui.add_html(logo_html)

        # Matches orchestrator_node.py's qos_state: current_task/current_state are only
        # published on change, so TRANSIENT_LOCAL is needed here too to get the last value
        # on connect (e.g. if this node starts/restarts after the orchestrator already moved
        # past a task/state) - a VOLATILE subscriber does not get replayed history even
        # against a TRANSIENT_LOCAL publisher.
        orchestrator_ui_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=3,
        )

        # Subscriptions
        self.create_subscription(MarkerArray, '/rviser/left/trajectory', self.left_traj_cb, 10)
        self.create_subscription(MarkerArray, '/rviser/right/trajectory', self.right_traj_cb, 10)
        self.create_subscription(Marker, '/foundation_pose/object_marker', self.object_marker_cb, 10)
        self.create_subscription(PoseStamped, '/robot_pose', self.robot_pose_cb, 10)
        self.create_subscription(String, 'robot_description', self.urdf_cb,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Image, self.get_parameter('camera_topic').value, self.img_cb, 10)
        self.create_subscription(JointState, self.get_parameter('opensot_topic').value, self.opensot_joint_cb, 10)
        self.create_subscription(JointState, self.get_parameter('joint_topic').value, self.joint_cb, 10)
        self.create_subscription(String, '/orchestrator/ui/current_task', self.orchestrator_task_cb, orchestrator_ui_qos)
        self.create_subscription(String, '/orchestrator/ui/current_state', self.orchestrator_state_cb, orchestrator_ui_qos)
        self.create_subscription(PathMsg, '/local_plan', self.local_plan_cb, 10)
        self.create_subscription(Float32, self.battery_level_topic, self.battery_level_callback, 10)
        self.create_subscription(BatteryState, self.laptop_battery_topic, self.laptop_battery_callback, 10)

        threading.Thread(target=self.viser_sync_loop, daemon=True).start()

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'first_person':
                self.first_person = param.value
            elif param.name == 'show_opensot':
                self.show_opensot = param.value
        return SetParametersResult(successful=True)

    # (min percentage, color) pairs, checked highest-first. The robot shuts itself off at
    # 40%, so its "red" warning kicks in well above where a laptop's would.
    ROBOT_BATTERY_THRESHOLDS = [(70, "#22c55e"), (50, "#ff9040"), (0, "#ef4444")]
    # Regular consumer-electronics convention (matches typical OS low-battery warnings).
    LAPTOP_BATTERY_THRESHOLDS = [(50, "#22c55e"), (20, "#ff9040"), (0, "#ef4444")]

    def battery_level_callback(self, msg):
        with self.state_lock:
            val = msg.data
            # Accept either a 0-1 normalized level or an already-percent value.
            if 0.0 < val <= 1.0:
                self.latest_battery_pct = int(val * 100)
            else:
                self.latest_battery_pct = int(val)

    def laptop_battery_callback(self, msg):
        with self.state_lock:
            val = msg.percentage
            # BatteryState.percentage is nominally 0-1, but accept 0-100 too, same as
            # the robot's Float32 topic.
            if 0.0 < val <= 1.0:
                self.latest_laptop_battery_pct = int(val * 100)
            else:
                self.latest_laptop_battery_pct = int(val)

    def _battery_color(self, val, thresholds):
        for min_pct, color in thresholds:
            if val >= min_pct:
                return color
        return thresholds[-1][1]

    def _render_battery_html(self, percentage, thresholds, icon, bottom_px):
        if percentage is None:
            val = "--"
            width = 0
            color = "gray"
        else:
            val = int(percentage)
            width = max(0, min(100, val))
            color = self._battery_color(val, thresholds)

        return f'''
        <div style="position: fixed; bottom: {bottom_px}px; right: 15px; z-index: 9999;
                    display: flex; align-items: center; gap: 8px;
                    background: rgba(0, 0, 0, 0.6); padding: 8px 12px;
                    border-radius: 8px; color: white; font-family: sans-serif; font-weight: bold; pointer-events: none;">
            <div style="font-size: 16px; line-height: 1;">{icon}</div>
            <div style="width: 30px; height: 14px; border: 2px solid white; border-radius: 3px; position: relative; padding: 1px;">
                <div style="width: {width}%; height: 100%; background: {color}; border-radius: 1px; transition: width 0.3s ease;"></div>
                <div style="position: absolute; right: -4px; top: 3px; width: 2px; height: 6px; background: white; border-radius: 0 2px 2px 0;"></div>
            </div>
            <div style="font-size: 14px; min-width: 40px;">{val}%</div>
        </div>
        '''

    def _render_logo_html(self):
        logo_path = self.assets_dir / "Inr_logo_fr_rouge.png"
        if not logo_path.exists():
            self.get_logger().warn(f"Logo not found: {logo_path}")
            return None

        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")

        return f'''
        <img src="data:image/png;base64,{logo_b64}"
             style="position: fixed; top: 15px; left: 15px; z-index: 9999;
                    height: 80px; pointer-events: none;" />
        '''

    def _toggle_first_person(self):
        new_value = not self.first_person
        self.set_parameters([Parameter('first_person', Parameter.Type.BOOL, new_value)])

        self.first_person_button.label = "Disable First Person" if new_value else "Enable First Person"
        self.first_person_button.color = "red" if new_value else "green"
        self.get_logger().info(f"First-person view {'enabled' if new_value else 'disabled'} via GUI button.")

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
            if msg.mesh_resource and msg.action != Marker.DELETE:
                scene_path = f"/robot/markers/{msg.ns}_{msg.id}"
                self.marker_last_seen[scene_path] = time.monotonic()

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

    def local_plan_cb(self, msg):
        with self.state_lock:
            if not msg.poses:
                self.latest_local_plan = None
                self.local_plan_frame = None
            else:
                self.latest_local_plan = np.array(
                    [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                    dtype=np.float32,
                )
                self.local_plan_frame = msg.header.frame_id
            self.local_plan_dirty = True

    def _load_scene_assets(self):
        """Loads static scene object meshes (from scene_assets/) at their latest
        exported pose (from exported_poses/<object_key>/pose_*.json, written by
        scene_manager's manual_adjust_node). These are placed once at startup,
        as root-level scene nodes in the "map" frame."""
        if not self.exported_poses_dir.is_dir():
            return

        fridge_key_to_state = {v: k for k, v in self.FRIDGE_SCENE_ASSET_KEYS.items()}

        for object_dir in sorted(self.exported_poses_dir.iterdir()):
            if not object_dir.is_dir():
                continue

            pose_files = sorted(object_dir.glob("pose_*.json"))
            if not pose_files:
                continue
            latest_pose_file = pose_files[-1]

            object_key = object_dir.name
            mesh_path = self.scene_assets_dir / f"{object_key}.obj"
            if not mesh_path.exists():
                self.get_logger().warn(f"Scene asset mesh not found for '{object_key}': {mesh_path}")
                continue

            try:
                with open(latest_pose_file) as f:
                    pose = json.load(f)
                position = pose["position"]
                orientation = pose["orientation"]


                mesh = trimesh.load(str(mesh_path), force='mesh', process=False, skip_materials=False)
            except Exception as e:
                self.get_logger().error(f"Failed to load scene asset '{object_key}': {e}")
                continue

            handle = self.server.scene.add_mesh_trimesh(
                f"/scene_assets/{object_key}",
                mesh=mesh,
                position=(position["x"], position["y"], position["z"]),
                wxyz=(orientation["w"], orientation["x"], orientation["y"], orientation["z"]),
            )
            self.get_logger().info(
                f"Loaded scene asset '{object_key}' at exported pose {latest_pose_file.name}"
            )

            fridge_state = fridge_key_to_state.get(object_key)
            if fridge_state is not None:
                # Both variants are loaded up front so swapping is instant (no trimesh
                # reload); only the "closed" one starts visible.
                handle.visible = (fridge_state == "closed")
                self.fridge_mesh_handles[fridge_state] = handle

    def _set_fridge_open(self, is_open: bool):
        closed_handle = self.fridge_mesh_handles.get("closed")
        open_handle = self.fridge_mesh_handles.get("open")
        if closed_handle is not None:
            closed_handle.visible = not is_open
        if open_handle is not None:
            open_handle.visible = is_open

    def _clear_hand_trajectories(self):
        """Drops the last-drawn gripper trajectory lines so they don't linger in the
        scene once the pick/place skill that produced them has finished."""
        with self.state_lock:
            self.latest_traj = {"left": None, "right": None}
            self.latest_traj_colors = {"left": None, "right": None}
            self.latest_traj_frame = {"left": None, "right": None}
            self.traj_dirty = {"left": True, "right": True}

    def orchestrator_task_cb(self, msg):
        with self.state_lock:
            self.latest_orchestrator_task = msg.data
            task, state = self.latest_orchestrator_task, self.latest_orchestrator_state
        self.status_html.content = self._render_orchestrator_status_html(task, state)

    def orchestrator_state_cb(self, msg):
        with self.state_lock:
            self.latest_orchestrator_state = msg.data
            task, state = self.latest_orchestrator_task, self.latest_orchestrator_state
        self.status_html.content = self._render_orchestrator_status_html(task, state)

        # The fridge mesh only flips once its skill has actually finished (TaskDoneState),
        # i.e. once the physical door motion is done - see pick_place_orchestrator.py's
        # "open_fridge"/"close_fridge" tasks and their ExecuteSkillState -> WaitSkillState
        # -> TaskDoneState transition.
        if state == "TaskDoneState":
            if task == "open_fridge":
                self._set_fridge_open(True)
            elif task == "close_fridge":
                self._set_fridge_open(False)

            # The hand trajectory line is only useful while the pick/place is in
            # flight; once the skill finishes, clear it so it doesn't stay drawn
            # at the gripper's last pose.
            if task.startswith("pick_") or task.startswith("place_"):
                self._clear_hand_trajectories()

    def _orchestrator_category(self, task_name, state_name):
        category = self.ORCHESTRATOR_STATE_CATEGORIES.get(state_name)
        if category is not None:
            return category

        # Remaining states (ExecuteSkillState/WaitSkillState/TaskDoneState) depend on
        # which task is actually running the skill.
        if task_name.startswith("pick_"):
            return "picking"
        if task_name.startswith("place_"):
            return "placing"
        if task_name == "open_fridge":
            return "opening"
        if task_name == "close_fridge":
            return "closing"
        return "idle"

    def _render_orchestrator_status_html(self, task_name, state_name):
        if not task_name and not state_name:
            category, label = "idle", "IDLE"
        else:
            category = self._orchestrator_category(task_name, state_name)
            label = category.upper()
        color = self.ORCHESTRATOR_CATEGORY_COLORS.get(category, "#6b7280")

        # "solevita" is the orchestrator's internal object key for the juice carton
        # (see pick_place_orchestrator.py: "pick_solevita"/"place_solevita") - show the
        # friendlier "juice" name instead, without touching the real task name used above.
        display_task = task_name.replace("solevita", "juice") if task_name else task_name

        return f'''
        <div style="background:{color}; color:white; font-weight:600; text-align:center;
                    padding:10px 6px; border-radius:6px; font-family:sans-serif;
                    font-size:14px; letter-spacing:0.5px;">
          {label}
          <div style="font-size:11px; font-weight:400; opacity:0.85; margin-top:2px;">
            {display_task or "&mdash;"}
          </div>
        </div>
        '''

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
                    current_marker_last_seen = self.marker_last_seen.copy()
                    current_battery = self.latest_battery_pct
                    current_laptop_battery = self.latest_laptop_battery_pct

                    # Copy trajectory data but DO NOT reset self.latest_traj to None
                    current_traj = self.latest_traj.copy()
                    current_traj_colors = self.latest_traj_colors.copy()
                    current_traj_frame = self.latest_traj_frame.copy()
                    dirty_flags = self.traj_dirty.copy()

                    current_local_plan = self.latest_local_plan
                    current_local_plan_frame = self.local_plan_frame
                    local_plan_dirty = self.local_plan_dirty

                    self.latest_img = None
                    self.latest_joints = None
                    self.latest_object_marker = None
                    self.traj_dirty = {"left": False, "right": False}
                    self.local_plan_dirty = False

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
                                    client.camera.look_at = (float(camera_look_at[0] + 0.7), float(camera_look_at[1]), float(camera_look_at[2]))
                                    client.camera.fov = float(np.deg2rad(self.first_person_fov_deg))
                            except Exception:
                                pass
                elif self._first_person_active_last_frame:
                    default_position = tuple(float(v) for v in self.server.initial_camera.position)
                    default_look_at = tuple(float(v) for v in self.server.initial_camera.look_at)
                    default_fov = float(self.server.initial_camera.fov)

                    for client in self.server.get_clients().values():
                        try:
                            with client.atomic():
                                client.camera.position = default_position
                                client.camera.look_at = default_look_at
                                client.camera.fov = default_fov
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
                    "left": (np.array([30, 170, 255], dtype=np.uint8), "/robot/trajectory/left"),
                    "right": (np.array([255, 120, 30], dtype=np.uint8), "/robot/trajectory/right"),
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
                        # graph_inference_node publishes "<target>_exec_anchor" frames via a static
                        # transform rooted at "opensot/base_link" (OpenSoT's own TF tree), not the
                        # plain "base_link" tree used everywhere else here - look those up accordingly.
                        tf_parent_frame = 'opensot/base_link' if header_frame.endswith('_exec_anchor') else 'base_link'
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                tf_parent_frame, header_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.01)
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

                # 4b. Update local plan visualization (Nav2's /local_plan). Added at the
                # scene root (like the scene_assets meshes) rather than under "/robot", since
                # a nav plan is a path through the world, not something attached to the robot body.
                if local_plan_dirty:
                    points = current_local_plan
                    header_frame = current_local_plan_frame

                    if points is None or header_frame is None or points.shape[0] < 2:
                        if self.local_plan_handle is not None:
                            self.local_plan_handle.remove()
                            self.local_plan_handle = None
                    else:
                        points_transformed = points
                        if header_frame != 'map':
                            try:
                                transform = self.tf_buffer.lookup_transform(
                                    'map', header_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.01)
                                )
                                trans = transform.transform.translation
                                rot = transform.transform.rotation
                                tf_matrix = self._pose_matrix(
                                    [trans.x, trans.y, trans.z], [rot.w, rot.x, rot.y, rot.z]
                                )
                                points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
                                points_transformed = (tf_matrix @ points_homogeneous.T).T[:, :3]
                            except Exception as e:
                                self.get_logger().debug(f"TF lookup failed for local_plan: {e}")

                        if self.local_plan_handle is not None:
                            self.local_plan_handle.remove()

                        segment_points = np.stack([points_transformed[:-1], points_transformed[1:]], axis=1)
                        self.local_plan_handle = self.server.scene.add_line_segments(
                            "/local_plan",
                            points=segment_points,
                            colors=np.array([0, 0, 0], dtype=np.uint8),  # black, distinct from gripper trajectories
                            line_width=4.0,
                        )

                # 5. Add meshes from object_marker topic (Now Async)
                if current_marker is not None and current_marker.mesh_resource and current_marker.action == Marker.DELETE:
                    scene_path = f"/robot/markers/{current_marker.ns}_{current_marker.id}"
                    cached = self.marker_mesh_handles.pop(scene_path, None)
                    if cached is not None:
                        cached[0].remove()
                    with self.state_lock:
                        self.marker_last_seen.pop(scene_path, None)
                    current_marker_last_seen.pop(scene_path, None)

                elif current_marker is not None and current_marker.mesh_resource:
                    mesh = self.mesh_cache.get(current_marker.mesh_resource)
                    scene_path = f"/robot/markers/{current_marker.ns}_{current_marker.id}"

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

                        cached = self.marker_mesh_handles.get(scene_path)
                        if cached is not None and cached[1] == current_marker.mesh_resource:
                            # Same mesh already in the scene: just move it, don't re-upload geometry/texture.
                            handle = cached[0]
                            handle.wxyz = tuple(wxyz)
                            handle.position = tuple(pos)
                        else:
                            if cached is not None:
                                cached[0].remove()
                            handle = self.server.scene.add_mesh_trimesh(
                                scene_path,
                                mesh=mesh,
                                wxyz=tuple(wxyz),
                                position=tuple(pos),
                            )
                            self.marker_mesh_handles[scene_path] = (handle, current_marker.mesh_resource)

                # Remove marker meshes that haven't been refreshed recently (e.g. tracking lost/occluded),
                # so they don't stay stuck at their last known pose forever.
                now = time.monotonic()
                stale_paths = [
                    path for path in self.marker_mesh_handles
                    if now - current_marker_last_seen.get(path, 0.0) > self.marker_timeout_sec
                ]
                for path in stale_paths:
                    cached = self.marker_mesh_handles.pop(path, None)
                    if cached is not None:
                        cached[0].remove()
                    with self.state_lock:
                        self.marker_last_seen.pop(path, None)

                # 6. Battery indicator overlays
                if current_battery is not None and current_battery != self.last_rendered_battery:
                    self.battery_html.content = self._render_battery_html(
                        current_battery, self.ROBOT_BATTERY_THRESHOLDS, "🤖", bottom_px=15
                    )
                    self.last_rendered_battery = current_battery

                if current_laptop_battery is not None and current_laptop_battery != self.last_rendered_laptop_battery:
                    self.laptop_battery_html.content = self._render_battery_html(
                        current_laptop_battery, self.LAPTOP_BATTERY_THRESHOLDS, "💻", bottom_px=60
                    )
                    self.last_rendered_laptop_battery = current_laptop_battery

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