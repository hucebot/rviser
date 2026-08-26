#!/usr/bin/env python3

import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Credentials wifi
# Name: OpossumNet
# Pwd: Strong_EU_Coop4?

SERVER = "192.168.253.48"
PORT = 8001
PATH = "message"
ROBOT_NAME = "tiago_inria"

# Friendly name for what's being handled, derived from the orchestrator's task name
# (see pick_place_orchestrator.py: PickPlaceOrchestrator.tasks).
TASK_OBJECTS = {
    "open_fridge": "the fridge",
    "close_fridge": "the fridge",
    "pick_milk": "the milk",
    "place_milk": "the milk",
    "pick_banana": "the banana",
    "place_banana": "the banana",
    "pick_baguette": "the baguette",
    "place_baguette": "the baguette",
    "pick_apple": "the apple",
    "place_apple": "the apple",
}

# task name prefix/name -> (verb, present participle used for "just finished ...")
TASK_VERBS = {
    "pick_": ("pick", "picking up"),
    "place_": ("place", "placing"),
    "open_fridge": ("open", "opening"),
    "close_fridge": ("close", "closing"),
}

# Orchestrator state class name -> narration phase, for phases that read the same
# regardless of which task is active (see pick_place_orchestrator.py: BaseState subclasses).
STATE_PHASES = {
    "StartPreNavAndMeshState": "navigating",
    "WaitPreNavState": "navigating",
    "StartHomeAndNavState": "navigating",
    "WaitHomeAndNavState": "navigating",
    "WaitDockPoseState": "docking",
    "ExecuteDockState": "docking",
    "WaitDockState": "docking",
    "VerifyMeshState": "localizing object",
    "WaitTargetPoseState": "localizing object",
    "TaskDoneState": "task_done",
    "DoneState": "all_done",
}


def _task_verb(task_name):
    """('pick_milk') -> ('pick', 'picking up')."""
    for prefix, verb_info in TASK_VERBS.items():
        if task_name == prefix or task_name.startswith(prefix):
            return verb_info
    return ("work on", "working on")


def narrate(task_name, state_name):
    """Turns an (task, state) pair from the orchestrator into a plain-language
    sentence for the TV narrator - no state-machine or robotics jargon."""
    if not task_name:
        return "The robot is getting ready to start."

    obj = TASK_OBJECTS.get(task_name, task_name.replace("_", " "))
    verb, verb_ing = _task_verb(task_name)
    phase = STATE_PHASES.get(state_name)

    if phase == "navigating":
        return f"The robot is on its way to {obj}."
    if phase == "docking":
        return f"The robot is getting into position near {obj}."
    if phase == "localizing object":
        return f"The robot is spotting {obj} and planning its next move."
    if phase == "task_done":
        return f"Done! The robot just finished {verb_ing} {obj}."
    if phase == "all_done":
        return "All tasks complete — the robot has finished its run!"

    # Remaining states (ExecuteSkillState / WaitSkillState) are the actual skill in progress.
    if verb == "pick":
        return f"The robot is picking up {obj}."
    if verb == "place":
        return f"The robot is placing {obj} in the fridge."
    if verb == "open":
        return f"The robot is opening {obj}."
    if verb == "close":
        return f"The robot is closing {obj}."
    return f"The robot is working on {obj}."


class ServerMessageBridge(Node):

    def __init__(self):
        super().__init__("server_message_bridge")

        self.latest_task = ""
        self.latest_state = ""
        self._last_sent_message = None

        self.create_subscription(
            String, "/orchestrator/ui/current_task", self._task_callback, 1
        )
        self.create_subscription(
            String, "/orchestrator/ui/current_state", self._state_callback, 1
        )

        self.get_logger().info(
            "Subscribed to orchestrator task/state topics"
        )

    def _task_callback(self, msg):
        self.latest_task = msg.data
        self._send_narration()

    def _state_callback(self, msg):
        self.latest_state = msg.data
        self._send_narration()

    def _send_narration(self):
        message = narrate(self.latest_task, self.latest_state)

        # Task and state arrive as separate messages, so skip re-sending the
        # same sentence if the other topic just fired without changing the meaning.
        if message == self._last_sent_message:
            return
        self._last_sent_message = message

        payload = {
            "robot_name": ROBOT_NAME,
            "message": message
        }

        try:
            response = requests.post(
                f"http://{SERVER}:{PORT}/{PATH}",
                json=payload,
                timeout=0.2
            )

            self.get_logger().info(
                f"Sent narration: \"{message}\" "
                f"(HTTP {response.status_code})"
            )

        except requests.exceptions.ConnectionError as e:
            self.get_logger().error(f"Connection error: {e}")

        except requests.exceptions.Timeout as e:
            self.get_logger().warning(f"Server timeout: {SERVER}:{PORT} - {e}")

        except requests.exceptions.RequestException as e:
            self.get_logger().warning(
                f"Server not reachable at {SERVER}:{PORT}: {e}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = ServerMessageBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
