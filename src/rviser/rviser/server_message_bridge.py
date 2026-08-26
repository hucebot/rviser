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


class ServerMessageBridge(Node):

    def __init__(self):
        super().__init__("server_message_bridge")

        self.subscription = self.create_subscription(
            String,
            "/orchestrator/ui/current_state",self.state_callback,1) # or current_task

        self.get_logger().info(
            "Subscribed to orchestrator"
        )

    def state_callback(self, msg):
        robot_state = msg.data

        payload = {
            "robot_name": ROBOT_NAME,
            "message": robot_state
        }

        try:
            response = requests.post(
                f"http://{SERVER}:{PORT}/{PATH}",
                json=payload,
                timeout=0.2
            )

            self.get_logger().info(
                f"Sent state: {robot_state} "
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
