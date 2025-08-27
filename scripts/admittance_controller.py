#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Point, PoseStamped
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np

class AdmittanceController(Node):
    def __init__(self):
        super().__init__('admittance_controller')

        # Parameters
        self.declare_parameter('mass', 1.0)
        self.declare_parameter('damping', 50.0)
        self.declare_parameter('rate', 100.0)  # Hz
        self.declare_parameter('goal_kp', 0.5)  # proportional gain for goal tracking

        self.mass = self.get_parameter('mass').value
        self.damping = self.get_parameter('damping').value
        self.dt = 1.0 / self.get_parameter('rate').value
        self.goal_kp = self.get_parameter('goal_kp').value

        # State
        self.velocity = np.zeros(3)  # velocity in x,y,z
        self.position = np.zeros(3)  # integrated position (goal pose)
        self.goal_position = None  # Latest goal from RViz

        # Fake constant Jacobian for testing (3x6)
        self.J = np.array([
            [0.1, 0, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0, 0],
            [0, 0, 0.1, 0, 0, 0]
        ])

        # Subscribers
        self.force_sub = self.create_subscription(
            WrenchStamped,
            '/ft_data',
            self.force_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )

        # Publishers
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10
        )

        self.marker_pub = self.create_publisher(
            Marker,
            '/force_marker',
            10
        )

        self.goal_marker_pub = self.create_publisher(
            Marker,
            '/goal_pose_marker',
            10
        )

        self.get_logger().info('Admittance controller node started.')

        # Timer for publishing goal marker at fixed rate
        self.create_timer(self.dt, self.publish_goal_marker)

    def goal_pose_callback(self, msg):
        # Update goal position from RViz PoseStamped message
        self.goal_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.get_logger().info(f"Received new goal: {self.goal_position}")

    def force_callback(self, msg):
        # Extract force vector
        F_ext = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])

        # Admittance control: acceleration
        acc = (F_ext - self.damping * self.velocity) / self.mass

        # Update velocity and position
        self.velocity += acc * self.dt
        self.position += self.velocity * self.dt

        # If a goal pose is set, generate velocity command toward goal with proportional control
        if self.goal_position is not None:
            error = self.goal_position - self.position
            goal_velocity = self.goal_kp * error
        else:
            goal_velocity = np.zeros(3)

        # Combine admittance velocity and goal tracking velocity
        combined_velocity = self.velocity + goal_velocity

        # Compute joint velocities via Jacobian transpose
        joint_velocities = self.J.T @ combined_velocity

        # Publish joint velocities (pad to 6 joints)
        cmd_msg = Float64MultiArray()
        cmd_msg.data = joint_velocities.tolist()
        while len(cmd_msg.data) < 6:
            cmd_msg.data.append(0.0)
        self.vel_pub.publish(cmd_msg)

        # Publish force visualization marker (red arrow)
        self.publish_force_marker(F_ext)

    def publish_force_marker(self, force_vec):
        force_mag = np.linalg.norm(force_vec)
        if force_mag < 1e-6:
            force_dir = np.zeros(3)
        else:
            force_dir = force_vec / force_mag

        marker = Marker()
        marker.header.frame_id = 'tool0'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'force'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # shaft diameter
        marker.scale.y = 0.02  # head diameter
        marker.scale.z = 0.02  # head length
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        start_point = Point()
        start_point.x = 0.0
        start_point.y = 0.0
        start_point.z = 0.0

        end_point = Point()
        end_point.x = force_dir[0] * min(force_mag * 0.01, 0.2)
        end_point.y = force_dir[1] * min(force_mag * 0.01, 0.2)
        end_point.z = force_dir[2] * min(force_mag * 0.01, 0.2)

        marker.points = [start_point, end_point]

        self.marker_pub.publish(marker)

    def publish_goal_marker(self):
        # Green sphere at the current admittance position (goal pose)
        marker = Marker()
        marker.header.frame_id = 'tool0'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'goal_pose'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.position.x = float(self.position[0])
        marker.pose.position.y = float(self.position[1])
        marker.pose.position.z = float(self.position[2])
        marker.pose.orientation.w = 1.0  # neutral orientation

        self.goal_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = AdmittanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Admittance controller stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

