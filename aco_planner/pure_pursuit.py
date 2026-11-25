import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Path, Odometry

import numpy as np
import math


def yaw_from_quaternion(x, y, z, w):
    yaw = math.atan2(2*(w*z + x*y), (1 - 2*(y**2 + z**2)))
    return yaw
  

def to_robot_frame(px, py, rx, ry, yaw):
    # punkt (px, py) w metrach → (x_r, y_r) w układzie robota
    dx = px - rx
    dy = py - ry

    x_r = math.cos(yaw)*dx + math.sin(yaw)*dy
    y_r = - math.sin(yaw)*dx + math.cos(yaw)*dy

    return x_r,y_r

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        self.path = []          # lista punktów (x, y) w metrach
        self.target_idx = 0     # do którego punktu ścieżki już doszliśmy
        self.pose = None        # (x, y, yaw) robota; None = jeszcze nie przyszło /odom

        self.declare_parameter('lookahead', 0.2)
        self.lookahead = self.get_parameter('lookahead').value

        self.declare_parameter('v', 0.15)
        self.v = self.get_parameter('v').value

        self.declare_parameter('goal_tolerance', 0.1)
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        self.declare_parameter('max_omega', 1.5)
        self.max_omega = self.get_parameter('max_omega').value

        # przesunięcie układu odom względem map (pozycja startowa robota w Gazebo)
        self.declare_parameter('odom_offset_x', -2.0)
        self.offset_x = self.get_parameter('odom_offset_x').value

        self.declare_parameter('odom_offset_y', -0.5)
        self.offset_y = self.get_parameter('odom_offset_y').value

        self.path_sub = self.create_subscription(Path, '/planned_path', self.on_path, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.vel_pub = self.create_publisher(TwistStamped,'/cmd_vel', 10)

        self.timer = self.create_timer(0.05, self.control_loop)


    def on_path(self,msg):       #zapisz punkty jako listę (x, y), wyzeruj indeks postępu
        self.path = []
        for p in msg.poses:

            pos = p.pose.position

            self.path.append((pos.x, pos.y))

        if len(self.path) == 0:
            self.get_logger().warn('Odebrana ścieżka ma długość 0')
            return
        
        self.target_idx = 0

        self.get_logger().info(f'Nowa ścieżka N punktów: {len(self.path)}')

    def on_odom(self, msg):      # zapisz x, y, yaw robota

        pos = msg.pose.pose.position
        x,y = pos.x, pos.y
        q = msg.pose.pose.orientation # (q.x, q.y, q.z, q.w)

        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

        self.pose = (x + self.offset_x, y + self.offset_y, yaw)

    def control_loop(self): # pętla sterowania robotem
        
        if not self.path or self.pose is None: return

        x, y, yaw = self.pose

        # sprawdzenie czy dojechaliśmy do punktu
        gx, gy = self.path[-1]
        if math.hypot(gx - x, gy - y) < self.goal_tolerance:
            self.publish_cmd(0.0, 0.0)
            self.path = []
            self.get_logger().info('Cel został osiągnięty')
            return

        tx, ty = self.find_target()

        x_r, y_r = to_robot_frame(tx, ty, x, y, yaw) # cel w koordynatach robota

        self.get_logger().info(
            f"robot=({x:.2f},{y:.2f},{yaw:.2f}) target=({tx:.2f},{ty:.2f}) "
            f"x_r={x_r:.2f} y_r={y_r:.2f}",
            throttle_duration_sec=1.0)


        if x_r < 0:  
            v = 0.0
            w = self.max_omega if y_r >= 0 else -self.max_omega
        else:
            kappa = 2 * y_r / self.lookahead**2
            v = self.v
            w = v * kappa

        w = max(-self.max_omega, min(w, self.max_omega)) # ograniczenie w do [-max_omega, max_omega]

        self.publish_cmd(v, w)

    def find_target(self):     # szukaj punktu ≥ L_d przed robotem, od ostatniego indeksu w przód
        x, y, _ = self.pose

        for i in range(self.target_idx, len(self.path)):
            px, py = self.path[i]
            if math.hypot(px - x, py - y) >= self.lookahead:
                self.target_idx = i
                return (px, py)
            
        return self.path[-1] #ostatni punkt ścieżki

    def publish_cmd(self, v, w): 
        msg = TwistStamped()

        msg.header.frame_id = 'base_link'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.twist.linear.x = float(v) #prędkość poruszania
        msg.twist.angular.z = float(w) #prędkość obrotu

        self.vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args) # uruchomienie komunikacji ROS2
    node = PurePursuit() # Utworzenie węzła - wykonanie init klasy PurePursuit

    try:
        rclpy.spin(node) # czekaj na zdarzenie
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()