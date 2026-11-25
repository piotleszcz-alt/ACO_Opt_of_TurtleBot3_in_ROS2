import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from aco_planner.map_data import MapData
import aco_planner.inflation as inflation
import aco_planner.config as config
from aco_planner.aco_opt import ACOPlanner
import aco_planner.tuning as tuning

import numpy as np 
import time



class AcoNode(Node):
    def __init__(self):
        super().__init__('aco_node')
        
        self.declare_parameter('map_yaml','')
        yaml_path = self.get_parameter('map_yaml').value

        if not yaml_path:
            self.get_logger().error("Brak parametru map_yaml. Uruchom z: --ros-args -p map_yaml:=<ścieżka>")
            raise RuntimeError("Brak parametru map_yaml")

        self.map = MapData(yaml_path)

        self.grid = self.map.binarize_map()
        self.dilated_grid = inflation.dilatation(config.ROBOT_SAFE_RADIUS, self.grid, self.map.get_resolution())

        self.get_logger().info(
            f"Mapa: {self.dilated_grid.shape}, przeszkody: {int(self.dilated_grid.sum())} komórek, "
            f"res = {self.map.get_resolution()} m")

        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid,'/custom_map', map_qos) # publisher do tworzenia mapy

        self.path_pub = self.create_publisher(Path, '/planned_path',10) # publisher do tworzenia trajektorii ruchu 

        self.publish_map()

        self.start = None

        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_initialpose, 10)

        

        self.get_logger().info('aco_node uruchomiony')

    def _to_cell(self, x, y, name):
        cell = self.map.world_to_grid(x,y)

        if cell is None: 
            self.get_logger().warn("Punkt jest poza mapą")
            return None
        
        r, c = cell 
        
        if self.dilated_grid[r,c] == 1:
            self.get_logger().warn("Punkt jest na przeszkodzie")
            return None
        
        self.get_logger().info(f"{name}: ({x:.2f}, {y:.2f}) m -> komórka {cell}, wolna")
        return cell 

    def publish_map(self):
        msg = OccupancyGrid()

        # nagłówek
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        # mapa
        o_x, o_y, _ = self.map.get_origin()
        msg.info.resolution = float(self.map.get_resolution())
        msg.info.height, msg.info.width = self.dilated_grid.shape
        msg.info.origin.position.x = float(o_x)
        msg.info.origin.position.y = float(o_y)
        msg.info.origin.orientation.w = 1.0 # brak obrotu

        msg.data = (self.dilated_grid.astype(np.int8) * 100).flatten().tolist()

        self.map_pub.publish(msg)
        self.get_logger().info('Mapa opublikowana przez ROS na /custom_map')

        # dane mojej mapy były w formie 0/1 - wolne/zajete --> ROS oczekuje danych w formie -1/0/100 (nieznane, wolne zajęte)



    def on_initialpose(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        cell = self._to_cell(x, y, "start")
        if cell is None:
            return
        # Zapamiętaj start po kliknięciu 2D Pose Estimate
        self.start = cell

    def on_goal(self,msg):
         # Czy start w ogóle został kliknięty
        if self.start is None:
            self.get_logger().warn("Najpierw trzeba ustawić start (2D Pose Estimate)")
            return

        x = msg.pose.position.x
        y = msg.pose.position.y
        goal = self._to_cell(x, y, "cel")
        if goal is None:
            return

        self.get_logger().info(f"Rozpoczynanie wyliczania optymalnej ścieżki z punktu: {self.start} -> {goal}")

        path = self._plan_path(self.start, goal)
        
        if path is None:
            return

        self.publish_path(path)



    def _plan_path(self, start, goal):
        params, _ = tuning.scaled_params(start, goal, config.ACO_PARAMS)

        t0 = time.perf_counter()
        try:
            planner = ACOPlanner(self.dilated_grid, start=start, goal=goal, **params)
            best_path = planner.run()
        except ValueError as e:
            self.get_logger().warn(f"ACO odrzuciło zadanie: {e}")
            return None

        dt = time.perf_counter() - t0

        if best_path is None:
            self.get_logger().warn(f"Nie znaleziono ścieżki z czasem {dt}")
            return None

        self.get_logger().info(f"Długość ścieżki w komórkach: {len(best_path)}, koszt J: {planner.best_cost:.2f}, czas obliczeń: {dt:.2f}")

        return best_path

    def publish_path(self,path):
        msg = Path()

        # nagłówek
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for r,c in path:
            pose = PoseStamped()
            pose.header = msg.header
            x,y = self.map.grid_to_world(r,c)

            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        # wysłanie publishera
        self.path_pub.publish(msg)




def main(args=None):
    rclpy.init(args=args) # uruchomienie komunikacji ROS2
    node = AcoNode() # Utworzenie węzła - wykonanie init klasy AcoNode

    try:
        rclpy.spin(node) # czekaj na zdarzenie
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()        
