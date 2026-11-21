import yaml
import numpy as np
import PIL.Image as Image


pgm_file_path = "/opt/ros/jazzy/share/turtlebot3_navigation2/map/map.pgm"
yaml_file_path = "/opt/ros/jazzy/share/turtlebot3_navigation2/map/map.yaml"

with open(yaml_file_path, 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

img = Image.open(pgm_file_path)
data = np.array(img)

print(yaml_data)


