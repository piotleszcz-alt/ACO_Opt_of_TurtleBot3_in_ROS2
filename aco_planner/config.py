import os

ROBOT_SAFE_RADIUS = 0.2

RUN_TUNING = False

# Parametry do areny
ACO_PARAMS_ARENA = {
    "beta": 5.0,
    "q_0": 0.95,
    "max_iter": 30,
    "alfa": 1.0,
    "lam": 0.2
}


ACO_PARAMS_MAZE = {
    "beta": 2.0,
    "q_0": 0.7,
    "max_iter": 60,
    "alfa": 3.0,
    "lam": 0.0,
}

SCALING_ARENA = {
    "ants_per_px": 0.25, "ants_min": 10, "ants_max": 50,
    "steps_factor": 4.0, "steps_min": 100, "scale_Q": False,
}

SCALING_MAZE = {
    "ants_per_px": 0.25, "ants_min": 20, "ants_max": 50,
    "steps_factor": 15.0, "steps_min": 100, "scale_Q": False,
}

PROFILES = {
    "arena": {
        "map": "/opt/ros/jazzy/share/turtlebot3_navigation2/map/map.yaml",
        "params": ACO_PARAMS_ARENA,
        "scaling": SCALING_ARENA,
    },
    "maze": {
        "map": os.path.expanduser("~/ros2_ws/src/aco_planner/maps/maze/maze.yaml"),
        "params": ACO_PARAMS_MAZE,
        "scaling": SCALING_MAZE,
    },
}

PROFILE = "maze"