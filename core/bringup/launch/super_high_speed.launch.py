"""SUPER high-speed benchmark scene on the existing tailsitter dev stack.

Launches the same plant and lidar_sim nodes as dev_sim.launch.py. The supported
entry point is `ap-pnc dev-sim prepare --scene super-high-speed`, which copies
planning_super_high_speed.yaml onto the job's planning.yaml and this sensor
file onto lidar_sim.yaml, then mounts the converted PLY. The default circle
mission is not this scene's route.

simple_sim has no collision. box_sfc or map_sfc (ground-truth /lidar_sim/map)
are the corridor paths; this launch does not start SUPER's FSM.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup = Path(get_package_share_directory("bringup"))
    return LaunchDescription([
        DeclareLaunchArgument("config", description="Absolute run-local practical plant config"),
        DeclareLaunchArgument(
            "lidar_config",
            default_value=str(bringup / "config/lidar_sim_super_high_speed.yaml"),
        ),
        DeclareLaunchArgument("rviz", default_value="false"),
        Node(package="simple_sim", executable="simple_sim_node", name="simple_sim", output="screen",
             parameters=[{"config": ParameterValue(LaunchConfiguration("config"), value_type=str)}]),
        Node(package="lidar_sim", executable="lidar_sim_node", name="lidar_sim", output="screen",
             parameters=[{"use_sim_time": True,
                          "config": ParameterValue(LaunchConfiguration("lidar_config"), value_type=str)}]),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(bringup / "rviz/dev_sim.rviz")],
             parameters=[{"use_sim_time": True}], condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
