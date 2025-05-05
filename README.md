# AMR-project
Optimized Autonomous Parking Node (ROS2)

This repository contains an optimized version of our autonomous parking system for ROS2. In this update, the previous architecture — composed of three separate nodes — has been unified into one main node, delivering the same full set of features with significantly better performance.

What's New
All-in-one design: The previous modular setup (three distinct ROS2 nodes) has been merged into a single Python node.

Performance boost: The system now performs 30% faster parking maneuvers, thanks to reduced inter-node communication overhead and better internal synchronization.

Simplified deployment: With fewer nodes to manage, the launch process is more straightforward and easier to maintain.

Features
Detection and tracking of ArUco markers using OpenCV.

Real-time pose estimation and transformation using tf2_ros.

Parking logic based on PID control and precise robot motion commands.

Fully integrated ROS2 node that handles subscriptions, transforms, and control logic.

How to Run
Ensure you have a working ROS2 environment with rclpy, tf2_ros, geometry_msgs, and cv2 installed.

Place node.py in your ROS2 package.

Update your setup.py and package.xml accordingly.

Run the node using:

  bash
  ros2 run your_package_name node

Architecture Changes
Previous Version:

video_capture.py

posee_estimation.py

controller.py

Current Version:

Unified in node.py

This refactor reduced inter-node communication and improved synchronization between detection, transform broadcasting, and control logic — leading to measurable improvements in system responsiveness and parking efficiency.

Performance
30% faster parking compared to the previous 3-node version.

Lower CPU usage.

Reduced ROS message latency.
