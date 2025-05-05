#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import time
import signal
import sys

class MarkerFollowerNode(Node):
    def __init__(self):
        super().__init__('marker_follower_node')
        
        # Publisher for processed image
        self.processed_image_publisher = self.create_publisher(Image, '/pose_estimation/output_image', 10)
        
        # Publisher for robot velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # CvBridge for converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            return
        
        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # Camera calibration parameters
        self.intrinsic_camera = np.array(((933.15867, 0, 657.59), 
                                         (0, 933.1586, 400.36993), 
                                         (0, 0, 1)))
        self.distortion = np.array((-0.43948, 0.18514, 0, 0))
        
        # 3D pose estimation parameters
        self.marker_size = 0.02
        self.obj_points = np.array([
            [-self.marker_size / 2, self.marker_size / 2, 0],
            [self.marker_size / 2, self.marker_size / 2, 0],
            [self.marker_size / 2, -self.marker_size / 2, 0],
            [-self.marker_size / 2, -self.marker_size / 2, 0]
        ], dtype=np.float32)
        
        # Target parameters for marker following
        self.target_marker_size = 100  # Target size in pixels
        self.target_x_position = 640   # Center of 1280-width frame
        self.target_y_position = 360   # Center of 720-height frame
        
        # PID controller parameters - REDUCED VALUES
        self.x_pid = PIDController(kp=0.0005, ki=0.00005, kd=0.0001)  # For x-axis positioning
        self.y_pid = PIDController(kp=0.0005, ki=0.00005, kd=0.0001)  # For y-axis positioning
        self.z_pid = PIDController(kp=0.0005, ki=0.00005, kd=0.0001)  # For distance/size control
        
        # Robot movement parameters
        self.max_linear_speed = 0.2  # Reduced from 0.3 for more stability
        self.max_angular_speed = 0.3  # Reduced from 0.5 for more stability
        
        # Search mode parameters
        self.search_mode = False
        self.last_marker_time = self.get_clock().now()
        self.marker_timeout = 5.0  # 5 seconds without detection before starting search
        self.marker_found = False  # Track if a marker has ever been found
        self.was_tracking = False  # Flag to ensure we send stop commands when switching modes
        
        # Create a timer for periodic processing (30 FPS)
        self.timer = self.create_timer(0.033, self.timer_callback)
        
        self.get_logger().info(f"OpenCV version: {cv2.__version__}")
        self.get_logger().info("Unified marker follower node started...")

    def timer_callback(self):
        # Capture frame
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame")
            return
        
        # Process the frame to detect and estimate pose of markers
        processed_frame = self.process_frame(frame)
        
        # Publish the processed image
        output_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding="bgr8")
        self.processed_image_publisher.publish(output_msg)
        
        # Show the frame with debug info
        cv2.imshow("Marker Follower", processed_frame)
        cv2.waitKey(1)

    def process_frame(self, frame):
        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Initialize velocity command
        twist = Twist()
        
        if ids is not None and len(corners) > 0:
            # Draw detected markers on frame
            cv2.aruco.drawDetectedMarkers(frame, corners)
            
            # Marker detected - reset search mode and update last detection time
            if self.search_mode:
                self.get_logger().info("Marker found after searching")
                # Reset PID controllers to avoid accumulated error
                self.x_pid.reset()
                self.y_pid.reset()
                self.z_pid.reset()
                
            self.search_mode = False
            self.last_marker_time = self.get_clock().now()
            self.marker_found = True
            self.was_tracking = True
            
            # Process each detected marker (but only use the first one for control)
            for i in range(len(ids)):
                marker_corners = corners[i][0]
                
                # Estimate 3D pose
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, marker_corners, self.intrinsic_camera, self.distortion)
                
                if success:
                    # Draw pose axes on the frame
                    cv2.drawFrameAxes(frame, self.intrinsic_camera, self.distortion, rvec, tvec, 0.01)
                    
                    # Log pose information (only for the first marker)
                    if i == 0:
                        self.get_logger().info(f"Marker ID: {ids[i]}")
                        self.get_logger().info(f"Translation: x={tvec[0][0]:.4f}, y={tvec[1][0]:.4f}, z={tvec[2][0]:.4f} meters")
                        self.get_logger().info(f"Rotation: x={rvec[0][0]:.4f}, y={rvec[1][0]:.4f}, z={rvec[2][0]:.4f} radians")
                else:
                    self.get_logger().warning(f"Pose estimation failed for Marker ID: {ids[i]}")
            
            # Use the first marker for following behavior
            marker_corners = corners[0][0]
            
            # Calculate marker center
            marker_center_x = np.mean(marker_corners[:, 0])
            marker_center_y = np.mean(marker_corners[:, 1])
            
            # Calculate marker size (average of width and height)
            width = np.linalg.norm(marker_corners[0] - marker_corners[1])
            height = np.linalg.norm(marker_corners[1] - marker_corners[2])
            marker_size = (width + height) / 2
            
            # Calculate errors
            x_error = self.target_x_position - marker_center_x  # Positive: marker is to the left
            y_error = self.target_y_position - marker_center_y  # Positive: marker is above
            z_error = self.target_marker_size - marker_size     # Positive: marker is too small (too far)
            
            # Update PID controllers
            x_correction = self.x_pid.update(x_error)
            y_correction = self.y_pid.update(y_error)
            z_correction = self.z_pid.update(z_error)
            
            # Convert corrections to robot velocities
            # X correction controls rotation
            angular_z = x_correction
            # Z correction controls forward/backward movement
            linear_x = z_correction
            
            # Apply limits
            twist.linear.x = max(min(linear_x, self.max_linear_speed), -self.max_linear_speed)
            twist.angular.z = max(min(angular_z, self.max_angular_speed), -self.max_angular_speed)
            
            # Apply a minimum threshold to avoid micro-movements
            if abs(twist.linear.x) < 0.01:
                twist.linear.x = 0.0
            if abs(twist.angular.z) < 0.01:
                twist.angular.z = 0.0
            
            # Log control information
            self.get_logger().info(f"Marker detected: Center({marker_center_x:.1f}, {marker_center_y:.1f}), Size: {marker_size:.1f}")
            self.get_logger().info(f"Errors: X:{x_error:.1f}, Y:{y_error:.1f}, Size:{z_error:.1f}")
            self.get_logger().info(f"Commands: LinearX:{twist.linear.x:.3f}, AngularZ:{twist.angular.z:.3f}")
            
            # Draw target indicators on the frame
            cv2.circle(frame, (int(marker_center_x), int(marker_center_y)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (self.target_x_position, self.target_y_position), 5, (0, 0, 255), -1)
            cv2.line(frame, (int(marker_center_x), int(marker_center_y)), 
                     (self.target_x_position, self.target_y_position), (255, 0, 0), 2)
            
        else:
            # No marker detected
            if self.was_tracking:
                # Just switched from tracking to not tracking, ensure PIDs are reset
                self.x_pid.reset()
                self.y_pid.reset()
                self.z_pid.reset()
                self.was_tracking = False
                
            current_time = self.get_clock().now()
            time_since_last_marker = (current_time - self.last_marker_time).nanoseconds / 1e9
            
            if self.marker_found and time_since_last_marker > self.marker_timeout:
                if not self.search_mode:
                    self.get_logger().warning(f"No marker detected for {self.marker_timeout} seconds - starting search")
                    self.search_mode = True
                # Apply search behavior only after timeout and if we've seen a marker before
                twist.angular.z = 0.1  # Slow rotation to search for markers
                twist.linear.x = 0.0  # Don't move forward during search
            else:
                # Just stop if marker recently disappeared or never detected
                search_status = "waiting" if self.marker_found else "no marker ever detected"
                self.get_logger().warning(f"No marker detected - {search_status} ({time_since_last_marker:.1f}/{self.marker_timeout} sec)")
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        
        # Add status text to the frame
        status_text = "Searching" if self.search_mode else "Tracking" if ids is not None else "Waiting"
        cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Publish velocity command
        self.cmd_vel_publisher.publish(twist)
        
        return frame

    def destroy_node(self):
        self.get_logger().info("Shutting down marker follower node...")
        
        # Release camera
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Send a zero velocity command multiple times before shutting down
        stop_twist = Twist()
        for _ in range(10):  # Send multiple times for reliability
            self.cmd_vel_publisher.publish(stop_twist)
            time.sleep(0.5)  # Add delay between commands
            
        self.get_logger().info("Node shutdown complete")
        super().destroy_node()


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        
        self.previous_error = 0
        self.integral = 0
        
    def update(self, error):
        # Calculate P term
        p_term = self.kp * error
        
        # Calculate I term
        self.integral += error
        # Limit integral windup
        self.integral = max(min(self.integral, 1000), -1000)
        i_term = self.ki * self.integral
        
        # Calculate D term
        d_term = self.kd * (error - self.previous_error)
        self.previous_error = error
        
        # Return total correction
        return p_term + i_term + d_term
    
    def reset(self):
        """Reset the controller state"""
        self.previous_error = 0
        self.integral = 0


def main(args=None):
    rclpy.init(args=args)
    node = MarkerFollowerNode()
    
    # Add signal handlers for proper termination
    def signal_handler(sig, frame):
        node.get_logger().info("Received termination signal")
        stop_twist = Twist()
        for _ in range(10):
            node.cmd_vel_publisher.publish(stop_twist)
            time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt detected")
    finally:
        # Send stop command before exiting
        stop_twist = Twist()
        for _ in range(10):
            node.cmd_vel_publisher.publish(stop_twist)
            time.sleep(0.5)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()