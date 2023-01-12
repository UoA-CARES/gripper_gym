# Gripper Code - CARES Summer Research Project 

This repository contains the code used to control and test the 3 fingered gripper currently being designed and used in the CARES lab at UOA. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with small changes to the code.   
 
## Languages and Libraries Used: 

- Python 3.10.5
- Dynamixel SDK 
- Numpy 1.23.0 
- cares_lib
- cares_reinforcement_learning

### Setup 

The current setup uses 9 Dynamixel XL-320 servo motors (shown in image below), which are connected to a U2D2 Powerhub Board, and uses a baudrate of 57600 bps. 

![Picture of a CAD assembly that shows a rig that is holding a three fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)

### File Structure Information

testing_loop: contains code to complete the training and testing for TD3 (also currently contains the actor and critic as there were problems getting them from examples in cares_reinforcement_learning)

gripper_environment: contains methods to do common reinforcement learning actions, tries to be consistent with openai gym naming
- __init__
- reset
- reward_function
- step

Gripper: contains methods to do actions with all 9 (or specified no of) motors
- __init__
- setup 
- angles_to_steps
- verify_steps
- move
- current_positions
- home
- gripper_moving_check
- close

Servo: contains methods for individual servo motors
addresses is global to servo
- __init__
- turn_on_LED
- limit_torque
- enable_torque
- disable_torque
- limit_speed
- moving_check
- present_position
- verify_step
- process_result (print out successes and errors in communication between servos and code)






