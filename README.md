# Gripper Code - CARES Summer Research Project 

This repository contains the code used to control and test the 3 fingered gripper currently being designed and used in the CARES lab at UOA. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with small changes to the code.   
 
## Languages and Libraries Used: 

- Python 3.10.5
- Dynamixel SDK 
- Numpy 1.23.0

## File Structure

gripperFunctions is a collation of functions that are used in almost all of the testing files. This is currently kept in the same folder and imported in. 

### Setup 
In the testing files there are a few global variables that can be changed depending on the given system. These are: 
- NUM_MOTORS --> specifies the number of motors 
- BAUDRATE --> baudrate of the motors, which can be found in the dynamixel wizard
- DEVICENAME --> the communication connection point
- MAX_VELOCITY_VALUE --> maximum speed of the motors
- LIM_TORQUE_VALUE --> maximum torque of the motors

The current setup uses 9 Dynamixel XL-320 servo motors (shown in image below), which are connected to a U2D2 Powerhub Board, and uses a baudrate of 57600 bps. 

![Picture of a CAD assembly that shows a rig that is holding a three fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)



