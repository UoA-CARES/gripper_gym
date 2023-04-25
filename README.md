# CARES Gripper Package

This repository contains the code used to control and test the grippers (Two-finger and Three-Finger) currently being designed and used in the CARES lab at UOA. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with small changes to the code.   
 
## Installation

<!-- TODO? 
pipreqs /home/project/location to update the requirements.txt file on optimus prime pc
-->


`git clone` the repository

Run `pip3 install -r requirements.txt` in the **root directory** of the package

This repo also uses two CARES libraries which need to be installed as instructed in [CARES Lib](https://github.com/UoA-CARES/cares_lib) and [CARES Reinforcement Learning Package](https://github.com/UoA-CARES/cares_reinforcement_learning).

Create local directory for gripper local storage, copy and past the config_example folder and modify for your local version. This 

## Hardware Setup 
The current setup uses Dynamixel XL-320 servo motors (4 for Two-Finger and 9 for Three-Finger Gripper), which are being controlled using a [U2D2](https://emanual.robotis.com/docs/en/parts/interface/u2d2/). 
### BOM
A list of items required to build the grippers can be found in [Grippers BOM](https://docs.google.com/spreadsheets/d/1GFGDXZwodSCUbbnDEK6e9giJs_8Xy-eVyAdYuDRv4Qk/edit#gid=1627805202).

###  STL files
3D printed parts for both grippers can be found in [Two-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link) and [Three-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link).

![Picture of a CAD assembly that shows a rig that is holding a three-fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)


## Usage

Consult the repository [wiki]() for a guide on how to use the package

<!-- TODO: 
WIKI Planning
need to explain and give instructions for:

- config: env, gripper, learning
- env: current ones: rotation and translation. can extend environment for different tasks but changing choose goal and reward function
- network is where you can play around with your own ideas
- training_loop.py
- to test and play with gripper - gripper example
 -->


## Package Structure

```
cares_reinforcement_learning/
├─ config/
│  ├─ env_xDOF_config_ID.json
│  ├─ gripper_xDOF_config_ID.json
│  ├─ learning_config_ID.json
│  ├─ camera_distortion.txt
│  ├─ camera_matrix.txt
├─ environments/
│  ├─ Environment.py
│  ├─ RotationEnvironment.py
│  ├─ TranslationEnvironment.py
│  ├─ ...
├─ networks/
│  ├─ ...

```
`config/`: various configuration files for environment, gripper, training and camera. Instructions for these configs can be seen in [wiki]().

`environments/`: currently for rotation and translation tasks. Can extend environment class for different tasks by changing choose goal and reward function.

`networks/`: can contain your own neural networks that can be used with each algorithm.
