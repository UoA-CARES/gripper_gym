# CARES Gripper Package

This repository contains the code used to control and test the grippers (Two-finger and Three-Finger) currently being designed and used in the CARES lab at UOA. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with minor changes to the code.   
 
## Installation

`git clone` the repository

Run `pip3 install -r requirements.txt` in the **root directory** of the package

This repo also uses two CARES libraries which need to be installed as instructed in [CARES Lib](https://github.com/UoA-CARES/cares_lib) and [CARES Reinforcement Learning Package](https://github.com/UoA-CARES/cares_reinforcement_learning).

Create local directory for gripper local storage, copy and paste the config_example folder and modify for your local version.

## Usage

Consult the repository [wiki](https://github.com/UoA-CARES/Gripper-Code/wiki) for a guide on how to use the package

## Hardware Setup 
The current setup uses Dynamixel XL-320 servo motors (4 for Two-Finger and 9 for Three-Finger Gripper), which are being controlled using a [U2D2](https://emanual.robotis.com/docs/en/parts/interface/u2d2/). 
Below is an example of the wiring setup for the three-fingered gripper. This is easily changed for other configurations, with a maximum of 4 daisy chains coming from the U2D2. 
<img src="https://github.com/UoA-CARES/Gripper-Code/assets/105029122/994e451f-8459-42e2-9aa7-c27b7d10af29" width="400" />

### Magnetic Encoder Setup
An AS5600 magnetic encoder can be used to get the object angle during training. 3D printed object valve suitable for using this encoder can be found in the STL files folder below. 

To set this up 
1. Connect the encoder with an arduino board. VCC - 3.3V; GND - GND; DIR - GND; SCL - SCL; SDA - SDA; (see wiring digram below)
2. Upload magnetic_encoder_object.ino onto the Arduino.
3. Check device name and modify the object_config file accordingly

   <img src="https://github.com/UoA-CARES/Gripper-Code/assets/105029122/305bc589-e68e-4433-9fbd-919544614493" alt="wiring diagram for connecting an as5600 magnetic encoder to an Arduino Mega" width="400" />
   


### BOM
A list of items required to build the grippers can be found in [Grippers BOM](https://docs.google.com/spreadsheets/d/1GFGDXZwodSCUbbnDEK6e9giJs_8Xy-eVyAdYuDRv4Qk/edit#gid=1627805202).

###  STL files
3D printed parts for both grippers can be found in [Two-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link) and [Three-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link).

![Picture of a CAD assembly that shows a rig that is holding a three-fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)

## Package Structure

```
cares_gripper_package/
├─ config_examples/
│  ├─ env_xDOF_config_IDx.json
│  ├─ gripper_xDOF_config_IDx.json
│  ├─ learning_config_IDx.json
│  ├─ objecy_config.IDx.json
│  ├─ camera_distortion.txt
│  ├─ camera_matrix.txt
├─ environments/
│  ├─ Environment.py
│  ├─ RotationEnvironment.py
│  ├─ TranslationEnvironment.py
│  ├─ ...
├─ networks/
│  ├─ Actor.py
│  ├─ Critic.py
├─ magnetic_encoder_object
│  ├─ magnetic_encoder_object.ino
├─ tools
│  ├─ error_handlers.py
│  ├─ gripper_example.py
│  ├─ utils.py
├─ configurations.py
├─ Gripper.py
├─ Objects.py
├─ training_loop.py


```
`config/`: various configuration file examples for environment, gripper, training and camera. Instructions for these configs can be found in [wiki]().

`environments/`: currently for rotation and translation tasks. Can extend environment class for different tasks by changing choose goal and reward function.

`networks/`: can contain your own neural networks that can be used with each algorithm.

`magneti encoder object/`: currently only contains arduino code for the magnetic encoder.

`tools/`: helper functions and example
