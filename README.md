<h1 align="center">CARES Gripper Package</h1>

<h3 align="center">

[![Python 3.11.4](https://img.shields.io/badge/python-3.11.4-blue.svg)](https://www.python.org/downloads/release/python-3114/)

<a href="https://www.youtube.com/watch?v=0kii1EJjOzw&feature=youtu.be">Video Demo</a>
</h3>

<div align="center">
This repository contains the code used to control and test the grippers (Two-finger and Three-Finger) currently being designed and used in the <a href="https://cares.blogs.auckland.ac.nz/">CARES lab</a> at the <a href="https://www.auckland.ac.nz">The University of Auckland</a>. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with minor changes to the code.

<br/>
See the gripper in action, learning to rotate the valve by 90 degrees:

| Exploration Phase                                                                      | During Training                                                                     | Final Policy                                                                      |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="readme_wiki_media/exploration_phase_final.gif" alt="explore" height="500px"> | <img src="readme_wiki_media/during_training_final.gif" alt="during" height="500px"> | <img src="readme_wiki_media/trained_policy_final.gif" alt="final" height="500px"> |
</div>

## Contents
- [Contents](#contents)
- [📋 Requirements](#-requirements)
- [👩‍🏫 Getting Started](#-getting-started)
- [� Usage](#-usage)
- [⚙️ Hardware Setup](#️-hardware-setup)
  - [Magnetic Encoder Setup](#magnetic-encoder-setup)
  - [BOM](#bom)
  - [STL files](#stl-files)
- [📦 Package Structure](#-package-structure)

## 📋 Requirements
The repository was tested using Python 3.11.4 on a machine running Ubuntu 22.04.2 LTS with Intel Core i9-10900 CPU and NVIDIA GeForce RTX 3080 GPU. It is recommended to use a Linux machine. The repository relies on [Pytorch](https://pytorch.org/). While the use of [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-zone) is supported, it is optional. Instructions for enabling CUDA in Pytorch can be found [here](https://pytorch.org/get-started/locally/).

A comprehensive list of dependencies is available in `requirements.txt`. Ensure that the hardware components for the gripper are turned on and connected to the machine.

## 👩‍🏫 Getting Started

1. Clone the repository using `git clone`.

2. Run `pip3 install -r requirements.txt` in the **root directory** of the package.

3. Install the two CARES libraries as instructed in [CARES Lib](https://github.com/UoA-CARES/cares_lib) and [CARES Reinforcement Learning Package](https://github.com/UoA-CARES/cares_reinforcement_learning).

4. Create a folder named `your_folder_name` to store config files. To get started, copy and paste the files in `scripts/config_examples` into the folder. For a guide on changing configs, see the [wiki](https://github.com/UoA-CARES/Gripper-Code/wiki/Configuration-Files).

## 📖 Usage

Consult the repository's [wiki](https://github.com/UoA-CARES/Gripper-Code/wiki) for a guide on how to use the package.

## ⚙️ Hardware Setup

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

### STL files

3D printed parts for both grippers can be found in [Two-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link) and [Three-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link).

![Picture of a CAD assembly that shows a rig that is holding a three-fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)

## 📦 Package Structure

```
cares_gripper_package/scripts/
├─ config_examples/
│  ├─ env_xDOF_config_IDx.json
│  ├─ gripper_9DOF_config_ID2.json (3 Fingered Gripper)
│  ├─ gripper_xDOF_config_IDx.json
│  ├─ learning_config_IDx.json
│  ├─ object_config_IDx.json
│  ├─ camera_distortion.txt
│  ├─ camera_matrix.txt
├─ environments/
│  ├─ Environment.py
│  ├─ RotationEnvironment.py
│  ├─ TranslationEnvironment.py
│  ├─ ...
├─ networks/
│  ├─ DDPG/
│  ├─ ├─ Actor.py
│  ├─ ├─ Critic.py
│  ├─ SAC/
│  ├─ ├─ ...
│  ├─ TD3/
│  ├─ ├─ ...
├─ magnetic_encoder_object
│  ├─ magnetic_encoder_object.ino
├─ tools
│  ├─ error_handlers.py
│  ├─ utils.py
├─ configurations.py
├─ evaluation_loop.py
├─ gripper_example.py
├─ GripperTrainer.py
├─ Objects.py
├─ training_loop.py
```

`config_examples/`: Various configuration file examples for the environment, gripper, training and camera. Instructions for these configs can be found in [wiki]().

`environments/`: Currently for rotation and translation tasks. Can extend environment class for different tasks by changing choose goal and reward function.

`networks/`: Can contain your own neural networks that can be used with each algorithm. Currently, we support the DDPG, SAC, and TD3 RL algorithms.

`magnetic_encoder_object/`: Currently only contains arduino code for the magnetic encoder.

`tools/`: Includes helper functions for I/O, plotting, and Slack integration in `utils.py`. Functions to handle various gripper errors and Slack monitoring can be found in `error_handlers.py`.
