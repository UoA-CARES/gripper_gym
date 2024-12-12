<h1 align="center">CARES Gripper Gym</h1>

<div align="center">

[![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![Pytorch 1.13.1](https://img.shields.io/badge/pytorch-1.13.1-blue)](https://pytorch.org/)

</div>

<div align="center">
<h3>
<a href="https://www.youtube.com/watch?v=0kii1EJjOzw&feature=youtu.be" target="_blank">Video Demo</a>
</h3>
</div>

<div align="center">
This repository contains the code used to control and train the grippers (Two-finger and Three-Finger) currently being designed and used in the <a href="https://cares.blogs.auckland.ac.nz/">CARES lab</a> at the <a href="https://www.auckland.ac.nz">The University of Auckland</a>. 
While being written for this specific system, it also intends to be applicable to many dynamixel servo systems with minor changes to the code.

<br/>
<br/>
See the gripper in action, learning to rotate the valve by 90 degrees:
<br/>
<br/>

| Exploration Phase                                                                      | During Training                                                                     | Final Policy                                                                      |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="readme_wiki_media/exploration_phase_final.gif" alt="explore" height="500px"> | <img src="readme_wiki_media/during_training_final.gif" alt="during" height="500px"> | <img src="readme_wiki_media/trained_policy_final.gif" alt="final" height="500px"> |

</div>

## Contents

- [Contents](#contents)
- [📋 Requirements](#-requirements)
- [👩‍🏫 Getting Started](#-getting-started)
- [📖 Usage](#-usage)
- [⚙️ Hardware Setup](#️-hardware-setup)
  - [BOM](#bom)
  - [STL files](#stl-files)
- [🗃️ Results](#️-results)

## 📋 Requirements

The repository was tested using Python 3.11.4 on a machine running Ubuntu 22.04.2 LTS. The repository relies on [Pytorch](https://pytorch.org/) relying on [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-zone). Instructions for enabling CUDA in Pytorch can be found [here](https://pytorch.org/get-started/locally/).

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

### BOM

A list of items required to build the grippers can be found in [Grippers BOM](https://docs.google.com/spreadsheets/d/1GFGDXZwodSCUbbnDEK6e9giJs_8Xy-eVyAdYuDRv4Qk/edit#gid=1627805202).

### STL files

3D printed parts for both grippers can be found in [Two-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link) and [Three-Finger STL](https://drive.google.com/drive/folders/1AuBA8254ImEZFrz9au1Tdciz5qx39S2c?usp=share_link).

![Picture of a CAD assembly that shows a rig that is holding a three-fingered gripper with the fingers hanging down](https://user-images.githubusercontent.com/105029122/205157459-ef70f9fb-dcea-464a-af8a-14d66047497a.png)

## 🗃️ Results

You can specify the folder to save results in using the `local_results_path` argument; otherwise, it defaults to `{home_path}/gripper_training`. The folder containing the results is named according to the following convention:

```
{date}_{robot_id}_{environment_type}_{observation_type}_{seed}_{algorithm}
```

Results are stored using the format as specified by standardised CARES RL format.
