# gripper_gym

This repository contains the code used to control and train custom dynamixel grippers currently being designed and used in the <a href="https://robotlearningteam.org/">Robot Learning Team</a> at the <a href="https://www.auckland.ac.nz">The University of Auckland</a>. 

See an example of the three finger gripper in action, learning to rotate a valve by 90 degrees:

<div align="center">
<h3>
<a href="https://www.youtube.com/watch?v=0kii1EJjOzw&feature=youtu.be" target="_blank">Video Demo</a>
</h3>
</div>

| Exploration Phase                                                                      | During Training                                                                     | Final Policy                                                                      |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="readme_wiki_media/exploration_phase_final.gif" alt="explore" height="300px"> | <img src="readme_wiki_media/during_training_final.gif" alt="during" height="300px"> | <img src="readme_wiki_media/trained_policy_final.gif" alt="final" height="300px"> |

# Usage
Consult the repository's [wiki](https://github.com/UoA-CARES/Gripper-Code/wiki) for a guide on how to use the package.

# Installation Instructions
![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)

Install the general CARES library as instructed in [CARES Lib](https://github.com/UoA-CARES/cares_lib). 

`git clone` the repository into your desired directory on your local machine

Run `pip3 install -r requirements.txt` in the **root directory** of the package

To make the module **globally accessible** in your working environment run `pip3 install --editable .` in the **project root**

# Gripper Environments
This package provides the baseline code for the gripper environments - you run these envrionments through the general training package [gymnasium_envrionments](https://github.com/UoA-CARES/gymnasium_envrionments). 

The domain defines the gripper you are working with, and the task defines the type of task the gripper is trying to learn. It is important to use the right gripper configurations with the right tasks. We currently support two variations of grippers with multiple tasks for each. The grippers themselves also have variations based on additonal components - touch sensing, depth cameras, etc. 

## Two Finger Gripper
The two finger gripper has three supported tasks. 

### Translation

```python
python run.py train cli gripper --domain two_finger --task translation SAC
```
<div align="center">
<img src="readme_wiki_media/translation_task.gif" alt="during" height="200px">
</div>

### Rotation

```python
python run.py train cli gripper --domain two_finger --task rotation SAC
```

<div align="center">
<img src="readme_wiki_media/rotation_task1.gif" alt="during" height="200px">
</div>

### Suspended Translation

```python
python run.py train cli gripper --domain two_finger --task suspended_translation SAC
```

## Four Finger Gripper
The four finger gripper supports four different tasks. 

### Translation

```python
python run.py train cli gripper --domain four_finger --task translation SAC
```

### Rotation

```python
python run.py train cli gripper --domain four_finger --task rotation SAC
```

### Suspended Translation

```python
python run.py train cli gripper --domain four_finger --task suspended_translation SAC
```

### Suspended Rotation

```python
python run.py train cli gripper --domain four_finger --task suspended_rotation SAC
```

# Setting up Grippers
To simplify connecting to the grippers we use UDEV rules to handle connections to the various USB devices. 

## UDEV Rules
Assign unique persistent names to USB devices (Dynamixel U2D2s, Arduino, Camera)

1. Find USB device details. Run the command below to identify device details:
    - Command. Note: Modify "/dev/ttyUSB0" to your device node.
    ```
    udevadm info --query=all --name=/dev/ttyUSB0  
    ```
    - Record the devices **idVendor**, **idProduct** and **idserial**, or some slight variation of those three keywords.
2. Create new UDEV rule file:
    - Create udev file. Note: "99-dynamixel" is able to be modified to suit the context. 
    ```
    sudo nano /etc/udev/rules.d/99-dynamixel.rules" 
    ```
    - Add rules as per the format below:
    ```
    SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6015", ATTRS{serial}=="A12345", SYMLINK+="gripper1"
    ```
    Note: the SYMLINK argument determines the custom name, e.g. gripper1 in this instance.

    - File should have the format like below, e.g. "0403" and "6015" are Vendor and Product ids for the dynamixel U2D2. Serial can be full or shortened version.

    ```
    SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6015", ATTRS{serial}=="A12345", SYMLINK+="gripper1"
    SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6015", ATTRS{serial}=="B67890", SYMLINK+="gripper2" 
    ```
    
3. Reload the UDEV rules
    ```
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    ```
    
4. Verify the rules:
    - Unplug and replug device then check the symbolic links
    ```
    ls -l /dev/dynamixel* 
    ```
    - Re-log user if still unchanged