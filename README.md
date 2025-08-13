<h1 align="center">CARES Gripper Gym</h1>

This repository contains the code used to control and train custom dynamixel grippers currently being designed and used in the <a href="https://robotlearningteam.org/">Robot Learning Team</a> at the <a href="https://www.auckland.ac.nz">The University of Auckland</a>. 

See an example of the three finger gripper in action, learning to rotate a valve by 90 degrees:

<div align="center">
<h3>
<a href="https://www.youtube.com/watch?v=0kii1EJjOzw&feature=youtu.be" target="_blank">Video Demo</a>
</h3>
</div>

| Exploration Phase                                                                      | During Training                                                                     | Final Policy                                                                      |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| <img src="readme_wiki_media/exploration_phase_final.gif" alt="explore" height="500px"> | <img src="readme_wiki_media/during_training_final.gif" alt="during" height="500px"> | <img src="readme_wiki_media/trained_policy_final.gif" alt="final" height="500px"> |

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

The domain defines the gripper you are working with, and the task defines the type of task the gripper is trying to learn. It is important to use the right gripper configurations with the right tasks.

## Two Finger Gripper

### Translation

```python
python run.py train cli gripper --domain two_finger --task translation SAC
```

### Rotation

### Suspended Translation

## Four Finger Gripper

### Translation

### Rotation

### Suspended Translation

### Suspended Rotation