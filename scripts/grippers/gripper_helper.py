import logging
from cares_lib.slack_bot.SlackBot import SlackBot
from pydantic import BaseModel
from typing import List, Optional

class GripperConfig(BaseModel):
    gripper_type: int
    gripper_id: int
    device_name: str
    baudrate: int
    torque_limit: int
    speed_limit: int
    num_motors: int
    min_value: List[int]
    max_value: List[int]
    home_pose: List[int]
    actuated_target: bool

def message_slack(message):
    with open('slack_token.txt') as file: 
        slack_token = file.read()

    slack_bot = SlackBot(slack_token=slack_token)

    slack_bot.post_message(channel="#cares-chat-bot", message=message)

def handle_gripper_error(error):
    logging.warning(error)
    logging.info("Please fix the gripper and press enter to try again or x to quit: ")
    message_slack(f"{error}, please fix before the programme continues")
    value  = input()
    if value == 'x':
        logging.info("Giving up correcting gripper")
        return True 
    return False

def create_gripper(config : GripperConfig):
    if config.gripper_type == 0:# U2D2
        return U2D2Gripper(config)
    elif config.gripper_type == 1:# Arduino
        return ArduinoGripper(config)
    
from grippers.ArduinoGripper import ArduinoGripper
from grippers.U2D2Gripper import U2D2Gripper