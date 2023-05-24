import logging
import os
logging.basicConfig(level=logging.INFO)
from pytimedinput import timedInput
from environments.Environment import EnvironmentError
from Gripper import GripperError
import cv2

def reboot(environment, slack_bot):
    try:
        logging.info("Rebooting Gripper")
        environment.gripper.reboot()
        logging.info("Rebooting succeeded")
    except (EnvironmentError , GripperError):
        warning_message = "Reboot failed"
        logging.warning(warning_message)
        slack_bot.post_message("#bot_terminal", warning_message)

def home(environment, slack_bot):
    try:
        logging.info("Trying to home")
        environment.gripper.home()
        logging.info("Home succeeded")
    except (EnvironmentError , GripperError):
        warning_message = "Home failed"
        logging.warning(warning_message)
        slack_bot.post_message("#bot_terminal", warning_message)

def wiggle_home(environment, slack_bot):
    try:
        logging.info("Trying to wiggle home")
        environment.gripper.wiggle_home()
        logging.info("Wiggle home succeeded")
    except (EnvironmentError , GripperError):
        warning_message = "Wiggle home failed"
        logging.warning(warning_message)
        slack_bot.post_message("#bot_terminal", warning_message)

def get_reward_plot(environment, slack_bot, file_path):
    if os.path.exists(f"{file_path}/reward.png"):
        slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: current progress", f"{file_path}/", "reward.png")
    else:
        slack_bot.post_message("#cares-chat-bot", f"#{environment.gripper.gripper_id}: Result plot not ready yet or doesn't exist")

def get_distance_plot(environment, slack_bot, file_path):
    if os.path.exists(f"{file_path}/distance.png"):
        slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: current progress", f"{file_path}/", "distance.png")
    else:
        slack_bot.post_message("#cares-chat-bot", f"#{environment.gripper.gripper_id}: Result plot not ready yet or doesn't exist")

def get_frame(environment, slack_bot, file_path):
    cv2.imwrite(f"{file_path}/current_frame.png", environment.camera.get_frame())
    if os.path.exists(f"{file_path}/current_frame.png"):
        slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: current_frame", f"{file_path}/", "current_frame.png")
    else:
        slack_bot.post_message("#cares-chat-bot", f"#{environment.gripper.gripper_id}: Having trouble accessing current frame")

def handle_gripper_error_home(environment, error_message, slack_bot, file_path):
    warning_message = f"Error handling has been initiated because of: {error_message}. Attempting to solve by home sequence."
    logging.warning(warning_message)
    slack_bot.post_message("#bot_terminal", warning_message)
    
    try :
        if not environment.gripper.wiggle_home():
            warning_message = f"#{environment.gripper.gripper_id}: Wiggle home failed, rebooting"
            logging.warning(warning_message)
            slack_bot.post_message("#bot_terminal", warning_message)

            environment.gripper.reboot()
        return True
    except (EnvironmentError , GripperError):
        warning_message = f"#{environment.gripper.gripper_id}: Auto wiggle fix or reboot failed, going to manual error handler"
        logging.warning(warning_message)
        slack_bot.post_message("#bot_terminal", warning_message)
        return handle_gripper_error(environment, error_message, slack_bot, file_path)
    
def handle_gripper_error(environment, error_message, slack_bot, file_path):
    logging.error(f"Error handling has been initiated because of: {error_message}")
    help_message = "Fix the gripper then press: c to continue | x to quit \
                    Commands: h to home | w to wiggle | r to reboot | p for progress | d for distance | f for current frame |"
    logging.error(help_message)
    slack_bot.post_message("#cares-chat-bot", f"{error_message}, {help_message}")
    
    while True:
        value, timed_out = timedInput(timeout=10)
        if timed_out:
            value = read_slack(slack_bot, environment.gripper.gripper_id)

        if value == 'c':
            logging.info("Gripper fixed continuing onwards")
            return True
        elif value == 'x':
            logging.info("Giving up correcting gripper")
            return False
        elif value == "r":
            reboot(environment, slack_bot)
        elif value  == "h":
            home(environment, slack_bot)
        elif value  == "w":
            wiggle_home(environment, slack_bot)
        elif value == "p":
            get_reward_plot(environment, slack_bot, file_path)
        elif value == "d":
            get_distance_plot(environment, slack_bot, file_path)
        elif value == "f":
            get_frame(environment, slack_bot, file_path)
            

def read_slack(slack_bot, gripper_id):
    message = slack_bot.get_message("cares-chat-bot")
    
    if message is not None:
        message = message.split(",") 
    else:
        return None

    if message[0] == str(gripper_id):
        return message[1]
    return None
