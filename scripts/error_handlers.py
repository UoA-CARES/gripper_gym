import logging
logging.basicConfig(level=logging.INFO)
from pytimedinput import timedInput
from environments.Environment import EnvironmentError
from Gripper import GripperError
import cv2

def handle_gripper_error_home(environment, error_message, slack_bot, result_plot_filename):
    warning_message = f"Error handling has been initiated because of: {error_message}. Attempting to solve by home sequence."
    logging.warning(warning_message)
    
    slack_bot.post_message("#bot_terminal", warning_message)
    
    try :
        environment.gripper.wiggle_home()
        return True
    except (EnvironmentError , GripperError):
        warning_message = f"Auto wiggle fix failed, going to final handler"
        logging.warning(warning_message)
        slack_bot.post_message("#bot_terminal", warning_message)
        return handle_gripper_error(environment, error_message, slack_bot, result_plot_filename)
    
    

def handle_gripper_error(environment, error_message, slack_bot, file_name):
    logging.error(f"Error handling has been initiated because of: {error_message}")
    help_message = "Please fix the gripper and press | c to try again | x to quit | w to wiggle:"
    logging.error(help_message)
    slack_bot.post_message("#cares-chat-bot", f"{error_message}, {help_message}")
    result_plot_filename = f"{file_name}.png"
    
    while True:
        value, timed_out = timedInput(timeout=10)
        if timed_out:
            value = read_slack(slack_bot)

        if value == 'c':
            logging.info("Gripper Fixed continuing onwards")
            return True
        elif value == 'x':
            logging.info("Giving up correcting gripper")
            return False
        elif value == "reboot" or value == "r":
            try:
                logging.info("Rebooting servos")
                environment.gripper.reboot()
            except (EnvironmentError , GripperError):
                warning_message = "Your commanded reboot failed, try again"
                logging.warning(warning_message)
                slack_bot.post_message("#bot_terminal", warning_message)
                return True # stay in loop and try again
            return True
        elif value  == "w":
            try:
                environment.gripper.wiggle_home()
            except (EnvironmentError , GripperError):
                warning_message = "Your commanded wiggle home failed, try again"
                logging.warning(warning_message)
                slack_bot.post_message("#bot_terminal", warning_message)
                return True
            return True
        elif value  == "w2":
            try:
                environment.gripper.move(environment.gripper.max_values)
            except (EnvironmentError , GripperError):
                warning_message = "Your commanded wiggle home 2 failed, try again"
                logging.warning(warning_message)
                slack_bot.post_message("#bot_terminal", warning_message)
                return True
            return True
        elif value == "p":
            slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: current progress", "results_plots/", result_plot_filename)
        elif value == "f":
            cv2.imwrite('current_frame.png', environment.camera.get_frame())
            slack_bot.upload_file("#cares-chat-bot", f"#{environment.gripper.gripper_id}: current_frame", "", "current_frame.png")

def read_slack(slack_bot):
    message = slack_bot.get_message("cares-chat-bot")
    
    if message is not None:
        message = message.split(",") 
    else:
        return None

    gripper_id = 9
    if message[0] == str(gripper_id):
        return message[1]
    return None
