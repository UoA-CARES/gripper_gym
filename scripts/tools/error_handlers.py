import logging
from time import sleep

logging.basicConfig(level=logging.INFO)
from cares_lib.dynamixel.Gripper import GripperError
from cares_lib.dynamixel.Servo import DynamixelServoError
from environments.environment import EnvironmentError
from pytimedinput import timedInput

WAIT_TIME = 5  # wait 5 seconds for auto sequences
NUM_REPEAT = 8


def auto_reboot_sequence(environment):
    try:
        reboot(environment)
        sleep(WAIT_TIME)
        home(environment)
        sleep(WAIT_TIME)
        return True
    except (GripperError, EnvironmentError, DynamixelServoError):
        return False


def auto_wiggle_sequence(environment):
    try:
        reboot(environment)
        sleep(WAIT_TIME)
        wiggle_home(environment)
        sleep(WAIT_TIME)
        return True
    except (GripperError, EnvironmentError, DynamixelServoError):
        return False


def reboot(environment):
    try:
        logging.info("Rebooting Gripper")
        environment.gripper.reboot()
        logging.info("Rebooting succeeded")
    except (EnvironmentError, GripperError):
        warning_message = "Reboot failed"
        logging.warning(warning_message)
        raise GripperError(warning_message)


def home(environment):
    try:
        logging.info("Trying to home")
        environment.gripper.home()
        logging.info("Home succeeded")
    except (EnvironmentError, GripperError):
        warning_message = "Home failed"
        logging.warning(warning_message)
        raise GripperError(warning_message)


def wiggle_home(environment):
    try:
        logging.info("Trying to wiggle home")
        environment.gripper.wiggle_home()
        logging.info("Wiggle home succeeded")
    except (EnvironmentError, GripperError):
        warning_message = "Wiggle home failed"
        logging.warning(warning_message)
        raise GripperError(warning_message)


def handle_gripper_error_home(environment, error_message, file_path):
    warning_message = f"Error handling has been initiated because of: {error_message}. Attempting to solve by home sequence."
    logging.warning(warning_message)

    try:
        if not environment.gripper.wiggle_home():
            warning_message = (
                f"#{environment.gripper.gripper_id}: Wiggle home failed, rebooting"
            )
            logging.warning(warning_message)
        return True
    except (EnvironmentError, GripperError):
        # Repeat this sequence n times before resorting to manual error handler
        for _ in range(NUM_REPEAT):
            # Try auto reboot first
            if auto_reboot_sequence(environment):
                logging.info(
                    f"#{environment.gripper.gripper_id}: Auto Reboot Sequence success"
                )
                return True

            reboot_failed_message = (
                f"#{environment.gripper.gripper_id}: Auto Reboot Sequence failed"
            )
            logging.warning(reboot_failed_message)

            # Try auto wiggle if auto reboot fails
            if auto_wiggle_sequence(environment):
                logging.info(
                    f"#{environment.gripper.gripper_id}: Auto Wiggle Sequence success"
                )
                return True

        warning_message = f"#{environment.gripper.gripper_id}: Auto Reboot and Wiggle both failed, going to manual error handler"
        logging.warning(warning_message)
        return handle_gripper_error(environment, error_message, file_path)


def handle_gripper_error(environment, error_message, file_path):
    logging.error(f"Error handling has been initiated because of: {error_message}")
    help_message = "Fix the gripper then press: c to continue | x to quit \
                    Commands: h to home | w to wiggle | r to reboot | p for progress | d for distance | f for current frame |"
    logging.error(help_message)

    while True:
        try:
            value, _ = timedInput(timeout=10)
            if value == "c":
                logging.info("Gripper fixed continuing onwards")
                return True
            elif value == "x":
                logging.info("Giving up correcting gripper")
                return False
            elif value == "r":
                reboot(environment)
            elif value == "h":
                home(environment)
            elif value == "w":
                wiggle_home(environment)
        except (EnvironmentError, GripperError) as error:
            # Error was encountered after user selects operation, allow them to select again
            retry_error_message = (
                f"Error encountered during manual error handling with message: {error}"
            )
            logging.error(retry_error_message)
            logging.error(help_message)
            continue
