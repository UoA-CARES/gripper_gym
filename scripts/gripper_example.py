import logging
logging.basicConfig(level=logging.DEBUG)

from Gripper import Gripper, GripperError

# Example of how to use Gripper
def main():
    gripper = Gripper(device_name="/dev/ttyACM0")
    # These calls to setup the gripper will work or alert the operator to any issues to resolve them, if operator can't resolve it will simply crash with an exception
    gripper.ping()
    gripper.led()
    gripper.home()

    # Will run and alert the operator if it can't be resolved, if operate can't resolve it then it will return to this except block to exit gracefully outside of the gripper class.
    try:
        target_steps1 = [412, 350, 650, 412, 350, 650, 412, 350, 650]
        target_steps2 = [512, 512, 512, 512, 512, 512, 512, 512, 512]
        target_steps3 = [611, 673, 373, 611, 673, 373, 611, 673, 373]

        for i in range(0,10):
            gripper.move(target_steps1)
            gripper.move(target_steps2)
            gripper.move(target_steps3)
    except GripperError as error:
        # Handle the error gracefully here as required...
        logging.error(error)
        exit() # kill the program entirely as gripper is unrecoverable for whatever reason

if __name__ == "__main__":
    main()
