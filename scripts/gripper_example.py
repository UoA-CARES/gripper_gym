import logging
logging.basicConfig(level=logging.DEBUG)

from Gripper import Gripper, GripperError

# Example of how to use Gripper
def main():
    gripper = Gripper(device_name="COM4")
    # These calls to setup the gripper will work or alert the operator to any issues to resolve them, if operator can't resolve it will simply crash with an exception
    gripper.enable()
    gripper.home()

    # Will run and alert the operator if it can't be resolved, if operate can't resolve it then it will return to this except block to exit gracefully outside of the gripper class.
    try:
        target_steps = [412, 350, 650, 412, 350, 650, 412, 350, 650]
        gripper.move(target_steps)
    except GripperError as error:
        # Handle the error gracefully here as required...
        print(error)
        exit() # kill the program entirely as gripper is unrecoverable for whatever reason

if __name__ == "__main__":
    main()
