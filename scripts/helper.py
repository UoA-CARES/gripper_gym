import logging
from cares_lib.slack_bot.SlackBot import SlackBot

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

def verify_steps(self, steps):
        # check all actions are within min max of each servo
        for id, servo in self.servos.items():
            if not servo.verify_step(steps[id]):
                logging.warn(f"Gripper#{self.gripper_id}: step for servo {id + 1} is out of bounds")
                return False
        return True

def action_to_steps(self, action):
    steps = action
    max_action = 1
    min_action = -1
    for i in range(0, len(steps)):
        max = self.servos[i].max
        min = self.servos[i].min
        #steps[i] = steps[i] * (max - min) + min
        steps[i] = int((steps[i] - min_action) * (max - min) / (max_action - min_action)  + min)
    return steps