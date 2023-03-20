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