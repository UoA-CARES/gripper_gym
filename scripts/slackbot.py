import slack

class SlackBot:
  def __init__(self, slack_token):
    self.client = slack.WebClient(token=slack_token)
    
  def post_message(self, channel, message):
    self.client.chat_postMessage(channel=channel,text=message)
  
  def upload_file(self, channel, message, filepath, filename):
    upload_text_file = self.client.files_upload(channels=channel, title=filename, file=filepath+filename, initial_comment=message)