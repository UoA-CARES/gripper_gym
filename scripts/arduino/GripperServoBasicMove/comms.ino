// C++ does not have a built in split function, and Arduino variations are mixed bags. This is clean and works nicely enough.
String getValue(String data, char separator, int index) {
  int found = 0;
  int strIndex[] = { 0, -1 };
  int maxIndex = data.length() - 1;

  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == separator || i == maxIndex) {
      found++;
      strIndex[0] = strIndex[1] + 1;
      strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }
  return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}

String poseToString(std::vector<int> pose) {
  String pose_s;
  bool first = true;
  for (int p : pose) {
    if (first)
      pose_s += String(p);
    else
      pose_s += "," + String(p);
    first = false;
  }
  return pose_s;
}

String getState() {
  return "state";
  //  return poseToString(gripper.currentPosition());
}

String stopGripper(String command) {
  //stop gripper
  int dxl_comm_result = 0;
  uint8_t dxl_error = 0;
  gripper->stopMoving(dxl_comm_result, dxl_error);
  return getState();
}

std::vector<int> commandToPose(String command) {
  std::vector<int> target_position;
  //TODO remove hard coded 9 for gripper sero num
  for (int i = 0; i < 9; i++) {
    int pos_i = getValue(command, ',', i + 1).toInt();
    target_position.push_back(pos_i);
  }
  return target_position;
}

String moveGripper(String command) {
  std::vector<int> target_position = commandToPose(command);
  return poseToString(target_position);
  //  return getState();
}

String processCommand(String command) {
  String response;
  //this isn't perfect as spam data can be interpretted as a given action but whatever
  Command action = Command(getValue(command, ',', 0).toInt());
  switch (action) {
    case Command::STOP:
      response = stopGripper(command);
      break;
    case Command::MOVE:
      response = moveGripper(command);
      break;
    case Command::GET_STATE:
      response = getState();
      break;
    default:
      return String(Response::ERROR_STATE) + ",Unkown Command: " + command;
  }
  return String(Response::SUCCEEDED) + "," + response;
}

void run_gripper() {

  int joint_pos1[9] = {512, 300, 300, 400, 400, 512, 512, 300, 512};
  int joint_pos2[9] = {512, 300, 300, 400, 400, 512, 512, 512, 300};
  int joint_pos3[9] = {512, 623, 623, 653, 750, 750, 512, 400, 512 };

  int dxl_comm_result = 0;
  uint8_t dxl_error = 0;

  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    
    while (command = MOVE){
         gripper->move(joint_pos1);
         delay(1000);
         gripper->move(joint_pos3);
         delay(1000);
         digitalWrite(led_pin, HIGH);
         if(command = STOP)
            gripper->stopMoving(dxl_comm_result, dxl_error);
            digitalWrite(led_pin, LOW);
            break;
  
        }

    String response = processCommand(command);
    Serial.println(response);
  }
    //led_state = !led_state;

    
  }
