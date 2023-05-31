//
//    FILE: AS5600_demo.ino
//  AUTHOR: Rob Tillaart
// PURPOSE: demo
//    DATE: 2022-05-28


#include "AS5600.h"
#include "Wire.h"

AS5600 as5600;

bool led_state = true;
int led_pin = 13;

enum Command {
  GET_YAW,
  OFFSET
};


// C++ does not have a built in split function, and Arduino variations are mixed bags. This is clean and works nicely enough.
String getValue(String data, char separator, int index){
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;
    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;
        }
    }
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}

void get_yaw(){
  Serial.println("YAW,"+String(as5600.rawAngle() * AS5600_RAW_TO_DEGREES));
}

float calculate_offset(float goal, float current) {
    float offset = goal - current;
    if (offset<0){
      offset = offset + 360;
    }
    return offset;
}

void set_offset(String command){
      as5600.setOffset(0);
      float aruco = atof(getValue(command, ',', 1).c_str());
      float magnet = as5600.rawAngle() * AS5600_RAW_TO_DEGREES;
      
      float offset = calculate_offset(aruco, magnet); 
      as5600.setOffset(offset);

      float difference = abs((as5600.rawAngle() * AS5600_RAW_TO_DEGREES) - aruco ) < 1.0;
 
      Serial.println("OFFSET,"+String(difference));
}

void setup()
{
  digitalWrite(LED_BUILTIN, LOW);
  led_pin = true;
  
  Serial.begin(115200);
  as5600.begin(4);//set direction pin.
  as5600.setDirection(AS5600_CLOCK_WISE);//default, just be explicit.
}

void processCommand(String command){
  Command action = Command(getValue(command, ',', 0).toInt());
  switch(action){
    case Command::GET_YAW:
      get_yaw();
      break;
    case Command::OFFSET:
      set_offset(command);
      break;
    default:
      Serial.println("ERROR CMD");
  }
}

void loop(){
  if(Serial.available()){
    String command = Serial.readStringUntil('\n');

    if(led_state)
      digitalWrite(LED_BUILTIN, HIGH);
    else
      digitalWrite(LED_BUILTIN, LOW);
    led_state = !led_state;

    processCommand(command);
  }
}
