#include <algorithm>
#include <vector> 
#include <Dynamixel2Arduino.h>

//This namespace is required to use DYNAMIXEL Control table item name definitions
using namespace ControlTableItem;

// https://emanual.robotis.com/docs/en/parts/controller/opencm904/
// https://emanual.robotis.com/docs/en/dxl/x/xl320/#control-table-data-address

// Control table address could be differ by Dynamixel Series
#define ADDRESS_TORQUE_ENABLE           24
#define ADDRESS_GOAL_POSITION           30 
#define ADDRESS_PRESENT_POSITION        37
#define ADDRESS_SPEED_LIMIT             32
#define ADDRESS_TORQUE_LIMIT            15  
#define ADDRESS_LEDS                    25

// Data Byte Length
#define LENGTH_GOAL_POSITION            2
#define LENGTH_PRESENT_POSITION         2


#define PROTOCOL_VERSION                2.0 // Protocol version
#define DXL_MOVING_STATUS_THRESHOLD     20  // Dynamixel moving status threshold
#define DXL_SERIAL                      Serial3 //OpenCM9.04 EXP Board's DXL port Serial. (Serial1 for the DXL port on the OpenCM 9.04 board)

const int DXL_DIR_PIN = 22; //OpenCM9.04 EXP Board's DIR PIN. (28 for the DXL port on the OpenCM 9.04 board)

const uint8_t BROADCAST_ID = 254;

const uint16_t SR_START_ADDR = ADDRESS_PRESENT_POSITION;
const uint16_t SR_ADDR_LEN   = 2;

const uint16_t SW_START_ADDR = ADDRESS_GOAL_POSITION;
const uint16_t SW_ADDR_LEN   = 2;

// Syncread and syncwrite data packets
typedef struct sr_data{
  uint16_t present_position;
} __attribute__((packed)) sr_data_t;
typedef struct sw_data{
  uint16_t goal_position;
} __attribute__((packed)) sw_data_t;

// Needs to match the enum order in Gripper.py
enum Command {
  PING,
  STOP,
  MOVE,
  MOVE_SERVO,
  GET_STATE
};

// Needs to match the enum order in Gripper.py
enum Response {
  SUCCEEDED,
  ERROR_STATE
};

class Gripper{
  public:
    Gripper(int baudrate=9600,int torque_limit=80, int speed_limit=100){

      int max[NUM_SERVOS] = {900, 750, 769, 900, 750, 802, 900, 750, 794};
      int min[NUM_SERVOS] = {100, 250, 130, 100, 198, 152, 100, 250, 140};
      
      this->torque_limit = torque_limit;
      this->speed_limit = speed_limit;

      // Fill the members of structure to syncRead using external user packet buffer
      this->sr_infos.packet.p_buf = user_pkt_buf;
      this->sr_infos.packet.buf_capacity = user_pkt_buf_cap;
      this->sr_infos.packet.is_completed = false;
      this->sr_infos.addr = SR_START_ADDR;
      this->sr_infos.addr_length = SR_ADDR_LEN;
      this->sr_infos.p_xels = info_xels_sr;
      this->sr_infos.xel_count = 0;  

      for(int i = 0; i < NUM_SERVOS; i++){
        this->info_xels_sr[i].id = SERVO_IDS[i];
        this->info_xels_sr[i].p_recv_buf = (uint8_t*)&sr_data[i];
        this->sr_infos.xel_count++;
      }
      this->sr_infos.is_info_changed = true;

      // Fill the members of structure to syncWrite using internal packet buffer
      this->sw_infos.packet.p_buf = nullptr;
      this->sw_infos.packet.is_completed = false;
      this->sw_infos.addr = SW_START_ADDR;
      this->sw_infos.addr_length = SW_ADDR_LEN;
      this->sw_infos.p_xels = info_xels_sw;
      this->sw_infos.xel_count = 0;

      for(int i = 0; i < NUM_SERVOS; i++){
        this->info_xels_sw[i].id = SERVO_IDS[i];
        this->info_xels_sw[i].p_data = (uint8_t*)&sw_data[i].goal_position;
        this->sw_infos.xel_count++;
      }
      this->sw_infos.is_info_changed = true;

      this->error_message = "";

      this->dxl = new Dynamixel2Arduino(DXL_SERIAL, DXL_DIR_PIN);
      this->dxl->begin(baudrate);
      this->dxl->setPortProtocolVersion(PROTOCOL_VERSION);

      // ControlTableItem::
      // this->dxl->writeControlTableItem(uint8_t item_idx, uint8_t id, int32_t data);
    }

    String getErrorMessage(){
      return error_message;
    }

    bool ping(){
      this->error_message = "";
      DYNAMIXEL::InfoFromPing_t ping_info[32];
      uint8_t count_pinged = dxl->ping(DXL_BROADCAST_ID, ping_info, sizeof(ping_info)/sizeof(ping_info[0]));
      if(count_pinged < NUM_SERVOS){
        this->error_message = "Only "+String(count_pinged)+" Servos found: ";
        for (int i = 0; i < count_pinged; i++){
            this->error_message += String(ping_info[i].id) +" ";
        }
        return false;
      }
      return true;
    }

    bool enableServos(){
      this->error_message = "";
      bool success = true;      
      success &= this->dxl->torqueOff(BROADCAST_ID);
      success &= this->dxl->setOperatingMode(BROADCAST_ID, OP_POSITION);
      
      for(int i = 0; i < NUM_SERVOS; i++){
        int id = SERVO_IDS[i];
        success &= this->dxl->writeControlTableItem(TORQUE_LIMIT, id, this->torque_limit);
      }

      for(int i = 0; i < NUM_SERVOS; i++){
        int id = SERVO_IDS[i];
        success &= this->dxl->writeControlTableItem(VELOCITY_LIMIT, id, this->speed_limit);
      }

      for(int i = 0; i < NUM_SERVOS; i++){
        int id = SERVO_IDS[i];
        success &= this->dxl->writeControlTableItem(LED, id, LEDS[i]);
      }
      
      success &= this->dxl->torqueOn(BROADCAST_ID);
      if(!success)
        this->error_message = "Failed to setup servos";
      return success;
    }

    bool move(int servo_id, int goal_position, bool wait=true, long int timeout=5000){
      this->error_message = "";
      if(std::count(std::begin(SERVO_IDS), std::end(SERVO_IDS), servo_id) == 0){
        this->error_message = "Servo "+String(servo_id)+" not found";
        return false;
      }

      if(!this->dxl->setGoalPosition(servo_id, goal_position)){
        this->error_message = "Write error to servo "+String(servo_id)+" Error: "+String(this->dxl->getLastLibErrCode());
        return false;
      }

      long int start_time = millis();
      while(wait && this->isMoving(servo_id)){ 
        if(millis() > start_time + timeout){
          this->error_message = "moving timeout";
          return false;
        }
        delay(100);
      }
  
      return true;
    }

    bool move(std::vector<int> goal_position, bool wait=true, long int timeout=5000){
      this->error_message = "";
      for(int i = 0; i < NUM_SERVOS; i++){
        this->sw_data[i].goal_position = goal_position[i];
      }
      this->sw_infos.is_info_changed = true;

      if(!dxl->syncWrite(&this->sw_infos)){
        this->error_message = "SyncWrite Error: "+String(this->dxl->getLastLibErrCode());
        return false;
      }

      long int start_time = millis();
      while(wait && this->isMoving()){ 
        if(millis() > start_time + timeout){
          this->error_message = "moving timeout";
          return false;
        }
        delay(100);
      }
      return true;
    }

    void stopMoving(int &dxl_comm_result, uint8_t &dxl_error){
      this->error_message = "";
      for(int i = 0; i < NUM_SERVOS; i++){
        
      }
    }

    bool isMoving(int servo_id){
      this->error_message = "";
      if(std::count(std::begin(SERVO_IDS), std::end(SERVO_IDS), servo_id) == 0){
        this->error_message = "Servo "+String(servo_id)+" not found";
        return false;
      }

      int goal_position = this->sw_data[servo_id-1].goal_position;
      int present_position = this->dxl->getPresentPosition(servo_id);
      if(this->dxl->getLastLibErrCode() > 0){
        this->error_message = "Servo "+String(servo_id)+" Error: "+String(this->dxl->getLastLibErrCode());
        return false;
      }
      return abs(present_position - goal_position) > DXL_MOVING_STATUS_THRESHOLD;
    }

    bool isMoving(){
      this->error_message = "";
      bool moving = false;
      for(int i = 0; i < NUM_SERVOS; i++){
        int id = SERVO_IDS[i];
        moving |=  this->isMoving(id);
      }
      return moving;
    }

    bool currentPosition(std::vector<int> &current_positions){
      this->error_message = "";
      for(int i = 0; i < NUM_SERVOS; i++){
        int servo_id = SERVO_IDS[i];
        int present_position = this->dxl->getPresentPosition(servo_id);
        if(this->dxl->getLastLibErrCode() > 0){
          this->error_message = "Servo "+String(servo_id)+" Error: "+String(this->dxl->getLastLibErrCode());
          return false;
        }
        current_positions.push_back(present_position);
      }
      return true;
    }
    
  private:
    static const int NUM_SERVOS = 9;
    int SERVO_IDS[NUM_SERVOS] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int LEDS[NUM_SERVOS] = {1, 1, 1, 2, 2, 2, 4, 4, 4};
    int torque_limit;
    int speed_limit;

    sr_data_t sr_data[NUM_SERVOS];
    DYNAMIXEL::InfoSyncReadInst_t sr_infos;
    DYNAMIXEL::XELInfoSyncRead_t info_xels_sr[NUM_SERVOS];

    sw_data_t sw_data[NUM_SERVOS];
    DYNAMIXEL::InfoSyncWriteInst_t sw_infos;
    DYNAMIXEL::XELInfoSyncWrite_t info_xels_sw[NUM_SERVOS];

    static const uint16_t user_pkt_buf_cap = 128;
    uint8_t user_pkt_buf[user_pkt_buf_cap];

    String error_message;

    Dynamixel2Arduino *dxl;
};

Gripper *gripper;

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

String ping(){
  if(!gripper->ping())
    return String(Response::ERROR_STATE)+","+gripper->getErrorMessage();  
  return String(Response::SUCCEEDED);
}

String getState() {
  std::vector<int> current_position;
  if(!gripper->currentPosition(current_position)){
    return String(Response::ERROR_STATE)+","+gripper->getErrorMessage();  
  }
  return String(Response::SUCCEEDED)+","+poseToString(current_position);
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

String moveGripper(String command){
  std::vector<int> target_position = commandToPose(command);
  
  if(!gripper->move(target_position)){
    return String(Response::ERROR_STATE)+","+gripper->getErrorMessage();  
  }
  return getState();
}

String moveServo(String command){
  int servo_id = getValue(command, ',', 1).toInt();
  int target_position = getValue(command, ',', 2).toInt();
  
  if(!gripper->move(servo_id, target_position)){
    return String(Response::ERROR_STATE)+","+gripper->getErrorMessage();  
  }
  return getState();
}

String processCommand(String command){
  String response;
  //this isn't perfect as spam data can be interpretted as a given action but whatever
  Command action = Command(getValue(command, ',', 0).toInt());
  switch(action){
    case Command::PING:
      return ping();
    case Command::STOP:
      return stopGripper(command);
    case Command::MOVE:
      return moveGripper(command);
    case Command::MOVE_SERVO:
      return moveServo(command);
    case Command::GET_STATE:
      return getState();
    default:
      return String(Response::ERROR_STATE)+",Unkown Command: "+command;
  }
}

void setup() {
  Serial.begin(115200);
  while(!Serial);

  while(!Serial.available()){
    Serial.read();
  }

  gripper = new Gripper();
  delay(100);
  if(!gripper->ping()){
    Serial.println(String(Response::ERROR_STATE)+","+gripper->getErrorMessage());
  }
  if(!gripper->enableServos()){
    Serial.println(String(Response::ERROR_STATE)+","+gripper->getErrorMessage());
  }
  String response = getState();
  Serial.println(response);
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    String response = processCommand(command);
    Serial.println(response);
  }
}