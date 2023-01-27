/*
code initally written by henry to move the python code over to arduino. 
beth has been filling in the rest. 
glhf
*/
#include <vector> 
#include <DynamixelSDK.h>

// Control table address could be differ by Dynamixel Series
#define ADDRESS_TORQUE_ENABLE           24
#define ADDRESS_GOAL_POSITION           30 
#define ADDRESS_PRESENT_POSITION        37
#define ADDRESS_SPEED_LIMIT             32
#define ADDRESS_TORQUE_LIMIT            15  
#define ADDRESS_LEDS                     25

// Data Byte Length
#define LENGTH_GOAL_POSITION            2
#define LENGTH_PRESENT_POSITION         2

// Protocol version
#define PROTOCOL_VERSION                2.0                 // See which protocol version is used in the Dynamixel

// Default setting
#define DXL_IDS                         [1,2,3,4,5,6,7,8,9]

#define BAUDRATE                        57600
#define DEVICENAME                      "3"                 // Check which port is being used on your controller
                                                            // DEVICENAME "1" -> Serial1
                                                            // DEVICENAME "2" -> Serial2
                                                            // DEVICENAME "3" -> Serial3(OpenCM 485 EXP)

#define TORQUE_ENABLE                   1                   // Value for enabling the torque
#define TORQUE_DISABLE                  0                   // Value for disabling the torque
#define DXL_MINIMUM_POSITION_VALUE      250                 // Dynamixel will rotate between this value
#define DXL_MAXIMUM_POSITION_VALUE      750                // and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
#define DXL_MOVING_STATUS_THRESHOLD     20                  // Dynamixel moving status threshold

#define ESC_ASCII_VALUE                 0x1b

enum Command {
  STOP,
  MOVE, 
  GET_STATE
};

enum Response {
  SUCCEEDED,
  ERROR_STATE
};

bool led_state = true;
int led_pin = 14;

class ServoMotor{
  public:
    ServoMotor(int id, dynamixel::PortHandler *portHandler, dynamixel::PacketHandler *packetHandler, int max, int min){
      this->id = id;
      this->portHandler = portHandler;
      this->packetHandler = packetHandler;
      this->min = min;
      this->max = max;

      int dxl_comm_result; 
      uint8_t dxl_error;
    }


/*
    //NOTE this is for a single servo :) 
    void move(int target_step, int servo_id, int &dxl_comm_result, uint8_t &dxl_error, bool wait=false, int timeout=5){
      this->target_position = target_step;
      int current_position = this->currentPosition(dxl_comm_result, dxl_error);
      //TODO make the actual call to move the servo
      groupSyncRead.addParam(servo_id)
      groupSyncWrite.addParam(servo_id, target_position)
      
      //txPacket moves the servos to the goal position
      dxl_comm_result = groupSyncWrite.txPacket();
      if (dxl_comm_result != COMM_SUCCESS) packetHandler->getTxRxResult(dxl_comm_result);
         print(dxl_comm_result)
      // Clear syncwrite parameter storage
      groupSyncWrite.clearParam();
      long int start_time = millis() * 1000;
      if(wait && this->isMoving(dxl_comm_result, dxl_error) && millis() * 1000 < start_time + timeout){
        
      }
    }
  */  

    void stopMoving(int &dxl_comm_result, uint8_t &dxl_error){
      //TODO make the actual call to move the servo - just send current position
    }

    bool isMoving(int &dxl_comm_result, uint8_t &dxl_error){
      return abs(this->target_position - this->currentPosition(dxl_comm_result, dxl_error)) < DXL_MOVING_STATUS_THRESHOLD;
    }

    int currentPosition(int &dxl_comm_result, uint8_t &dxl_error){
      return 0;
    }

    double currentLoad(int &dxl_comm_result, uint8_t &dxl_error){
      return 0;
    }

    void limitTorque(int torque_limit, int &dxl_comm_result, uint8_t &dxl_error){
      //Serial.print("limiting torque / ");
      dxl_comm_result = this->packetHandler->write1ByteTxRx(this->portHandler, this->id, ADDRESS_TORQUE_LIMIT, torque_limit, &dxl_error);
    }

    void enableTorque(int &dxl_comm_result, uint8_t &dxl_error){
      //Serial.print("enabling torque of ");
      //Serial.print(this->id) ;
      dxl_comm_result = this->packetHandler->write1ByteTxRx(this->portHandler, this->id, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
    }

    void disableTorque(int &dxl_comm_result, uint8_t &dxl_error){
      dxl_comm_result = this->packetHandler->write1ByteTxRx(this->portHandler, this->id, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
    }

    void limitSpeed(int speed_limit, int &dxl_comm_result, uint8_t &dxl_error){
      dxl_comm_result = this->packetHandler->write1ByteTxRx(this->portHandler, this->id, ADDRESS_SPEED_LIMIT, speed_limit, &dxl_error);
    }

    void setLED(int color, int &dxl_comm_result, uint8_t &dxl_error){
      //Serial.println(" / turning on LED ");
      dxl_comm_result = this->packetHandler->write1ByteTxRx(this->portHandler, this->id, ADDRESS_LEDS, color, &dxl_error);
    }

    bool verifyStep(int step){
      return this->min <= step and step <= this->max;
    }

    double step2Angle(int step){
      return double((step - 511.5) / 3.41);
    }

    double angle2step(double angle){
      return int((3.41 * angle) + 511.5);
    }

  private:
    int id;
    dynamixel::PortHandler *portHandler;
    dynamixel::PacketHandler *packetHandler;
    int min;
    int max;
    int current_position;
    int target_position;
};

class Gripper{
  public:
    Gripper(dynamixel::PortHandler *portHandler, dynamixel::PacketHandler *packetHandler, int torque_limit=80, int speed_limit=100){
       this->portHandler = portHandler;
       this->packetHandler = packetHandler;

      groupSyncWrite = new dynamixel::GroupSyncWrite(this->portHandler, this->packetHandler, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION);
      groupSyncRead = new dynamixel::GroupSyncRead(this->portHandler, this->packetHandler, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      int max[NUM_SERVOS] = {900, 750, 769, 900, 750, 802, 900, 750, 794};
      int min[NUM_SERVOS] = {100, 250, 130, 100, 198, 152, 100, 250, 140};
      
      this->torque_limit = torque_limit;
      this->speed_limit = speed_limit;

      for(int id = 0; id < NUM_SERVOS; id++){
        servos[id] = new ServoMotor(id+1, portHandler, packetHandler, min[id], max[id]);
      }
    }

    bool enableServos(){

      int leds[NUM_SERVOS] = {0, 3, 2, 0, 7, 5, 0, 4, 6};
      // currently just ignores any issues with servo communication here...
      for(int servo_id = 0; servo_id < (NUM_SERVOS); servo_id++){
        int dxl_comm_result = 0;
        uint8_t dxl_error = 0;
        servos[servo_id]->limitTorque(this->torque_limit, dxl_comm_result, dxl_error);
        servos[servo_id]->limitSpeed(this->speed_limit, dxl_comm_result, dxl_error);
        servos[servo_id]->enableTorque(dxl_comm_result, dxl_error);
        servos[servo_id]->setLED(leds[servo_id], dxl_comm_result, dxl_error);
      }
      return true;
    }
    
//TODO figue out later
/*
    void moveServo(int servo_id, int target_step, bool wait=true, double timeout=5){
      int dxl_comm_result = 0;
      uint8_t dxl_error = 0;
      this->servos[servo_id]->move(target_step, dxl_comm_result, dxl_error, wait=wait, timeout=timeout);
    }
*/

//TODO enum for motor id? 
    void move(int target_steps[], bool wait=true, long int timeout=5){
      int dxl_comm_result = 0;
      uint8_t dxl_error = 0;
      uint8_t param_goal_position[4];

      for(int id = 0; id < NUM_SERVOS; id++){
        this->groupSyncRead->addParam(id+1);
      }
      
      for(int id = 0; id < NUM_SERVOS; id++){
        
        param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(target_steps[id]));
        param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(target_steps[id]));
//        param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[index]));
//        param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[index]));
   
        this->groupSyncWrite->addParam(id+1, param_goal_position);
      }
      //TODO make the actual call to move the servos
      this->groupSyncWrite->txPacket();

      this->groupSyncWrite->clearParam();
      
      long int start_time = millis() * 1000;
      if(wait && this->isMoving(dxl_comm_result, dxl_error) && millis() * 1000 < start_time + timeout){ 
      }
    }

    void stopMoving(int &dxl_comm_result, uint8_t &dxl_error){
      for(int id = 0; id < NUM_SERVOS; id++){
        this->servos[id]->stopMoving(dxl_comm_result, dxl_error);
      }
    }

    bool isMoving(int &dxl_comm_result, uint8_t &dxl_error){
      for(int id = 0; id < NUM_SERVOS; id++){
        int dxl_comm_result = 0;
        uint8_t dxl_error = 0;
        if(this->servos[id]->isMoving(dxl_comm_result, dxl_error))
          return true;
      }
      return false;
    }

    std::vector<int> currentPosition(int &dxl_comm_result, uint8_t &dxl_error){
      std::vector<int> current_positions(NUM_SERVOS);  
      for(int id = 0; id < NUM_SERVOS; id++){
        current_positions[id] = this->servos[id]->currentPosition(dxl_comm_result, dxl_error);
      }
      return current_positions;
    }
    
  private:
    static const int NUM_SERVOS = 9;
    ServoMotor *servos[NUM_SERVOS];
    int LEDS[NUM_SERVOS] = {0, 3, 2, 0, 7, 5, 0, 4, 6};
    int torque_limit;
    int speed_limit;
    
    dynamixel::PortHandler *portHandler;
    dynamixel::PacketHandler *packetHandler;
    dynamixel::GroupSyncWrite *groupSyncWrite;
    dynamixel::GroupSyncRead *groupSyncRead;
};

// TODO move stuff above into other files etc

Gripper *gripper;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  while(!Serial);


  //Serial.println("Start..");
  // Need to set device name with parameter
  dynamixel::PortHandler *portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
  dynamixel::PacketHandler *packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
  
  if(!portHandler->openPort()){
    //Serial.println("Failed to open the port!");
    return;
  }
  //Serial.println("Succeeded to open the port!");

  // Set port baudrate
  if(!portHandler->setBaudRate(BAUDRATE)){
    //Serial.println("Failed to change the baudrate!");
    return;
  }

  gripper = new Gripper(portHandler, packetHandler); 

  gripper->enableServos();
  
}   

void loop(){

   
  /*
   gripper->move(joint_pos1);
   delay(1000);
   gripper->move(joint_pos2);
   delay(1000);
   gripper->move(joint_pos3);
   delay(1000);
   */

   run_gripper();

   Serial.end();

   


  
}
