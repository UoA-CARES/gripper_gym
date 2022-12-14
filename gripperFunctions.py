import dynamixel_sdk as dxl         # Uses Dynamixel SDK library

"""
The servo class contains methods used to change attributes of the servo motors
most useful for setting up the servos to move when they are chained togerther
Beth Cutler
"""


class Servo(object):

    def __init__(self, portHandler, packetHandler, LED_colour, addresses, motor_id):
        self.portHandler = portHandler
        self.packetHandler = packetHandler

        self.LED_colour = LED_colour
        self.addresses = addresses
        self.motor_id = motor_id
        # TODO fix
        self.device_name = "COM5"

    def turn_on_LED(self):

        #ledColours = [0, 3, 2, 0, 7, 5, 0, 4, 6]

        # write and read to servos
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.motor_id, self.addresses["led"], self.LED_colour)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully LED turned on" %
                  self.motor_id)

    # limit the torque of the motors

    def limit_torque(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, self.motor_id, self.addresses["torque_limit"], 180)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque limited" %
                  self.motor_id)

    # enable the torque of the motors
    def enable_torque(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, self.motor_id, self.addresses["torque_enable"], 1)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully torque enabled" %
                  self.motor_id)

    # disable the torque of the motors

    def disable_torque(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.motor_id, self.addresses["torque_enable"], 0)
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully disabled torque" %
                  self.motor_id)

    # limit the speed of the servos
    def limit_speed(self):

        # write and read to servos
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, self.motor_id, self.addresses["moving_speed"], 90)
        # verify write read successful
        if dxl_comm_result != dxl.COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel#%d has been successfully speed limited" %
                  self.motor_id)

    def moving_check(self):
        # write and read to servos
        dxl_moving_result, dxl_comm_result, dxl_error = self.packetHandler.read1ByteTxRx(
            self.portHandler, self.motor_id, self.addresses["moving"])

        print(dxl_moving_result)
        return int(dxl_moving_result) == 1
