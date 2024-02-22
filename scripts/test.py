
from cares_lib.dynamixel.Servo import Servo, DynamixelServoError, OperatingMode
from cares_lib.dynamixel.servos_addresses import addresses
import dynamixel_sdk as dxl


def main():
    protocol = 2
    device_name = '/dev/ttyUSB1'
    port_handler = dxl.PortHandler(device_name)
    packet_handler = dxl.PacketHandler(protocol)
    servo = Servo(port_handler, packet_handler, protocol, 1, 1, 200, 100, 1000, 0, model="XM430-W350")

    servo.port_handler.openPort()
    print(servo.port_handler.is_open)
    servo.enable()
    servo.reboot()
    servo.move(200)


if __name__ == '__main__':
    main()
