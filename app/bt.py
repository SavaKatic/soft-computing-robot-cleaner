#!/usr/bin/python

import logging
import logging.handlers
import argparse
import sys
import os
import random
import threading
import time
import data
from bluetooth import *

class LoggerHelper(object):
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        pass

def setup_logging():
    # Default logging settings
    LOG_FILE = "./raspibtsrv.log"
    LOG_LEVEL = logging.INFO

    # Define and parse command line arguments
    argp = argparse.ArgumentParser(description="Raspberry PI Bluetooth Server")
    argp.add_argument("-l", "--log", help="log (default '" + LOG_FILE + "')")

    # Grab the log file from arguments
    args = argp.parse_args()
    if args.log:
        LOG_FILE = args.log

    # Setup the logger
    logger = logging.getLogger(__name__)
    # Set the log level
    logger.setLevel(LOG_LEVEL)
    # Make a rolling event log that resets at midnight and backs-up every 3 days
    handler = logging.handlers.TimedRotatingFileHandler(LOG_FILE,
        when="midnight",
        backupCount=3)

    # Log messages should include time stamp and log level
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    # Attach the formatter to the handler
    handler.setFormatter(formatter)
    # Attach the handler to the logger
    logger.addHandler(handler)

    # Replace stdout with logging to file at INFO level
    # sys.stdout = LoggerHelper(logger, logging.INFO)
    # Replace stderr with logging to file at ERROR level
    sys.stderr = LoggerHelper(logger, logging.ERROR)

class BTConnect(object):
    def __init__(self):
        self.uuid = "7be1fcb3-5776-42fb-91fd-2ee7b5bbb86d"

    def start(self):
        thread = threading.Thread(target=self.run)
        thread.start()

    def makeBluetoothSockets(self):
        while True:
            # Create a new server socket using RFCOMM protocol
            server_sock = BluetoothSocket(RFCOMM)
            # Bind to any port
            server_sock.bind(("", PORT_ANY))
            # Start listening
            server_sock.listen(1)

            # Get the port the server socket is listening
            port = server_sock.getsockname()[1]

            # Start advertising the service
            advertise_service(server_sock, "RaspiBtSrv",
                              service_id=self.uuid,
                              service_classes=[self.uuid, SERIAL_PORT_CLASS],
                              profiles=[SERIAL_PORT_PROFILE])

            print("Waiting for connection on RFCOMM channel %d" % port)

            # This will block until we get a new connection
            client_sock, client_info = server_sock.accept()
            print("Accepted connection from ", client_info)

            # Write the data to the client
            self.sendingThread = threading.Thread(target=self.sendData, args=(client_sock,))
            self.receivingThread = threading.Thread(target=self.receiveData, args=(client_sock,))

            self.sendingThread.start()
            self.receivingThread.start()

            self.sendingThread.join()

    def run(self):
        # Setup logging
        # setup_logging()

        # We need to wait until Bluetooth init is done
        time.sleep(10)

        # Make device visible
        os.system("sudo hciconfig hci0 piscan")

        self.makeBluetoothSockets()

    def receiveData(self, socket):
        while(True):
            time.sleep(1)
            receivedData = socket.recv(1024)
            if len(receivedData) != 0:
                receivedString = receivedData.decode('ASCII')
                print("Received string: ", receivedString)

                if "stop" in receivedString:
                    print("******POSLAO JE STOM DATI************")
                    data.application_state = data.ApplicationState.STOP
                elif "start" in receivedString:
                    data.application_state = data.ApplicationState.START

            else:
                print("No command from phone.")


    def sendData(self, socket):
        while (True):
            side = data.table_side
            dataToSend = "Power voltage left:" + str(data.power_voltage_left) + ";Temperature left: " + str(data.driver_temp_left) +  ";Actual current left: " + str(data.actual_current_left) +\
                         ";Coordinates: " + str(round(data.x_current, 2)) + "," + str(round(data.y_current, 2)) + "," + str(round(data.f_current, 2)) +\
                         ";Power voltage right:" + str(data.power_voltage_right) + ";Temperature right: " + str(data.driver_temp_right) + ";Actual current right: " + str(data.actual_current_right) +\
                         ";" + str(data.emotional_state.name) + ";Y: " + str(round(data.y_difference, 2)) + ";Distance: " + str(round(data.cordinates_from_camera[0], 2))
            socket.send(dataToSend)
            # print("Data sent: ", dataToSend)
            time.sleep(1)


if __name__ == '__main__':
    btcon = BTConnect()
    btcon.start()

