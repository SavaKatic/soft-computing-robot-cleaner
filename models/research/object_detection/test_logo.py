import os, sys, time, traceback
import math
import cv2
import numpy as np
import tensorflow as tf
import threading, queue
import heapq
from multiprocessing.pool import ThreadPool
from utils import label_map_util
import data
from threading import Thread

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Maximum number of boxes that can be considered on a single image
MAX_BOXES_NUMBER = 10

# Minimum likelihood box shows correct class that has to be satisfied
MIN_SCORE_THRESH=0.68

# CAMERA CONSTANTS
LEFT_CAMERA_SOURCE = 1
RIGHT_CAMERA_SOURCE = 0
ANGLE_WIDTH = 42
ANGLE_HEIGHT = 36
FRAME_RATE = 30
PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
CAMERA_HEIGHT = 30
CAMERA_SEPARATION = 19.5
POSSIBLE_PEANUT_DISTANCE_Y = 15

class CameraThread:
    # IMPORTANT: a queue is much more efficient than a deque
    # the queue version runs at 35% of 1 processor
    # the deque version ran at 108% of 1 processor

    # ------------------------------
    # User Instructions
    # ------------------------------

    # Using the user variables (see below):
    # Set the camera source number (default is camera 0).
    # Set the camera pixel width and height (default is 640x480).
    # Set the target (max) frame rate (default is 30).
    # Set the number of frames to keep in the buffer (default is 4).
    # Set buffer_all variable: True = no frame loss, for reading files, don't read another frame until buffer allows
    #                          False = allows frame loss, for reading camera, just keep most recent frame reads

    # Start camera thread using self.start().

    # Get next frame in using self.next(black=True,wait=1).
    #    If black, the default frame value is a black frame.
    #    If not black, the default frame value is None.
    #    If timeout, wait up to timeout seconds for a frame to load into the buffer.
    #    If no frame is in the buffer, return the default frame value.

    # Stop the camera using self.stop()

    # ------------------------------
    # User Variables
    # ------------------------------

    # camera setup
    camera_source = 0
    camera_width = 640
    camera_height = 480
    camera_frame_rate = 10
    camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    # buffer setup
    buffer_length = 5
    buffer_all = False

    # ------------------------------
    # System Variables
    # ------------------------------

    # camera
    camera = None

    # buffer
    buffer = None

    # control states
    frame_grab_run = False
    frame_grab_on = False

    # counts and amounts
    frame_count = 0
    frames_returned = 0
    current_frame_rate = 0
    loop_start_time = 0

    # ------------------------------
    # Functions
    # ------------------------------

    def start(self):

        # buffer
        if self.buffer_all:
            self.buffer = queue.Queue(self.buffer_length)
        else:
            # last frame only
            self.buffer = queue.Queue(1)

        # camera setup
        self.camera = cv2.VideoCapture(self.camera_source)
        self.camera.set(3, self.camera_width)
        self.camera.set(4, self.camera_height)
        self.camera.set(5, self.camera_frame_rate)
        self.camera.set(6, self.camera_fourcc)
        time.sleep(0.5)

        # camera image vars
        self.camera_width = int(self.camera.get(3))
        self.camera_height = int(self.camera.get(4))
        self.camera_frame_rate = int(self.camera.get(5))
        self.camera_mode = int(self.camera.get(6))
        self.camera_area = self.camera_width * self.camera_height

        # black frame (filler)
        self.black_frame = np.zeros((self.camera_height, self.camera_width, 3), np.uint8)

        # set run state
        self.frame_grab_run = True

        # start thread
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    def stop(self):

        # set loop kill state
        self.frame_grab_run = False

        # let loop stop
        while self.frame_grab_on:
            time.sleep(0.1)

        # stop camera if not already stopped
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        self.camera = None

        # drop buffer
        self.buffer = None

    def loop(self):

        # load start frame
        frame = self.black_frame
        if not self.buffer.full():
            self.buffer.put(frame, False)

        # status
        self.frame_grab_on = True
        self.loop_start_time = time.time()

        # frame rate
        fc = 0
        t1 = time.time()

        # loop
        while 1:

            # external shut down
            if not self.frame_grab_run:
                break

            # true buffered mode (for files, no loss)
            if self.buffer_all:

                # buffer is full, pause and loop
                if self.buffer.full():
                    time.sleep(1 / self.camera_frame_rate)

                # or load buffer with next frame
                else:

                    grabbed, frame = self.camera.read()

                    if not grabbed:
                        break

                    self.buffer.put(frame, False)
                    self.frame_count += 1
                    fc += 1

            # false buffered mode (for camera, loss allowed)
            else:

                grabbed, frame = self.camera.read()
                if not grabbed:
                    break

                # open a spot in the buffer
                if self.buffer.full():
                    self.buffer.get()

                self.buffer.put(frame, False)
                self.frame_count += 1
                fc += 1

            # update frame read rate
            if fc >= 10:
                self.current_frame_rate = round(fc / (time.time() - t1), 2)
                fc = 0
                t1 = time.time()

        # shut down
        self.loop_start_time = 0
        self.frame_grab_on = False
        self.stop()

    def next(self, black=True, wait=0):

        # black frame default
        if black:
            frame = self.black_frame

        # no frame default
        else:
            frame = None

        # get from buffer (fail if empty)
        try:
            frame = self.buffer.get(timeout=wait)
            self.frames_returned += 1
        except queue.Empty:
            # print('Queue Empty!')
            # print(traceback.format_exc())
            pass

        # done
        return frame


class FrameAngles:
    # ------------------------------
    # User Instructions
    # ------------------------------

    # Set the pixel width and height.
    # Set the angle width (and angle height if it is disproportional).
    # These can be set during init, or afterwards.

    # Run build_frame.

    # Use angles_from_center(self,x,y,top_left=True,degrees=True) to get x,y angles from center.
    # If top_left is True, input x,y pixels are measured from the top left of frame.
    # If top_left is False, input x,y pixels are measured from the center of the frame.
    # If degrees is True, returned angles are in degrees, otherwise radians.
    # The returned x,y angles are always from the frame center, negative is left,down and positive is right,up.

    # Use pixels_from_center(self,x,y,degrees=True) to convert angle x,y to pixel x,y (always from center).
    # This is the reverse of angles_from_center.
    # If degrees is True, input x,y should be in degrees, otherwise radians.

    # Use frame_add_crosshairs(frame) to add crosshairs to a frame.
    # Use frame_add_degrees(frame) to add 10 degree lines to a frame (matches target).
    # Use frame_make_target(openfile=True) to make an SVG image target and open it (matches frame with degrees).

    # ------------------------------
    # User Variables
    # ------------------------------

    pixel_width = 640
    pixel_height = 480

    angle_width = 60
    angle_height = None

    # ------------------------------
    # System Variables
    # ------------------------------

    x_origin = None
    y_origin = None

    x_adjacent = None
    x_adjacent = None

    # ------------------------------
    # Init Functions
    # ------------------------------

    def __init__(self, pixel_width=None, pixel_height=None, angle_width=None, angle_height=None):

        # full frame dimensions in pixels
        if type(pixel_width) in (int, float):
            self.pixel_width = int(pixel_width)
        if type(pixel_height) in (int, float):
            self.pixel_height = int(pixel_height)

        # full frame dimensions in degrees
        if type(angle_width) in (int, float):
            self.angle_width = float(angle_width)
        if type(angle_height) in (int, float):
            self.angle_height = float(angle_height)

        # do initial setup
        self.build_frame()

    def build_frame(self):

        # this assumes correct values for pixel_width, pixel_height, and angle_width

        # fix angle height
        if not self.angle_height:
            self.angle_height = self.angle_width * (self.pixel_height / self.pixel_width)

        # center point (also max pixel distance from origin)
        self.x_origin = int(self.pixel_width / 2)
        self.y_origin = int(self.pixel_height / 2)

        # theoretical distance in pixels from camera to frame
        # this is the adjacent-side length in tangent calculations
        # the pixel x,y inputs is the opposite-side lengths
        self.x_adjacent = self.x_origin / math.tan(math.radians(self.angle_width / 2))
        self.y_adjacent = self.y_origin / math.tan(math.radians(self.angle_height / 2))

    # ------------------------------
    # Pixels-to-Angles Functions
    # ------------------------------

    def angles(self, x, y):

        return self.angles_from_center(x, y)

    def angles_from_center(self, x, y, top_left=True, degrees=True):

        # x = pixels right from left edge of frame
        # y = pixels down from top edge of frame
        # if not top_left, assume x,y are from frame center
        # if not degrees, return radians

        if top_left:
            x = x - self.x_origin
            y = self.y_origin - y

        xtan = x / self.x_adjacent
        ytan = y / self.y_adjacent

        xrad = math.atan(xtan)
        yrad = math.atan(ytan)

        if not degrees:
            return xrad, yrad

        return math.degrees(xrad), math.degrees(yrad)

    def pixels_from_center(self, x, y, degrees=True):

        # this is the reverse of angles_from_center

        # x = horizontal angle from center
        # y = vertical angle from center
        # if not degrees, angles are radians

        if degrees:
            x = math.radians(x)
            y = math.radians(y)

        return int(self.x_adjacent * math.tan(x)), int(self.y_adjacent * math.tan(y))

    # ------------------------------
    # 3D Functions
    # ------------------------------

    def distance(self, *coordinates):
        return self.distance_from_origin(*coordinates)

    def distance_from_origin(self, *coordinates):
        return math.sqrt(sum([x ** 2 for x in coordinates]))

    def intersection(self, pdistance, langle, rangle, degrees=False):

        # return (X,Y) of target from left-camera-center

        # pdistance is the measure from left-camera-center to right-camera-center (point-to-point, or point distance)
        # langle is the left-camera  angle to object measured from center frame (up/right positive)
        # rangle is the right-camera angle to object measured from center frame (up/right positive)
        # left-camera-center is origin (0,0) for return (X,Y)
        # X is measured along the baseline from left-camera-center to right-camera-center
        # Y is measured from the baseline

        # fix degrees
        if degrees:
            langle = math.radians(langle)
            rangle = math.radians(rangle)

        # fix angle orientation (from center frame)
        # here langle is measured from right baseline
        # here rangle is measured from left  baseline
        langle = math.pi / 2 - langle
        rangle = math.pi / 2 + rangle

        # all calculations using tangent
        ltan = math.tan(langle)
        rtan = math.tan(rangle)

        # get Y value
        # use the idea that pdistance = ( Y/ltan + Y/rtan )
        Y = pdistance / (1 / ltan + 1 / rtan)

        # get X measure from left-camera-center using Y
        X = Y / ltan

        # done
        return X, Y

    def location(self, pdistance, lcamera, rcamera, center=False, degrees=True, height=False):

        # return (X,Y,Z,D) of target from left-camera-center (or baseline midpoint if center-True)

        # pdistance is the measure from left-camera-center to right-camera-center (point-to-point, or point distance)
        # lcamera = left-camera-center (Xangle-to-target,Yangle-to-target)
        # rcamera = right-camera-center (Xangle-to-target,Yangle-to-target)
        # left-camera-center is origin (0,0) for return (X,Y)
        # X is measured along the baseline from left-camera-center to right-camera-center
        # Y is measured from the baseline
        # Z is measured vertically from left-camera-center (should be same as right-camera-center)
        # D is distance from left-camera-center (based on pdistance units)

        # separate values
        lxangle, lyangle = lcamera
        rxangle, ryangle = rcamera

        # yangle should be the same for both cameras (if aligned correctly)
        yangle = (lyangle + ryangle) / 2

        # fix degrees
        if degrees:
            lxangle = math.radians(lxangle)
            rxangle = math.radians(rxangle)
            yangle = math.radians(yangle)

        # get X,Z (remember Y for the intersection is Z frame)
        X, Z = self.intersection(pdistance, lxangle, rxangle, degrees=False)

        # get Y
        # using yangle and 2D distance to target
        Y = math.tan(yangle) * self.distance_from_origin(X, Z)

        # baseline-center instead of left-camera-center
        if center:
            X -= pdistance / 2

        # get 3D distance
        D = self.distance_from_origin(X, Y, Z)

        if height:
            D = math.sqrt(abs(D ** 2 - CAMERA_HEIGHT ** 2))

        # done
        return X, Y, Z, D

    # ------------------------------
    # Tertiary Functions
    # ------------------------------

    def frame_add_crosshairs(self, frame):

        # add crosshairs to frame to aid in aligning

        cv2.line(frame, (0, self.y_origin), (self.pixel_width, self.y_origin), (0, 255, 0), 1)
        cv2.line(frame, (self.x_origin, 0), (self.x_origin, self.pixel_height), (0, 255, 0), 1)

        cv2.circle(frame, (self.x_origin, self.y_origin), int(round(self.y_origin / 8, 0)), (0, 255, 0), 1)

    def frame_add_degrees(self, frame):

        # add lines to frame every 10 degrees (horizontally and vertically)
        # use this to test that your angle values are set up properly

        for angle in range(10, 95, 10):

            # calculate pixel offsets
            x, y = self.pixels_from_center(angle, angle)

            # draw verticals
            if x <= self.x_origin:
                cv2.line(frame, (self.x_origin - x, 0), (self.x_origin - x, self.pixel_height), (255, 0, 255), 1)
                cv2.line(frame, (self.x_origin + x, 0), (self.x_origin + x, self.pixel_height), (255, 0, 255), 1)

            # draw horizontals
            if y <= self.y_origin:
                cv2.line(frame, (0, self.y_origin - y), (self.pixel_width, self.y_origin - y), (255, 0, 255), 1)
                cv2.line(frame, (0, self.y_origin + y), (self.pixel_width, self.y_origin + y), (255, 0, 255), 1)

    def frame_make_target(self, outfilename='targeting_angles_frame_target.svg', openfile=False):

        # this will make a printable target that matches the frame_add_degrees output
        # use this to test that your angle values are set up properly

        # svg size
        ratio = self.pixel_height / self.pixel_width
        width = 1600
        height = 1600 * ratio

        # svg frame locations
        x_origin = width / 2
        y_origin = height / 2
        distance = width * 0.5

        # start svg
        svg = '<svg xmlns="http://www.w3.org/2000/svg"\n'
        svg += 'xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        svg += 'width="{}px"\n'.format(width)
        svg += 'height="{}px">\n'.format(height)

        # crosshairs
        svg += '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="1" stroke="green"/>\n'.format(0, width, y_origin,
                                                                                                  y_origin)
        svg += '<line x1="{}" x2="{}" y1="{}" y2="{}" stroke-width="1" stroke="green"/>\n'.format(x_origin, x_origin, 0,
                                                                                                  height)

        # center circle
        svg += '<circle cx="{}" cy="{}" r="{}" stroke="green" stroke-width="1" fill="none"/>'.format(x_origin, y_origin,
                                                                                                     y_origin / 8)

        # distance from screen line
        svg += '<line x1="{0}" x2="{1}" y1="{2}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(
            x_origin - distance / 2, x_origin + distance / 2, y_origin - y_origin / 8)
        svg += '<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(
            x_origin - distance / 2, y_origin - y_origin / 16, y_origin - y_origin / 8)
        svg += '<line x1="{0}" x2="{0}" y1="{1}" y2="{2}" stroke-width="1" stroke="red"/>\n'.format(
            x_origin + distance / 2, y_origin - y_origin / 16, y_origin - y_origin / 8)

        # add degree lines
        for angle in range(10, 95, 10):
            pixels = distance * math.tan(math.radians(angle))

            # draw verticals
            if pixels <= x_origin:
                svg += '<line x1="{0}" x2="{0}" y1="0" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(
                    x_origin - pixels, height)
                svg += '<line x1="{0}" x2="{0}" y1="0" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(
                    x_origin + pixels, height)

            # draw horizontals
            if pixels <= y_origin:
                svg += '<line x1="0" x2="{0}" y1="{1}" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(width,
                                                                                                            y_origin - pixels)
                svg += '<line x1="0" x2="{0}" y1="{1}" y2="{1}" stroke-width="1" stroke="black"/>\n'.format(width,
                                                                                                            y_origin + pixels)

        # end svg
        svg += '</svg>'

        # write file
        outfile = open(outfilename, 'w')
        outfile.write(svg)
        outfile.close()

        # open file
        if openfile:
            import webbrowser
            webbrowser.open(os.path.abspath(outfilename))


def get_logos(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)
    all_contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_logo = []
    contours = []
    for contour in all_contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if width > 20 and width < 70 and height > 30 and height < 70:
            contours_logo.append((center[1] * -1, center[0], 100 * size[0] * size[1] / (PIXEL_HEIGHT * PIXEL_WIDTH)))
            contours.append(contour)

    #cv2.drawContours(frame, contours, -1, (255, 0, 0), 1)
    return contours_logo


if __name__ == '__main__':
    ct1 = CameraThread()
    ct1.camera_source = LEFT_CAMERA_SOURCE
    ct1.camera_width = PIXEL_WIDTH
    ct1.camera_height = PIXEL_HEIGHT
    ct1.camera_frame_rate = FRAME_RATE

    ct2 = CameraThread()
    ct2.camera_source = RIGHT_CAMERA_SOURCE
    ct2.camera_width = PIXEL_WIDTH
    ct2.camera_height = PIXEL_HEIGHT
    ct2.camera_frame_rate = FRAME_RATE

    ct1.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    ct2.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    ct1.start()
    ct2.start()

    # pause to stabilize
    time.sleep(0.5)

    while True:
        # get frames from both cameras
        frame1 = ct1.next(black=True, wait=2)
        frame2 = ct2.next(black=True, wait=2)

        left_camera_logos = get_logos(frame1)
        right_camera_logos = get_logos(frame2)

        print(left_camera_logos)
        print(right_camera_logos)

        # display frame
        cv2.imshow("Left Camera 1", frame1)
        cv2.imshow("Right Camera 2", frame2)

        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty('Left Camera 1', cv2.WND_PROP_VISIBLE) < 1:
            break
        elif cv2.getWindowProperty('Right Camera 2', cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key == ord('q'):
            break
        elif key != 255:
            print('KEY PRESS:', [chr(key)])
