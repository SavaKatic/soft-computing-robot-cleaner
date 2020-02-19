
# ------------------------------
# Imports
# ------------------------------

import os, sys, time, traceback
import math
import cv2
import numpy as np
import tensorflow as tf
import threading, queue
import heapq
from multiprocessing.pool import ThreadPool
from utils import label_map_util
sys.path.insert(1, '/home/mi2019')
import data
from threading import Thread

sys.path.append("..")

MODEL_NAME = 'inference_graph'

CWD_PATH = os.getcwd()

NUM_CLASSES = 1

MAX_BOXES_NUMBER = 10

MIN_SCORE_THRESH=0.68

LEFT_CAMERA_SOURCE = 0
RIGHT_CAMERA_SOURCE = 2
ANGLE_WIDTH = 42
ANGLE_HEIGHT = 36
FRAME_RATE = 30
PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480
CAMERA_HEIGHT = 30
CAMERA_SEPARATION = 19.5
POSSIBLE_PEANUT_DISTANCE_Y = 15

# Path to frozen detection graph .pb file
PATH_TO_CKPT = os.path.join('/home/pi/proba/models/research/object_detection',MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join('/home/pi/proba/models/research/object_detection','training','labelmap.pbtxt')

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class PeanutDetectionModel:
    def __init__(self):
        self.load_labels()
        self.load_frozen_graph()
        self.load_tensors()

    def load_labels(self):
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def load_frozen_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

    def load_tensors(self):
        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_bounding_boxes(self, frame):
        frame_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        result = [] # min-heap data structure
        result2 = []
        for i in range(MAX_BOXES_NUMBER):
            if (scores[0][i] > MIN_SCORE_THRESH):
                ymin, xmin, ymax, xmax = boxes[0][i]

                (left, right, bottom, top) = (xmin * PIXEL_WIDTH, xmax * PIXEL_WIDTH,
                                              ymin * PIXEL_HEIGHT, ymax * PIXEL_HEIGHT)


                box_width = right - left
                box_height = top - bottom
                box_area = box_width * box_height

                p = 100 * box_area / (PIXEL_HEIGHT * PIXEL_WIDTH)
                tx = left + int(box_width / 2)
                ty = (bottom + int(box_height / 2)) * -1

                if p > 0.025:
                    if classes[0][i] == 1: # is it a peanut?
                        heapq.heappush(result, (ty, tx, p))
                    elif classes[0][i] == 2: # is it a logo?
                        heapq.heappush(result2, (ty, tx, p))

        if len(result) >= 4:
            data.emotional_state = data.RobotEmotion.ANGRY

        return result, result2


class Targeter:
    def __init__(self):
        self.model = PeanutDetectionModel()

        self.set_up_cameras()

        self.angler = FrameAngles(PIXEL_WIDTH, PIXEL_HEIGHT, ANGLE_WIDTH, ANGLE_HEIGHT)

    def set_up_cameras(self):
        self.ct1 = CameraThread()
        self.ct1.camera_source = LEFT_CAMERA_SOURCE
        self.ct1.camera_width = PIXEL_WIDTH
        self.ct1.camera_height = PIXEL_HEIGHT
        self.ct1.camera_frame_rate = FRAME_RATE

        self.ct2 = CameraThread()
        self.ct2.camera_source = RIGHT_CAMERA_SOURCE
        self.ct2.camera_width = PIXEL_WIDTH
        self.ct2.camera_height = PIXEL_HEIGHT
        self.ct2.camera_frame_rate = FRAME_RATE

        self.ct1.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.ct2.camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    def start(self):
        t2 = threading.Thread(target=self.run)
        t2.start()

    def get_logos(self, frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)
        image, all_contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_logo = []
        for contour in all_contours:  # za svaku konturu
            center, size, angle = cv2.minAreaRect(contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
            width, height = size
            if width > 20 and width < 70 and height > 30 and height < 70:
                contours_logo.append((center[1] * -1, center[0], 100 * size[0] * size[1]/ (PIXEL_HEIGHT * PIXEL_WIDTH) ))
        return contours_logo

    def onStart(self):
        print("Pokrecem kamere...")
        # start cameras
        self.ct1.start()
        self.ct2.start()

        # pause to stabilize
        time.sleep(0.5)


    def get_info(self):

        # get frames from both cameras
        frame1 = self.ct1.next(black=True, wait=1)
        frame2 = self.ct2.next(black=True, wait=1)

        # find targets for both frames
        # targets1 = self.model.get_bounding_boxes(frame1)
        # targets2 = self.model.get_bounding_boxes(frame2)

        # pool = ThreadPool(processes=1)
        # process = pool.apply_async(self.model.get_bounding_boxes, (frame1,))
        # targets2 = self.model.get_bounding_boxes(frame2)
        # targets1 = process.get()

        modelThread = ThreadWithReturnValue(target=self.model.get_bounding_boxes, args=(frame2,))
        modelThread.start()
        left_camera_peanuts, left_camera_logos = self.model.get_bounding_boxes(frame1)
        right_camera_peanuts, right_camera_logos = modelThread.join()


        print("Na levoj nasao kikiriki: ", left_camera_peanuts)
        print("Na desnoj nasao kikiriki: ", right_camera_peanuts)

        print("Na levoj nasao logo: ", left_camera_logos)
        print("Na desnoj nasao logo: ", right_camera_logos)

        data.y_difference = 0

        # check - object detected in both pictures?
        data.cordinates_from_camera = self.findDistance(left_camera_peanuts, right_camera_peanuts) # detect peanut
        data.cordinates_for_unloading = self.findDistance(left_camera_logos, right_camera_logos)

    def findDistance(self, targets1, targets2):
        possible_difference = 0.2
        if len(targets1) and len(targets2):
            pairs = []
            for position in targets1:
                for otherPosition in targets2:
                    if abs(position[0] * -1 - otherPosition[
                        0] * -1) < POSSIBLE_PEANUT_DISTANCE_Y:
                        xlangle, ylangle = self.angler.angles_from_center(position[1], position[0] * -1, top_left=True,
                                                                          degrees=True)
                        xrangle, yrangle = self.angler.angles_from_center(otherPosition[1], otherPosition[0] * -1,
                                                                          top_left=True, degrees=True)
                        angle_at_top = (180 - (90 - xlangle) - (90 + xrangle))
                        if angle_at_top > 6:
                            angle_at_top = angle_at_top * -1
                            # heapq.heappush(pairs, (angle_at_top, position, otherPosition))
                            pairs.append((position, otherPosition))

            if len(pairs) == 0:  # vidi po jedan na kamerama ali su razliciti
                y1, x1, s1 = targets1[0]
                y2, x2, s2 = targets2[0]

                y1 = y1 * -1
                y2 = y2 * -1

                if (y1 > y2):  # PROVERITI
                    xangle, yangle = self.angler.angles_from_center(x1, y1, top_left=True, degrees=True)
                else:
                    xangle, yangle = self.angler.angles_from_center(x2, y2, top_left=True, degrees=True)

                return [1, xangle * -1]

            else:
                firstpair = pairs[0]
                y1, x1, s1 = firstpair[0]
                y2, x2, s2 = firstpair[1]

                y1 = y1 * -1
                y2 = y2 * -1

                data.y_difference = abs(y1 - y2)

                if abs(s1 - s2) <= possible_difference:
                    # get angles from camera centers
                    xlangle, ylangle = self.angler.angles_from_center(x1, y1, top_left=True, degrees=True)
                    xrangle, yrangle = self.angler.angles_from_center(x2, y2, top_left=True, degrees=True)

                    # triangulation - calculate depth
                    X, Y, Z, D = self.angler.location(CAMERA_SEPARATION, (xlangle, ylangle), (xrangle, yrangle),
                                                      center=True, degrees=True, height=True)

                    # calculate angle from center of the robot
                    a = CAMERA_SEPARATION / 2  # half the side between cameras
                    right_camera_angle = 90 + xrangle  # inverse angle for right camera
                    angle_at_top = math.degrees(np.arcsin(a * math.sin(math.radians(right_camera_angle)) / D))
                    center_angle = 180 - right_camera_angle - angle_at_top
                    print("Saljem {0:.3f} milimetara distancu i {1:.3f} ugao".format(D * 10, center_angle - 90))
                    print("---------------------------------------")
                    return [D*10, center_angle - 90]
                else:
                    return [0, 0]
        elif len(targets1) or len(targets2):
            targets = targets1 if len(targets1) else targets2
            y1, x1, s1 = targets[0]
            xangle, yangle = self.angler.angles_from_center(x1, y1, top_left=True, degrees=True)

            print("Saljem 1 milimetara distancu i {0:.3f} ugao".format(xangle * -1))
            return [1, xangle * -1]
        else:
            return [0, 0]

    def run(self):
        self.onStart()

        # loop
        while True:
            self.get_info()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key != 255:
                print('KEY PRESS:', [chr(key)])



class CameraThread:
  # THREAD FOR LOOPING CAMERA
   
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
            pass

        # done
        return frame


class FrameAngles:
  # TRIANGULATION CALCULATOR

    pixel_width = 640
    pixel_height = 480

    angle_width = 60
    angle_height = None

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
# Testing
# ------------------------------

if __name__ == '__main__':
    targeter = Targeter()
    targeter.start()
