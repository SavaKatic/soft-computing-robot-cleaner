from flask import Flask, request, Response, jsonify
import os, sys
import queue
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from PIL import Image
import io, base64
import json
import jsonpickle
import queue
from multiprocessing.pool import ThreadPool

app = Flask(__name__)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Maximum number of boxes that can be considered on a single image
MAX_BOXES_NUMBER = 10

# Minimum likelihood box shows correct class that has to be satisfied
MIN_SCORE_THRESH=0.90

PIXEL_WIDTH = 640
PIXEL_HEIGHT = 480

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
        # Load the Tensorflow model into memory.
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

    def load_tensors(self):
        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_bounding_boxes(self, frame):
        frame_expanded = np.expand_dims(frame, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: frame_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.90)


        result = [] # max-heap data structure
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
                ty = bottom + int(box_height / 2)

                result.append((tx, ty, p))

        return result

model = PeanutDetectionModel()
frame1 = []
frame2 = []

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def convert_image_to_np_array(bytes):
    return toRGB(stringToImage(bytes))

@app.route('/')
def index():
    return "Hello WORLD!"

@app.route('/peanut-model/api/boxes', methods=['POST'])
def getBoxes():
    f1 = request.json['frame1']
    f2 = request.json['frame2']

    frame1 = convert_image_to_np_array(f1)
    frame2 = convert_image_to_np_array(f2)

    pool = ThreadPool(processes=1)
    process = pool.apply_async(model.get_bounding_boxes, (frame1,))
    targets2 = model.get_bounding_boxes(frame2)
    targets1 = process.get()

    return jsonify({'targets1': targets1, 'targets2': targets2}), 200
    # return Response(response={'targets1': targets1, 'targets2': targets2}, status=200, mimetype="application/json")

if __name__ == "__main__":
    app.run(debug=True)

