# from https://heartbeat.fritz.ai/end-to-end-object-detection-using-efficientdet-on-raspberry-pi-3-part-3-2bd6a7a6614d
# (nib_box_track) environment
# (nib_box_track)

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import io
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

import os, sys
# os.environ['PYTHONPATH'] += "./models"

# import sys
# sys.path.append("./models")


from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


# %matplotlib inline


def load_image_into_numpy_array(image):
    (im_width, im_height, channel) = image.shape
    return image.astype(np.uint8)

def load_image_into_numpy_array_tf_io(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_keypoint_tuples(eval_config):
    """Return a tuple list of keypoint edges from the eval config.

    Args:
    eval_config: an eval config containing the keypoint edges

    Returns:
    a list of edge tuples, each in the format (start, end)
    """
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
        tuple_list.append((edge.start, edge.end))
    return tuple_list


# recover our saved model
pipeline_config = './pipeline_v_mep.config'
# generally you want to put the last ckpt from training in here
model_dir = './checkpoint/'
print(os.getcwd())
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
    model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
    model=detection_model)
ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


detect_fn = get_model_detection_function(detection_model)

# map labels for inference decoding
label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

# run detector on test image
# it takes a little longer on the first run and then runs at normal speed.


import random
import numpy as np
import cv2
import tensorflow as tf

cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./outputff.avi', fourcc, 20.0, (640, 480))

while (True):
    # Capture frame-by-frame

    ret, frame = cap.read()
    # Our operations on the frame come here
    image_np = load_image_into_numpy_array(frame)

    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in detections:
        keypoints = detections['detection_keypoints'][0].numpy()
        keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        # agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=get_keypoint_tuples(configs['eval_config'])
    )

    # Display the resulting frame
    out.write(image_np_with_detections)
    cv2.imshow('frame', image_np_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()