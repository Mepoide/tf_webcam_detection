# TFlite Webcam Detection

To use tf_lite models:

src/tflite_webcam_detection.python

To use tf models:

src/real_time.py


## Installation

TFlite Installation on RPi Zero W: https://github.com/cloudwiser/TensorFlowLiteRPIZero

How to Perform Object Detection with TensorFlow Lite on Raspberry Pi: https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588

Object Detection API: https://www.tensorflow.org/lite/examples/object_detection/overview

Running on mobile with TensorFlow Lite: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md

Running TF2 Detection API Models on mobile: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md

TensorFlow Lite converter: https://www.tensorflow.org/lite/convert

tflite_convert_commands_Windows: src/tflite_convert_commands_Windows.txt

Training Custom Object Detector: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#

“TypeError: Expected Operation, Variable, or Tensor, got level_5”: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/issues.html#export-error

## Usage

```Anaconda Prompt
(nib_box_track) G:\Mi unidad\Newral\repos\nibble_box_tracking\object_detection_API\exported-models\package_ssd_mobilenet_v2_fpnlite_320>python ..\..\..\..\tf_lite_webcam_detection\src\real_time.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)