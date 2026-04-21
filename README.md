# Object-Detection
Object Detection using YOLOv4 and OpenCV

Project Description

This project is about detecting objects in an image using the YOLOv4 model in Python. The user selects an image, and the system identifies objects in that image. It draws boxes around the objects and shows their names along with confidence scores.

Features

Upload an image from your computer
Detect multiple objects in one image
Show object names with confidence values
Draw bounding boxes around detected objects
Display the result using a graph window
Save the output image automatically
Technologies Used

Python
OpenCV
NumPy
Tkinter
Matplotlib
Requirements

Make sure you have the following installed:

Python

Required Python libraries:

opencv-python
numpy
matplotlib
You also need these YOLOv4 files:

yolov4.weights
yolov4.cfg
coco.names
How to Run the Project

Install the required libraries using pip
Place YOLO files in your project folder
Run the Python script
When you run the program:

A window will open to select an image
After selecting the image, detection will start
The result will be displayed with boxes and labels
The output image will also be saved
Parameters

INPUT_SIZE: Controls image size for detection (higher value gives better accuracy but slower speed)
CONF_THRESH: Minimum confidence to detect objects
NMS_THRESH: Used to remove duplicate boxes
Limitations

It may run slow on systems without a GPU
Detection accuracy depends on the model and image quality
File paths must be correct
Future Work

Add live detection using webcam
Improve speed and accuracy
Use latest YOLO versions
Add different colors for different objects
Author

Attique Ahmed BS Artificial Intelligence
