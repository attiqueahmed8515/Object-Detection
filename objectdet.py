import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Function to upload an image
def upload_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    return file_path

# Step 1: Load YOLO
cfg_file = "D:/8 semester/Computer Vision/Unconfirmed 338392.crdownload" # Path to the yolov4.cfg file
weights_file = "D:\8 semester\Computer Vision\yolov4.weights"  # Path to the yolov4.weights file
names_file = "D:\8 semester\Computer Vision\coco.names" # Path to the coco.names file

# Inference tuning (accuracy focused; single-image use is fine)
INPUT_SIZE = 608  # multiples of 32; 608 tends to improve accuracy vs 416
CONF_THRESH = 0.25  # lower to catch harder objects (like people in busy scenes)
NMS_THRESH = 0.50

# Load YOLO
net = cv2.dnn.readNet(weights_file, cfg_file)

# Getting the output layer names correctly (robust to OpenCV version differences)
layer_names = net.getLayerNames()
unconnected = net.getUnconnectedOutLayers()
unconnected = unconnected.flatten() if hasattr(unconnected, "flatten") else unconnected
output_layers = [layer_names[i - 1] for i in unconnected]

# Step 2: Load Class Labels (Coco dataset classes)
with open(names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Step 3: Upload Image
image_path = upload_image()
if not image_path:
    print("No image selected. Exiting.")
    exit()

# Step 4: Load and Prepare the Image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image. Check the file path and integrity.")
    exit()

orig_h, orig_w = image.shape[:2]

def letterbox_image(img, new_size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y

letterboxed, scale, pad_x, pad_y = letterbox_image(image, INPUT_SIZE)
blob = cv2.dnn.blobFromImage(letterboxed, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), (0, 0, 0), swapRB=True, crop=False)

# Step 5: Forward Pass Through YOLO Network
net.setInput(blob)
outs = net.forward(output_layers)

# Step 6: Process YOLO Outputs
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        class_score = float(scores[class_id])
        objectness = float(detection[4])
        confidence = objectness * class_score

        if confidence > CONF_THRESH:
            center_x = detection[0] * INPUT_SIZE
            center_y = detection[1] * INPUT_SIZE
            w = detection[2] * INPUT_SIZE
            h = detection[3] * INPUT_SIZE

            x = (center_x - w / 2 - pad_x) / scale
            y = (center_y - h / 2 - pad_y) / scale
            w = w / scale
            h = h / scale

            x = int(max(0, min(orig_w - 1, x)))
            y = int(max(0, min(orig_h - 1, y)))
            w = int(max(1, min(orig_w - x, w)))
            h = int(max(1, min(orig_h - y, h)))

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

# Step 7: Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

# Step 8: Draw Bounding Boxes and Display Results
if len(indices) > 0:
    indices = indices.flatten()
else:
    indices = []

for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        # Draw rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw label with a filled background so it stays readable
        text = f"{label} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y - 10
        if ty - th - baseline < 0:
            ty = y + th + 10  # move label inside the box if it would go off-image
        tx = x
        if tx + tw >= orig_w:
            tx = max(0, orig_w - tw - 1)
        cv2.rectangle(
    image,
    (tx, ty - th - baseline),
    (tx + tw, ty + baseline),
    (0, 255, 0),
    thickness=-1,
)
        
        cv2.putText(
    image,
    text,
    (tx, ty),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 0),  # current text color (black)
    2,
)

print(f"Detections kept after NMS: {len(indices)}")

# Step 9: Show the Final Image with Detected Objects
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("YOLO Object Detection")
plt.show()

# Optional: Save the output image
cv2.imwrite('output_image.jpg', image)