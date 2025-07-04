import cv2
import numpy as np
import os
import time
import onnxruntime as ort
import supervision as sv

# from ultralytics import YOLO
# model = YOLO('weights/yolov8m.pt')
# model.export(format='onnx', opset=12)

# Load ONNX with GPU provider
session = ort.InferenceSession("weights/yolov8m.onnx", providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name

# Supervision tools
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture('video/yolo_test_video.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/5)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/5)
fps = cap.get(cv2.CAP_PROP_FPS)

output_folder = 'output_video/8'
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
predict_start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Preprocess input
    image = cv2.resize(frame, (640, 640))  # ONNX 輸入大小
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 640, 640)

    # ONNX Inference
    outputs = session.run(None, {input_name: image})
    pred = outputs[0]  # shape: (1, num_boxes, 85)

    # Filter predictions (YOLOv8格式: xywh + conf + 80 classes)
    boxes = pred[0][pred[0][:, 4] > 0.5]  # confidence threshold

    xyxys = []
    confidences = []
    class_ids = []
    for row in boxes:
        x, y, w, h = row[:4]
        conf = row[4]
        cls_scores = row[5:]
        class_id = int(np.argmax(cls_scores))
        x1 = int((x - w/2) * frame_width)
        y1 = int((y - h/2) * frame_height)
        x2 = int((x + w/2) * frame_width)
        y2 = int((y + h/2) * frame_height)
        xyxys.append([x1, y1, x2, y2])
        confidences.append(conf)
        class_ids.append(class_id)

    if xyxys:
        detections = sv.Detections(
            xyxy=np.array(xyxys),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )
    else:
        detections = sv.Detections.empty()

    annotated_image = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:05d}.png'), annotated_image)
    cv2.imshow("Webcam", annotated_image)
    cv2.waitKey(1)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Total time: {time.time() - predict_start_time:.2f}s")

# Combine images to video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video/7/reconstructed_video.mp4', fourcc, fps, (frame_height, frame_width))

for i in range(frame_count):
    frame = cv2.imread(os.path.join(output_folder, f'frame_{i:05d}.png'))
    out.write(frame)

out.release()
print("Video reconstruction complete.")
