import cv2
import supervision as sv
from ultralytics import YOLO
# from ultralytics import YOLOv10
import distance
import time
import os

model = YOLO('weights/yolov8x.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture('video/yolo_test_video.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 192)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 108)

if not cap.isOpened():
    print("Can't open the video file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/5)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/5)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1.0 / fps

output_folder = 'output_video/7'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0

predict_start_time = time.time()

processing_time_list = []

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (frame_width, frame_height))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('frame', frame)
    # results = model(frame, verbose=True, imgsz=(1920, 1080))[0]
    results = model.predict(source=frame, verbose=True, imgsz=(640, 384))[0]
    processing_time_list.append(float(results.speed['preprocess']) + float(results.speed['inference']) + float(results.speed['postprocess']))
    boxes = len(results.boxes.xywh.tolist())

    dist = []
    # if boxes != 0:
    #     dist = distance.distance(results=results)
    #     print(f'Distance: {dist}')

    detections = sv.Detections.from_ultralytics(results)
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)

    # dist = [1.2]

    # for i, box in enumerate(results.boxes.xywh.tolist()):
    #     x, y, w, h = box
    #     label_text = f"Distance: {dist[0]:.2f}m" if boxes != 0 else "NO distance"
    #     cv2.putText(annotated_image, label_text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.8, (255, 255, 255), 2)

    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    frame_filename = os.path.join(output_folder, f'frame_{frame_count:05d}.png')
    cv2.imwrite(frame_filename, annotated_image)
    frame_count += 1

    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # delay_time = max(int((frame_duration - elapsed_time) * 1000), 1)

    cv2.imshow('Webcam', annotated_image)
    cv2.waitKey(1)

    # if cv2.waitKey(delay_time) & 0xFF == 27:
    #     print("Escape hit, closing...")
    #     break

cap.release()
cv2.destroyAllWindows()

predict_end_time = time.time()
print(f"Total time for prediction: {predict_end_time - predict_start_time:.2f} seconds")
 # sec
print(f"FPS: {1 / (sum(processing_time_list) / len(processing_time_list) / 1000)}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video/7/reconstructed_video.mp4', fourcc, fps, (frame_height, frame_width))

for i in range(frame_count):
    frame_filename = os.path.join(output_folder, f'frame_{i:05d}.png')
    frame = cv2.imread(frame_filename)
    out.write(frame)

out.release()
print("Video reconstruction complete.")