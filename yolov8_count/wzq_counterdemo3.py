# 项目说明：该项目是基于yolov8代码由微智启软件工作室改进，支持鼠标绘制区域计数
# 操作指南
# 1、鼠标中键点击窗口，就可以绘制区域点了

import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import os
import math
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from copy import deepcopy
from collections import Counter
from collections import deque
# 用于存储跟踪历史记录
track_history = defaultdict(list)
# 变量初始化为None，表示当前没有选中的区域
current_region = None  # 当前区域
current_point = None  # 当前鼠标点
up = 0
down = 0
# 两个区域的计数值，颜色，拖拽状态

new_regions = [
    {
        'polygon': None,
        'counts': 0,
        'dragging': False,
        'region_color': None,
        'text_color': (0, 0, 0),
    }
]
# 它接受五个参数：事件类型、鼠标坐标x、鼠标坐标y、标志位和参数。
def mouse_callback(event, x, y, flags, param):
    global current_region
    global current_point
    current_region=new_regions
    # 鼠标左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
            # 判断鼠标的xy坐标是否在循环遍历的区域new_regions[0]['polygon'][0]
            if new_regions[0]['polygon'] is not None:
                # if new_regions[0]['polygon'].contains(Point((x, y))):
                    new_regions[0]['dragging'] = True
                    new_regions[0]['offset_x'] = x
                    new_regions[0]['offset_y'] = y
    # 当鼠标移动时执行。如果当前选中的区域不为空且处于拖拽状态，则计算鼠标移动的距离，并根据偏移量更新区域的多边形形状。同时更新偏移量。
    elif event == cv2.EVENT_MOUSEMOVE:
        if new_regions[0] is not None and new_regions[0]['dragging']:
            dx = x - new_regions[0]['offset_x']
            dy = y - new_regions[0]['offset_y']
            new_regions[0]['polygon'] = ([(p[0] + dx, p[1] + dy) for p in new_regions[0]['polygon']])
            new_regions[0]['offset_x'] = x
            new_regions[0]['offset_y'] = y

    # 当鼠标左键抬起时执行。如果当前选中的区域不为空且处于拖拽状态，则将拖拽状态设置为False
    elif event == cv2.EVENT_LBUTTONUP:
        if new_regions[0] is not None and new_regions[0]['dragging']:
            new_regions[0]['dragging'] = False
    # 鼠标中键按下,记录所有点的坐标，加入新的坐标数组
    elif event == cv2.EVENT_MBUTTONDOWN:
        current_point = (x, y)

        if new_regions[0]['polygon'] is not None:
            if len(new_regions[0]['polygon'])>1:
                print("坐标点已超过两个，请按空格键形成线条")
            else:
                new_regions[0]['polygon'].append((x, y))

        else:
            new_regions[0]['polygon']=[(x,y)]

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def run(
        weights='yolov8n.pt',
        source="",
        device='',
        view_img=False,  # 是否显示图像
        save_img=False,  # 是否保存图像
        exist_ok=False,  # 是否存在
        classes=None,
        line_thickness=2,
        track_thickness=2,

        region_thickness=2,
):
    # 初始化帧计数器为0
    vid_frame_count = 0
    path_record = {}
    global current_point  # 使用全局变量
    global up
    global down
    # 检查视频源路径是否存在，如果不存在则抛出FileNotFoundError异常
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # 设置YOLO模型，并根据设备类型选择使用CPU或GPU
    model = YOLO(f'{weights}')
    model.to('cuda') if device == '0' else model.to('cpu')

    # 提取模型中的类别名称
    names = model.model.names

    # 打开视频文件并获取帧宽度、帧高度、帧率和编码格式
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    # 设置输出目录，并在其中创建一个新视频文件
    save_dir = increment_path(Path('output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    # 遍历视频的每一帧，进行目标检测和跟踪
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        # 获取窗口宽度
        window_width = frame.shape[1]
        # 计算文本在右侧的位置
        text_position = (window_width - 280, 30)
        vid_frame_count += 1
        if current_point:
            cv2.circle(frame, current_point, 5, (0, 255, 0), 2)
            if new_regions[0]['polygon'] is not None:
                for i in range(1, len(new_regions[0]['polygon'])):
                    cv2.line(frame, new_regions[0]['polygon'][i - 1], new_regions[0]['polygon'][i], (0, 255, 0), 2)
        cv2.putText(frame, 'UP: {}'.format(up), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    line_thickness)

        cv2.putText(frame, 'Down {}'.format(down), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        #绘制线段
        if new_regions[0]['polygon'] is not None:
            if len(new_regions[0]['polygon']) > 1:
                cv2.line(frame, new_regions[0]['polygon'][0], new_regions[0]['polygon'][1], (0, 0, 255), 10)

        # Extract the results
        results = model.track(frame, persist=True, classes=classes)
        # 如果检测结果中包含目标框，则提取目标框的位置、ID和类别
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            # 创建一个注释器对象，在图像上增加线条和文字，用于在图像上绘制目标框和跟踪线
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            # 遍历每个目标框、跟踪ID和类别，绘制目标框、跟踪线和区域计数
            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                # #将图像坐标，转为笛卡尔坐标系
                # origin_point = (bbox_center)
                if track_id not in path_record:
                    path_record[track_id] = deque(maxlen=2)
                    # total_track = track_id
                path_record[track_id].append(bbox_center)
                before_point = path_record[track_id][0]
                # origin_previous_midpoint = (before_point[0], frame.shape[0] - before_point[1])
                if new_regions[0]['polygon'] is not None:
                    if len(new_regions[0]['polygon'])>1:
                        if intersect(bbox_center, before_point, new_regions[0]['polygon'][0], new_regions[0]['polygon'][1]):
                            # 计算两者的向量
                            x = bbox_center[0] - before_point[0]
                            y = bbox_center[1] - before_point[1]
                            angle = math.degrees(math.atan2(y, x))
                            if angle>0:
                                down+=1
                            elif angle<0:
                                up+=1
                track = track_history[track_id]  # 跟踪线
                track.append((float(bbox_center[0]), float(bbox_center[1])))  # 追踪物体的xy坐标
                if len(track) > 30:
                    track.pop(0)
                #如果记录的数量大于50个，那么删除最开始的元素，减轻内存
                if len(path_record) >30:
                    del path_record[list(path_record)[0]]
                # 将跟踪线中的点坐标连接起来
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # 在视频帧上绘制跟踪线
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
        # 如果需要显示图像，则显示带有目标框和跟踪线的图像
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('YOLOv8 Counter wzq')
                cv2.setMouseCallback('YOLOv8 Counter wzq', mouse_callback)
            cv2.imshow('YOLOv8 Counter wzq', frame)
        # 如果需要保存图像，则将当前帧写入输出视频文件
        if save_img:
            video_writer.write(frame)
        # 重置每个区域的计数
        # for region in all_regions:  # Reinitialize count for each region
        new_regions[0]['counts'] = 0
        # 如果按下'q'键，则退出循环
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            new_regions[0]['polygon'] = []
            # 清空当前坐标点
            current_point = None

    # 释放帧计数器、视频写入器和视频捕获器，并关闭所有OpenCV窗口
    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', default='person.mp4', type=str, help='video file path')
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--track-thickness', type=int, default=2, help='Tracking line thickness')
    parser.add_argument('--region-thickness', type=int, default=4, help='Region thickness')
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
