# 项目说明：该项目是基于yolov8代码由微智启软件工作室改进，支持鼠标绘制区域计数
#画线区域计数,总共定义了两个区域，第一个是上行的区域，第二个是下行的区域
#测试资料：链接：https://pan.baidu.com/s/15WKMnAeP0DjxQpQ5bP8w3A?pwd=6666 提取码：6666
#链接：https://pan.baidu.com/s/14-sKGo5fT9CXLKVfh603JQ?pwd=6666   提取码：6666
#如果链接失效，可以联系技术客服在群文件下载：3447362049

import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import os
from collections import deque
import math
import cv2
from PIL import ImageFont, ImageDraw, Image
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import random
from copy import deepcopy
from shapely.geometry import Polygon
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry.point import Point
# 用于存储跟踪历史记录
track_history = defaultdict(list)
# 变量初始化为None，表示当前没有选中的区域
current_region = None  # 当前区域
current_point = None  # 当前鼠标点
detect_count = 0  # 统计检测到的总数
up = 0 #上行数量
down =0#下行数量
total_passing = []  #走过区域的总个数
total_region=0 #现在在区域内的个数
#all_regions和new_regions用于记录描绘点的坐标和拖动的位置
all_regions = []
new_regions = [
    {
        'polygon': [],
        'counts': 0,
        'dragging': False,
        'region_color': None,
        'text_color': (0, 0, 0),
    }
]
# 这个是鼠标监听事件，它接受五个参数：事件类型、鼠标坐标x、鼠标坐标y、标志位和参数。
def mouse_callback(event, x, y, flags, param):
    global current_region
    global current_point
    # 鼠标左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        for region in all_regions:
            # 判断鼠标的xy坐标是否在循环遍历的区域
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y
    # 当鼠标移动时执行。如果当前选中的区域不为空且处于拖拽状态，则计算鼠标移动的距离，并根据偏移量更新区域的多边形形状。同时更新偏移量。
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y

    # 当鼠标左键抬起时执行。如果当前选中的区域不为空且处于拖拽状态，则将拖拽状态设置为False
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False
    # 鼠标中键按下,记录所有点的坐标，加入新的坐标数组
    elif event == cv2.EVENT_MBUTTONDOWN:
        current_point = (x, y)
        new_regions[0]['polygon'].append((x, y))
def run(
        weights='yolov8n.pt',
        device='',
        source="",
        save_img=True,  # 是否保存检测结果
        view_img=True,  # 是否显示图像
        exist_ok=False,  # 是否存在
        classes=None,
        line_thickness=2,
        track_thickness=2,
        region_thickness=2
):
    # 初始化帧计数器为0
    vid_frame_count = 0
    # 使用global关键字，把变量提升为全局变量
    global current_point
    global detect_count
    global up
    global down
    global total_passing
    global total_region
    scale = 1 #视频的缩放比例，如果想缩小比例，用小数点表示即可，例如：0.8
    path_record = {} #用于记录每个目标的最近两次位置，通过对比可以得知是上行还是下行
    # 如果检测的资源是摄像头，那么就会是一串数字字符，把他们转成数字格式
    if source.isdigit():
        source = int(source)
    elif not Path(source).exists():
        print("检测资源路径错误")
    # 设置YOLO模型，并根据设备类型选择使用CPU或GPU
    model = YOLO(f'{weights}')
    model.to('cuda') if device == '0' else model.to('cpu')
    # 提取模型中的类别名称
    names = model.model.names
    # 打开摄像头
    videocapture = cv2.VideoCapture(source)
    #获取检测帧的宽高
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    #获取视频帧的帧率
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')
    # 设置输出目录，并在其中创建一个新视频文件
    save_dir = increment_path(Path('output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source,int):
        source=str(source)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (int(frame_width*scale), int(frame_height*scale)))
    #设置显示字体的样式
    font = ImageFont.truetype("Alibaba-PuHuiTi-Bold.ttf", size=20, encoding="unic")
    # 遍历视频的每一帧，进行目标检测和跟踪
    while videocapture.isOpened():
        success, frame = videocapture.read()  # 读取检测资源帧
        if not success: #如果检测不到，一般是检测完了，所以跳出循环
            break
        vid_frame_count += 1
        # 如果高度和宽度太高，那么可以稍微缩小一下比例，方便操作
        if frame_width >= 1920 or frame_height >= 1080:
            # 设置缩小比例
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        #对视频帧进行目标追踪检测
        results = model.track(frame, persist=True, classes=classes)
        # 如果检测结果中包含目标框，则提取目标框的位置、ID和类别
        if results[0].boxes.id is not None:
            #这行代码提取目标框的位置信息，并将其存储在boxes变量中。xyxy表示目标框的坐标，cpu()表示将数据从GPU转移到CPU
            boxes = results[0].boxes.xyxy.cpu()
            #这行代码提取目标框的ID信息，并将其存储在track_ids变量中。id表示目标框的ID，int()表示将数据转换为整数类型，tolist()表示将数据转换为列表形式。
            track_ids = results[0].boxes.id.int().cpu().tolist()
            #标签的序号，存放在{}里面
            clss = results[0].boxes.cls.cpu().tolist()
            #统计检测到的数量
            detect_count = len(clss)
            # 创建一个注释器对象，在图像上增加线条和文字，用于在图像上绘制目标框和跟踪线
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            # 遍历每个目标框、跟踪ID和类别，绘制目标框、跟踪线和区域计数
            for box, track_id, cls in zip(boxes, track_ids, clss):
                #这个是描绘检测框和对应的文本等内容
                annotator.box_label(box, str(names[cls])+str(track_id), color=colors(cls, True))
                #目标的中点
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                # 检查目标框是否在指定的区域内，如果是，则增加该区域的计数
                for i, region in enumerate(all_regions):
                    if region['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                        region['counts'] += 1
                        #判断之前有没有记录过改用户ID，没有的话记录
                        if track_id not in total_passing:
                            total_passing.append(track_id)
                            # 由于我们定义的第一个区域是上行区域，所以当遍历的索引是0时，那么就是上行的数量
                            if i==0:
                                up+=1
                            elif i==1:
                                down+=1

                #记录目标的轨迹，方便后期绘制跟踪先，也就是小尾巴
                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))  # 追踪物体的xy坐标
                #当目标轨迹点超过30个的时候，删掉最初的第一个，如果希望小尾巴长一点，可以把数值调大
                if len(track) > 30:
                    track.pop(0)
                # 如果记录的数量大于30个，那么删除最开始的元素，减轻内存
                if len(path_record) > 30:
                    del path_record[list(path_record)[0]]
                #如果记录经过区域的个数超过30个，那么删除第一个
                if len(total_passing)>30:
                    del total_passing[list(total_passing)[0]]
                # 将跟踪线中的点坐标连接起来
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                # 在视频帧上绘制跟踪线
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
        for i, region in enumerate(all_regions):
            total_region+=region['counts']
            region_label = str(region['counts'])
            region_color = region['region_color']
            region_text_color = region['text_color']
            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)
            text_size, _ = cv2.getTextSize(region_label,
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.7,
                                           thickness=line_thickness)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5),
                          (text_x + text_size[0] + 5, text_y + 5),
                          region_color, -1)

            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
                        line_thickness)
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color,
                          thickness=region_thickness)


        # 如果需要显示图像，则显示带有目标框和跟踪线的图像
        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('YOLOv8 Counter wzq')
                cv2.setMouseCallback('YOLOv8 Counter wzq', mouse_callback)
            # 当点存在，显示、绘制当前点坐标到窗口上
            if current_point:
                cv2.circle(frame, current_point, 5, (0, 255, 0), 2)

            # 将 OpenCV 图像转换为 PIL 图像
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 在 PIL 图像上绘制文字
            draw = ImageDraw.Draw(frame_pil)
            txt= f"检测到物体的总数是：{detect_count}\n现在总共有{total_region}个在划线区域内\n上行的数量是：{up}\n下行的数量是：{down}\n总共有{len(total_passing)}个经过区域"

            draw.text((20, 10), txt, font=font, fill=(255, 0, 0))
            # 将 PIL 图像转换回 OpenCV 图像
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            #遍历绘制区域的线条
            for i in range(1, len(new_regions[0]['polygon'])):
                cv2.line(frame, new_regions[0]['polygon'][i - 1], new_regions[0]['polygon'][i], (0, 255, 0), 2)


            cv2.imshow('YOLOv8 Counter wzq', frame)
        # 重置总人数个数
        detect_count = 0
        total_region = 0
        for region in all_regions:
            region['counts'] = 0
        #保存检测结果
        if save_img:
            video_writer.write(frame)
        key = cv2.waitKey(1)
        # 如果按下'q'键，或者点击右上角的关闭按钮，则退出循环
        if key == ord('q') or cv2.getWindowProperty('YOLOv8 Counter wzq', cv2.WND_PROP_VISIBLE) < 1:
            break
            # 当按下空格键时，如果存在未连接的点，把它们连起来
        elif key == ord(' '):
            # 如果坐标大于等于3个，可以把它们封装成Polygon对象
            if len(new_regions[0]['polygon']) >= 3:
                new_regions[0]['polygon'] = Polygon(new_regions[0]['polygon'])
                # 置随机颜色
                random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_regions[0]['region_color'] = random_color
                # 把坐标加入all_regions
                # all_regions.append(new_regions[0])
                all_regions.append(deepcopy(new_regions[0]))
                # 清空new_regions里面的坐标
                new_regions[0]['polygon'] = []
                # 清空当前坐标点
                current_point = None
            else:
                print("坐标个数必须大于3")
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, default="car.mp4", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')
    parser.add_argument('--save-img', action='store_true',default=True, help='save results')
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
