# 项目说明：该项目是基于yolov8代码由微智启软件工作室改进，支持鼠标绘制区域计数
#划线计数，可统计上行和下行的数量
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import os
from collections import deque
import math
from PIL import ImageFont, ImageDraw, Image
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

# 用于存储跟踪历史记录
track_history = defaultdict(list)
# 变量初始化为None，表示当前没有选中的区域
current_region = None  # 当前区域
current_point = None  # 当前鼠标点
count = 0  # 统计检测到的总数
up = 0 #上行数量
down = 0 #下行数量
passing = 0  # 通过（触线）的人数
#all_regions和new_regions用于记录描绘点的坐标和拖动的位置
all_regions = []
new_regions = [
    {
        'polygon': None,
        'counts': 0,
        'dragging': False,
        'region_color': None,
        'text_color': (0, 0, 0),
    }
]
# 这个是鼠标监听事件，它接受五个参数：事件类型、鼠标坐标x、鼠标坐标y、标志位和参数。
def mouse_callback(event, x, y, flags, param):
    # 使用global关键字，把变量提升为全局变量
    global current_point
    # 鼠标左键点击
    if event == cv2.EVENT_LBUTTONDOWN:
        # 判断鼠标的xy坐标是否选中了横线（也就是自己画的线），如果选中了那么修改偏移后的xy的坐标
        if new_regions[0]['polygon'] is not None:
            new_regions[0]['dragging'] = True
            new_regions[0]['offset_x'] = x
            new_regions[0]['offset_y'] = y

    # 鼠标移动事件，跟上面差不多，也是修改移动后的xy坐标
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
            if len(new_regions[0]['polygon']) > 1:
                print("坐标点已超过两个，请按空格键形成线条")
            else:
                new_regions[0]['polygon'].append((x, y))
        else:
            new_regions[0]['polygon'] = [(x, y)]

#以下两个方法是用于计算A, B, C, D的点有没有相交，也就是用于判断目标有没有撞线
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

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
    global count
    global up
    global down
    global passing
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
    # 设置显示字体的样式
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
            count = len(clss)
            # 创建一个注释器对象，在图像上增加线条和文字，用于在图像上绘制目标框和跟踪线
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            # 遍历每个目标框、跟踪ID和类别，绘制目标框、跟踪线和区域计数
            for box, track_id, cls in zip(boxes, track_ids, clss):
                #这个是描绘检测框和对应的文本等内容
                annotator.box_label(box, str(names[cls])+str(track_id), color=colors(1, True))
                #目标的中点
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                #如果之前没有记录过该目标的路径，那么创建2个空的容器
                if track_id not in path_record:
                    path_record[track_id] = deque(maxlen=2)
                #记录目标的坐标
                path_record[track_id].append(bbox_center)
                #查询当前目标上一次的坐标位置信息
                before_point = path_record[track_id][0]
                #判断有没有划线
                if new_regions[0]['polygon'] is not None:
                    #点的坐标大于1，才能成一条线，所以判断点的坐标有没有超过1个
                    if len(new_regions[0]['polygon']) > 1:
                        #判断当前目标是否相交（撞线）
                        if intersect(bbox_center, before_point, new_regions[0]['polygon'][0],
                                     new_regions[0]['polygon'][1]):
                            #如果撞线了，那么撞线数+1
                            passing += 1
                            # 计算两者的向量，判断是上行还是下行，并进行对应的计数
                            x = bbox_center[0] - before_point[0]
                            y = bbox_center[1] - before_point[1]
                            angle = math.degrees(math.atan2(y, x))
                            if angle > 0:
                                down += 1
                            elif angle < 0:
                                up += 1
                #记录目标的轨迹，方便后期绘制跟踪先，也就是小尾巴
                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))  # 追踪物体的xy坐标
                #当目标轨迹点超过30个的时候，删掉最初的第一个，如果希望小尾巴长一点，可以把数值调大
                if len(track) > 30:
                    track.pop(0)
                # 如果记录的数量大于30个，那么删除最开始的元素，减轻内存
                if len(path_record) > 30:
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
            # 当点存在，并且点的个数小于2，显示、绘制当前点坐标到窗口上
            if current_point and len(new_regions[0]['polygon']) < 2:
                cv2.circle(frame, current_point, 5, (0, 255, 0), 2)
            # 下面的几个cv2是用于显示窗口左上角的文字的
            # cv2.putText(frame, 'total: {}'.format(count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
            #             line_thickness)

            # 将 OpenCV 图像转换为 PIL 图像
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # 在 PIL 图像上绘制文字
            draw = ImageDraw.Draw(frame_pil)
            txt = f"检测到物体的总数是：{count}\n总共有{passing}个目标经过区域\n上行的数量是：{up}\n下行的数量是：{down}"

            draw.text((20, 10), txt, font=font, fill=(255, 0, 0))
            # 将 PIL 图像转换回 OpenCV 图像
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)



            if new_regions[0]['polygon'] is not None:
                if len(new_regions[0]['polygon']) > 1:
                    cv2.line(frame, new_regions[0]['polygon'][0], new_regions[0]['polygon'][1], (0, 0, 255), 10)


            cv2.imshow('YOLOv8 Counter wzq', frame)
            # 重置总人数个数
        count = 0
        #保存检测结果
        if save_img:
            video_writer.write(frame)
        key = cv2.waitKey(1)
        # 如果按下'q'键，或者点击右上角的关闭按钮，则退出循环
        if key == ord('q') or cv2.getWindowProperty('YOLOv8 Counter wzq', cv2.WND_PROP_VISIBLE) < 1:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='initial weights path')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, default="0", help='file/dir/URL/glob/screen/0(webcam)')
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
