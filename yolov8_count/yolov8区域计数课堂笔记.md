:::info
### 一、快捷键操作指南：
:::

- 描点：**鼠标中键**描绘所需点，数量要大于3个（三点才能成一个区域嘛）
- 创建封闭区域：按**键盘空格键**，即可把瞄点的区域形成一个封闭的区域进行计数
- 清除坐标：当画错想清除点的坐标时，可以在**英文状态输入法**下，按**键盘C**即可清除未封闭的点坐标
- 退出检测窗口：英文状态下，按**键盘q**，即可退出检测
:::info
### 二、运行以及自定义修改文件
:::

1. 在使用程序前，需要安装对应的依赖
2. 如果还没安装依赖，可以在anaconda创建独立环境
```
conda create -n yolo8 python=3.8 -y
```

3. 然后再安装requirements.txt里面的依赖
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/  
```

4. 安装完依赖后，运行目录下的wzq_counter.py即可启动

如果播放的视频过大，可以使用读取完视频帧的时候，cv2缩放窗口大小
例如：
```
   while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1
  
        # 设置缩小比例
        scale = 0.3
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
```

如果想使用自己的权重或者检测其他视频，可以找到下方代码，修改对应的路径，然后运行测试
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1706241165122-073be709-da3a-47ea-9f03-8fab031e9aa3.png#averageHue=%23202125&clientId=uc3954646-e944-4&from=paste&height=227&id=ub681c110&originHeight=227&originWidth=1485&originalType=binary&ratio=1&rotation=0&showTitle=false&size=64920&status=done&style=none&taskId=u1659faad-997b-4995-9cad-a7b823e5f9c&title=&width=1485)


每次新建区域时，采用随机颜色，如果想自定义使用固定颜色，可以把
```
new_regions = [
    {
        'polygon': [],
        'counts': 0,
        'dragging': False,
        'region_color': None,
        'text_color': (0, 0, 0),
    }
]
```
这个数组里面的region_color写上固定颜色
再把这两行代码删除即可
```
random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                new_regions[0]['region_color'] = random_color
```

:::info
### 三、自定义修改代码案例
:::
一、把统计的数量显示在窗口左上角或者右上角，参考wzq_counterdemo1.py   
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1706259961507-b28c53b2-f022-4e0d-af28-8fe6e5e4aede.png#averageHue=%23938567&clientId=u8de5e4a7-9601-4&from=paste&height=1104&id=ue7a9f286&originHeight=1104&originWidth=1420&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1302689&status=done&style=none&taskId=ua21940b2-72ae-4285-8e3b-6d4570af2b2&title=&width=1420)

1. 先创建一个变量，用于求和全部区域的数量，count=0 #统计人数
2. 在需要求和的方法里面，把变量升级为全局变量，方便统一赋值：global count
3. 当程序读取完每一帧值后，在页面上通过cv2增加文字，如下代码：
```
cv2.putText(frame, 'people {}'.format(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
```
4、在170行附近，程序会遍历所有封闭的区域数组，所以只需在循环里面，把所有区域的计数累加即可得到总数
```
for region in all_regions:
    count+=region['counts']
```

5、由于同一个物体在连续的几帧时间里，会重复出现在区域里面，一直累计是不对的，所以显示完数据后，需要重置计数：
```
count = 0
```
**扩展思考：**
如果想把文字添加到右上角应该怎么操作呢？
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1706260873853-986c08e7-1244-4c65-97d9-7e34fec53af9.png#averageHue=%23727b79&clientId=u8de5e4a7-9601-4&from=paste&height=360&id=u73d7dd74&originHeight=360&originWidth=546&originalType=binary&ratio=1&rotation=0&showTitle=false&size=163005&status=done&style=none&taskId=u98496ee1-2107-49b2-a7ce-c3f86e5a302&title=&width=546)

只需把cv2.putText方法里面的第三个参数的坐标即可
```
cv2.putText(frame, 'people {}'.format(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
```
也可以获取窗口的大小，然后使用相对大小进行灵活放置
例如在这个读取完帧的后面
```
 while videocapture.isOpened():
        success, frame = videocapture.read()
```
加上
```
# 获取窗口宽度
    window_width = frame.shape[1]

    # 计算文本在右侧的位置
    text_position = (window_width - 270, 30)
```

二、实现汽车上行和下行的数量统计，参考wzq_counterdemo2.py 
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1706273666562-6fecae93-594b-4aa4-82b0-6d93c5f0a3d0.png#averageHue=%236a6552&clientId=uede16496-d2e7-4&from=paste&height=687&id=u6a6dbd7d&originHeight=687&originWidth=1160&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1237672&status=done&style=none&taskId=uade1a368-3573-47de-b231-6e3ddd45e70&title=&width=1160)
实现思路，先定义两个空的数组用于存放追踪的ID，然后把追踪到的ID存入数组中，通过len统计上行和下行的数量，使用cv2展示到窗口上。**为了区分，我们把第一个画的区域为上行，第二个画的位置为下行。**如果是十字路口，需要统计4个的话，那么程序可以按照顺时针的方式来区分，定义哪个是第一个。
1、创建空数组
```
up=[]
down=[]
```

```
path_record={}

global up
global down
```

2、在追踪完毕的时候，得到追踪的id。判断区域的位置是否包含追踪的ID坐标。然后再判断之前是否加过了，如果加过了那么就不用再加入数组了。
```
                for i, region in enumerate(all_regions):
                    # 如果是在第一个区域，那么把追踪的ID加入上行的数组
                    if i == 0:
                        if track_id not in up:
                            up.append(track_id)
                    # 如果是第二个区域，那么加入下行的数组
                    elif i == 1:
                        if track_id not in down:
                            down.append(track_id)
```

3、获取完视频帧后，用cv2把二者的数量显示在窗口上。
```
        cv2.putText(frame, 'UP: {}'.format(len(up)), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255) ,
                        line_thickness)

        cv2.putText(frame, 'Down {}'.format(len(down)), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


```

扩展思考：如果是来往的人群，那么怎么统计上下行走的数量呢？
大致思路，定义两个空的数组和变量，再划两个并行的区域。
如果走过了第一个的区域，把id加入上行的数组里面，再判断下行的数组里面有没有这个id，如果有说明他是从下行的

三、通过一条直线，进行数据统计，包括上行和下行
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1707206483559-7000e82e-a523-4a4b-9f45-0ceb7460dd60.png#averageHue=%237b7a6b&clientId=u352805a5-cc4d-4&from=paste&height=646&id=u4bd750ff&originHeight=808&originWidth=1777&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1154736&status=done&style=none&taskId=u24f22ef0-6ff3-4a83-b639-56efd6cf4ff&title=&width=1421.6)
基本原理：通过记录追踪id的最近两次坐标，然后对比二者的相对位置（向量），从而得知是上行还是下行，并记录到对应的数组中。
视频太大，先缩小视频
```
        # 设置缩小比例
        scale = 0.3
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
```
1、创建上行和下行的变量，用于记录上和下的人数
```
up = 0
down = 0
```
2、创建path_record变量记录每个id的路径，并且把up和down转为全局变量
```
path_record={}

global up
global down
```
3、修改对象格式：
```
new_regions = [
    {
        'polygon': None,
        'counts': 0,
        'dragging': False,
        'region_color': None,
        'text_color': (0, 0, 0),
    }
]
```

5、移动鼠标事件
左键
```
# 判断鼠标的xy坐标是否在循环遍历的区域new_regions[0]['polygon'][0]
        if new_regions[0]['polygon'] is not None:
            # if new_regions[0]['polygon'].contains(Point((x, y))):
            new_regions[0]['dragging'] = True
            new_regions[0]['offset_x'] = x
            new_regions[0]['offset_y'] = y
```
移动
```
 if new_regions[0] is not None and new_regions[0]['dragging']:
            dx = x - new_regions[0]['offset_x']
            dy = y - new_regions[0]['offset_y']
            new_regions[0]['polygon'] = ([(p[0] + dx, p[1] + dy) for p in new_regions[0]['polygon']])
            new_regions[0]['offset_x'] = x
            new_regions[0]['offset_y'] = y
```
鼠标左键抬起
```
        if new_regions[0] is not None and new_regions[0]['dragging']:
            new_regions[0]['dragging'] = False
```
6、鼠标中键事件
```
current_point = (x, y)
        if new_regions[0]['polygon'] is not None:
            if len(new_regions[0]['polygon']) > 1:
                print("坐标点已超过两个，请按空格键形成线条")
            else:
                new_regions[0]['polygon'].append((x, y))
            
        else:
            new_regions[0]['polygon'] = [(x, y)]
```
7、断4个点的坐标是否相交（网上copy的代码）
```
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
```
8、记录用户路径的字典
```
path_record={}
```
9、当程序读取完视频帧之后，添加上行和下行的人数，并且绘制线段
```
# 获取窗口宽度
window_width = frame.shape[1]
# 计算文本在右侧的位置
text_position = (window_width - 280, 30)
cv2.putText(frame, 'UP: {}'.format(up), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    line_thickness)
cv2.putText(frame, 'Down {}'.format(down), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
if new_regions[0]['polygon'] is not None:
    if len(new_regions[0]['polygon']) > 1:
        cv2.line(frame, new_regions[0]['polygon'][0], new_regions[0]['polygon'][1], (0, 0, 255), 10)
```
10、当程序追踪完目标后，记录追踪目标最近的两次坐标点，通过对比两次的坐标，进行判断是上行还是下行
```
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
```
11、为了避免内存溢出，所以当记录的个数超过30个时，删除第一个元素
```
#如果记录的数量大于50个，那么删除最开始的元素，减轻内存
if len(path_record) >30:
    del path_record[list(path_record)[0]]
```
导入相关的依赖
```
from collections import deque
import math
```

四：最终两次大更新，代码分别封装在了**wzq_counterdemo4.py（划线计数）**和**wzq_counterdemo5.py（区域计数）**
本次增加了摄像头检测功能以及其他的一些优化
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1708480366471-114ee2b1-20e6-459d-a924-fb7baa43f893.png#averageHue=%2374766b&clientId=u49b7b309-7717-4&from=paste&height=568&id=u1f4e6834&originHeight=710&originWidth=1717&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1010299&status=done&style=none&taskId=ua4cbf8a2-9c18-43fb-8b15-a9da00e7dcb&title=&width=1373.6)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1157688/1708479572154-47795cf3-fd11-4279-adf2-f7bde7d6d214.png#averageHue=%236b6655&clientId=u49b7b309-7717-4&from=paste&height=550&id=uf9e52ddc&originHeight=687&originWidth=1168&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1239306&status=done&style=none&taskId=u7f44690f-4fab-4e74-9b42-045438c53c4&title=&width=934.4)


技术客服：3447362049
发送购买记录，可以让客服拉入yolo交流群：850471252

