import json
from typing import Dict, List, Tuple, Set

import numpy as np
import cv2
import skimage

from ci2n.items import NAME, Device, Junction, Circuit, Port, Connection, MosTypes
from PIL import Image
from tensorflow.keras.preprocessing import image
import igraph as ig

IMG = np.ndarray


def binary_the_img(img: IMG) -> IMG:
    """图片二值化"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def extract_skeleton(binary_img: IMG) -> IMG:
    """骨架提取"""
    # 提取骨架
    skeleton = skimage.morphology.skeletonize(binary_img)
    # 归一化
    skeleton = skeleton.astype(np.uint8) * 255
    return skeleton


def remove_small_connected_components(binary_img: IMG) -> IMG:
    """删除小的连通域"""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    new_img = np.zeros((binary_img.shape[0], binary_img.shape[1]), np.uint8)
    # 所有联通域面积中的最大值的x分之一
    threshold = stats[:, 4][1:].max() // 20

    for i in range(1, num_labels):
        if stats[i][4] > threshold:
            new_img[labels == i] = 255
    return new_img


def get_skeleton_with_items(img: IMG) -> IMG:
    img = binary_the_img(img)
    skeleton = extract_skeleton(img)

    return skeleton


def items_to_rectangles(skeleton_img: IMG, devices: Dict[NAME, Device]) -> IMG:
    new_img = skeleton_img.copy()

    for device in devices.values():
        xmin, ymin = device.top_left
        xmax, ymax = device.bottom_right

        new_img[ymin:ymax, xmin:xmax] = 0
        # 边框画白色正方形
        new_img[ymin, xmin:xmax] = 255
        new_img[ymax - 1, xmin:xmax] = 255
        new_img[ymin:ymax, xmin] = 255
        new_img[ymin:ymax, xmax - 1] = 255

    return new_img


def remove_items(skeleton_img: IMG, devices: Dict[NAME, Device]) -> IMG:
    new_img = skeleton_img.copy()

    for device in devices.values():
        xmin, ymin = device.top_left
        xmax, ymax = device.bottom_right

        new_img[ymin:ymax, xmin:xmax] = 0

    return new_img


# 寻找骨架图的所有交点
def find_junction_points(skeleton_img: IMG) -> List[Tuple[int, int]]:
    # 遍历骨架图的所有点，如果他有超过2个相邻的点，则是交点，输出坐标
    junction_points = []
    for i in range(1, skeleton_img.shape[0] - 1):
        for j in range(1, skeleton_img.shape[1] - 1):
            if skeleton_img[i][j] == 255:
                if np.sum(skeleton_img[i - 1:i + 2, j - 1:j + 2] == 255) > 3:
                    junction_points.append((j, i))
    return junction_points


def remove_duplicate_points(junction_points: List[Tuple[int, int]], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """连在一起的点去重"""
    # 把这些点组成一个二值化图片
    junction_points_img = np.zeros(shape, np.uint8)
    for i, j in junction_points:
        junction_points_img[j][i] = 255

    # 连通域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_points_img, connectivity=8)
    new_junction_points = []
    for i in range(1, num_labels):
        # 取出每个连通域的所有点
        points = np.argwhere(labels == i)
        new_junction_points.append((points[:, 1].mean().astype(int), points[:, 0].mean().astype(int)))

    return new_junction_points


def get_junction_locations(skeleton_img: IMG) -> List[Tuple[int, int]]:
    shape = (skeleton_img.shape[0], skeleton_img.shape[1])

    junction_points = find_junction_points(skeleton_img)
    junction_points = remove_duplicate_points(junction_points, shape)

    return junction_points


def draw_junctions(skeleton_img: IMG, junctions: List[Tuple[int, int]], size: int = 8) -> IMG:
    new_img = skeleton_img.copy()
    for i, j in junctions:
        # 画空心正方形（画四条线）如果空间不够就画到边框为止
        new_img[max(0, j - size):min(new_img.shape[0], j + size), max(0, i - size)] = 0
        new_img[max(0, j - size):min(new_img.shape[0], j + size), min(new_img.shape[1] - 1, i + size)] = 0
        new_img[max(0, j - size), max(0, i - size):min(new_img.shape[1] - 1, i + size)] = 0
        new_img[min(new_img.shape[0] - 1, j + size), max(0, i - size):min(new_img.shape[1], i + size)] = 0
    return new_img


def get_crossed_pics(img: IMG, junctions: List[Junction], size: int = 24) -> List[IMG]:
    # 图片加大一圈白边
    img = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    junction_points_imgs = []
    for junction in junctions:
        # junction中的坐标 是原图的坐标，需要加上size
        i, j = junction.location
        i += size
        j += size
        img_ = img[max(0, j - size):min(img.shape[0], j + size), max(0, i - size):min(img.shape[1], i + size)]
        assert img_.shape == (2 * size, 2 * size, 3), img_.shape
        junction_points_imgs.append(img_)

    return junction_points_imgs


def get_crossed_pics_with_red_square(imgs: List[IMG], size: int = 8) -> List[IMG]:
    new_imgs = []
    for img in imgs:
        new_img = img.copy()
        new_img = cv2.rectangle(new_img, (size, size), (new_img.shape[1] - size, new_img.shape[0] - size), (0, 0, 255),
                                2)
        new_imgs.append(new_img)

    return new_imgs


NotMentioned = True


def judge_img_junction_style(junctions: List[Junction]) -> Set[Junction.JunctionType]:
    """返回的是不联通的，是不能忽略的"""
    circle, flat, bridge = False, False, False
    for junction in junctions:
        if junction.junction_type == Junction.JunctionType.circle:
            circle = True
        elif junction.junction_type == Junction.JunctionType.flat:
            flat = True
        elif junction.junction_type == Junction.JunctionType.bridge:
            bridge = True

    if bridge:
        return {Junction.JunctionType.bridge}

    if circle and flat:
        return {Junction.JunctionType.flat}

    return set()


PORT_OF_JUNCTION = Tuple[int, int]
POINT = Tuple[int, int]


def judge_out_degree(junction: Junction, skeleton_img: IMG, square_size: int = 8) -> Tuple[
    int, List[PORT_OF_JUNCTION], IMG]:
    # 骨架图补一圈宽度为square_size的0（以后可以优化）
    skeleton_img = cv2.copyMakeBorder(skeleton_img, square_size, square_size, square_size, square_size,
                                      cv2.BORDER_CONSTANT, value=0)
    i, j = junction.location
    i += square_size
    j += square_size

    out_degree = 0
    # 抠图
    junction_img = skeleton_img[j - square_size:j + square_size, i - square_size:i + square_size]
    # 删掉小一圈的图的内容
    junction_img_judge = junction_img.copy()
    junction_img_judge[1:-1, 1:-1] = 0
    # 找到所有的联通域的中心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_img_judge, connectivity=8)
    ports = []
    for i in range(1, num_labels):
        # 取出每个连通域的所有点
        points = np.argwhere(labels == i)
        out_degree += 1
        port = (points[:, 1].mean().astype(int), points[:, 0].mean().astype(int))
        port = (port[0] + junction.location[0] - square_size, port[1] + junction.location[1] - square_size)
        ports.append(port)

    # 把junction_img中所有junction_img_judge中的点都变成红色
    junction_img[junction_img_judge == 255] = 100

    return out_degree, ports, junction_img


def pair_points_clockwise(rectangle_top_left: POINT, rectangle_bottom_right: POINT, points: List[POINT]) -> List[
    Tuple[POINT, POINT]]:
    # 顺时针方向重新排序四个点的函数
    def sort_points_clockwise(points: List[POINT]) -> List[POINT]:
        # 计算矩形中心
        rect_center = ((rectangle_top_left[0] + rectangle_bottom_right[0]) / 2,
                       (rectangle_top_left[1] + rectangle_bottom_right[1]) / 2)

        # 按角度排序，确保按顺时针顺序
        sorted_points = sorted(points,
                               key=lambda point: (cv2.fastAtan2(point[1] - rect_center[1], point[0] - rect_center[0])))
        return sorted_points

    # 对四个点进行顺时针排序
    sorted_points = sort_points_clockwise(points)

    # 标记点 1, 2, 3, 4，并配对
    paired_points = [(sorted_points[0], sorted_points[2]), (sorted_points[1], sorted_points[3])]

    # 返回配对的点
    return paired_points


def get_pair_points_clockwise(rectangle_top_left: POINT, rectangle_bottom_right: POINT, ports: Dict[NAME, Port]) -> \
        List[
            Tuple[NAME, NAME]]:
    (A1, A2), (B1, B2) = pair_points_clockwise(rectangle_top_left, rectangle_bottom_right,
                                               [port.top_left for port in ports.values()])

    def get_name(point: POINT) -> NAME:
        for name, port in ports.items():
            if port.top_left == point:
                return name
        raise ValueError("Point not found")

    return [(get_name(A1), get_name(A2)), (get_name(B1), get_name(B2))]


def junction_logic(circuit: Circuit) -> Circuit:
    def get_connection(port_name: Tuple[NAME, NAME], connections: Set[Connection]) -> Connection:
        for connection in connections:
            if port_name in connection.ports:
                return connection
        return Connection(ports=frozenset())

    junctions = {
        name: device for name, device in circuit.devices.items() if device.device_type == Device.DeviceType.bridge
    }
    for name, junction in junctions.items():
        (A1, A2), (B1, B2) = get_pair_points_clockwise(
            junction.top_left, junction.bottom_right, junction.ports
        )
        for NAME1, NAME2 in [(A1, A2), (B1, B2)]:
            NET1 = get_connection((name, NAME1), circuit.connections)
            NET2 = get_connection((name, NAME2), circuit.connections)
            # 删除这两个网络，再加入他们的并集

            circuit.connections.discard(NET1)
            circuit.connections.discard(NET2)

            # 两个网络中除去含有NAME1和NAME2的端口
            NET1.ports -= {(name, NAME1)}
            NET2.ports -= {(name, NAME2)}

            circuit.connections.add(Connection(ports=NET1.ports | NET2.ports))

    # 删除所有的bridge
    circuit.devices = {name: device for name, device in circuit.devices.items() if
                       device.device_type != Device.DeviceType.bridge}

    return circuit


def get_device_direction_and_ports(img: IMG, device: Device, model):
    # 图像预处理函数
    def preprocess_image(img):
        img = img.resize((150, 150))  # 调整图像大小与训练时一致
        img_array = image.img_to_array(img)  # 将 PIL 图像转换为 Numpy 数组
        img_array = np.expand_dims(img_array, axis=0)  # 扩展维度以适应模型输入
        img_array /= 255.0  # 与训练时的归一化一致
        return img_array

    # 预测函数，使用TensorFlow模型进行分类
    def predict_image(cropped_img, _model):
        img_array = preprocess_image(cropped_img)  # 确保图像已转换为 Numpy 数组
        predictions = _model.predict(img_array, verbose=False)  # 使用TensorFlow Keras模型预测
        predicted_class = np.argmax(predictions, axis=1)[0]

        # 将0-3映射到'd', 'l', 'r', 'u'
        class_labels = {0: 'd', 1: 'l', 2: 'r', 3: 'u'}
        return class_labels.get(predicted_class, 'Unknown')

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if device.device_type in MosTypes:
        x1, y1 = device.top_left
        x2, y2 = device.bottom_right

        # 计算器件框的宽度和高度
        width = x2 - x1
        height = y2 - y1

        # 计算扩展10%的值
        width_expand = int(width * 0.1)
        height_expand = int(height * 0.1)

        # 获取图像的宽度和高度，确保不会越界
        img_width, img_height = img.size

        # 扩展后的坐标，确保不会越界
        x1_expanded = max(0, x1 - width_expand)  # 左边界不能小于0
        y1_expanded = max(0, y1 - height_expand)  # 上边界不能小于0
        x2_expanded = min(img_width, x2 + width_expand)  # 右边界不能超过图像宽度
        y2_expanded = min(img_height, y2 + height_expand)  # 下边界不能超过图像高度

        # 裁剪器件框内的图像，使用扩展后的坐标
        cropped_img = img.crop((x1_expanded, y1_expanded, x2_expanded, y2_expanded))

        # 使用TensorFlow模型进行预测
        predicted_direction = predict_image(cropped_img, model)

        # 将预测结果保存在device的direction属性中
        device.direction = predicted_direction
        # print(f"Device {device_name} ({device.device_type.name}) predicted direction: {predicted_direction}")
    elif device.device_type == Device.DeviceType.gnd or device.device_type == Device.DeviceType.port:
        pass

    else:
        raise ValueError(f"Device type {device.device_type} not supported")

    return device


def get_device_ports(device: Device):
    # 仅处理属于 mos_types 的设备
    if device.device_type in MosTypes:
        # 获取器件框的左上角和右下角坐标
        x1, y1 = device.top_left
        x2, y2 = device.bottom_right

        # 计算上下左右边的中点和中心点
        top_mid = ((x1 + x2) // 2, y1)  # 上边中点
        bottom_mid = ((x1 + x2) // 2, y2)  # 下边中点
        left_mid = (x1, (y1 + y2) // 2)  # 左边中点
        right_mid = (x2, (y1 + y2) // 2)  # 右边中点
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # 中心点

        # 根据方向为 l、u、r、d 进行不同处理
        if device.direction == 'l':
            # 添加端口 g, ds1, ds2
            device.ports['g'] = Port(top_left=device.top_left, bottom_right=bottom_mid)
            device.ports['ds1'] = Port(top_left=top_mid, bottom_right=right_mid)
            device.ports['ds2'] = Port(top_left=center, bottom_right=device.bottom_right)

        elif device.direction == 'u':
            device.ports['g'] = Port(top_left=device.top_left, bottom_right=right_mid)
            device.ports['ds1'] = Port(top_left=center, bottom_right=device.bottom_right)
            device.ports['ds2'] = Port(top_left=left_mid, bottom_right=bottom_mid)

        elif device.direction == 'r':
            device.ports['g'] = Port(top_left=top_mid, bottom_right=device.bottom_right)
            device.ports['ds1'] = Port(top_left=left_mid, bottom_right=bottom_mid)
            device.ports['ds2'] = Port(top_left=device.top_left, bottom_right=center)

        elif device.direction == 'd':
            device.ports['g'] = Port(top_left=left_mid, bottom_right=device.bottom_right)
            device.ports['ds1'] = Port(top_left=device.top_left, bottom_right=center)
            device.ports['ds2'] = Port(top_left=top_mid, bottom_right=right_mid)

    elif device.device_type == Device.DeviceType.gnd or device.device_type == Device.DeviceType.port:
        # 如果是GND或端口，端口就是本身的位置
        device.ports[device.device_type.name] = Port(
            top_left=device.top_left,
            bottom_right=device.bottom_right
        )

    else:
        raise ValueError(f"Device type {device.device_type} not supported")

    return device


# 创建一个函数来沿器件框边缘和内部附近寻找导线像素并保存导线部分
def search_wire(x: int, y: int, visited: np.ndarray, skeleton_img: IMG) -> Set[Tuple[int, int]]:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    stack = [(x, y)]
    wire = set()  # 保存单根导线的像素坐标，使用集合避免重复
    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True

        # 检查是否为导线部分
        if skeleton_img[y, x] >= 128:
            wire.add((x, y))  # 使用集合存储导线像素

            # 在当前像素周围进行局部搜索
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                # 检查新坐标是否在图像范围内，且没有被访问过
                if 0 <= nx < skeleton_img.shape[1] and 0 <= ny < skeleton_img.shape[0] and not visited[ny, nx]:
                    # 如果邻居像素也是导线部分（白色），将其加入栈中继续搜索
                    if skeleton_img[ny, nx] >= 128:
                        stack.append((nx, ny))
    return wire


def search_around_device(device: Device, skeleton_img: IMG, found_wires, visited: np.ndarray, margin=5) -> List[
    Set[Tuple[int, int]]]:
    """
        扩展搜索范围，沿器件框外部指定的区域（宽高各加 margin 像素）进行检索。
        :param visited: 访问标记数组
        :param device: 当前器件
        :param found_wires: 已经找到的导线集合
        :param skeleton_img: 骨架图
        :param margin: 扩展搜索范围的像素数
        :return: 找到的导线集合
        """
    wires_found = []
    x1, y1 = device.top_left
    x2, y2 = device.bottom_right

    # 搜索范围为器件框外部区域，框内不搜索
    x_min = max(0, x1 - margin)  # 左边界
    x_max = min(x2 + margin, skeleton_img.shape[1])  # 右边界
    y_min = max(0, y1 - margin)  # 上边界
    y_max = min(y2 + margin, skeleton_img.shape[0])  # 下边界

    # 搜索框的上方、下方、左侧、右侧的外扩区域，不包括框内部
    # 上方扩展范围
    for y in range(y_min, y1):  # 从扩展范围的顶部到框顶部
        for x in range(x_min, x_max):  # 全部 x 范围
            if skeleton_img[y, x] >= 128 and not visited[y, x]:
                wire = search_wire(x, y, visited, skeleton_img)
                if wire and wire not in found_wires:
                    wires_found.append(wire)

    # 下方扩展范围
    for y in range(y2, y_max):  # 从框底部到扩展范围的底部
        for x in range(x_min, x_max):  # 全部 x 范围
            if skeleton_img[y, x] >= 128 and not visited[y, x]:
                wire = search_wire(x, y, visited, skeleton_img)
                if wire and wire not in found_wires:
                    wires_found.append(wire)

    # 左侧扩展范围
    for x in range(x_min, x1):  # 从扩展范围的左边到框的左边
        for y in range(y_min, y_max):  # 全部 y 范围
            if skeleton_img[y, x] >= 128 and not visited[y, x]:
                wire = search_wire(x, y, visited, skeleton_img)
                if wire and wire not in found_wires:
                    wires_found.append(wire)

    # 右侧扩展范围
    for x in range(x2, x_max):  # 从框的右边到扩展范围的右边
        for y in range(y_min, y_max):  # 全部 y 范围
            if skeleton_img[y, x] >= 128 and not visited[y, x]:
                wire = search_wire(x, y, visited, skeleton_img)
                if wire and wire not in found_wires:
                    wires_found.append(wire)

    return wires_found


def find_wires(circuit: Circuit, skeleton_img: IMG) -> List[Set[Tuple[int, int]]]:
    found_wires = []
    # 定义一个标记数组，用于记录哪些像素已被访问过
    visited = np.zeros_like(skeleton_img, dtype=bool)

    for device_name, device in circuit.devices.items():
        found_wires_for_device = search_around_device(device, skeleton_img, found_wires, visited)
        for wire in found_wires_for_device:
            # 检查该导线是否已经被找到，避免重复
            if wire not in found_wires:
                found_wires.append(wire)  # 保存导线

    return found_wires


def draw_lines(img: IMG, lines: List[Set[Tuple[int, int]]]) -> IMG:
    # 创建一个新的有颜色的图像，和原来的图像一样大小（原来的是1通道，新的是3通道），初始化为黑色
    new_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for line in lines:
        # 每个线不同颜色
        color = np.random.randint(0, 255, 3)
        for x, y in line:
            new_img[y, x] = color

    return new_img


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """计算两个点的欧几里得距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_wire_length(wire: Set[Tuple[int, int]]) -> int:
    """
    :return: 导线的总像素数，即导线的长度
    """
    return len(wire)


def search_nearby_ports(wire: Set[Tuple[int, int]], device: Device) -> NAME:
    """
    搜索器件的端口，找到距离导线最近的端口框
    :param wire: 导线的像素点集合
    :param device: 当前器件
    :return: 与导线连接的端口名称，如果无效返回None
    """
    closest_port = None
    min_distance = float('inf')

    # 遍历器件的每一个端口
    for port_name, port in device.ports.items():
        # 计算每个端口框的中心
        port_center = ((port.top_left[0] + port.bottom_right[0]) // 2,
                       (port.top_left[1] + port.bottom_right[1]) // 2)

        # 遍历导线中的每个点，计算其与端口的距离
        for wx, wy in wire:
            dist = distance((wx, wy), port_center)

            # 如果是最近的距离，则认为找到最近的端口
            if dist <= min_distance:
                min_distance = dist
                closest_port = port_name

    return closest_port


def add_connection(circuit, dev1, port1, dev2, port2):
    """
    添加连接到电路的connection列表中
    :param circuit: 电路对象
    :param dev1: 器件1名称
    :param port1: 器件1的端口名称
    :param dev2: 器件2名称
    :param port2: 器件2的端口名称
    """
    # 使用 frozenset 确保连接的端口集可以被哈希，且不区分顺序
    connection = Connection(ports=frozenset({(dev1, port1), (dev2, port2)}))

    # 检查是否已经存在相同的连接，避免重复添加
    if connection not in circuit.connections:
        circuit.connections.add(connection)
        print(f"添加有效连接：{dev1}的端口{port1}和{dev2}的端口{port2}")
    # else:
    # print(f"连接已存在：{dev1}的端口{port1}和{dev2}的端口{port2}")


def get_connections(found_wires: List[Set[Tuple[int, int]]], devices: Dict[NAME, Device], skeleton_img: IMG) -> Set[
    Connection]:
    visited = np.zeros_like(skeleton_img, dtype=bool)
    connections_graph = []

    def mark_visited(wire, wx, wy, radius=3):
        """
        标记导线中半径为 radius 内的所有像素为已访问
        :param wire: 导线的像素点集合
        :param wx: 当前像素点的 x 坐标
        :param wy: 当前像素点的 y 坐标
        :param radius: 搜索半径
        """
        for x, y in wire:
            if abs(x - wx) <= radius and abs(y - wy) <= radius:
                visited[y, x] = True

    count = 0
    count2 = 0
    count3 = 0
    count4 = 0

    # 遍历每根导线，检查是否与器件端口相连
    for wire in found_wires:
        connected_ports = []

        # 遍历导线中的每个像素点，寻找与导线相连的器件框及端口
        for wx, wy in wire:
            # 检查当前像素点是否已经被访问过
            if visited[wy, wx]:
                continue  # 如果已经访问，跳过
            count += 1

            for device_name, device in devices.items():
                # 检查导线像素是否位于设备框外3个像素的范围内
                if (device.top_left[0] - 3 <= wx <= device.bottom_right[0] + 3 and
                        device.top_left[1] - 3 <= wy <= device.bottom_right[1] + 3):
                    count2 += 1
                    # 搜索最近的端口
                    port = search_nearby_ports({(wx, wy)}, device)

                    # 如果找到有效的端口连接
                    if port:
                        connected_ports.append((device_name, port))
                        count3 += 1

                    # 标记导线中该像素点为中心，半径为3的所有像素为已访问
                    mark_visited(wire, wx, wy, radius=3)

        # 如果导线连接了两个或以上器件端口
        if len(connected_ports) > 1:
            for i in range(len(connected_ports)):
                for j in range(i + 1, len(connected_ports)):
                    dev1, port1 = connected_ports[i]
                    dev2, port2 = connected_ports[j]

                    # 如果两个端口属于同一器件，则计算导线总长度
                    if dev1 == dev2:
                        device = devices[dev1]

                        # 计算导线的总长度
                        wire_length = calculate_wire_length(wire)

                        # 计算器件的周长
                        device_width = abs(device.top_left[0] - device.bottom_right[0])
                        device_height = abs(device.top_left[1] - device.bottom_right[1])
                        device_perimeter = 2 * (device_width + device_height)

                        # 如果导线长度小于等于器件周长的1/4，则视为无效连接
                        if wire_length <= device_perimeter / 4:
                            continue  # 忽略无效连接

                    # 如果是不同器件的端口，或同器件的端口有效，添加到连接中
                    connections_graph.append((f"{dev1};{port1}", f"{dev2};{port2}"))
                    count4 += 1

        # 如果导线只连接到一个端口，则忽略该导线
        # elif len(connected_ports) == 1:
        #     print(f"忽略导线：仅连接到 {connected_ports[0][0]} 的端口 {connected_ports[0][1]}")
    g = ig.Graph.TupleList(connections_graph, directed=False)
    all_components = g.components()

    connections = set()
    for subgraph in all_components.subgraphs():
        vertex_set = set(subgraph.vs["name"])
        vertex_set = {tuple(vertex.split(";")) for vertex in vertex_set}
        connections.add(Connection(ports=vertex_set))

    return connections


def draw_devices_and_ports(img: IMG, devices: Dict[NAME, Device]) -> IMG:
    new_img = img.copy()

    for device in devices.values():
        xmin, ymin = device.top_left
        xmax, ymax = device.bottom_right

        new_img[ymin:ymax, xmin:xmax] = 0
        # 边框画白色正方形
        new_img[ymin, xmin:xmax] = 255
        new_img[ymax - 1, xmin:xmax] = 255
        new_img[ymin:ymax, xmin] = 255
        new_img[ymin:ymax, xmax - 1] = 255

        for port in device.ports.values():
            xmin, ymin = port.top_left
            xmax, ymax = port.bottom_right

            if xmin == xmax:
                xmax += 1
            if ymin == ymax:
                ymax += 1

            new_img[ymin:ymax, xmin:xmax] = 0
            # 边框画灰色正方形
            new_img[ymin, xmin:xmax] = 128
            new_img[ymax - 1, xmin:xmax] = 128
            new_img[ymin:ymax, xmin] = 128
            new_img[ymin:ymax, xmax - 1] = 128

    return new_img


def shift_num_to_char(num) -> str:
    # 将数字转换为字母, 先小写后大写，超过52后则变成两位数
    # 1 -> a, 2 -> b, ..., 26 -> z, 27 -> A, 28 -> B, ..., 52 -> Z, 53 -> aa, 54 -> ab, ...
    if num <= 26:
        return chr(num + 96)
    elif num <= 52:
        return chr(num + 38)
    else:
        return shift_num_to_char(num // 52) + shift_num_to_char(num % 52)


def draw_connections(img: IMG, circuit: Circuit) -> IMG:
    new_img = img.copy()

    for i, connection in enumerate(circuit.connections):
        color = np.random.randint(0, 255, 3)
        color = color.tolist()
        name = shift_num_to_char(i + 1)
        ports = list(connection.ports)

        for port in ports:
            x1, y1 = circuit.devices[port[0]].ports[port[1]].top_left
            x2, y2 = circuit.devices[port[0]].ports[port[1]].bottom_right
            if x1 == x2:
                x2 += 1
            if y1 == y2:
                y2 += 1

            x = (x1 + x2) // 2
            y = (y1 + y2) // 2

            # 画长方形框
            new_img[y1, x1:x2] = color
            new_img[y2 - 1, x1:x2] = color
            new_img[y1:y2, x1] = color
            new_img[y1:y2, x2 - 1] = color

            # 画端口名
            new_img = cv2.putText(
                new_img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return new_img


def circuit_to_json(circuit: Circuit) -> str:
    target_json = dict()

    for name, device in circuit.devices.items():
        target_json[name] = {
            "component_type": device.device_type.name,
            "port_connection": {}
        }

    for i, connection in enumerate(circuit.connections):
        net_name = shift_num_to_char(i + 1)
        ports = list(connection.ports)

        for port in ports:
            device_name, port_name = port
            target_json[device_name]["port_connection"][port_name] = net_name

    target_json = [
        item for item in target_json.values()
    ]
    return json.dumps(target_json, indent=4)
