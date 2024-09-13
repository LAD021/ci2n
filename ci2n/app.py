import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import torch
from loguru import logger
from pydantic import BaseModel
from torchvision import transforms
from ultralytics import YOLO
import tensorflow as tf

import ci2n.line_algo as la
from ci2n.items import Device, Junction, Port, Circuit, class_name_to_device_type


class FakeLogger:
    def trace(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class Settings(BaseModel):
    yolo_model_path: Path
    item_classifier_model_path: Path
    junction_classifier_model_path: Path
    verbose_path: Path


def circuit_image_to_netlist(image: np.ndarray, settings: Settings, verbose: bool = True) -> str:
    algorithm_step = 0
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 设置日志
    if not verbose:
        lg = FakeLogger()
    else:
        lg = logger

        # 自定义的日志格式化函数
        start_time = time.time()

        def custom_format(record):
            # 获取当前时间与程序启动时间的差值
            elapsed_time = time.time() - start_time
            record["elapsed"] = f"{elapsed_time:.3f}s"

            format_string = "{elapsed} | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
            if record["level"].name == "DEBUG":
                format_string = "------ | " + format_string
            elif record["level"].name == "TRACE":
                format_string = "--------------- | " + format_string
            return format_string

        # 修改日志格式
        lg.remove()
        lg.add(sys.stderr, format=custom_format, level="TRACE")

    # 开始算法
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    lg.info(f'Starting algorithm at {start_time_str}')
    lg_path = settings.verbose_path / time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(start_time))
    lg_path.mkdir(parents=True, exist_ok=True)

    # 显示图片信息
    lg.debug(f'Image shape: {image.shape}')
    # 加载模型
    yolo_model = YOLO(str(settings.yolo_model_path))
    lg.debug(f'(MODEL) Yolo model loaded')

    item_classifier_model = tf.keras.models.load_model(str(settings.item_classifier_model_path))
    lg.debug(f'(MODEL) Item classifier model loaded')
    junction_classifier_model = torch.load(settings.junction_classifier_model_path).to(torch_device)
    junction_classifier_model.eval()
    lg.debug(f'(MODEL) Node classifier model loaded')
    settings.verbose_path.mkdir(parents=True, exist_ok=True)
    lg.debug(f'Verbose path created at path: {settings.verbose_path}')

    circuit = Circuit(devices={}, connections=set())

    # 识别器件
    results = yolo_model(image, verbose=False)[0]
    lg.info(f'Yolo task finished')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_yolo_results.jpg"

        lg_path.mkdir(parents=True, exist_ok=True)
        results.save(filename=str(lg_path / filename))
        lg.debug(f'Yolo results saved at {lg_path / filename}')

    for i, box in enumerate(results.boxes):
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        class_name = yolo_model.names[int(box.cls[0])]

        device_type = class_name_to_device_type.get(class_name.lower(), None)
        if device_type is None:
            lg.warning(f'Unknown device type {class_name} detected at {xmin, ymin, xmax, ymax}')
            continue

        device_name = f'{device_type.name}:{i}'
        circuit.devices[device_name] = Device(
            device_type=device_type,
            ports={},
            top_left=(xmin, ymin),
            bottom_right=(xmax, ymax),
            direction="r",
            mirror=0,
        )
        lg.trace(f'Device {device_name} detected at {xmin, ymin, xmax, ymax}')

    # 识别器件的方向和端口
    for name, device in circuit.devices.items():
        try:
            device = la.get_device_direction_and_ports(image, device, item_classifier_model)
            device = la.get_device_ports(device)
        except Exception as e:
            lg.warning(f'Error when processing device {name}: {e}')
        lg.trace(f'Device {name} direction: {device.direction}, ports: {device.ports}')
        circuit.devices[name] = device

    # 提取骨架图
    skeleton_img = la.get_skeleton_with_items(image)
    lg.info(f'Skeleton image generated')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_skeleton_img.jpg"
        cv2.imwrite(str(lg_path / filename), skeleton_img)
        lg.debug(f'Skeleton image saved at {lg_path / filename}')

    # 器件变方框
    skeleton_img = la.items_to_rectangles(skeleton_img, circuit.devices)
    lg.info(f'Skeleton image without items generated')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_skeleton_img_items_to_rectangles.jpg"
        cv2.imwrite(str(lg_path / filename), skeleton_img)
        lg.debug(f'Skeleton image items to rectangles saved at {lg_path / filename}')

    # 扣除小的联通域
    skeleton_img = la.remove_small_connected_components(skeleton_img)
    lg.info(f'Small connected components removed')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_skeleton_img_without_small_connected_components.jpg"
        cv2.imwrite(str(lg_path / filename), skeleton_img)
        lg.debug(f'Skeleton image without small connected components saved at {lg_path / filename}')

    # 扣除器件
    skeleton_img_without_items = la.remove_items(skeleton_img, circuit.devices)
    lg.info(f'Skeleton image without items generated')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_skeleton_img_without_items.jpg"
        cv2.imwrite(str(lg_path / filename), skeleton_img_without_items)
        lg.debug(f'Skeleton image without items saved at {lg_path / filename}')
    # 识别交点，提取交点坐标，以及交点与外界的交接的联通域数量
    junction_locations = la.get_junction_locations(skeleton_img_without_items)
    lg.info(f'Junctions detected')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_junctions.jpg"
        junction_img = la.draw_junctions(image, junction_locations)
        cv2.imwrite(str(lg_path / filename), junction_img)
        lg.debug(f'Junctions saved at {lg_path / filename}')

    # 制作交点特征图
    junctions = []
    for junction_location in junction_locations:
        junction = Junction()
        junction.location = junction_location
        junctions.append(junction)

    junction_feature_imgs = la.get_crossed_pics(image, junctions)
    junction_feature_imgs = la.get_crossed_pics_with_red_square(junction_feature_imgs)
    for i, junction in enumerate(junctions):
        junction.feature_img = junction_feature_imgs[i]

    lg.info(f'Junction feature images generated, {len(junction_feature_imgs)} junctions')

    algorithm_step += 1
    if verbose:
        dirname = f"{algorithm_step:02d}_junction_feature"
        Path(lg_path / dirname).mkdir(parents=True, exist_ok=True)
        lg.debug(f'Junction classified images saving at {lg_path / dirname}')

    # 识别交点的类型
    mean = torch.tensor([0.8449, 0.8448, 0.8450])
    std = torch.tensor([0.2928, 0.2932, 0.2918])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    for i, junction in enumerate(junctions):
        with torch.no_grad():
            img = transform(junction.feature_img).unsqueeze(0).to(torch_device)
            output = junction_classifier_model(img)
        _, predicted = torch.max(output.data, 1)
        junction.junction_type = Junction.JunctionType(predicted.item() + 1)
        lg.trace(f'Junction {junction.location} classified as {junction.junction_type.name}')

        if verbose:
            dirname = f"{algorithm_step:02d}_junction_feature"
            cv2.imwrite(str(lg_path / dirname / f"{junction.junction_type.name}_{junction.location}.jpg"),
                        junction.feature_img)

    lg.info(f'Junctions classified')
    # 图像风格判断
    should_not_be_neglected = la.judge_img_junction_style(junctions)
    # 剔除全连接的交点
    should_not_be_neglected_junctions = []
    for i, junction in enumerate(junctions):
        if junction.junction_type in should_not_be_neglected:
            should_not_be_neglected_junctions.append(junction)

    lg.info(f'Style judgement finished, {should_not_be_neglected} junctions should not be neglected')
    lg.info(f'Junctions left: {len(should_not_be_neglected_junctions)}')
    for i, junction in enumerate(should_not_be_neglected_junctions):
        lg.trace(f'Junction {junction.junction_type.name} at {junction.location} should not be neglected')

    # 剔除3通的交点
    algorithm_step += 1
    if verbose:
        dirname = f"{algorithm_step:02d}_junction_out_degree_feature"
        Path(lg_path / dirname).mkdir(parents=True, exist_ok=True)
        lg.debug(f'Junction degree judged images saving at {lg_path / dirname}')

    without_three_out_degree_junctions = []
    for junction in should_not_be_neglected_junctions:
        out_degree, ports, img = la.judge_out_degree(junction, skeleton_img_without_items)
        lg.trace(f'Junction {junction.location} out degree: {out_degree}, ports: {ports}')
        junction.ports = [
            Port(top_left=port, bottom_right=port) for port in ports
        ]
        without_three_out_degree_junctions.append(junction)
        if verbose:
            dirname = f"{algorithm_step:02d}_junction_out_degree_feature"
            cv2.imwrite(str(lg_path / dirname / f"{out_degree}_{junction.location}.jpg"), img)

    junctions_as_devices = [
        Device(
            device_type=Device.DeviceType.bridge,
            ports={f"{port.top_left[0]}_{port.top_left[1]}": port for port in junction.ports},
            top_left=[junction.location[0] - 8, junction.location[1] - 8],
            bottom_right=[junction.location[0] + 8, junction.location[1] + 8],
            direction="r",
            mirror=0,
        ) for junction in without_three_out_degree_junctions
    ]
    circuit.devices.update({f"bridge:{i}": device for i, device in enumerate(junctions_as_devices)})

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_all_devices_and_junctions_and_ports.jpg"
        cv2.imwrite(str(lg_path / filename), la.draw_devices_and_ports(image, circuit.devices))
        lg.debug(f'All devices and junctions saved at {lg_path / filename}')

    # 扣掉交点
    skeleton_img_without_items_and_junctions = la.remove_items(skeleton_img_without_items, circuit.devices)
    lg.info(f'Skeleton image without items and junctions generated')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_skeleton_img_without_items_and_junctions.jpg"
        cv2.imwrite(str(lg_path / filename), skeleton_img_without_items_and_junctions)
        lg.debug(f'Skeleton image without items and junctions saved at {lg_path / filename}')

    # 识别线
    wires = la.find_wires(circuit, skeleton_img_without_items_and_junctions)
    lg.info(f'Wires detected, {len(wires)} lines')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_lines.jpg"
        lines_img = la.draw_lines(skeleton_img, wires)
        cv2.imwrite(str(lg_path / filename), lines_img)
        lg.debug(f'Lines saved at {lg_path / filename}')

    # 获得连接关系
    connections = la.get_connections(wires, circuit.devices, skeleton_img_without_items_and_junctions)
    circuit.connections = connections
    lg.info(f'Connections detected, {len(connections)} connections')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_connections.jpg"
        connections_img = la.draw_connections(image, circuit)
        cv2.imwrite(str(lg_path / filename), connections_img)
        lg.debug(f'Connections saved at {lg_path / filename}')

    # 执行交点逻辑
    circuit = la.junction_logic(circuit)
    lg.info(f'Junction logic executed')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_connections_with_junction_logic.jpg"
        connections_img = la.draw_connections(image, circuit)
        cv2.imwrite(str(lg_path / filename), connections_img)
        lg.debug(f'Connections saved at {lg_path / filename}')
    # 整理输出
    target_json = la.circuit_to_json(circuit)
    lg.info(f'Output generated')

    algorithm_step += 1
    if verbose:
        filename = f"{algorithm_step:02d}_output.json"
        with open(lg_path / filename, 'w') as f:
            f.write(target_json)
        lg.debug(f'Output saved at {lg_path / filename}')

    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    lg.info(f'Algorithm finished at {end_time_str}, elapsed time: {end_time - start_time:.3f}s')
    return target_json


def circuit_image_to_netlist_from_file(file_path: os.PathLike, settings: Settings, verbose: bool = True):
    image = cv2.imread(str(file_path))
    if verbose:
        logger.info(f'Processing image from file {file_path}')
    return circuit_image_to_netlist(image, settings, verbose)


def main():
    # 读取图片
    target_json = circuit_image_to_netlist_from_file(
        Path('test_circuit_img.jpg'),
        Settings(
            yolo_model_path=Path('yolo_model.pt'),
            item_classifier_model_path=Path('item_classifier.h5'),
            junction_classifier_model_path=Path('junction_classifier.pt'),
            verbose_path=Path('verbose')
        )
    )


if __name__ == '__main__':
    main()
