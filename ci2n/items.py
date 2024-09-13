from typing import Dict, Tuple, Set, List, FrozenSet

import numpy as np
from pydantic import BaseModel
from enum import Enum, auto

# 定义端口名称类型
NAME = str
PORT_NAME = Tuple[NAME, NAME]


# 定义端口类
class Port(BaseModel):
    top_left: Tuple[int, int]  # 端口方框的左上角坐标 (x1, y1)
    bottom_right: Tuple[int, int]  # 端口方框的右下角坐标 (x2, y2)


# 定义器件类
class Device(BaseModel):
    # 器件类型的枚举类
    class DeviceType(int, Enum):
        pmos = auto()
        nmos = auto()
        npn = auto()
        pnp = auto()
        port = auto()
        capacitor = auto()
        inductor = auto()
        gnd = auto()
        voltage = auto()
        voltage_lines = auto()
        current = auto()
        diode = auto()
        single_end_amp = auto()
        single_input_single_end_amp = auto()
        diff_amp = auto()
        pmos_cross = auto()
        pmos_bulk = auto()
        nmos_cross = auto()
        nmos_bulk = auto()
        npn_cross = auto()
        pnp_cross = auto()
        capacitor_3 = auto()
        inductor_3 = auto()
        switch = auto()
        switch_3 = auto()
        antenna = auto()
        resistor2 = auto()
        resistor1_3 = auto()
        resistor2_3 = auto()
        bridge = auto()

    device_type: DeviceType  # 器件类型
    ports: Dict[NAME, Port]  # 存储多个端口
    top_left: Tuple[int, int]  # 器件框的左上角坐标 (x1, y1)
    bottom_right: Tuple[int, int]  # 器件框的右下角坐标 (x2, y2)
    direction: str  # 方向，u, d, l, r之一
    mirror: int  # 镜像，0或1


class_name_to_device_type = {
    'pmos': Device.DeviceType.pmos,
    'nmos': Device.DeviceType.nmos,
    'npn': Device.DeviceType.npn,
    'pnp': Device.DeviceType.pnp,
    'port': Device.DeviceType.port,
    'capacitor': Device.DeviceType.capacitor,
    'inductor': Device.DeviceType.inductor,
    'gnd': Device.DeviceType.gnd,
    'voltage': Device.DeviceType.voltage,
    'voltage_lines': Device.DeviceType.voltage_lines,
    'current': Device.DeviceType.current,
    'diode': Device.DeviceType.diode,
    'single_end_amp': Device.DeviceType.single_end_amp,
    'single_input_single_end_amp': Device.DeviceType.single_input_single_end_amp,
    'diff_amp': Device.DeviceType.diff_amp,
    'pmos_cross': Device.DeviceType.pmos_cross,
    'pmos_bulk': Device.DeviceType.pmos_bulk,
    'nmos_cross': Device.DeviceType.nmos_cross,
    'nmos-cross': Device.DeviceType.nmos_cross,
    'nmos_bulk': Device.DeviceType.nmos_bulk,
    'npn_cross': Device.DeviceType.npn_cross,
    'pnp_cross': Device.DeviceType.pnp_cross,
    'capacitor_3': Device.DeviceType.capacitor_3,
    'inductor_3': Device.DeviceType.inductor_3,
    'switch': Device.DeviceType.switch,
    'switch_3': Device.DeviceType.switch_3,
    'antenna': Device.DeviceType.antenna,
    'resistor2': Device.DeviceType.resistor2,
    'resistor1_3': Device.DeviceType.resistor1_3,
    'resistor2_3': Device.DeviceType.resistor2_3,
    'bridge': Device.DeviceType.bridge,
}

MosTypes = set()
MosTypes.add(Device.DeviceType.pmos)
MosTypes.add(Device.DeviceType.nmos)
MosTypes.add(Device.DeviceType.pmos_cross)
MosTypes.add(Device.DeviceType.pmos_bulk)
MosTypes.add(Device.DeviceType.nmos_cross)
MosTypes.add(Device.DeviceType.nmos_bulk)


# 定义器件间的连接类
class Connection(BaseModel):
    ports: FrozenSet[PORT_NAME]  # 使用frozenset存储端口连接关系

    def __hash__(self):
        return hash(frozenset(self.ports))

    def __eq__(self, other):
        if isinstance(other, Connection):
            return frozenset(self.ports) == frozenset(other.ports)
        return False


# 定义电路类
class Circuit(BaseModel):
    devices: Dict[NAME, Device]  # 存储电路中的所有器件，key是器件名称
    connections: Set[Connection]  # 存储器件端口之间的连接关系


class Junction:
    class JunctionType(Enum):
        circle = auto()
        flat = auto()
        bridge = auto()
        others = auto()

    def __init__(self):
        self.location: Tuple[int, int] = (0, 0)
        self.junction_type: "Junction.JunctionType" = Junction.JunctionType.others
        self.feature_img: np.ndarray = np.zeros((1, 1, 3), dtype=np.uint8)
        self.ports: List[Port] = []
