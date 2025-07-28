'''
gdsfactory version: 8.8.9
author: wuxiao

'''

import math
from math import pi

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.generic_tech import LAYER


@gf.cell
def add_2x2MMI_1(
    core_length: float = 87,
    core_width: float = 10.2,
    separation: float = 3.24,
    taper_length: float = 30,
    taper_width: float = 2.6,
    end_width: float = 1.17,
    layer: tuple = LAYER.WG,
):
    """创建一个2x2多模干涉器(MMI)器件

    该函数生成一个标准的2x2 MMI分光器/合光器结构，包含矩形多模区域和四个渐变器(taper)。
    MMI器件可用于光信号的分光、合光或功分功能。

    Args:
        core_length (float, optional): MMI核心区域的长度，单位为微米。默认值为87μm。
        core_width (float, optional): MMI核心区域的宽度，单位为微米。默认值为10.2μm。
        separation (float, optional): 输入/输出端口之间的中心间距，单位为微米。默认值为3.24μm。
        taper_length (float, optional): 渐变器(taper)的长度，单位为微米。默认值为30μm。
        taper_width (float, optional): 连接MMI核心区的taper宽端宽度，单位为微米。默认值为2.6μm。
        end_width (float, optional): 端口处的波导宽度(taper窄端)，单位为微米。默认值为1.17μm。
        layer (tuple, optional): 器件所在的工艺层，格式为(layer, datatype)。默认为LAYER.WG。

    Returns:
        gf.Component: 包含2x2 MMI结构的GDSFactory组件对象，具有四个光学端口：
                     - 'o1': 左上端口
                     - 'o2': 右上端口
                     - 'o3': 右下端口
                     - 'o4': 左下端口

    Structure:
        o1 ──taper──┐           ┌──taper── o2
                   ║           ║
                   ║           ║
                 ┌─╫───────────╫─┐
                 │ ║    MMI    ║ │
                 │ ║   核心区   ║ │
                 └─╫───────────╫─┘
                   ║           ║
                   ║           ║
        o4 ──taper──┘           └──taper── o3

    Note:
        - 默认生成负胶版图(LAYER.WG层)
        - 如需正胶版图，需要使用Region取轮廓
        - MMI核心区为矩形结构，通过渐变器连接单模波导
        - 端口orientation设置：左侧端口为180°，右侧端口为0°
        - 适用于1550nm波长的硅光子平台
    """
    c = gf.Component()

    c.add_polygon(
        [
            [-core_length / 2, core_width / 2],
            [core_length / 2, core_width / 2],
            [core_length / 2, -core_width / 2],
            [-core_length / 2, -core_width / 2],
        ],
        layer=layer,
    )
    c.add_port(
        'core_1',
        center=(-core_length / 2, separation / 2),
        width=taper_width,
        orientation=180,
        layer=layer,
    )
    c.add_port(
        'core_2',
        center=(core_length / 2, separation / 2),
        width=taper_width,
        orientation=0,
        layer=layer,
    )
    c.add_port(
        'core_3',
        center=(core_length / 2, -separation / 2),
        width=taper_width,
        orientation=0,
        layer=layer,
    )
    c.add_port(
        'core_4',
        center=(-core_length / 2, -separation / 2),
        width=taper_width,
        orientation=180,
        layer=layer,
    )
    taper1 = gf.components.taper2(
        length=taper_length, width1=end_width, width2=taper_width, layer=layer
    )
    t1 = c.add_ref(taper1)
    t2 = c.add_ref(taper1)
    t3 = c.add_ref(taper1)
    t4 = c.add_ref(taper1)
    t1.connect('o2', c.ports['core_1'])
    t2.connect('o2', c.ports['core_2'])
    t3.connect('o2', c.ports['core_3'])
    t4.connect('o2', c.ports['core_4'])
    c.add_port(name='o1', port=t1.ports['o1'])
    c.add_port(name='o2', port=t2.ports['o1'])
    c.add_port(name='o3', port=t3.ports['o1'])
    c.add_port(name='o4', port=t4.ports['o1'])

    return c


@gf.cell
def add_1x2MMI_1(
    core_length: float = 23.6,
    core_width: float = 5.7,
    separation: float = 3,
    taper_length: float = 15,
    taper_width: float = 1.68,
    end_width: float = 1,
    layer: tuple = LAYER.WG,
):
    """
    创建一个1x2多模干涉器(MMI)器件

    该函数生成一个标准的1x2 MMI分光器结构，包含矩形多模区域和三个渐变器(taper)。
    器件左侧有一个输入端口，右侧有两个输出端口，用于将单路光信号分成两路。

    Args:
        core_length (float, optional): MMI核心区域的长度，单位为微米。默认值为23.6μm。
        core_width (float, optional): MMI核心区域的宽度，单位为微米。默认值为5.7μm。
        separation (float, optional): 输出端口之间的中心间距，单位为微米。默认值为3μm。
        taper_length (float, optional): 渐变器(taper)的长度，单位为微米。默认值为15μm。
        taper_width (float, optional): 连接MMI核心区的taper宽端宽度，单位为微米。默认值为1.68μm。
        end_width (float, optional): 端口处的波导宽度(taper窄端)，单位为微米。默认值为1μm。
        layer (tuple, optional): 器件所在的工艺层，格式为(layer, datatype)。默认为LAYER.WG。

    Returns:
        gf.Component: 包含1x2 MMI结构的GDSFactory组件对象，具有三个光学端口：
                     - 'o1': 左侧输入端口
                     - 'o2': 右上输出端口
                     - 'o3': 右下输出端口

    Structure:
                        o2
                        |
                        |
    o1 --------MMI----------
                        |
                        |
                        o3

    Note:
        - 默认生成负胶版图(LAYER.WG层)
        - MMI核心区为矩形结构，通过渐变器连接单模波导
        - 输入端口orientation为180°，输出端口orientation为0°
        - 输入端口位于核心区中心高度，输出端口按separation间距分布
        - 适用于1550nm波长的硅光子平台的功分器应用

    """
    c = gf.Component()
    c.add_polygon(
        [
            [-core_length / 2, core_width / 2],
            [core_length / 2, core_width / 2],
            [core_length / 2, -core_width / 2],
            [-core_length / 2, -core_width / 2],
        ],
        layer=layer,
    )
    c.add_port(
        'core_1',
        center=(-core_length / 2, 0),
        width=taper_width,
        orientation=180,
        layer=layer,
    )
    c.add_port(
        'core_2',
        center=(core_length / 2, separation / 2),
        width=taper_width,
        orientation=0,
        layer=layer,
    )
    c.add_port(
        'core_3',
        center=(core_length / 2, -separation / 2),
        width=taper_width,
        orientation=0,
        layer=layer,
    )
    taper1 = gf.components.taper2(
        length=taper_length, width1=end_width, width2=taper_width, layer=layer
    )
    t1 = c.add_ref(taper1)
    t2 = c.add_ref(taper1)
    t3 = c.add_ref(taper1)
    t1.connect('o2', c.ports['core_1'])
    t2.connect('o2', c.ports['core_2'])
    t3.connect('o2', c.ports['core_3'])
    c.add_port(name='o1', port=t1.ports['o1'])
    c.add_port(name='o2', port=t2.ports['o1'])
    c.add_port(name='o3', port=t3.ports['o1'])
    return c


@gf.cell
def add_wg_1(
    wg_length: float = 100,
    wg_width: float = 0.9,
    layer: tuple = LAYER.WG,
    et_width: float = 18,
    et_layer: tuple = (46, 0),
):
    '''
    c.add_wg_1
    为c添加一段波导
    默认层为LAYER.WG
    默认为负胶
    :param wg_length:
    :param wg_width:
    :return:
    '''
    s = gf.Component()

    path_0 = gf.Path()
    path_0 += gf.path.straight(length=wg_length)
    path_0_1 = gf.path.extrude(path_0, layer=layer, width=wg_width)
    s.add_ref(path_0_1)  # type: ignore
    s.add_port(name='o1', center=(0, 0), width=wg_width, orientation=180, layer=layer)
    s.add_port(
        name='o2', center=(wg_length, 0), width=wg_width, orientation=0, layer=layer
    )
    s.add_port(
        name='et_middle',
        center=(wg_length / 2, 0),
        width=et_width,
        orientation=0,
        layer=et_layer,
    )
    return s


@gf.cell
def add_gc_1(
    boxh: float = 15,
    gratingp: float = 0.94,
    duty: float = 0.725,
    gcl: int = 30,
    taper_length: float = 300,
    wg_width: float = 0.9,
    layer: tuple = LAYER.WG,
):
    '''
    nclad=1
    fiberradius=5.2
    fiberangle=8
    modeswitch=1
    toph=0.01
    gratingh=0.26       #刻蚀深度
    boxh=50           #齿长度
    gratingp=0.94       #周期
    duty=0.725          #刻下去的区域占空比
    gcl=28              #齿数
    taper_length=300     #taper长度
    wg_width=0.9

    创建一个TE偏振光栅耦合器(Grating Coupler)

    该函数生成一个用于光纤-芯片耦合的光栅耦合器结构，包含一个线性渐变器和周期性光栅齿结构。
    适用于TE偏振光的垂直耦合，是硅光子芯片与光纤连接的标准器件。

    Args:
        boxh (float, optional): 光栅齿的长度(高度)，单位为微米。默认值为15μm。
        gratingp (float, optional): 光栅周期，单位为微米。默认值为0.94μm。
        duty (float, optional): 光栅占空比，即刻蚀区域占周期的比例。默认值为0.725。
        gcl (int, optional): 光栅齿的数量。默认值为30个齿。
        taper_length (float, optional): 线性渐变器的长度，单位为微米。默认值为300μm。
        wg_width (float, optional): 连接波导的宽度，单位为微米。默认值为0.9μm。
        layer (tuple, optional): 器件所在的工艺层，格式为(layer, datatype)。默认为LAYER.WG。

    Returns:
        gf.Component: 包含光栅耦合器结构的GDSFactory组件对象，具有一个光学端口：
                     - 'o1': 波导连接端口(左侧)

    Structure:
        光纤耦合区域        渐变器              波导连接
        ║║║║║║║║║║ ◄────────────── o1
        ┴┴┴┴┴┴┴┴┴┴
        光栅齿结构    线性taper     单模波导

    Parameters Detail:
        - boxh: 光栅齿长度，影响耦合效率和角度容忍度
        - gratingp: 光栅周期，决定布拉格波长和耦合角度
        - duty: 占空比，影响耦合强度和带宽
        - gcl: 齿数，影响耦合效率和反射
        - taper_length: 渐变器长度，影响模式转换效率
        - wg_width: 输出波导宽度，需匹配后续器件

    Note:
        - 默认生成负胶版图(LAYER.WG层)
        - 优化参数适用于1550nm波长TE偏振光
        - 光栅周期0.94μm对应约8°的光纤角度
        - 占空比0.725提供良好的耦合效率
        - 端口orientation为180°，适合从左侧连接波导

    '''

    gc = gf.Component()
    gc.add_polygon(
        [
            [0, 0],
            [0 + taper_length, 0 - (boxh - wg_width) / 2],
            [0 + taper_length, 0 - (boxh - wg_width) / 2 + boxh],
            [0, 0 + wg_width],
        ],
        layer=layer,
    )
    gc.add_polygon(
        [
            [0 + taper_length, 0 - (boxh - wg_width) / 2],
            [0 + taper_length, 0 - (boxh - wg_width) / 2],
            [0 + taper_length, 0 - (boxh - wg_width) / 2 + boxh],
            [0 + taper_length, 0 - (boxh - wg_width) / 2 + boxh],
        ],
        layer=layer,
    )
    g_1 = 0 + taper_length + duty * gratingp  # 第一个齿的左下角x值
    for num in range(gcl):
        g_pos = g_1 + num * gratingp
        gc.add_polygon(
            [
                [g_pos, 0 - (boxh - wg_width) / 2],
                [g_pos + gratingp * (1 - duty), 0 - (boxh - wg_width) / 2],
                [g_pos + gratingp * (1 - duty), 0 - (boxh - wg_width) / 2 + boxh],
                [g_pos, 0 - (boxh - wg_width) / 2 + boxh],
            ],
            layer=layer,
        )
    gc1 = gf.Component()
    gc_1 = gc1.add_ref(gc)

    gc1.add_port(
        name='o1',
        center=(0, wg_width / 2),
        width=wg_width,
        orientation=180,
        layer=layer,
    )

    return gc1


@gf.cell
def add_gc_2(
    period: float = 1,
    n: int = 20,
    duty_ratio: float = 0.5,
    angle: float = 10,
    length_taper: float = 40,
    length_end: float = 50,
    wg_width: float = 0.9,
    layer: tuple = LAYER.WG,
) -> gf.Component:
    '''
    创建一个偏振不敏感弧形光栅耦合器

    该函数生成一个弧形光栅耦合器结构，具有多个周期性光栅齿排列在弧形路径上，
    适用于偏振不敏感的光纤-芯片耦合应用。

    Args:
        period (float, optional): 光栅周期，单位为微米。默认值为1μm。
        n (int, optional): 光栅齿的数量。默认值为20个齿。
        duty_ratio (float, optional): 光栅占空比，刻蚀区域占周期的比例。默认值为0.5。
        angle (float, optional): 弧形光栅的张角，单位为度。默认值为10°。
        length_taper (float, optional): 弧形区域的半径长度，单位为微米。默认值为40μm。
        length_end (float, optional): 连接波导的长度，单位为微米。默认值为50μm。
        wg_width (float, optional): 输出波导的宽度，单位为微米。默认值为0.9μm。
        layer (tuple, optional): 器件所在的工艺层，格式为(layer, datatype)。默认为LAYER.WG。

    Returns:
        gf.Component: 包含弧形光栅耦合器结构的GDSFactory组件对象，具有一个光学端口：
                     - 'o1': 波导连接端口

    Parameters Detail:
        - period: 光栅周期，影响工作波长和耦合效率
        - n: 光栅齿数，影响带宽和反射特性
        - duty_ratio: 占空比，影响耦合强度
        - angle: 弧形张角，影响模场匹配和偏振特性
        - length_taper: 弧形半径，决定器件尺寸
        - length_end: 连接段长度，便于与其他器件连接
        - wg_width: 输出波导宽度，需匹配后续器件

    Note:
        - 弧形设计提供更好的偏振不敏感特性
        - 通过多个Section构建复杂的光栅齿结构
        - 使用弧形路径(gf.path.arc)创建弯曲光栅
        - 端口orientation为0°，适合从右侧连接波导
        - 适用于需要偏振不敏感耦合的应用场景

    '''
    c = gf.Component()
    secs_teeth = []
    for i in range(n):
        secs_teeth.append(
            gf.Section(
                width=period * duty_ratio,
                offset=(i - duty_ratio / 2 + 1) * period,
                layer=layer,
            )
        )
    secs_teeth = tuple(secs_teeth)
    cs_teeth = gf.cross_section.cross_section(
        width=length_taper, offset=-length_taper / 2, layer=layer, sections=secs_teeth
    )
    p_teeth = gf.path.arc(radius=length_taper, angle=angle, start_angle=180 - angle / 2)
    ref_teeth = c << p_teeth.extrude(cross_section=cs_teeth)  # type: ignore
    end = gf.components.straight(
        length=length_end + length_taper / 2, layer=layer, width=wg_width
    )
    ref_end = c << end
    ref_teeth.xmax = int(ref_end.xmin + length_taper / 2)
    ref_teeth.y = ref_end.y
    port = ref_end['o2']
    port.name = 'o1'
    c.add_port(name='o1', port=port)

    return c


@gf.cell
def add_ring_1(
    wg_width: float = 0.9,
    gap: float = 1.9,
    L: float = 453,
    Lc: float = 8,
    straight_wg_length: float = 80,
    layer: tuple = LAYER.WG,
    layer_heater: tuple = LAYER.HEATER,
    h_width: float = 8,
):
    '''
    创建一个带上下直波导的微环谐振器

    该函数生成一个标准的微环谐振器结构，包含一个环形谐振腔和上下两条直波导。
    微环由两段直波导和两个180°圆弧组成，支持热光调谐功能。

    Args:
        wg_width (float, optional): 波导宽度(包括直波导和环形波导)，单位为微米。默认值为0.9μm。
        gap (float, optional): 直波导与微环的耦合间距(中心到中心距离)，单位为微米。默认值为1.9μm。
        L (float, optional): 微环的总周长，单位为微米。默认值为453μm。
        Lc (float, optional): 微环耦合区的直波导长度，单位为微米。默认值为8μm。
        straight_wg_length (float, optional): 上下直波导的长度，单位为微米。默认值为80μm。
        layer (tuple, optional): 波导所在的工艺层，格式为(layer, datatype)。默认为LAYER.WG。
        layer_heater (tuple, optional): 加热电极所在的工艺层，格式为(layer, datatype)。默认为LAYER.HEATER。
        h_width (float, optional): 加热电极的宽度，单位为微米。默认值为8μm。

    Returns:
        gf.Component: 包含微环谐振器结构的GDSFactory组件对象，具有六个端口：
                     - 'o1': 左上波导端口
                     - 'o2': 右上波导端口
                     - 'o3': 右下波导端口
                     - 'o4': 左下波导端口
                     - 'h1': 左侧加热电极端口
                     - 'h2': 右侧加热电极端口

    Structure:
        o1 ────────────────────────────── o2
                   ┌─────────────┐
                  ╱               ╲
                 ╱                 ╲
                ╱       微环        ╲
               ╱        Ring        ╲
              ╱         核心         ╲
              ╲                     ╱
               ╲                   ╱
                ╲                 ╱
                 ╲_______________╱
                    h1       h2
        o4 ────────────────────────────── o3

    Parameters Detail:
        - wg_width: 影响传输损耗和模式特性
        - gap: 控制耦合强度，决定器件的品质因子和自由频谱范围
        - L: 决定谐振波长和自由频谱范围
        - Lc: 影响耦合区域的耦合系数
        - straight_wg_length: 便于与其他器件连接
        - h_width: 加热电极宽度，影响热调谐效率

    Note:
        - 微环半径自动计算：R = (L - 2*Lc) / (2*π)
        - 微环由两段直波导(长度Lc)和两个180°圆弧组成
        - 加热电极位于微环左右两侧的弯曲中心
        - 端口orientation: 左侧为180°，右侧为0°
        - 适用于滤波器、开关、调制器等应用


    wg_width=0.9                            #波导宽度，包括了直波导和弯曲波导
    gap=0.9                                 #直波导和微环的耦合间距,指中心波导间距
    L=453                                   #微环的总长度
    Lc=8                                    #耦合区的微环直波导长度
    straight_wg_length=80                   #上下两直波导的长度
    h_width                                 #加热电极宽度
    '''
    R = (L - Lc * 2) / 2 / pi  # 微环除了耦合区为直波导，其他为圆，R为其半径

    s = gf.Component()
    path_0 = gf.Path()
    path_0 += gf.path.straight(length=straight_wg_length)
    path_1 = gf.Path()
    path_1 += gf.path.straight(length=Lc)
    path_1 += gf.path.arc(radius=R, angle=-180)
    path_1 += gf.path.straight(length=Lc)
    path_1 += gf.path.arc(radius=R, angle=-180)
    path_2 = path_0.copy()
    path_0.dmove([-straight_wg_length / 2, R + gap])
    path_1.dmove([-Lc / 2, R])
    path_2.dmove([-straight_wg_length / 2, -(R + gap)])
    path_0_1 = gf.path.extrude(path_0, layer=layer, width=wg_width)
    path_1_1 = gf.path.extrude(path_1, layer=layer, width=wg_width)
    path_2_1 = gf.path.extrude(path_2, layer=layer, width=wg_width)
    s.add_ref(path_0_1)  # type: ignore
    s.add_ref(path_1_1)  # type: ignore
    s.add_ref(path_2_1)  # type: ignore
    s.add_port(
        name='o1',
        center=(-straight_wg_length / 2, R + gap),
        width=wg_width,
        orientation=180,
        layer=layer,
    )
    s.add_port(
        name='o2',
        center=(straight_wg_length / 2, R + gap),
        width=wg_width,
        orientation=0,
        layer=layer,
    )
    s.add_port(
        name='o3',
        center=(straight_wg_length / 2, -(R + gap)),
        width=wg_width,
        orientation=0,
        layer=layer,
    )
    s.add_port(
        name='o4',
        center=(-straight_wg_length / 2, -(R + gap)),
        width=wg_width,
        orientation=180,
        layer=layer,
    )
    s.add_port(
        name='h1',
        center=(-(R + Lc / 2), 0),
        width=h_width,
        orientation=90,
        layer=layer_heater,
    )  # h1在微环左侧弯曲中心，用来连接Ti弯曲部分
    s.add_port(
        name='h2',
        center=((R + Lc / 2), 0),
        width=h_width,
        orientation=90,
        layer=layer_heater,
    )  # h2在微环右侧弯曲中心
    return s


@gf.cell
def add_loop_mirror_1(
    layer_wg: tuple = LAYER.WG, wg_width: float = 1, multi_wg_width: float = 4
):
    '''
    添加一个loop_mirror
    o1 ══════════════════════════════════════
        ║                                ║
    ┌─╫─┐    ╭─────────────────╮    ┌─╫─┐
    │MMI│    ╱                 ╲    │MMI│
    │ 1 │   ╱    Loop Mirror    ╲   │ 2 │
    └─╫─┘   ╲     环形镜       ╱   └─╫─┘
        ║      ╲_________________╱      ║
    o2 ══════════════════════════════════════
    :param layer_wg:
    :param wg_width:
    :param multi_wg_width:
    :return:
    '''
    layer_wg = LAYER.WG
    layer_ti = LAYER.HEATER
    layer_au = LAYER.SOURCE

    wg_width = 1
    multi_wg_width = 4
    h_width = 8  # ti的宽度
    au_width = 10

    loop_mirror = gf.Component()
    mmi1 = loop_mirror << add_2x2MMI_1(
        separation=3.24,
        layer=layer_wg,
        core_length=90,
        taper_width=2.6,
        taper_length=30,
        end_width=1.17,
        core_width=10.2,
    )
    mmi2 = loop_mirror << add_2x2MMI_1(
        separation=3.24,
        layer=layer_wg,
        core_length=90,
        taper_width=2.6,
        taper_length=30,
        end_width=1.17,
        core_width=10.2,
    )
    taper1 = gf.components.taper2(
        length=30, width1=1.17, width2=wg_width, layer=layer_wg
    )
    t1 = loop_mirror.add_ref(taper1)
    t2 = loop_mirror.add_ref(taper1)
    t1.connect('o1', mmi1.ports['o2'])
    t2.connect('o1', mmi2.ports['o3'])
    wg1 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg2 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg1.connect('o1', t1.ports['o2'])
    wg2.connect('o1', t2.ports['o2'])
    L_basaer = 180  # 贝塞尔曲线的长
    W_basaer = 60  # 贝塞尔曲线的宽
    section1 = gf.Section(width=wg_width, layer=layer_wg, port_names=('o1', 'o2'))
    x = gf.CrossSection(sections=tuple([section1]))
    b1 = gf.components.bezier(
        control_points=[
            (0, 0),
            (L_basaer / 2, 0),
            (L_basaer / 2, W_basaer),
            (L_basaer, W_basaer),
        ],
        npoints=1000,
        cross_section=x,
    )
    b2 = gf.components.bezier(
        control_points=[
            (0, 0),
            (L_basaer / 2, 0),
            (L_basaer / 2, -W_basaer),
            (L_basaer, -W_basaer),
        ],
        npoints=1000,
        cross_section=x,
    )
    basaer1 = loop_mirror.add_ref(b1)
    basaer2 = loop_mirror.add_ref(b2)
    basaer1.connect('o1', wg1.ports['o2'])
    basaer2.connect('o1', wg2.ports['o2'])
    wg3 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg4 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg3.connect('o1', basaer1.ports['o2'])
    wg4.connect('o1', basaer2.ports['o2'])
    taper2 = gf.components.taper2(
        length=150, width1=wg_width, width2=multi_wg_width, layer=layer_wg
    )
    t3 = loop_mirror.add_ref(taper2)
    t4 = loop_mirror.add_ref(taper2)
    t3.connect('o1', wg3.ports['o2'])
    t4.connect('o1', wg4.ports['o2'])
    wg5 = loop_mirror << add_wg_1(
        wg_length=400, wg_width=multi_wg_width, layer=layer_wg
    )
    wg6 = loop_mirror << add_wg_1(
        wg_length=400, wg_width=multi_wg_width, layer=layer_wg
    )
    wg5.connect('o1', t3.ports['o2'])
    wg6.connect('o1', t4.ports['o2'])
    taper3 = gf.components.taper2(
        length=150, width1=multi_wg_width, width2=wg_width, layer=layer_wg
    )
    t5 = loop_mirror.add_ref(taper3)
    t6 = loop_mirror.add_ref(taper3)
    t5.connect('o1', wg5.ports['o2'])
    t6.connect('o1', wg6.ports['o2'])
    wg5 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg6 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg5.connect('o1', t5.ports['o2'])
    wg6.connect('o1', t6.ports['o2'])
    basaer3 = loop_mirror.add_ref(b2)
    basaer4 = loop_mirror.add_ref(b1)
    basaer3.connect('o1', wg5.ports['o2'])
    basaer4.connect('o1', wg6.ports['o2'])
    wg7 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg8 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg7.connect('o1', basaer3.ports['o2'])
    wg8.connect('o1', basaer4.ports['o2'])
    taper4 = gf.components.taper2(
        length=30, width1=wg_width, width2=1.17, layer=layer_wg
    )
    t7 = loop_mirror.add_ref(taper4)
    t8 = loop_mirror.add_ref(taper4)
    t7.connect('o1', wg7.ports['o2'])
    t8.connect('o1', wg8.ports['o2'])
    mmi2.connect('o1', t7.ports['o2'])
    t9 = loop_mirror.add_ref(taper1)
    t10 = loop_mirror.add_ref(taper1)
    t9.connect('o1', mmi2.ports['o2'])
    t10.connect('o1', mmi2.ports['o3'])
    wg9 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg10 = loop_mirror << add_wg_1(wg_length=10, layer=layer_wg, wg_width=wg_width)
    wg9.connect('o1', t9.ports['o2'])
    wg10.connect('o1', t10.ports['o2'])
    L_basaer = 220
    W_basaer = 84
    section1 = gf.Section(width=wg_width, layer=layer_wg, port_names=('o1', 'o2'))
    x = gf.CrossSection(sections=tuple([section1]))
    b3 = gf.components.bezier(
        control_points=[
            (0, 0),
            (L_basaer / 2, 0),
            (L_basaer / 2, W_basaer),
            (L_basaer, W_basaer),
        ],
        npoints=1000,
        cross_section=x,
    )
    b4 = gf.components.bezier(
        control_points=[
            (0, 0),
            (L_basaer / 2, 0),
            (L_basaer / 2, -W_basaer),
            (L_basaer, -W_basaer),
        ],
        npoints=1000,
        cross_section=x,
    )
    basaer5 = loop_mirror.add_ref(b3)
    basaer6 = loop_mirror.add_ref(b4)
    basaer5.connect('o1', wg9.ports['o2'])
    basaer6.connect('o1', wg10.ports['o2'])
    euler1 = gf.components.bend_euler(
        angle=-90, p=0.2, layer=layer_wg, width=wg_width, radius=171.24 / 2
    )
    e1 = loop_mirror.add_ref(euler1)
    e1.connect('o1', basaer5.ports['o2'])
    euler2 = gf.components.bend_euler(
        angle=90, p=0.2, layer=layer_wg, width=wg_width, radius=171.24 / 2
    )
    e2 = loop_mirror.add_ref(euler2)
    e2.connect('o1', basaer6.ports['o2'])
    t11 = loop_mirror.add_ref(taper4)
    t12 = loop_mirror.add_ref(taper4)
    t11.connect('o2', mmi1.ports['o1'])
    t12.connect('o2', mmi1.ports['o4'])
    wg11 = loop_mirror << add_wg_1(wg_length=20, layer=layer_wg, wg_width=wg_width)
    wg12 = loop_mirror << add_wg_1(wg_length=20, layer=layer_wg, wg_width=wg_width)
    wg11.connect('o2', t11.ports['o1'])
    wg12.connect('o2', t12.ports['o1'])

    loop_mirror.add_port('o1', port=wg11.ports['o1'])
    loop_mirror.add_port('o2', port=wg12.ports['o1'])

    return loop_mirror


@gf.cell
def add_1x2MMItree(
    core_length: float = 23.6,
    core_width: float = 5.7,
    separation: float = 3,
    taper_length: float = 15,
    taper_width: float = 1.68,
    layer: tuple = LAYER.WG,
    num: int = 5,
    L_basaer: float = 180,
    W_basaer: float = 60,
    mmi_wg_width: float = 1,
    wg_width: float = 1,
    gc_pro: gf.Component | None = None,
    wg_lvbo: float = 0.8,
    wg_lvbo_length: float = 100,
):
    '''

    创建一个1x2 MMI分光器树状网络

    该函数生成一个多级1x2 MMI分光器的树状结构，通过级联多个1x2 MMI分光器
    和贝塞尔曲线连接，形成1对多输出的分光网络。每个输出端都连接光栅耦合器。

    Args:
        core_length (float, optional): 每个MMI核心区域的长度，单位为微米。默认值为23.6μm。
        core_width (float, optional): 每个MMI核心区域的宽度，单位为微米。默认值为5.7μm。
        separation (float, optional): MMI输出端口之间的中心间距，单位为微米。默认值为3μm。
        taper_length (float, optional): 每个MMI渐变器的长度，单位为微米。默认值为15μm。
        taper_width (float, optional): MMI连接处的taper宽度，单位为微米。默认值为1.68μm。
        layer (tuple, optional): 器件所在的工艺层，格式为(layer, datatype)。默认为LAYER.WG。
        num (int, optional): MMI分光器的级联层数。默认值为5级。
        L_basaer (float, optional): 贝塞尔曲线的水平长度，单位为微米。默认值为180μm。
        W_basaer (float, optional): 贝塞尔曲线的垂直偏移，单位为微米。默认值为60μm。
        wg_width (float, optional): 连接波导的宽度，单位为微米。默认值为1μm。
        gc_pro (gf.Component, optional): 光栅耦合器组件对象，如果为None则自动生成。默认为None。
        当gc_pro为None时，函数会自动创建一个光栅耦合器,下列参数此时有用:
        wg_lvbo (float, optional): 光栅耦合器连接波导
        wg_lvbo_length (float, optional): 光栅耦合器连接波导长度，单位为微米。默认值为100μm。

    Returns:
        None: 函数直接显示生成的器件，不返回组件对象

    Structure:
        GC ── MMI1 ──┬── Bezier ── MMI2 ──┬── Bezier ── MMI3 ──┬── GC
                     │                    │                    │
                     └── Bezier ── GC     └── Bezier ── GC     └── GC

        输入: 1个光栅耦合器
        输出: 2^num 个光栅耦合器 (级联num级MMI)

    Parameters Detail:
        - core_length/core_width: 控制每个MMI的尺寸和分光比
        - separation: 影响MMI输出端口间距
        - num: 决定最终输出端口数量 (2^num个输出)
        - L_basaer/W_basaer: 控制级间连接的贝塞尔曲线形状
        - wg_width: 统一的波导连接宽度

    Note:
        - 每级MMI都使用相同的几何参数
        - 贝塞尔曲线提供平滑的级间连接，减少损耗
        - 自动在每个输出端添加光栅耦合器用于测试
        - 函数末尾调用c1.show()直接显示器件
        - 适用于1对多功分器、阵列波导光栅前端等应用

    '''

    c1 = gf.Component()
    ##############################################################################
    # 构建mmi_pro，即原本的MMI每个taper后加一个wg_width宽的直波导
    mmi_pro = gf.Component()
    mmi1 = mmi_pro << add_1x2MMI_1(
        core_length=core_length,
        core_width=core_width,
        layer=layer,
        taper_length=taper_length,
        taper_width=taper_width,
        separation=separation,
        end_width=mmi_wg_width,
    )
    taper_wg_mmi = gf.components.taper2(
        length=20, width1=wg_width, width2=mmi_wg_width, layer=layer
    )
    taper_wg_mmi_ref1 = mmi_pro.add_ref(taper_wg_mmi)
    taper_wg_mmi_ref2 = mmi_pro.add_ref(taper_wg_mmi)
    taper_wg_mmi_ref3 = mmi_pro.add_ref(taper_wg_mmi)

    taper_wg_mmi_ref1.connect('o2', mmi1.ports['o1'])
    taper_wg_mmi_ref2.connect('o2', mmi1.ports['o2'])
    taper_wg_mmi_ref3.connect('o2', mmi1.ports['o3'])
    mmi_pro.add_port(name='o1', port=taper_wg_mmi_ref1.ports['o1'])
    mmi_pro.add_port(name='o2', port=taper_wg_mmi_ref2.ports['o1'])
    mmi_pro.add_port(name='o3', port=taper_wg_mmi_ref3.ports['o1'])

    ##############################################################################
    if gc_pro is None:
        gc_pro = gf.Component()
        gc = gc_pro << add_gc_1(wg_width=wg_width, layer=layer)
        taper_wg_lvbo = gf.components.taper2(
            length=20, width1=wg_width, width2=wg_lvbo, layer=layer
        )
        taper_wg_lvbo_ref1 = gc_pro.add_ref(taper_wg_lvbo)
        taper_wg_lvbo_ref2 = gc_pro.add_ref(taper_wg_lvbo)
        taper_wg_lvbo_ref1.connect('o1', gc.ports['o1'])
        wg1 = gc_pro << add_wg_1(
            wg_length=wg_lvbo_length, wg_width=wg_lvbo, layer=layer
        )
        wg1.connect('o1', taper_wg_lvbo_ref1.ports['o2'])
        taper_wg_lvbo_ref2.connect('o2', wg1.ports['o2'])
        gc_pro.add_port(name='o1', port=taper_wg_lvbo_ref2.ports['o1'])
    ##############################################################################
    mmi1 = c1.add_ref(mmi_pro)
    gc1 = c1.add_ref(gc_pro)
    # L_basaer = 180
    # W_basaer = 60
    section1 = gf.Section(width=wg_width, layer=layer, port_names=('o1', 'o2'))
    x = gf.CrossSection(sections=tuple([section1]))
    b1 = gf.components.bezier(
        control_points=[
            (0, 0),
            (L_basaer / 2, 0),
            (L_basaer / 2, W_basaer),
            (L_basaer, W_basaer),
        ],
        npoints=1000,
        cross_section=x,
    )
    b2 = gf.components.bezier(
        control_points=[
            (0, 0),
            (L_basaer / 2, 0),
            (L_basaer / 2, -W_basaer),
            (L_basaer, -W_basaer),
        ],
        npoints=1000,
        cross_section=x,
    )
    basaer_1 = c1.add_ref(b1)
    basaer_2 = c1.add_ref(b2)
    basaer_1.connect('o1', mmi1.ports['o2'])
    basaer_2.connect('o1', mmi1.ports['o3'])
    gc1.connect('o1', mmi1.ports['o1'])
    gc1 = c1.add_ref(gc_pro)
    gc1.connect('o1', basaer_2.ports['o2'])
    # N = N  # 级联的MMI数量
    for i in range(num - 1):
        mmi1 = c1.add_ref(mmi_pro)
        mmi1.connect('o1', basaer_1.ports['o2'])
        basaer_1 = c1.add_ref(b1)
        basaer_2 = c1.add_ref(b2)
        basaer_1.connect('o1', mmi1.ports['o2'])
        basaer_2.connect('o1', mmi1.ports['o3'])
        gc1 = c1.add_ref(gc_pro)
        gc1.connect('o1', basaer_2.ports['o2'])
    gc1 = c1.add_ref(gc_pro)
    gc1.connect('o1', basaer_1.ports['o2'])
    c1.show()


@gf.cell
def add_multi_wg_tlet(
    multi_wg_width: float,
    multi_wg_layer: tuple,
    multi_wg_length: float,
    et_layer: tuple,
    et_separation: float,
    et_width: tuple,
    et_length_d: float = 350,
):
    """创建一个带有GSG电极结构的多模波导器件

    该函数生成一个电光调制器结构，包含一个多模波导和配套的GSG电极系统。
    电极通过弯曲路径连接到波导中心，形成完整的调制器件。

    Args:
        multi_wg_width (float): 多模波导的宽度，单位为微米
        multi_wg_layer (tuple): 多模波导所在的层，格式为(layer, datatype)
        multi_wg_length (float): 多模波导的总长度，单位为微米
        et_layer (tuple): 电极所在的层，格式为(layer, datatype)
        et_separation (float): 相邻电极之间的间距，单位为微米
        et_width (tuple): 三个电极的宽度，格式为(上电极宽度, 中心电极宽度, 下电极宽度)
        et_length_d (float): 中心电极的直通段长度相比多模波导少的距离，单位为微米，默认为350μm

    Returns:
        gf.Component: 包含多模波导和T型电极结构的GDSFactory组件对象，
                     具有端口'o1'和'o2'用于光学连接

    Structure:
        电极弯曲部分 ─┐        ┌─ 电极弯曲部分
                    │        │
                    │  多模  │
        o1 ─────────┼─波导───┼────────── o2
                    │        │
                    └────────┘
                 et_middle (中心连接点)

    Note:
        - 电极弯曲半径固定为217μm
        - 电极直通段长度为 multi_wg_length - et_length_d
        - 中心电极通过'et_middle'端口与波导中心连接
        - 适用于电光调制器、相位调制器等应用场景
    """
    c1 = gf.Component()
    # 添加多模波导
    wg1 = c1 << add_wg_1(
        wg_width=multi_wg_width,
        layer=multi_wg_layer,
        wg_length=multi_wg_length,
        et_width=et_width[1],
        et_layer=et_layer,
    )
    # 添加电极的第一个弯曲
    path1 = gf.Path()
    path1 += gf.path.arc(radius=217, angle=90, start_angle=90)
    s0 = gf.Section(width=et_width[1], layer=et_layer, port_names=('o1', 'o2'))
    s1 = gf.Section(
        width=et_width[0],
        offset=et_width[0] / 2 + et_separation + et_width[1] / 2,
        layer=et_layer,
    )
    s2 = gf.Section(
        width=et_width[2],
        offset=-(et_width[2] / 2 + et_separation + et_width[1] / 2),
        layer=et_layer,
    )
    x = gf.CrossSection(sections=tuple([s0, s1, s2]))
    path1_1 = gf.path.extrude(path1, cross_section=x)
    path1_ref = c1.add_ref(path1_1)  # type: ignore
    path1_ref.drotate(-90)
    # 添加电极的直通端和第二个弯曲
    path2 = gf.Path()
    path2 += gf.path.straight(length=multi_wg_length - et_length_d)
    path2 += gf.path.arc(radius=217, angle=90)
    s0 = gf.Section(width=et_width[1], layer=et_layer, port_names=('o1', 'o2'))
    s1 = gf.Section(
        width=et_width[0],
        offset=et_width[0] / 2 + et_separation + et_width[1] / 2,
        layer=et_layer,
    )
    s2 = gf.Section(
        width=et_width[2],
        offset=-(et_width[2] / 2 + et_separation + et_width[1] / 2),
        layer=et_layer,
    )
    x = gf.CrossSection(sections=tuple([s0, s1, s2]))
    path2_1 = gf.path.extrude(path2, cross_section=x)
    path2_1.add_port(
        name='et_middle',
        center=(
            (multi_wg_length - et_length_d) / 2,
            -(et_width[1] / 2 + et_separation / 2),
        ),
        layer=et_layer,
        width=et_width[1],
        orientation=180,
    )
    path2_ref = c1.add_ref(path2_1)  # type: ignore

    path2_ref.connect('et_middle', wg1.ports['et_middle'])
    path1_ref.connect('o2', path2_ref.ports['o1'])

    c1.add_port(name='o1', port=wg1.ports['o1'])
    c1.add_port(name='o2', port=wg1.ports['o2'])

    return c1


@gf.cell
def add_bentDC(
    waveguide_width: float,
    gap: float,  # 定向耦合器间距，为波导中心间距
    angle: float,
    radius1: float,
    x_position: float,
    y_position: float,
    layer_wg: tuple,
    # dwidth: float,
    # waveguide_gap: float | None = None,#上下波导间距
) -> Component:
    '''
    创建一个弯曲定向耦合器(Bent Directional Coupler)

    该函数生成一个弯曲型定向耦合器结构，由两对贝塞尔曲线组成，提供光功率分配和耦合功能。
    器件采用弯曲设计，可实现紧凑的布局和可调的耦合长度。

    Args:
        waveguide_width (float): 波导宽度，单位为微米
        gap (float): 定向耦合器间距，指波导中心到中心的距离，单位为微米
        angle (float): 弯曲角度，单位为度
        radius1 (float): 上臂弯曲半径，单位为微米
        x_position (float): 器件在x方向的位置偏移，单位为微米
        y_position (float): 器件在y方向的位置偏移，单位为微米
        layer_wg (tuple): 波导所在的工艺层，格式为(layer, datatype)

    Returns:
        Component: 包含弯曲定向耦合器结构的GDSFactory组件对象，具有四个光学端口：
                  - 'o1': 左上输入端口
                  - 'o2': 左下输入端口
                  - 'o3': 右上输出端口
                  - 'o4': 右下输出端口

    Structure:
        o1 ══════╗                        ╔══════ o3
                ║╲                      ╱║
                ║ ╲                    ╱ ║
                ║  ╲ Bent Directional ╱  ║
                ║   ╲   Coupler      ╱   ║
                ║    ╲              ╱    ║
                ║     ╲____________╱     ║
                ║     ╱            ╲     ║
                ║    ╱              ╲    ║
                ║   ╱                ╲   ║
                ║  ╱                  ╲  ║
                ║ ╱                    ╲ ║
                ║╱                      ╲║
        o2 ══════╝                        ╚══════ o4

    Parameters Detail:
        - waveguide_width: 影响单模传输特性和耦合效率
        - gap: 控制耦合强度，决定功率分配比
        - angle: 影响器件的紧凑性和弯曲损耗
        - radius1: 上臂半径，影响上臂的弯曲特性
        - x_position/y_position: 器件的全局位置偏移
        - layer_wg: 定义器件的制造层

    Note:
        - 使用贝塞尔曲线实现平滑的弯曲过渡
        - 自动计算下臂半径：radius2 = radius1 + gap
        - 自动补偿上下臂的长度差，确保相位匹配
        - 固定的内部角度参数：m = π/6 (30°)
        - 适用于功分器、合路器、光开关等应用
        - @gf.cell装饰器支持器件缓存和复用

    '''
    D = Component()
    dc = gf.Component()
    l = gf.Component()
    r = gf.Component()
    m = math.pi / 6

    hi = (4 / 3) * (1 - math.cos(m / 2)) / (math.sin(m / 2))
    car = math.sqrt(
        (1 - math.cos(m) - hi * math.sin(m)) ** 2
        + (math.sin(m) - hi * math.cos(m) - hi) ** 2
    )
    xs = gf.cross_section.cross_section(
        width=waveguide_width,
        offset=0,
        layer=layer_wg,
        port_types=('optical', 'optical'),
        port_names=("o1", "o2"),
    )
    m1 = (angle / 180) * math.pi
    radius2 = radius1 + gap

    c1 = gf.components.bezier(
        control_points=(
            (0, 0),
            (
                float(radius1) * (float(hi) * float(math.cos(m / 2))),
                radius1 * (-hi * math.sin(m / 2.0)),
            ),
            (
                radius1 * (hi * math.cos(m / 2.0) + car),
                radius1 * (-hi * math.sin(m / 2.0)),
            ),
            (radius1 * (2.0 * hi * math.cos(m / 2.0) + car), 0),
        ),
        npoints=201,
        with_manhattan_facing_angles=True,
        cross_section=xs,
        allow_min_radius_violation=False,
    )

    c2 = gf.components.bezier(
        control_points=(
            (0, 0),
            (radius1 * (-hi * math.cos(m / 2)), radius1 * (hi * math.sin(m / 2))),
            (radius1 * (-hi * math.cos(m / 2)), 2 * radius1 * (hi * math.sin(m / 2))),
            (
                radius1 * (-3 * hi * math.cos(m / 2)),
                2 * radius1 * (hi * math.sin(m / 2)),
            ),
        ),
        npoints=201,
        with_manhattan_facing_angles=True,
        cross_section=xs,
        allow_min_radius_violation=False,
    )

    c3 = gf.components.bezier(
        control_points=(
            (radius1 * (2 * hi * math.cos(m / 2) + car), 0),
            (
                radius1 * (hi * math.cos(m / 2))
                + radius1 * (2 * hi * math.cos(m / 2) + car),
                radius1 * (hi * math.sin(m / 2)),
            ),
            (
                radius1 * (hi * math.cos(m / 2))
                + radius1 * (2 * hi * math.cos(m / 2) + car),
                2 * radius1 * (hi * math.sin(m / 2)),
            ),
            (
                radius1 * (3 * hi * math.cos(m / 2))
                + radius1 * (2 * hi * math.cos(m / 2) + car),
                2 * radius1 * (hi * math.sin(m / 2)),
            ),
        ),
        npoints=201,
        with_manhattan_facing_angles=True,
        cross_section=xs,
        allow_min_radius_violation=False,
    )

    hi1 = (4 / 3) * (1 - math.cos(m1 / 2)) / (math.sin(m1 / 2))
    car1 = math.sqrt(
        (1 - math.cos(m1) - hi1 * math.sin(m1)) ** 2
        + (math.sin(m1) - hi1 * math.cos(m1) - hi1) ** 2
    )

    h1 = radius1 - math.sqrt(
        radius1**2 - ((radius1 * (2 * hi * math.cos(m / 2) + car)) / 2) ** 2
    )
    h2 = radius2 - math.sqrt(
        radius2**2 - ((radius2 * (2 * hi1 * math.cos(m1 / 2) + car1)) / 2) ** 2
    )

    c4 = gf.components.bezier(
        control_points=(
            (0, 0),
            (radius2 * (hi1 * math.cos(m1 / 2)), radius2 * (-hi1 * math.sin(m1 / 2))),
            (
                radius2 * (hi1 * math.cos(m1 / 2) + car1),
                radius2 * (-hi1 * math.sin(m1 / 2)),
            ),
            (radius2 * (2 * hi1 * math.cos(m1 / 2) + car1), 0),
        ),
        npoints=201,
        with_manhattan_facing_angles=True,
        cross_section=xs,
        allow_min_radius_violation=False,
    )

    c5 = gf.components.bezier(
        control_points=(
            (0, 0),
            (radius2 * (-hi1 * math.cos(m1 / 2)), radius2 * (hi1 * math.sin(m1 / 2))),
            (
                radius2 * (-hi1 * math.cos(m1 / 2) - hi1),
                radius2 * (hi1 * math.sin(m1 / 2)),
            ),
            (
                radius2 * (-hi1 * math.cos(m1 / 2) - 2 * hi1),
                radius2 * (hi1 * math.sin(m1 / 2)),
            ),
        ),
        npoints=201,
        with_manhattan_facing_angles=True,
        cross_section=xs,
        allow_min_radius_violation=False,
    )

    c6 = gf.components.bezier(
        control_points=(
            (radius2 * (2 * hi1 * math.cos(m1 / 2) + car1), 0),
            (
                radius2 * (hi1 * math.cos(m1 / 2))
                + radius2 * (2 * hi1 * math.cos(m1 / 2) + car1),
                radius2 * (hi1 * math.sin(m1 / 2)),
            ),
            (
                radius2 * (hi1 * math.cos(m1 / 2) + hi1)
                + radius2 * (2 * hi1 * math.cos(m1 / 2) + car1),
                radius2 * (hi1 * math.sin(m1 / 2)),
            ),
            (
                radius2 * (hi1 * math.cos(m1 / 2) + 2 * hi1)
                + radius2 * (2 * hi1 * math.cos(m1 / 2) + car1),
                radius2 * (hi1 * math.sin(m1 / 2)),
            ),
        ),
        npoints=201,
        with_manhattan_facing_angles=True,
        cross_section=xs,
        allow_min_radius_violation=False,
    )

    ##############################计算上下臂相差多长
    t = radius1 * (3 * hi * math.cos(m / 2)) + radius1 * (
        2 * hi * math.cos(m / 2) + car
    )
    t1 = (radius1 * (2 * hi * math.cos(m / 2) + car)) / 2 - (
        radius2 * (2 * hi1 * math.cos(m1 / 2) + car1)
    ) / 2
    t2 = radius2 * (hi1 * math.cos(m1 / 2) + 2 * hi1) + radius2 * (
        2 * hi1 * math.cos(m1 / 2) + car1
    )
    ###################################
    ##############################计算上下臂相差多高
    height = (
        2 * radius1 * hi * math.sin(m / 2)
        - radius2 * hi1 * math.sin(m1 / 2)
        - (-h1 + h2 - gap)
    )

    ###################################

    wg1 = gf.components.straight(length=10, npoints=2, cross_section=xs)  # 左上
    wg2 = gf.components.straight(
        length=10 + t - t1 - t2, npoints=2, cross_section=xs
    )  # 左下

    wg3 = gf.components.straight(length=10, npoints=2, cross_section=xs)  # 右上
    wg4 = gf.components.straight(
        length=10 + t - t1 - t2, npoints=2, cross_section=xs
    )  # 右下

    ###########################################################################################

    ###########################################################################################

    # gc = gf.components.grating_coupler_elliptical_uniform(n_periods=30, period=1, fill_factor=0.34902,taper_length=40.3,taper_angle=17.6,fiber_angle=8,cross_section=xs)

    # 绘制单元弯曲DC.把一个弯曲dc作为一个基本单元
    arc_up = dc.add_ref(c1)
    link_upl = dc.add_ref(c2)
    link_upr = dc.add_ref(c3)
    arc_down = dc.add_ref(c4)
    link_downl = dc.add_ref(c5)
    link_downr = dc.add_ref(c6)

    waveg1 = dc.add_ref(wg1)
    waveg2 = dc.add_ref(wg2)
    waveg3 = dc.add_ref(wg3)
    waveg4 = dc.add_ref(wg4)

    arc_down.dmove(
        (
            (radius1 * (2 * hi * math.cos(m / 2) + car)) / 2
            - (radius2 * (2 * hi1 * math.cos(m1 / 2) + car1)) / 2,
            -h1 + h2 - gap,
        )
    )
    link_downl.dmove(
        (
            (radius1 * (2 * hi * math.cos(m / 2) + car)) / 2
            - (radius2 * (2 * hi1 * math.cos(m1 / 2) + car1)) / 2,
            -h1 + h2 - gap,
        )
    )
    link_downr.dmove(
        (
            (radius1 * (2 * hi * math.cos(m / 2) + car)) / 2
            - (radius2 * (2 * hi1 * math.cos(m1 / 2) + car1)) / 2,
            -h1 + h2 - gap,
        )
    )
    arc_up.dmove((x_position, y_position))
    arc_down.dmove((x_position, y_position))
    link_upl.connect("o1", arc_up.ports["o1"])
    link_upr.connect("o1", arc_up.ports["o2"])
    link_downl.connect("o1", arc_down.ports["o1"])
    link_downr.connect("o1", arc_down.ports["o2"])
    waveg1.connect("o2", link_upl.ports["o2"])
    waveg2.connect("o2", link_downl.ports["o2"])
    waveg3.connect("o1", link_upr.ports["o2"])
    waveg4.connect("o1", link_downr.ports["o2"])
    # gc1 = dc.add_ref(gc)
    # gc2 = dc.add_ref(gc)

    # sbend1 = gf.components.bend_s(size=(7*(waveguide_gap-height)/2, (waveguide_gap-height)/2), cross_section=xs)
    # s1=D.add_ref(sbend1)
    # s2=D.add_ref(sbend1)
    # s3=D.add_ref(sbend1)
    # s4=D.add_ref(sbend1)
    #
    # s1.mirror_x()
    # s4.mirror_x()

    dc.add_port("o1", port=waveg1.ports["o1"])
    dc.add_port("o2", port=waveg2.ports["o1"])
    dc.add_port("o3", port=waveg3.ports["o2"])
    dc.add_port("o4", port=waveg4.ports["o2"])
    ##################################
    bentdc1 = D.add_ref(dc)

    # s1.connect("o1", bentdc1.ports["o1"])
    # s2.connect("o1", bentdc1.ports["o2"])
    # s3.connect("o1", bentdc1.ports["o3"])
    # s4.connect("o1", bentdc1.ports["o4"])
    # D.add_port("o1", port=s1.ports["o2"])
    # D.add_port("o2", port=s2.ports["o2"])
    # D.add_port("o3", port=s3.ports["o2"])
    # D.add_port("o4", port=s4.ports["o2"])
    return dc


import kfactory as kf


@gf.cell
def connect_ports_with_parallelogram(
    port1: kf.Port, port2: kf.Port, layer: tuple = LAYER.SOURCE
):
    '''
    自动生成平行四边形连接两个同宽度端口
    :param port1: Ports类型
    :param port2:
    :param layer:
    :return:
    # 使用示例
    if __name__ == "__main__":
    c = gf.Component()

    # 创建两个测试端口
    port1 = gf.Port(name="port1", center=(0, 0), width=0.5, orientation=0, layer=(1, 0))
    port2 = gf.Port(name="port2", center=(10, 5), width=0.5, orientation=45, layer=(1, 0))

    # 生成连接器
    connector = connect_ports_with_parallelogram(port1, port2)
    ref = c.add_ref(connector)

    c.show()
    '''
    c = gf.Component()

    # 提取参数
    p1 = np.array(port1.center) / 1000
    p2 = np.array(port2.center) / 1000
    width = port1.width / 1000
    ang1 = port1.orientation
    ang2 = port2.orientation - 180

    # 计算边缘向量
    def get_delta(angle):
        return (
            0.5
            * width
            * np.array([np.cos(np.deg2rad(angle + 90)), np.sin(np.deg2rad(angle + 90))])
        )

    delta1 = get_delta(ang1)
    delta2 = get_delta(ang2)

    # 生成顶点
    points = [
        list((p1 + delta1).astype(float)),
        list((p1 - delta1).astype(float)),
        list((p2 - delta2).astype(float)),
        list((p2 + delta2).astype(float)),
    ]

    # 添加多边形
    c.add_polygon(points, layer=layer)

    # 添加端口保持连接性
    c.add_port(
        name="port1", center=tuple(p1), width=width, orientation=ang1, layer=layer
    )
    c.add_port(
        name="port2", center=tuple(p2), width=width, orientation=ang2, layer=layer
    )

    return c


@gf.cell
def add_ring_2(
    gap: float = 1.6,
    length_x: float = 10,
    length_y: float = 1000,
    straight_wg_length: float = 100,
    p: float = 0.2,
    radius: float = 150,
    wg_width: float = 1,
    layer_wg: tuple = (1, 0),
):
    '''
    创建一个带Euler弯曲的矩形微环谐振器

    该函数生成一个采用Euler弯曲的矩形微环谐振器结构，包含一个矩形环形谐振腔和左右两条直波导。
    微环采用Euler弯曲替代传统圆弧，减少弯曲损耗，提供更好的传输特性。

    Args:
        gap (float, optional): 直波导与微环的耦合间距，单位为微米。默认值为1.6μm。
        length_x (float, optional): 微环矩形的短边长度，单位为微米。默认值为10μm。
        length_y (float, optional): 微环矩形的长边长度，单位为微米。默认值为1000μm。
        straight_wg_length (float, optional): 左右直波导的长度，单位为微米。默认值为100μm。
        p (float, optional): Euler弯曲的p参数，控制弯曲形状。默认值为0.2。
        radius (float, optional): Euler弯曲的半径，单位为微米。默认值为150μm。
        wg_width (float, optional): 波导宽度，单位为微米。默认值为1μm。
        layer_wg (tuple, optional): 波导所在的工艺层，格式为(layer, datatype)。默认值为(1, 0)。

    Returns:
        gf.Component: 包含矩形微环谐振器结构的GDSFactory组件对象，具有四个光学端口：
                     - 'o1': 左上直波导端口
                     - 'o2': 左下直波导端口
                     - 'o3': 右上直波导端口
                     - 'o4': 右下直波导端口

    Structure:
        o1 ──────────────────────────── o2
                ┌─────────────────┐
            ╱                   ╲
            ╱     Rectangular     ╲
            ╱        Ring with      ╲
            ╱       Euler Bends      ╲
        ╱                           ╲
        ╱           Ring             ╲
        ╱            Core              ╲
        ╲                             ╱
        ╲                           ╱
        ╲                         ╱
            ╲                       ╱
            ╲                     ╱
            ╲___________________╱
        o4 ──────────────────────────── o3

    Parameters Detail:
        - gap: 控制耦合强度，影响品质因子和耦合效率
        - length_x/length_y: 决定微环尺寸和谐振波长
        - p: Euler弯曲参数，影响弯曲损耗和模式匹配
        - radius: 弯曲半径，影响器件占用空间和传输特性
        - straight_wg_length: 便于与其他器件连接

    Note:
        - 使用四个90°Euler弯曲构成矩形微环
        - Euler弯曲提供更低的弯曲损耗
        - 自动计算并输出微环周长
        - 支持高精度路径描述(npoints=5000)
        - 适用于滤波器、延迟线、光学缓存等应用

    '''

    c = gf.Component()
    c1 = gf.Component()

    # ring
    path0 = gf.Path()
    path0 += gf.path.euler(angle=90, p=p, radius=radius, use_eff=True, npoints=5000)
    path0 += gf.path.straight(length=length_y)
    path0 += gf.path.euler(angle=90, p=p, radius=radius, use_eff=True, npoints=5000)
    path0 += gf.path.straight(length=length_x)
    path0 += gf.path.euler(angle=90, p=p, radius=radius, use_eff=True, npoints=5000)
    path0 += gf.path.straight(length=length_y)
    path0 += gf.path.euler(angle=90, p=p, radius=radius, use_eff=True, npoints=5000)
    path0 += gf.path.straight(length=length_x)

    print('微环周长：', path0.length())

    # 上面的直波导
    path1 = gf.Path()
    path1 += gf.path.straight(length=straight_wg_length)
    # 下面的直波导
    path2 = path1.copy()

    path0.drotate(angle=-90, center=(0, 0))
    path1.drotate(angle=-90, center=(0, 0))
    path2.drotate(angle=-90, center=(0, 0))

    path1.dmove([-(gap), (straight_wg_length + length_x) / 2])
    path2.dmove([(gap) + radius * 2 + length_y, (straight_wg_length + length_x) / 2])
    path0_1 = gf.path.extrude(path0, layer=layer_wg, width=wg_width)
    path1_1 = gf.path.extrude(path1, layer=layer_wg, width=wg_width)
    path2_1 = gf.path.extrude(path2, layer=layer_wg, width=wg_width)
    path0_1_ref = c1.add_ref(path0_1)  # 微环 # type: ignore
    path1_1_ref = c1.add_ref(path1_1)  # 左边直波导 # type: ignore
    path2_1_ref = c1.add_ref(path2_1)  # 右边直波导 # type: ignore

    c1.add_port("o1", port=path1_1_ref.ports["o1"])
    c1.add_port("o2", port=path1_1_ref.ports["o2"])
    c1.add_port("o3", port=path2_1_ref.ports["o1"])
    c1.add_port("o4", port=path2_1_ref.ports["o2"])
    return c1


@gf.cell
def add_circle_dbr(
    N: int = 6000,
    circle_gap: float = 1.5,
    circle_D: float = 0.5,
    circle_period: float = 1.3,
    wg_width: float = 1,
    layer_wg: tuple = (1, 0),
):
    '''
    创建一个圆柱形分布式布拉格反射器(DBR)

    该函数生成一个圆柱形DBR结构，通过在直波导两侧周期性排列圆形散射体，
    形成一维光子晶体结构，可用于激光器反射镜、滤波器等应用。

    Args:
        N (int, optional): DBR周期数(圆形散射体对数)。默认值为6000。
        circle_gap (float, optional): 圆形散射体中心到波导中心的间距，单位为微米。默认值为0.5μm。
        circle_D (float, optional): 圆形散射体的直径，单位为微米。默认值为0.5μm。
        circle_period (float, optional): DBR的周期长度，单位为微米。默认值为1.3μm。
        wg_width (float, optional): 中心波导的宽度，单位为微米。默认值为1μm。
        layer_wg (tuple, optional): 器件所在的工艺层，格式为(layer, datatype)。默认值为(1, 0)。

    Returns:
        gf.Component: 包含圆柱形DBR结构的GDSFactory组件对象，具有两个光学端口：
                     - 'o1': 左侧波导端口
                     - 'o2': 右侧波导端口

    Structure:
        o1 ─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─○─ o2
           │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
           ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○ ○

        上下对称排列的圆形散射体，形成周期性结构

    Parameters Detail:
        - N: 决定DBR的反射带宽和反射率强度
        - circle_gap: 控制耦合强度和工作波长
        - circle_D: 影响散射强度和带隙特性
        - circle_period: 决定布拉格波长 λ = 2*n_eff*Λ
        - wg_width: 影响模式特性和传输损耗
        - layer_wg: 定义制造工艺层

    Note:
        - 总DBR长度：L = N × circle_period
        - 波导总长度：dbr_wg_L = L + 40μm (包含20μm缓冲区)
        - 圆形散射体成对出现在波导上下两侧
        - 散射体中心位置：y = ±circle_gap
        - 适用于激光器、高Q谐振器、窄带滤波器等应用

    '''

    c1 = gf.Component()
    L = N * circle_period
    dbr_wg_L = L + 40
    dbr_wg = c1 << add_wg_1(wg_width=wg_width, layer=layer_wg, wg_length=dbr_wg_L)
    circle = gf.components.circle(
        radius=circle_D / 2, angle_resolution=1, layer=layer_wg
    )
    for i in range(N):
        x = (dbr_wg_L - L) / 2 + (i - 1) * circle_period
        circle_1 = c1.add_ref(circle)
        circle_1.dmove((x, circle_gap))
        circle_1 = c1.add_ref(circle)
        circle_1.dmove((x, -(circle_gap)))
    c1.add_port('o1', port=dbr_wg.ports['o1'])
    c1.add_port('o2', port=dbr_wg.ports['o2'])
    return c1


@gf.cell
def add_jingyuan(D, L):
    '''
    添加一个晶圆

    :param D: 晶圆直径[um]
    :param L: 晶圆长度[um]

    :return: 返回一个包含晶圆的GDSFactory组件对象
    '''
    c1 = gf.Component()

    R = D / 2
    distance = (R**2 - (L / 2) ** 2) ** 0.5
    angle = 2 * np.atan(L / 2 / distance) / np.pi * 180
    path0 = gf.Path()
    path0 += gf.path.arc(radius=R / 2, angle=360 - angle, npoints=1000)

    s0 = gf.Section(width=R, layer=(1, 0), port_names=('o1', 'o2'))
    x = gf.CrossSection(sections=tuple([s0]))
    path0_1 = gf.path.extrude(path0, cross_section=x)
    path0_1_ref = c1.add_ref(path0_1)  # type:ignore

    path0_1_ref.dmove((0, -R / 2))
    path0_1_ref.drotate(angle=angle / 2, center=(0, 0))

    poles = np.array([[0, 0], [L / 2, -distance], [-L / 2, -distance]])
    c1.add_polygon(poles, layer=(1, 0))
    return c1


def generate_orientation_marker(
    wafer_diameter, region_size, rows, marker_position, layer
):
    """
    生成晶向标记版图，基于输入的区域大小、每行的区域数和标记的相对位置。
    假设晶圆上每个区域大小相同，为每个区域添加晶向标记。

    Args:
        wafer_diameter (float): 晶圆直径，单位为 um。
        region_size (list): 每个区域的大小 [width, height]，单位为 um。
        rows (list): 每行的区域数，例如 [2, 3, 2]。
        marker_position (list): 标记在每个区域中的相对位置 [x, y]，单位为 um。
        layer (tuple): GDS 工件的层信息，格式为 (layer_number, data_type)。

    Returns:
        gdsfactory.Component: 包含所有晶向标记的 GDS 工件。
    """
    # wafer_diameter = 150000  # 晶圆直径，单位为 um
    layout = gf.Component()

    def create_marker():
        """创建晶向标记"""
        marker = gf.Component()
        # 添加箭头形状
        marker.add_polygon(
            np.array([(0, 0), (110, 0), (110, 350), (0, 350)]), layer=layer
        )  # 下箭头
        marker.add_polygon(
            np.array([(165, 0), (-55, 0), (55, -150)]), layer=layer
        )  # 上箭头
        return marker

    y_offset = 0
    for row_index, num_regions in enumerate(rows):
        row_width = num_regions * region_size[0]
        x_start = (wafer_diameter - row_width) / 2  # 使区域居中

        for col_index in range(num_regions):
            x = x_start + col_index * region_size[0] + marker_position[0]
            y = y_offset + marker_position[1]

            # 创建标记并添加到版图
            marker = create_marker()
            layout.add_ref(marker).dmove((x, y))

        y_offset += region_size[1]

    return layout


@gf.cell
def generate_number_layout(
    wafer_diameter, region_size, rows, number_position, text, layer_text
):
    """
    生成数字版图，基于输入的区域大小、每行的区域数和数字的相对位置。
    假设晶圆上每个区域大小相同，为每个左下角产生数字。

    Args:
        wafer_diameter (float): 晶圆直径，单位为 um。
        region_size (list): 每个区域的大小 [width, height]，单位为 um。
        rows (list): 每行的区域数，例如 [2, 3, 2]。从下到上算
        number_position (list): 数字在每个区域中的相对位置 [x, y]，单位为 um。
        text (str): 要添加的文本内容。
        layer_text (tuple): GDS 工件的层信息，格式为 (layer_number, data_type)。

    Returns:
        gdsfactory.Component: 包含所有数字版图的 GDS 工件。
    """
    # wafer_diameter = 150000  # 晶圆直径，单位为 um
    layout = gf.Component()

    y_offset = 0
    for row_index, num_regions in enumerate(rows):
        row_width = num_regions * region_size[0]
        x_start = (wafer_diameter - row_width) / 2  # 使区域居中

        for col_index in range(num_regions):
            x = x_start + col_index * region_size[0] + number_position[0]
            y = y_offset + number_position[1]

            # 创建数字并添加到版图
            number = gf.components.text(
                text + ' ' + str((row_index + 1) * 10 + col_index + 1),
                size=150,
                layer=layer_text,
            )
            layout.add_ref(number).dmove((x, y))

        y_offset += region_size[1]

    return layout


@gf.cell
def add_caliper(level, layer1, layer2, spacing) -> gf.Component:
    '''

    添加游标卡尺组件
    Args:
        level: 游标卡尺的级数
        layer1: 红色部分的layer
        layer2: 蓝色部分的layer
        spacing: 两个条周期之间的间距
    Returns:
        gf.Component: 返回游标卡尺组件

    '''
    # 创建一个新的组件
    c = gf.Component()

    # 参数设置
    bar_width = 3  # 每个条的宽度
    bar_length = 50  # 每个条的长度
    bar_period1 = 9.0
    bar_period2 = 9.0 + spacing

    # 绘制红色部分
    for i in range(level):

        y_offset = i * bar_period1
        if i == 0:
            x_offset = -bar_length - 50
        elif i % 5 == 0:
            x_offset = -bar_length - 20
        else:
            x_offset = -bar_length
        c.add_polygon(
            points=[
                [x_offset, y_offset],
                [0, y_offset],
                [0, y_offset + bar_width],
                [x_offset, y_offset + bar_width],
            ],
            layer=layer1,
        )

    # 绘制蓝色部分
    for i in range(level):
        y_offset = i * bar_period2
        if i == 0:
            x_offset = bar_length + 50
        elif i % 5 == 0:
            x_offset = bar_length + 20
        else:
            x_offset = bar_length
        c.add_polygon(
            points=[
                [x_offset, y_offset],
                [0, y_offset],
                [0, y_offset + bar_width],
                [x_offset, y_offset + bar_width],
            ],
            layer=layer2,
        )

        if i % 5 == 0 and i != 0:
            # 添加偏差值
            t = gf.components.text(
                str(spacing * i),
                position=(x_offset + 10, y_offset),
                size=15,
                layer=layer2,
            )
            c.add_ref(t)

    return c
