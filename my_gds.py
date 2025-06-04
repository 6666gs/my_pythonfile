import math
from math import pi

import gdsfactory as gf
import numpy as np
from gdsfactory.component import Component
from gdsfactory.generic_tech import LAYER


def add_2x2MMI_1(
    core_length: float = 87,
    core_width: float = 10.2,
    separation: float = 3.24,
    taper_length: float = 30,
    taper_width: float = 2.6,
    end_width: float = 1.17,
    layer: tuple = LAYER.WG,
):
    '''
    添加一个可供结构化组成一个整体的MMI，其有四个接口，分别为o1,o2,o3,o4。左上、右上、右下、左下
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
    添加的版图默认为LAYER.WG层，默认为负胶版图。
    如需正胶版图，需要使用Region取轮廓
    core_length=87                     #多模区域长128um
    core_width=9.8                      #多模区域宽12um
    separation=3.4                      #taper间距
    taper_length=30                     #taper长度
    taper_width=1.8                     #taper宽度
    wg_width=1.1                        #单模波导宽度
    :return: c
    '''
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


def add_1x2MMI_1(
    core_length: float = 23.6,
    core_width: float = 5.7,
    separation: float = 3,
    taper_length: float = 15,
    taper_width: float = 1.68,
    end_width: float = 1,
    layer: tuple = LAYER.WG,
):
    '''
    添加一个1x2MMI,左侧只有一个端口，o1;右侧两个端口，上o2，下o3
                        o2
                        |
                        |
    o1 --------MMI----------
                        |
                        |
                        o3
    :param core_length: 多模波导长度
    :param core_width: 多模波导宽度
    :param separation: taper间距
    :param taper_length: taper长度
    :param taper_width: taper宽度
    :param end_width: taper末端宽度
    :param layer: 层
    :return:
    '''
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
    添加一个光栅耦合器，适合进行结构体装配，针对TE偏振
    为光栅耦合器添加port，名字为o1
    默认层为LAYER.WG
    默认为负胶
    :param boxh:
    :param gratingp:
    :param duty:
    :param gcl:
    :param taper_length:
    :param wg_width:
    :return:

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
    添加一个偏振不敏感光栅耦合器
    :param period:
    :param n:
    :param duty_ratio:
    :param angle:
    :param length_taper:
    :param length_end:
    :param wg_width:
    :param layer:
    :return:
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
    c<<add_ring_1
    为c添加一个带上下两波导的微环，并携带四个端口，o1:左上；o2：右上；o3：右下；o4：左下。h1，h2为加热电极端口
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
    默认层为LAYER.WG
    默认为负胶
    :param wg_width:
    :param gap:
    :param L:
    :param Lc:
    :param straight_wg_length:
    :param h_width:
    :return:

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
    wg_width: float = 1,
):
    '''
    添加一个MMI树，层数为num
    :param core_length:MMI
    :param core_width:
    :param separation:
    :param taper_length:
    :param taper_width:
    :param layer:以上全部为MMI的参数
    :param num:MMI树的层数
    :param L_basaer:贝塞尔曲线的横向长度
    :param W_basaer:贝塞尔曲线的宽度
    :param wg_width:单模波导的宽度，顶宽
    :return:
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
        end_width=wg_width,
    )
    wg1 = mmi_pro << add_wg_1(wg_width=wg_width, layer=layer, wg_length=20)
    wg2 = mmi_pro << add_wg_1(wg_width=wg_width, layer=layer, wg_length=20)
    wg3 = mmi_pro << add_wg_1(wg_width=wg_width, layer=layer, wg_length=20)
    wg1.connect('o1', mmi1.ports['o1'])
    wg2.connect('o1', mmi1.ports['o2'])
    wg3.connect('o1', mmi1.ports['o3'])
    mmi_pro.add_port(name='o1', port=wg1.ports['o2'])
    mmi_pro.add_port(name='o2', port=wg2.ports['o2'])
    mmi_pro.add_port(name='o3', port=wg3.ports['o2'])

    ##############################################################################
    mmi1 = c1.add_ref(mmi_pro)
    gc1 = c1 << add_gc_1(wg_width=wg_width, layer=layer)
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
    gc1 = c1 << add_gc_1(wg_width=wg_width, layer=layer)
    gc1.connect('o1', basaer_2.ports['o2'])
    # N = N  # 级联的MMI数量
    for i in range(num - 1):
        mmi1 = c1.add_ref(mmi_pro)
        mmi1.connect('o1', basaer_1.ports['o2'])
        basaer_1 = c1.add_ref(b1)
        basaer_2 = c1.add_ref(b2)
        basaer_1.connect('o1', mmi1.ports['o2'])
        basaer_2.connect('o1', mmi1.ports['o3'])
        gc1 = c1 << add_gc_1(wg_width=wg_width, layer=layer)
        gc1.connect('o1', basaer_2.ports['o2'])
    gc1 = c1 << add_gc_1(wg_width=wg_width, layer=layer)
    gc1.connect('o1', basaer_1.ports['o2'])
    c1.show()


def add_multi_wg_tlet(
    multi_wg_width: float,
    multi_wg_layer: tuple,
    multi_wg_length: float,
    et_layer: tuple,
    et_separation: float,
    et_width: tuple,
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
        - 电极直通段长度为 multi_wg_length - 350
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
    path2 += gf.path.straight(length=multi_wg_length - 350)
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
        center=((multi_wg_length - 350) / 2, -(et_width[1] / 2 + et_separation / 2)),
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
    添加定向耦合器
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
    :param waveguide_width:
    :param gap:
    :param angle:
    :param radius1:
    :param x_position:
    :param y_position:
    :param dwidth:
    :return:
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
    添加一个euler弯曲的微环
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
    :param gap:
    :param length_x:
    :param length_y:
    :param straight_wg_length:
    :param p:
    :param radius:
    :param wg_width:
    :param layer_wg:
    :return:
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
