import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
import os
import string
import re
from scipy.signal import find_peaks

plt.rcParams['font.sans-serif'] = ['SimHei']


def dbm_mW(x):
    '''计算dbm转换为mW'''
    return 10 ** (x / 10)


def mW_dbm(x):
    '''计算mW转换为dbm'''
    return np.log10(x) * 10


def db_efficient(x):
    '''计算db转换为效率'''
    return 10 ** (x / 10)


def efficient_db(x):
    '''计算效率转换为db'''
    return np.log10(x) * 10


def excel_read(file_dir, sheet='Sheet1'):
    '''
    :param file_dir: excel文件地址
    :param sheet: 选取的sheet，例如'Sheet1'
    :return: 返回字典，键为每列标题，值为每列数据的列表。
    '''
    data = pd.read_excel(file_dir, sheet_name=sheet)
    df = pd.DataFrame(data)
    titles = df.columns.values
    dic = {}
    for title in titles:
        dic[title] = df.loc[:, title].values.tolist()
    return dic


def csv_read_to_array(file_dir):
    '''
    :param file_dir: csv文件的路径。
    :return:
        - df_array: 一个NumPy数组，包含csv文件中的数据，第一行表头除外，数据类型为float32。
    '''
    df = pd.read_csv(
        file_dir,
        encoding="utf-8",
        header=0,
        names=['Wavelength (microns)', 'Effective Index'],
    )
    df_array = np.array(df)  # 将pandas读取的数据转化为array
    df_array = np.float32(df_array[1:, :])
    return df_array


def find_3db(freorlam, power_spectrum, height=-20, distance=100):
    '''
    :param freorlam: 频率/Hz或波长/nm
    :param power_spectrum: 功率谱密度，单位为db
    :param height: 过滤峰值大小
    :return:
        - peaks: [峰值的序号]

        - bandwidths: [每个峰值的bandwidth]

        - all:[(peak在横坐标轴的序号,peak在横坐标轴的值,3db bandwidth)]
    '''
    # 找到所有峰值
    peaks_point, _ = find_peaks(
        power_spectrum, height=height, distance=distance
    )  # height参数可以根据实际情况调整
    freorlam = list(freorlam)
    # 计算每个峰值的3dB带宽
    all = []
    peaks = []
    bandwidths = []
    for peak in peaks_point:
        max_power = power_spectrum[peak]
        half_power = max_power - 3

        # 找到功率下降到3dB以下的频率范围
        try:
            left_idx = np.max(np.where(power_spectrum[:peak] <= half_power)[0])
        except:
            left_idx = 0
        try:
            right_idx = np.min(np.where(power_spectrum[peak:] <= half_power)[0]) + peak
        except:
            right_idx = len(power_spectrum) - 1

        bandwidth = freorlam[right_idx] - freorlam[left_idx]
        all.append((peak, freorlam[peak], bandwidth))
        peaks.append(peak)
        bandwidths.append(bandwidth)

    return peaks, bandwidths, all


def find_50(freorlam, power_spectrum, height=0.5, distance=100):
    '''
    :param freorlam: 频率/Hz或波长/nm
    :param power_spectrum: 功率谱密度，单位为幅度大小
    :param height: 过滤峰值大小
    :return:
        - peaks: [峰值的序号]

        - bandwidths: [每个峰值的bandwidth]

        - all:[(peak在横坐标轴的序号,peak在横坐标轴的值,3db bandwidth)]
    '''
    # 找到所有峰值
    peaks_point, _ = find_peaks(
        power_spectrum, height=height, distance=distance
    )  # height参数可以根据实际情况调整
    freorlam = list(freorlam)
    # 计算每个峰值的3dB带宽
    all = []
    peaks = []
    bandwidths = []
    for peak in peaks_point:
        max_power = power_spectrum[peak]
        half_power = max_power / 2

        # 找到功率下降到3dB以下的频率范围
        try:
            left_idx = np.max(np.where(power_spectrum[:peak] <= half_power)[0])
        except:
            left_idx = 0
        try:
            right_idx = np.min(np.where(power_spectrum[peak:] <= half_power)[0]) + peak
        except:
            right_idx = len(power_spectrum) - 1

        bandwidth = freorlam[right_idx] - freorlam[left_idx]
        all.append((peak, freorlam[peak], bandwidth))
        peaks.append(peak)
        bandwidths.append(bandwidth)

    return peaks, bandwidths, all


def read_csv_arrays(prefile, skiprows, readcoll):
    '''
    读取prefile目录中所有满足格式的文件
    格式要求:

        芯片名_<器件名_序号_%$#X>_stepX_rangeX.csv

        1、其中<>中需要为英文和数字，间隔使用下划线_
        2、实际并不包括<>
        3、<>内的内容加上_stepX_rangeX会变成读取的数组变量名
        4、读取列数由列表readcoll指定，对应不同的range
        5、跳过行数由列表skiprows指定，对应不同的range

    :param
        - prefile: 数据文件所在文件夹的绝对地址或相对地址，如
                        prefile = r'.\20250516LTdbr_ring'
                        或者
                        prefile = r'E:\\onedrive\\Project\\pycharm\\processdata\\DBR_RING\\20250516LTdbr_ring'

        - skiprows: 跳过的行数，字典格式，如

        - readcoll: 读取的列数，字典格式
    :return:
            1、每读取一个文件，即返回读取成功信息
            2、返回所有读取内容
            3、所有数组存放在词典data_dict中，通过data_dict['数组名']读取
    '''
    data_dict = {}
    loaded_count = 0
    conflict_counter = {}  # 新增冲突计数器（网页2/网页4）

    for filename in os.listdir(prefile):
        if filename.endswith('.csv'):
            # 增强正则表达式捕获step和range参数（网页1/网页6）
            pattern = r"^.*?_(.*?)_step(\d+(?:\.\d+)?[a-zA-Z]+)_range(\d+)_source(\d+)dbm\.csv$"
            match = re.match(pattern, filename)

            if match:
                core_name = match.group(1).replace('-', '_')
                step_val = match.group(2).lower()  # 统一小写处理（网页8）
                range_val = match.group(3)
                source_val = match.group(4)

                # 构建唯一键名（网页3/网页7）
                base_name = (
                    f"{core_name}_step{step_val}_range{range_val}_source{source_val}"
                )

                # 冲突检测与处理（网页2/网页4）
                if base_name in conflict_counter:
                    conflict_counter[base_name] += 1
                    unique_suffix = f"_{conflict_counter[base_name]}"
                else:
                    conflict_counter[base_name] = 0
                    unique_suffix = ""

                var_name = f"{base_name}{unique_suffix}_array"

                filepath = os.path.join(prefile, filename)
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding="utf-8",
                        skiprows=skiprows[f'{range_val}'],
                        usecols=readcoll[f'{range_val}'],
                        header=None,
                        engine='python',  # 增强解析兼容性（网页5）
                    )
                    data_dict[var_name] = df.to_numpy()
                    loaded_count += 1
                    print(f"已加载: {var_name} ← {filename}")
                except (KeyError, pd.errors.ParserError) as e:
                    print(f"错误: 文件 {filename} 读取失败 ({str(e)})")
            else:
                print(f"警告: 文件名 {filename} 格式不匹配")

    print(f"\n总计导入 {loaded_count} 个文件")
    return data_dict


def plt_ready(n: int = 1, cols: int = 2):
    """预先设置好绘图环境

    Args:
        n (_type_): figure的子图数量
        例如：n=4表示绘图时有4个子图

    Returns:
        - fig axs: 返回绘图的figure和子图数组axs
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 全局字体设置
    plt.rcParams.update(
        {
            'xtick.labelsize': 14,  # X轴刻度字体
            'ytick.labelsize': 14,  # Y轴刻度字体
            'axes.labelsize': 16,  # 坐标轴标签字体
            'axes.titlesize': 18,  # 子图标题字体
        }
    )

    # 创建绘图布局
    # cols = 2  # 每行显示2个子图
    if n == 0:
        return None, None

    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 5))
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
    # 隐藏空白子图
    for j in range(n, len(axs)):
        axs[j].axis('off')
    for j in range(n):
        axs[j].grid(True, alpha=0.3)
    return fig, axs


def select_file_gui():
    """弹出图形界面选择文件并返回文件路径

    该函数使用tkinter库弹出一个简单的文件选择窗口，允许用户通过图形界面选择任意文件（支持所有文件、txt、csv、dat等类型）。
    选择后窗口自动关闭，并返回所选文件的完整路径字符串。如果未选择文件则返回空字符串。

    Returns:
        str: 用户选择的文件路径（字符串），若未选择则返回空字符串。

    Example:
        ```python
        file_path = select_file_gui()
        if file_path:
            print(f"已选择文件: {file_path}")
        else:
            print("未选择文件")
        ```
    """
    import tkinter as tk
    from tkinter import filedialog

    file1name = ""

    def select_file():
        file_path = filedialog.askopenfilename(
            title="选择文件",
            filetypes=[
                ("所有文件", "*.*"),
                ("文本文件", "*.txt"),
                ("CSV文件", "*.csv"),
                ("数据文件", "*.dat"),
            ],
        )
        if file_path:
            nonlocal file1name
            file1name = file_path
            label.config(text=f"已选择文件：{file1name}")
            root.quit()  # 选中后关闭窗口

    root = tk.Tk()
    root.title("文件选择示例")
    root.geometry("400x150")

    button = tk.Button(root, text="选择文件", command=select_file)
    button.pack(pady=20)

    label = tk.Label(root, text="未选择文件")
    label.pack()

    root.mainloop()
    root.destroy()
    return file1name


def ring_T_D(
    L_1: float = 3500e-6,  # [m]
    l_eo_1: float = 0,  # [m]
    l_eo_2: float = 0,  # [m]
    delta_n_eo_1: float = 0,
    delta_n_eo_2: float = 0,
    tau_1: float = 0.8,
    tau_2: float = 0.8,
    sigma: float = 0,  # [db/m]
    lamda1: np.ndarray = np.array([]),  # [m]
    n_e1: np.ndarray = np.array([]),
):
    '''
    添加一个微环

    :param L_1: 周长
    :param delta_n_eo_1: 正对着入射光的电光折射率改变量
    :param delta_n_eo_2:
    :param tau_1: 直通系数，默认上耦合和下耦合相同
    :return:
        - T_1:直通端电场
        - D_1:下载端电场
    '''

    # tau_array_1 = np.ones(len(lamda1)) * tau_1
    # tau_array_2 = np.ones(len(lamda1)) * tau_2
    kai_1 = (1 - tau_1**2) ** 0.5
    kai_2 = (1 - tau_2**2) ** 0.5

    # l_eo_1 = L_1 / 2  # 与入射光正对的半环，设置此时两边加电电压不同
    # l_eo_2 = L_1 / 2
    # delta_n_eo_1 = 0
    # delta_n_eo_2 = delta_n_eo_1 *-(kai_1**2/tau_1**2+1)
    # beta = 2 * pi * n_e1 / lamda1
    beta_eo_1 = 2 * np.pi * (n_e1 + delta_n_eo_1) / lamda1
    beta_eo_2 = 2 * np.pi * (n_e1 + delta_n_eo_2) / lamda1

    alfa = np.exp(-L_1 * sigma)  # 功率损耗系数，sigma单位为1db/m

    # phi = (L_1 - l_eo_1 - l_eo_2) * beta
    phi_1 = l_eo_1 * beta_eo_1
    phi_2 = l_eo_2 * beta_eo_2
    # L_e1 = L_1 * (0.5 + (1 - kai_1[0] ** 2) / kai_1[0] ** 2)
    # print(f'R={L_1*1e6}um微环')
    # print(f'tau_1={tau_1:.3f}')
    # print(f'kai_1={kai_1:.3f}')
    # print(f'tau_2={tau_2:.3f}')
    # print(f'kai_2={kai_2:.3f}')
    # print(f'R={L_1 * 1e6}um微环有效长度：L_e,1={L_e1 * 1e6}um')

    T_1 = (tau_1 - alfa**0.5 * tau_2 * np.exp(-1j * (phi_1 + phi_2))) / (
        1 - alfa**0.5 * tau_1 * tau_2 * np.exp(-1j * (phi_1 + phi_2))
    )  # 直输出端传递函数
    D_1 = (-((alfa**0.5) ** 0.5) * kai_1 * kai_2 * np.exp(-1j * phi_1)) / (
        1 - alfa**0.5 * tau_1 * tau_2 * np.exp(-1j * (phi_1 + phi_2))
    )  # 下载端传递函数

    phi_T_1 = np.angle(T_1)
    A_T_1 = np.abs(T_1)
    phi_D_1 = np.angle(D_1)
    A_D_1 = np.abs(D_1)
    return phi_T_1, A_T_1, phi_D_1, A_D_1
