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
    :return: 返回字典，键为每列标题
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
    读取csv格式的内容。注意，第一行为表头，因此转换为array时无法包括
    :param file_dir:
    :return:
    '''
    df = pd.read_csv(file_dir, encoding="utf-8", header=0, names=['Wavelength (microns)', 'Effective Index'])
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
    peaks_point, _ = find_peaks(power_spectrum, height=height, distance=distance)  # height参数可以根据实际情况调整
    freorlam = list(freorlam)
    # 计算每个峰值的3dB带宽
    all = []
    peaks = []
    bandwidths = []
    for peak in peaks_point:
        max_power = power_spectrum[peak]
        half_power = max_power -3

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
    peaks_point, _ = find_peaks(power_spectrum, height=height, distance=distance)  # height参数可以根据实际情况调整
    freorlam = list(freorlam)
    # 计算每个峰值的3dB带宽
    all = []
    peaks=[]
    bandwidths=[]
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

    return peaks,bandwidths,all


def find_3db_minus(freorlam, power_spectrum, height=-20, distance=100):
    '''

    :param freorlam:
    :param power_spectrum:
    :param height:
    :param distance:
    :return:
    '''
    # 找到所有峰值
    peaks, _ = find_peaks(-power_spectrum, height=height, distance=distance)  # height参数可以根据实际情况调整
    # 计算每个峰值的3dB带宽
    bandwidths = []
    for peak in peaks:
        min_power = power_spectrum[peak]
        half_power = min_power + 3

        # 找到功率下降到3dB以下的频率范围
        left_idx = np.max(np.where(power_spectrum[:peak] >= half_power)[0])
        right_idx = np.min(np.where(power_spectrum[peak:] >= half_power)[0]) + peak

        bandwidth = freorlam[right_idx] - freorlam[left_idx]
        bandwidths.append((peak, freorlam[peak], bandwidth))

    return bandwidths

def read_csv_arrays(prefile,skiprows,readcoll):
    '''
    读取prefile目录中所有满足格式的文件
    格式要求：
        芯片名_<器件名_序号_%$#X>_stepX_rangeX.csv
        1、其中<>中需要为英文和数字，间隔使用下划线_
        2、实际并不包括<>
        3、<>内的内容加上_stepX_rangeX会变成读取的数组变量名
        4、读取列数由列表readcoll指定
        5、跳过行数由列表skiprows指定
    :param prefile: 数据文件所在文件夹的绝对地址或相对地址，如
                        prefile = r'.\20250516LTdbr_ring'
                        或者
                        prefile = r'E:\onedrive\Project\pycharm\processdata\DBR_RING\20250516LTdbr_ring'
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
            pattern = r"^.*?_(.*?)_step(\d+[a-zA-Z]+)_range(\d+)\.csv$"
            match = re.match(pattern, filename)

            if match:
                core_name = match.group(1).replace('-', '_')
                step_val = match.group(2).lower()  # 统一小写处理（网页8）
                range_val = match.group(3)

                # 构建唯一键名（网页3/网页7）
                base_name = f"{core_name}_step{step_val}_range{range_val}"

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
                        skiprows=skiprows,
                        usecols=readcoll,
                        header=None,
                        engine='python'  # 增强解析兼容性（网页5）
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




'''data=excel_read('.\透镜表.xlsx')
l=data['聚焦透镜后距离']
print(l)

x = db_efficient(-1)
print(x)

x=1000*math.cos(19.5/180*math.pi)+350*math.sin(19.5/180*math.pi)
y=1000*math.sin(19.5/180*math.pi)+350*math.cos(19.5/180*math.pi)
print(1.08/math.cos(19.5/180*math.pi))

print(math.tan(0.8493218*24/math.pi)*0.07)'''
