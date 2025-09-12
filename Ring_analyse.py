'''
## 对微环进行分析的类。
### Original author: Mozhenwu
### Second author: Wuxiao
### Date: 2025-09-12
### Version:0.1

### 功能
1. 计算自由光谱范围FSR
2. 计算Q因子，包括负载Q、本征Q、功率耦合系数、波导损耗
3. 绘制波长域和频谱域透射谱和下载谱
4. 计算色散参数D1、D2、beta2、积分色散

### 示例代码

import Ring_analyse as Ra
import my_math as mm

variable = {
    'variable': None,
    'file': r"E:\20250902LTliupian\H1\H1_sil_1_zhitong_1545_1555_step0.1pm_range2_source0dbm.csv",
    'mode': 'T',
    'type': 'with drop',
    'reference': r"E:\20250902LTliupian\H0_ref_0_0_1500_1630_step1pm_range2_source0dbm.csv",
    'nefile': r"E:\LTOIy_ne_400_240_1um.mat",
    'ngfile': r"E:\LTOIy_ng_400_240_1um.mat",
    'L': 3000e-6,
}

Ring1 = Ra.Ring(variable)
Ring1.cal_Q(holdon=False)
Ring1.cal_fsr(range_nm=(1530, 1550))
Ring1.plot_lambda(range_nm=(1548, 1550))
Ring1.cal_D()

'''

from scipy.constants import c
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import mat73
from typing import Tuple, Union
from matplotlib.figure import Figure


def plt_ready(
    n: int = 1, cols: int = 2, figsize=(8, 5)
) -> Union[Tuple[None, None], Tuple[Figure, Union[np.ndarray, list]]]:
    """预先设置好绘图环境
    Args:
        n: figure的子图数量
        cols: 每行显示的子图数量

    Returns:
        (fig, axs): 返回绘图的figure和子图数组axs
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
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
    fig, axs = plt.subplots(rows, cols, figsize=(cols * figsize[0], rows * figsize[1]))
    axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]
    # 隐藏空白子图
    for j in range(n, len(axs)):
        axs[j].axis('off')
    for j in range(n):
        axs[j].grid(True, alpha=0.3)
    return fig, axs


def decimal_places(num):
    '''
    获取小数的有效位数
    '''
    from decimal import Decimal

    try:
        return abs(Decimal(str(float(num))).as_tuple().exponent)  # type:ignore
    except ValueError:
        raise TypeError("The input must be a numeric value.")


class Ring:
    T: np.ndarray | None = None  # [db]
    D: np.ndarray | None = None  # [db]
    fre: np.ndarray | None = None  # [THz]
    lamda: np.ndarray | None = None  # [nm]
    type: str = 'unknown'  # 是否有下载端，有则为 'with_drop'，无则为 'no_drop'

    lambda0: np.ndarray | None = None
    lambda_step: float | None = None  # 波长步长
    ne: dict | None = None  # 有效折射率数据
    ng: dict | None = None  # 群折射率数据
    L: float | None = None  # 微环周长，单位米
    fsr_mean: float | None = None  # 平均自由光谱范围，单位nm

    def __init__(self, variable):
        '''
        ### 对微环进行分析的类。
        ### Original author: Mozhenwu
        ### Second author: Wuxiao
        ### Date: 2025-09-12
        ### Version:0.1

        ### 功能
        1. 计算自由光谱范围FSR
        2. 计算Q因子，包括负载Q、本征Q、功率耦合系数、波导损耗
        3. 绘制波长域和频谱域透射谱和下载谱
        4. 计算色散参数D

        初始化时，variable可以有两种方式提供数据：
        1. 直接提供variable字典，包含所有需要的变量。
        2. 提供file、mode、type参数，从文件中读取数据。
        3. 若选择方式1，则file、mode、type应设置为None。反之亦然。
        另外，还需要提供reference、nefile、ngfile和L参数。****nefile、ngfile、L必须提供****
        Args:
            variable: 字典，包含初始化所需的变量，需要包括:{'variable', 'file', 'mode', 'type', 'reference', 'nefile', 'ngfile', 'L'}
                - variable: dict or None, 包含初始化所需的变量
                - file: str or None, 透射谱或下载谱文件路径
                - mode: str or None, 'T'表示透射谱，'D'表示下载谱
                - type: str or None, 'with drop'表示有下载端，'without drop'表示无下载端
                - reference: str or None, 参考记录文件路径，用于去除系统误差
                - nefile: str, 有效折射率数据文件路径，从.mat文件中读取，该文件通过lumerical mode solver的频率扫描导出
                - ngfile: str, 群折射率数据文件路径，从.mat文件中读取，该文件通过lumerical mode solver的频率扫描导出
                - L: float, 微环周长，单位[米]
        '''

        # 检测输入变量内容是否满足要求
        def check_variable_keys(variable, required_keys):
            missing = [k for k in required_keys if k not in variable]
            if missing:
                raise ValueError(f"variable缺少以下键: {missing}")

        required_keys = [
            'variable',
            'file',
            'mode',
            'type',
            'reference',
            'nefile',
            'ngfile',
            'L',
        ]
        check_variable_keys(variable, required_keys)

        def get_all_class_attrs(cls):
            attrs = set()
            for base in cls.__bases__:
                attrs.update(get_all_class_attrs(base))
            attrs.update(
                {
                    k
                    for k, v in cls.__dict__.items()
                    if not callable(v) and not k.startswith('__')
                }
            )
            return attrs

        if variable['variable'] is None:
            if (
                variable['file'] is None
                or variable['type'] is None
                or variable['mode'] is None
            ):
                raise ValueError(
                    r"请提供variable或file\mode\type参数，两种方式至少完整提供一个。"
                )
            if os.path.isfile(variable['file']):
                data = pd.read_csv(
                    variable['file'],
                    encoding="utf-8",
                    skiprows=list(range(0, 15)),
                    usecols=[0, 2],
                    header=None,
                    engine='python',
                )
                self.lamda = np.array(data[0].values)
                if variable['mode'] == 'T':
                    self.T = np.array(data[2].values)
                elif variable['mode'] == 'D':
                    self.D = np.array(data[2].values)
                else:
                    raise ValueError("mode参数错误，仅支持 'T' 或 'D'。")
                self.type = variable['type']
            else:
                raise ValueError(f"文件 {variable['file']} 不存在，请检查路径。")
        elif variable['variable'] is not None:
            if not isinstance(variable['variable'], dict):
                raise TypeError("variable参数必须是字典类型。")
            # 获取所有父类的属性
            class_attrs = get_all_class_attrs(self.__class__)
            instance_attrs = set(self.__dict__.keys())
            keys = instance_attrs | class_attrs  # keys即所有属性

            # 初始化属性
            for key in keys:
                if key in variable['variable']:
                    setattr(self, key, variable['variable'][key])

        if self.lamda is not None and self.fre is None:
            self.fre = c / (self.lamda * 1e-9) / 1e12
        elif self.fre is not None and self.lamda is None:
            self.lamda = c / (self.fre * 1e12) * 1e9

        if variable['reference'] is not None:
            '''
            reference中为参考记录的插损
            将微环的所有数据减去该插损
            '''
            if os.path.isfile(variable['reference']):
                ref = {}
                data = pd.read_csv(
                    variable['reference'],
                    encoding="utf-8",
                    skiprows=list(range(0, 15)),
                    usecols=[0, 2],
                    header=None,
                    engine='python',
                )
                ref_lambda = np.array(data[0].values)
                ref_power = np.array(data[2].values)
                if self.lamda is not None:
                    start = self.lamda.min()
                    end = self.lamda.max()
                else:
                    raise ValueError("微环数据不存在波长信息，无法匹配参考数据。")
                ref_lambda_step = np.mean(np.diff(ref_lambda))
                self.lamda_step = np.mean(np.diff(self.lamda))
                if ref_lambda_step != self.lamda_step:
                    f_interp = interp1d(ref_lambda, ref_power, kind='cubic')
                    ref_lambda = np.round(
                        np.arange(start, end + 1e-6, self.lamda_step),
                        decimal_places(self.lamda_step),
                    )
                    ref_power = f_interp(ref_lambda)

                if start < ref_lambda.min() or end > ref_lambda.max():  # type:ignore
                    raise ValueError(
                        f"参考数据无效：微环数据的波长范围超出参考数据波长的范围。\nRing:{start} - {end} nm, Ref data range: {ref_lambda.min()} - {ref_lambda.max()} nm"  # type:ignore
                    )
                mask = (ref_lambda >= start) & (ref_lambda <= end)

                ref_lambda = ref_lambda[mask]
                ref_power = ref_power[mask]

                if self.T is not None:
                    self.T = self.T - ref_power
                if self.D is not None:
                    self.D = self.D - ref_power

            else:
                print(f"文件 {variable['reference']} 不存在，请检查路径。")

        self.ne = {}
        self.ng = {}
        if variable['nefile'] is None or variable['ngfile'] is None:
            raise ValueError("请提供有效折射率和群折射率数据文件路径。")
        else:
            data = list(mat73.loadmat(variable['nefile']).values())[0]
            self.ne['lamda'] = np.round(data['x0'] * 1000, 4)
            self.ne['ne'] = data['y0']

            data = list(mat73.loadmat(variable['ngfile']).values())[0]
            self.ng['lamda'] = np.round(data['x0'] * 1000, 4)
            self.ng['ng'] = data['y0']
        if variable['L'] is None:
            raise ValueError("请提供微环周长L，单位米。")
        else:
            self.L = variable['L']

    def cal_fsr(self, range_nm=None, display=True):
        '''
        根据透射谱或下载段谱计算自由光谱范围（FSR），并绘制FSR随波长和频率的变化图。
        Args:
            range_nm: 计算FSR以及最后显示的波长范围，格式为(start, end)，单位nm
            display: 是否显示FSR随波长和频率的变化图
        '''

        if self.fre is not None and self.lamda is not None:
            lamda = self.lamda
            fre = self.fre
            T = self.T
            D = self.D
            if range_nm is not None:
                start, end = range_nm
                mask = (self.lamda >= start) & (self.lamda <= end)
                lamda = self.lamda[mask]
                fre = self.fre[mask]
                if self.T is not None:
                    T = self.T[mask]
                if self.D is not None:
                    D = self.D[mask]
            assert self.L is not None and self.ng is not None
            fsr_theory = (
                np.mean(self.lamda * 1e-9) ** 2
                / (self.L * np.mean(self.ng['ng']))
                * 1e9
            )  # 理论FSR，单位nm
            step_size = np.mean(np.diff(lamda))
            distance_pts = int(fsr_theory * 0.7 / step_size)
            if T is not None:
                peaks, properties = find_peaks(-T, distance=distance_pts, prominence=2)
                bandwidths = []
                for peak in peaks:
                    max_power = -T[peak]
                    half_power = max_power - 3

                    # 找到功率下降到3dB以下的频率范围
                    try:
                        left_idx = np.max(np.where(-T[:peak] <= half_power)[0])
                    except:
                        left_idx = 0
                    try:
                        right_idx = np.min(np.where(-T[peak:] <= half_power)[0]) + peak
                    except:
                        right_idx = len(-T) - 1

                    bandwidth = lamda[right_idx] - lamda[left_idx]
                    bandwidths.append(bandwidth)
                self.T_bandwidths_mean = np.mean(np.array(bandwidths))
            elif D is not None:
                peaks, properties = find_peaks(D, distance=distance_pts, prominence=2)

            lambda_peaks = lamda[peaks]
            self.fsr_mean = float(np.mean(np.abs(np.diff(lambda_peaks))))
            self.lambda0 = lambda_peaks
            fre_peaks = fre[peaks]
            fsr_lambda = np.abs(np.diff(lambda_peaks))
            fsr_fre = np.abs(np.diff(fre_peaks))
            if display:
                fig, axes = plt_ready(4, 2, figsize=(8, 6))
                if fig is not None and axes is not None:
                    ax1, ax2, ax3, ax4 = axes
                    ax1.plot(lambda_peaks[:-1], fsr_lambda, 'o-')
                    ax1.set_xlabel('Wavelength (nm)')
                    ax1.set_ylabel('FSR (nm)')
                    ax1.set_title('Free Spectral Range vs Wavelength')
                    ax1.grid(True)

                    ax2.plot(fre_peaks[:-1], fsr_fre, 'o-')
                    ax2.set_xlabel('Frequency (THz)')
                    ax2.set_ylabel('FSR (THz)')
                    ax2.set_title('Free Spectral Range vs Frequency')
                    ax2.grid(True)
                    if T is not None:
                        ax3.plot(lamda, T, label='Transmission Spectrum')
                        ax3.plot(lambda_peaks, T[peaks], 'ro', label='Resonance Peaks')
                        ax3.axhline(
                            np.max(T),
                            color='gray',
                            linestyle='--',
                            label='Max Transmission',
                        )
                        ax3.text(
                            np.max(lamda),  # 横坐标设为右侧
                            np.max(T),
                            f'{np.max(T):.2f}',
                            va='center',
                            ha='left',
                            color='gray',
                            fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                        )
                        ax3.set_xlabel('Wavelength (nm)')
                        ax3.set_ylabel('dB')

                        ax4.plot(fre, T, label='Transmission Spectrum')
                        ax4.plot(fre_peaks, T[peaks], 'ro', label='Resonance Peaks')
                        ax4.set_xlabel('Frequency (THz)')
                        ax4.set_ylabel('dB')
                        ax4.axhline(
                            np.max(T),
                            color='gray',
                            linestyle='--',
                            label='Max Transmission',
                        )
                        ax4.text(
                            np.max(fre),  # 横坐标设为右侧
                            np.max(T),
                            f'{np.max(T):.2f}',
                            va='center',
                            ha='left',
                            color='gray',
                            fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
                        )
                    elif D is not None:
                        ax3.plot(lamda, D, label='Dispersion Spectrum')
                        ax3.plot(lambda_peaks, D[peaks], 'ro', label='Resonance Peaks')
                        ax3.set_xlabel('Wavelength (nm)')
                        ax3.set_ylabel('dB')

                        ax4.plot(fre, D, label='Dispersion Spectrum')
                        ax4.plot(fre_peaks, D[peaks], 'ro', label='Resonance Peaks')
                        ax4.set_xlabel('Frequency (THz)')
                        ax4.set_ylabel('dB')
                else:
                    print("Error creating plots.")
        else:
            print("Frequency and wavelength data are required to calculate FSR.")

    def cal_Q(self, holdon=False):
        '''
        根据透射谱计算Q因子。
        Args:
            holdon: 是否保留每个峰的拟合图像
        '''

        def lorentzian(lambda_, T0, A, lambda0, gamma):
            return T0 - A / (1 + ((lambda_ - lambda0) / gamma) ** 2)

        if self.type == 'without drop':
            '''
            没有下载端，只有直通的
            以后再写
            '''
            pass
        elif self.type == 'with drop':
            '''
            有下载端
            '''
            if self.fre is not None and self.lamda is not None:
                if self.T is None:
                    '''
                    利用透射谱来计算Q因子
                    '''
                    print("Transmission data (T) is required to calculate Q-factor.")
                    return
                else:
                    if self.fsr_mean is None:
                        self.cal_fsr(display=False)
                    assert self.fsr_mean is not None
                    step_size = np.mean(np.diff(self.lamda))
                    distance_pts = int(self.fsr_mean * 0.7 / step_size)
                    peaks, properties = find_peaks(
                        -self.T, distance=distance_pts, prominence=2
                    )
                    fit_results = []
                    range_nm = self.T_bandwidths_mean * 2
                    for peak in peaks:
                        # 选取拟合窗口范围
                        lambda0_guess = self.lamda[peak]
                        # range_nm = 0.05
                        mask = (self.lamda >= lambda0_guess - range_nm) & (
                            self.lamda <= lambda0_guess + range_nm
                        )
                        lambda_slice = self.lamda[mask]
                        T_slice = self.T[mask]
                        T_slice = 10 ** (T_slice / 10)

                        # 初始猜测参数
                        T0_guess = max(T_slice)
                        A_guess = T0_guess - min(T_slice)
                        lambda0_guess = self.lamda[peak]
                        gamma_guess = 0.002

                        figs = []
                        try:
                            popt, pcov = curve_fit(
                                lorentzian,
                                lambda_slice,
                                T_slice,
                                p0=[T0_guess, A_guess, lambda0_guess, gamma_guess],
                            )

                            # 计算每个极值点附近的FSR
                            if peak == peaks[0]:
                                fsr = self.lamda[peaks[1]] - self.lamda[peaks[0]]
                            elif peak == peaks[-1]:
                                fsr = self.lamda[peaks[-1]] - self.lamda[peaks[-2]]
                            else:

                                lamda1 = self.lamda[
                                    peaks[peaks.tolist().index(peak) + 1]
                                ]
                                lamda2 = self.lamda[
                                    peaks[peaks.tolist().index(peak) - 1]
                                ]
                                fsr = (lamda1 - lamda2) / 2

                            # 保存结果
                            fit_results.append(
                                {
                                    'lambda0': popt[2],
                                    'gamma': 2 * popt[3],
                                    'kappa2': np.pi * 2 * popt[3] / fsr,
                                    'Ql': popt[2] / (2 * popt[3]),
                                    'Qi': popt[2]
                                    / (2 * popt[3])
                                    * np.sqrt(1 / (1 - popt[1] / popt[0])),
                                    'params': popt,
                                }
                            )

                            # 可视化拟合结果
                            T_fit = lorentzian(lambda_slice, *popt)
                            fig, ax = plt.subplots()
                            ax.plot(
                                lambda_slice,
                                T_slice / popt[0],
                                'bo',
                                label='实验数据',
                            )
                            ax.plot(
                                lambda_slice,
                                T_fit / popt[0],
                                'r-',
                                label='洛伦兹拟合',
                            )
                            ax.grid(True)
                            figs.append(fig)
                        except RuntimeError:
                            print(f'峰 @ {self.lamda[peak]:.2f} nm 拟合失败')
                    if holdon is False:
                        plt.close('all')
                        del figs

                    # 处理kappa2, Ql, Qi，去除异常值
                    kappa2_list = [res["kappa2"] for res in fit_results]  # type:ignore
                    kappa2_list = [
                        kappa2
                        for kappa2 in kappa2_list
                        if kappa2 is not np.inf and kappa2 > 0 and kappa2 < 1
                    ]
                    data = np.array(kappa2_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    kappa2_filtered = data[
                        np.abs(data - mean) < 4 * std
                    ]  # 只保留在均值±3σ范围内的数据

                    Ql_list = [res["Ql"] for res in fit_results]  # type:ignore
                    Ql_list = [ql for ql in Ql_list if ql is not np.inf and ql > 0]
                    data = np.array(Ql_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    Ql_filtered = data[
                        np.abs(data - mean) < 4 * std
                    ]  # 只保留在均值±3σ范围内的数据

                    Qi_list = [res["Qi"] for res in fit_results]  # type:ignore
                    Qi_list = [qi for qi in Qi_list if qi is not np.inf and qi > 0]
                    data = np.array(Qi_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    Qi_filtered = data[
                        np.abs(data - mean) < 4 * std
                    ]  # 只保留在均值±3σ范围内的数据

                    # 计算损耗参数α
                    start = self.lamda.min()
                    end = self.lamda.max()

                    assert self.ne is not None
                    ref_lambda = self.ne['lamda']
                    ref_ne = self.ne['ne']
                    ref_lambda_step = np.mean(np.diff(ref_lambda))
                    if ref_lambda_step != self.lamda_step:
                        f_interp = interp1d(ref_lambda, ref_ne, kind='cubic')
                        ref_lambda = np.round(
                            np.arange(start, end + 1e-6, self.lamda_step),
                            decimal_places(self.lamda_step),
                        )
                        ref_ne = f_interp(ref_lambda)
                    if start < ref_lambda.min() or end > ref_lambda.max():
                        raise ValueError(
                            f"有效折射率数据无效：微环数据的波长范围超出有效折射率数据波长的范围。\nRing:{start} - {end} nm, ne data range: {ref_lambda.min()} - {ref_lambda.max()} nm"
                        )
                    mask = (ref_lambda >= start) & (ref_lambda <= end)
                    ref_lambda = ref_lambda[mask]
                    ref_ne = ref_ne[mask]
                    Qi_list = [res["Qi"] for res in fit_results]  # type:ignore
                    lamda0_list = [res["lambda0"] for res in fit_results]  # type:ignore

                    alpha_list = []
                    for Qi, lamda0 in zip(Qi_list, lamda0_list):
                        n_eff = ref_ne[
                            np.abs(ref_lambda - lamda0).argmin()
                        ]  # 有效折射率
                        lamda0 = lamda0 * 1e-9  # 转换为米
                        A = np.pi * n_eff * self.L / lamda0
                        x = Qi / A

                        # 解二次方程 x*y**2 + y - x = 0
                        a = x
                        b = 1
                        c = -x

                        # 求根公式
                        y1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
                        y2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

                        # 只取正根
                        y = y1 if y1 > 0 else y2
                        alpha = y**2
                        alpha_1 = -20 * np.log10(alpha) / self.L
                        alpha_list.append(alpha_1 / 100)
                    # 筛选
                    alpha_list = [
                        alpha
                        for alpha in alpha_list
                        if alpha is not np.inf and alpha > 0
                    ]
                    data = np.array(alpha_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    alpha_filtered = data[
                        np.abs(data - mean) < 5 * std
                    ]  # 只保留在均值±3σ范围内的数据

                    fig, axes = plt_ready(4, 2, figsize=(8, 6))
                    if fig is not None and axes is not None:
                        ax1, ax2, ax3, ax4 = axes
                        ax1.hist(
                            Ql_filtered,
                            bins=20,
                            color='mediumseagreen',
                            edgecolor='black',
                        )
                        ax1.set_title('Loaded Q-factor Distribution')
                        ax1.set_xlabel('Ql')
                        ax1.set_ylabel('Count')
                        ax1.grid(True)

                        ax2.hist(
                            Qi_filtered,
                            bins=20,
                            color='steelblue',
                            edgecolor='black',
                        )
                        ax2.set_title('Intrinsic Q-factor Distribution')
                        ax2.set_xlabel('Qi')
                        ax2.set_ylabel('Count')
                        ax2.grid(True)

                        ax3.hist(
                            kappa2_filtered,
                            bins=20,
                            color='coral',
                            edgecolor='black',
                        )
                        ax3.set_title('Coupling Coefficient (kappa^2) Distribution')
                        ax3.set_xlabel('kappa^2')

                        ax4.hist(
                            alpha_filtered,
                            bins=20,
                            color='gold',
                            edgecolor='black',
                        )
                        ax4.set_title('Loss Coefficient (alpha) Distribution')
                        ax4.set_xlabel('alpha db/cm')
                        ax4.set_ylabel('Count')
                        ax4.grid(True)
            else:
                print(
                    "Frequency and wavelength data are required to calculate Q-factor."
                )

    def plot_lambda(self, range_nm):
        '''
        横轴为波长，绘制透射谱和下载谱
        Args:
            range_nm: 显示的波长范围，格式为(start, end)，单位nm
        '''
        if self.lamda is None:
            print("Frequency data is not available.")
        else:
            lamda = self.lamda
            T = self.T
            D = self.D
            if range_nm is not None:
                start, end = range_nm
                mask = (self.lamda >= start) & (self.lamda <= end)
                lamda = self.lamda[mask]
                T = self.T[mask] if self.T is not None else None
                D = self.D[mask] if self.D is not None else None
            # 统计非None的数量
            plot_data = []
            if T is not None:
                plot_data.append(('Transmission Spectrum', lamda, T, 'b'))
            if D is not None:
                plot_data.append(('Drop Spectrum', lamda, D, 'orange'))
            n = len(plot_data)
            if n == 0:
                print("No data to plot.")
                return
            fig, axes = plt_ready(n, n, figsize=(8, 5))
            if fig is not None and axes is not None:
                for i, (title, x, y, color) in enumerate(plot_data):
                    ax = axes[i]
                    ax.plot(x, y, label=title, color=color)
                    ax.set_xlabel('Lambda (nm)')
                    ax.set_ylabel('dB')
                    ax.set_title(title)
                    ax.grid(True)
                    ax.legend()

    def plot_fre(self, range_THz):
        '''
        横轴为频率，绘制透射谱和下载谱
        Args:
            range_THz: 显示的频率范围，格式为(start, end)，单位THz
        '''
        if self.fre is None:
            print("Frequency data is not available.")
        else:
            fre = self.fre
            T = self.T
            D = self.D
            if range_THz is not None:
                start, end = range_THz
                mask = (self.fre >= start) & (self.fre <= end)
                fre = self.fre[mask]
                T = self.T[mask] if self.T is not None else None
                D = self.D[mask] if self.D is not None else None

            # 统计非None的数量
            plot_data = []
            if T is not None:
                plot_data.append(('Transmission Spectrum', fre, T, 'b'))
            if D is not None:
                plot_data.append(('Drop Spectrum', fre, D, 'orange'))
            n = len(plot_data)
            if n == 0:
                print("No data to plot.")
                return
            fig, axes = plt_ready(n, n, figsize=(8, 5))
            if fig is not None and axes is not None:
                for i, (title, x, y, color) in enumerate(plot_data):
                    ax = axes[i]
                    ax.plot(x, y, label=title, color=color)
                    ax.set_xlabel('Frequency (THz)')
                    ax.set_ylabel('dB')
                    ax.set_title(title)
                    ax.grid(True)
                    ax.legend()

    def cal_D(self):
        '''
        根据透射谱或下载谱计算色散参数D。
        '''
        if self.fre is None or self.lamda is None:
            print("Frequency and wavelength data are required to calculate dispersion.")
            return
        if self.T is None:
            print("Transmission data (T) is required to calculate dispersion.")
            return

        assert self.fre is not None and self.lamda is not None and self.T is not None

        self.cal_fsr(display=False)
        if self.lambda0 is None:
            print("Resonant wavelength (lambda0) is required to calculate dispersion.")
            return
        else:
            omega0_array = 2 * np.pi * c / (self.lambda0 * 1e-9)  # 转换为rad/s
            omega0_array = np.flip(omega0_array)
            frequency0_array = omega0_array / (2 * np.pi) / 1e12  # 转换为THz

            N = len(omega0_array)
            mu_array = np.arange(-(N // 2), N // 2 + (N % 2))

            # 拟合 ωμ = ω0 + D1μ + ½ D2μ² + ⅙ D3μ³
            coeffs = np.polyfit(mu_array, omega0_array, 3)
            omega_fit = np.polyval(coeffs, mu_array)
            omega0 = coeffs[3]
            D1 = coeffs[2]
            D2 = 2 * coeffs[1]
            ng = np.mean(self.ng['ng'])  # type:ignore
            beta2 = -D2 * ng / c / D1**2
            print(f'D1 = {D1 / (2 * np.pi * 1e9)} GHz ')
            print(f'D2 = {D2/(2 * np.pi * 1e6)} MHz')
            print(f'β2 = {beta2} s^2/m')

            Dint_array = omega0_array - (omega0 + D1 * mu_array)
            Dint = Dint_array / D1

            fig, axes = plt_ready(1, 1, figsize=(8, 6))
            if fig is not None and axes is not None:
                (ax1,) = axes
                ax1.plot(frequency0_array, Dint, 'o-')
                ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax1.set_title('Integrated Dispersion')
                ax1.set_xlabel('Mode Number (μ)')
                ax1.set_ylabel('Dint / D1')
                ax1.grid(True)
