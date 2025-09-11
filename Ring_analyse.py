'''
Original author: Mozhenwu
Second author: Wuxiao
Date: 2025-09-10
Version:0.1
    目前，仅支持有下载端的微环数据处理，所含的功能包括：
    1. 计算Q因子，包括负载Q、本征Q
    2. 计算功率耦合系数
    3. 计算色散参数D
    4. 计算损耗参数α
    5. 绘制透射谱和下载谱
    6. 计算自由光谱范围FSR
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
    fit_results: list | None = None
    lambda_step: float | None = None  # 波长步长

    def __init__(self, variable, file=None, type=None, reference=None):
        '''
        初始化函数。
        Args:
            variable: 字典，包含初始化所需的变量，需要包括:{'fre', 'lamda', 'T', 'D', 'type'}，其中fre和lamda至少需要一个
            file: 可选，Santec扫谱系统Raw Data保存的CSV文件路径，用于加载数据
            type: 可选，字符串，指明是否有下载端，有则为 'with drop'，无则为 'without drop'
            reference: 可选，Santec扫谱系统Raw Data的Reference参考文件路径，用于加载参考数据并进行插损校正
        '''

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

        if file is not None:
            if type is None:
                raise ValueError("请提供type参数，指明是否有下载端。")
            if os.path.isfile(file):
                data = pd.read_csv(
                    file,
                    encoding="utf-8",
                    skiprows=list(range(0, 15)),
                    usecols=[0, 2],
                    header=None,
                    engine='python',
                )
                variable['lamda'] = data[0].values
                variable['T'] = data[2].values
                variable['type'] = type
            else:
                raise ValueError(f"文件 {file} 不存在，请检查路径。")

        # 获取所有父类的属性
        class_attrs = get_all_class_attrs(self.__class__)
        instance_attrs = set(self.__dict__.keys())
        keys = instance_attrs | class_attrs  # keys即所有属性

        # 初始化属性
        for key in keys:
            if key in variable:
                setattr(self, key, variable[key])
        if self.lamda is not None and self.fre is None:
            self.fre = c / (self.lamda * 1e-9) / 1e12
        elif self.fre is not None and self.lamda is None:
            self.lamda = c / (self.fre * 1e12) * 1e9

        if reference is not None:
            '''
            reference中为参考记录的插损
            将微环的所有数据减去该插损
            '''
            if os.path.isfile(reference):
                ref = {}
                data = pd.read_csv(
                    reference,
                    encoding="utf-8",
                    skiprows=list(range(0, 15)),
                    usecols=[0, 2],
                    header=None,
                    engine='python',
                )
                ref['lamda'] = data[0].values
                ref['T'] = data[2].values
                if self.lamda is not None:
                    start = self.lamda.min()
                    end = self.lamda.max()
                else:
                    raise ValueError("微环数据不存在波长信息，无法匹配参考数据。")
                if (
                    start < ref['lamda'].min()  # type:ignore
                    or end > ref['lamda'].max()  # type:ignore
                ):
                    raise ValueError(
                        f"参考数据无效：微环数据的波长范围超出参考数据波长的范围。\nRing:{start} - {end} nm, Ref data range: {ref['lamda'].min()} - {ref['lamda'].max()} nm"  # type:ignore
                    )
                mask = (ref['lamda'] >= start) & (ref['lamda'] <= end)
                self.lamda_step = np.mean(np.diff(self.lamda))
                ref_lambda = ref['lamda'][mask]
                ref_power = ref['T'][mask]
                ref_lambda_step = np.mean(np.diff(ref_lambda))
                if ref_lambda_step != self.lamda_step:
                    f_interp = interp1d(ref_lambda, ref_power, kind='cubic')
                    ref_lambda = np.round(
                        np.arange(start, end + 1e-6, self.lamda_step),
                        decimal_places(self.lamda_step),
                    )
                    ref_power = f_interp(ref_lambda)
                if self.T is not None:
                    self.T = self.T - ref_power
                if self.D is not None:
                    self.D = self.D - ref_power

            else:
                print(f"文件 {reference} 不存在，请检查路径。")

    def cal_fsr(self, fsr_nm_g, range_nm):
        '''
        计算自由光谱范围（FSR），并绘制FSR随波长和频率的变化图。
        Args:
            fsr_nm_g: 理论FSR，单位nm
            range_nm: 计算FSR以及最后显示的波长范围，格式为(start, end)，单位nm
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

            delta_lambda_min = fsr_nm_g  # 单位: nm
            step_size = np.mean(np.diff(lamda))
            distance_pts = int(delta_lambda_min / step_size)
            if T is not None:
                peaks, properties = find_peaks(-T, distance=distance_pts, prominence=2)
            elif D is not None:
                peaks, properties = find_peaks(D, distance=distance_pts, prominence=2)
            lambda_peaks = lamda[peaks]
            fre_peaks = fre[peaks]
            fsr_lambda = np.abs(np.diff(lambda_peaks))
            fsr_fre = np.abs(np.diff(fre_peaks))

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

                    ax4.plot(fre, T, label='Transmission Spectrum')
                    ax4.plot(fre_peaks, T[peaks], 'ro', label='Resonance Peaks')
            else:
                print("Error creating plots.")
        else:
            print("Frequency and wavelength data are required to calculate FSR.")

    def cal_Q(self, delta_lambda_min=0.3, range_nm=0.05, holdon=False, display=True):
        '''
        根据透过射谱计算Q因子。
        Args:
            delta_lambda_min: fsr间距，单位nm，用于峰值检测
            range_nm: 拟合窗口范围，单位nm
            holdon: 是否保留每个峰的拟合图像
            display: 是否显示结果图像
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
                    # delta_lambda_min = 0.3  # 单位: nm
                    step_size = np.mean(np.diff(self.lamda))
                    distance_pts = int(delta_lambda_min / step_size)
                    peaks, properties = find_peaks(
                        -self.T, distance=distance_pts, prominence=2
                    )
                    self.fit_results = []

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
                            self.fit_results.append(
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
                            if display:
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
                    if display:
                        if holdon is False:
                            plt.close('all')
                            del figs

                    # 处理kappa2, Ql, Qi，去除异常值
                    kappa2_list = [res["kappa2"] for res in self.fit_results]
                    kappa2_list = [
                        kappa2
                        for kappa2 in kappa2_list
                        if kappa2 is not np.inf and kappa2 > 0
                    ]
                    data = np.array(kappa2_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    kappa2_filtered = data[
                        np.abs(data - mean) < 5 * std
                    ]  # 只保留在均值±3σ范围内的数据

                    Ql_list = [res["Ql"] for res in self.fit_results]
                    Ql_list = [ql for ql in Ql_list if ql is not np.inf and ql > 0]
                    data = np.array(Ql_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    Ql_filtered = data[
                        np.abs(data - mean) < 5 * std
                    ]  # 只保留在均值±3σ范围内的数据

                    Qi_list = [res["Qi"] for res in self.fit_results]
                    Qi_list = [qi for qi in Qi_list if qi is not np.inf and qi > 0]
                    data = np.array(Qi_list)
                    mean = np.mean(data)
                    std = np.std(data)
                    Qi_filtered = data[
                        np.abs(data - mean) < 5 * std
                    ]  # 只保留在均值±3σ范围内的数据
                    if display:
                        fig, axes = plt_ready(3, 2, figsize=(8, 6))
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
                        else:
                            print("Error creating plots.")
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
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
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
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
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
        计算色散参数D。
        '''
        if self.fre is None or self.lamda is None:
            print("Frequency and wavelength data are required to calculate dispersion.")
            return
        if self.T is None:
            print("Transmission data (T) is required to calculate dispersion.")
            return

        assert self.fre is not None and self.lamda is not None and self.T is not None

        if self.fit_results is None:
            self.cal_Q(holdon=False, display=False)
        else:
            pass

        lamda0_list = np.array(
            [res["lambda0"] for res in self.fit_results]  # type:ignore
        )
        omega0_list = 2 * np.pi * c / (lamda0_list * 1e-9)  # 转换为rad/s
        frequency0_list = omega0_list / (2 * np.pi) / 1e12  # 转换为THz

        N = len(omega0_list)
        mu_array = np.arange(-(N // 2), N // 2 + (N % 2))

        # 拟合 ωμ = ω0 + D1μ + ½ D2μ² + ⅙ D3μ³
        coeffs = np.polyfit(mu_array, omega0_list, 3)
        omega_fit = np.polyval(coeffs, mu_array)
        omega0 = coeffs[3]
        D1 = coeffs[2]
        print(f'D1 = {D1 / (2 * np.pi * 1e9)} GHz ')

        Dint_array = omega0_list - (omega0 + D1 * mu_array)
        Dint = Dint_array / D1

        fig, axes = plt_ready(1, 1, figsize=(8, 6))
        if fig is not None and axes is not None:
            (ax1,) = axes
            ax1.plot(frequency0_list, Dint, 'o-')
            ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax1.set_title('Integrated Dispersion')
            ax1.set_xlabel('Mode Number (μ)')
            ax1.set_ylabel('Dint / D1')
            ax1.grid(True)

    def cal_alpha(self, ne_file, L):
        '''
        计算损耗参数α。
        Args:
            ne_file: 有效折射率数据文件路径，支持.mat格式，通过Lumerical Mode Solutions导出
            L: 微环周长，单位米
        '''
        if self.lamda is None:
            print("Wavelength data is required to calculate loss.")
            return
        if ne_file is None or not os.path.isfile(ne_file):
            print("Effective index file is required to calculate loss.")
            return

        ref = {}
        data = list(mat73.loadmat(ne_file).values())[0]
        ref['lamda'] = np.round(data['x0'] * 1000, 4)
        ref['ne'] = data['y0']
        if self.lamda is not None:
            start = self.lamda.min()
            end = self.lamda.max()
        else:
            raise ValueError("微环数据不存在波长信息，无法匹配参考数据。")
        if start < ref['lamda'].min() or end > ref['lamda'].max():
            raise ValueError(
                f"有效折射率数据无效：微环数据的波长范围超出有效折射率数据波长的范围。\nRing:{start} - {end} nm, ne data range: {ref['lamda'].min()} - {ref['lamda'].max()} nm"
            )
        mask = (ref['lamda'] >= start) & (ref['lamda'] <= end)
        ref_lambda = ref['lamda'][mask]
        ref_ne = ref['ne'][mask]
        ref_lambda_step = np.mean(np.diff(ref_lambda))
        if ref_lambda_step != self.lamda_step:
            f_interp = interp1d(ref_lambda, ref_ne, kind='cubic')
            ref_lambda = np.round(
                np.arange(start, end + 1e-6, self.lamda_step),
                decimal_places(self.lamda_step),
            )
            ref_ne = f_interp(ref_lambda)
        if self.fit_results is None:
            self.cal_Q(holdon=False, display=False)
        else:
            pass
        Qi_list = [res["Qi"] for res in self.fit_results]  # type:ignore
        lamda0_list = [res["lambda0"] for res in self.fit_results]  # type:ignore

        alpha_list = []
        for Qi, lamda0 in zip(Qi_list, lamda0_list):
            n_eff = ref_ne[np.abs(ref_lambda - lamda0).argmin()]  # 有效折射率
            lamda0 = lamda0 * 1e-9  # 转换为米
            A = np.pi * n_eff * L / lamda0
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
            alpha_1 = -20 * np.log10(alpha) / L
            alpha_list.append(alpha_1 / 100)
        # 筛选
        alpha_list = [
            alpha for alpha in alpha_list if alpha is not np.inf and alpha > 0
        ]
        data = np.array(alpha_list)
        mean = np.mean(data)
        std = np.std(data)
        alpha_filtered = data[
            np.abs(data - mean) < 5 * std
        ]  # 只保留在均值±3σ范围内的数据

        fig, axes = plt_ready(1, 1, figsize=(8, 6))
        if fig is not None and axes is not None:
            (ax1,) = axes
            ax1.hist(
                alpha_filtered,
                bins=20,
                color='mediumseagreen',
                edgecolor='black',
            )
            ax1.set_title('波导损耗')
            ax1.set_xlabel('$\\alpha$ (dB/cm)')
            ax1.set_ylabel('Count')
            ax1.grid(True)

        else:
            print("Error creating plots.")
