import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import find_peaks
import my_math as mm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class SpectrumAnalyzer:
    t_values: np.ndarray
    signal: np.ndarray
    sample_rate: float
    N: int
    freqs: np.ndarray | None
    spectrum: np.ndarray | None

    def __init__(self, t_values, signal):
        """
        初始化频谱分析器

        :param t_values: 时间值数组
        :param signal: 信号数组（复数形式）
        :param sample_rate: 采样率 (Hz)
        """
        self.t_values = t_values
        self.signal = signal
        self.sample_rate = 1 / (t_values[1] - t_values[0]) if len(t_values) > 1 else 1.0
        self.N = len(signal)
        self.freqs = None
        self.spectrum = None

    def calculate_spectrum(self, single_sided=True):
        """计算信号的频谱"""
        # 执行FFT
        fft_result = fft(self.signal)
        freqs = fftfreq(self.N, 1 / self.sample_rate)
        if single_sided:
            self.spectrum = fft_result[: self.N // 2] / self.N  # type: ignore
            self.spectrum[  # type: ignore
                1:
            ] *= 2  # 单边谱需要除以N并乘以2（除DC分量外）
            self.spectrum = abs(self.spectrum)  # type: ignore
            self.freqs = freqs[: self.N // 2]
        else:
            self.spectrum = np.abs(fftshift(fft_result / self.N))  # type: ignore
            self.freqs = fftshift(freqs)

        return self.freqs, self.spectrum

    def find_peaks(self, min_height=0.1, min_distance=10, single_sided=True):
        """查找频谱峰值"""

        if self.spectrum is None:
            self.calculate_spectrum(single_sided=single_sided)
        assert self.freqs is not None, "频率轴未计算"
        assert self.spectrum is not None, "频谱未计算"

        peaks, _ = find_peaks(self.spectrum, height=min_height, distance=min_distance)
        peak_freqs = self.freqs[peaks]
        peak_vals = self.spectrum[peaks]

        return peak_freqs, peak_vals

    def plot_spectrum(
        self,
        title="Signal Spectrum",
        log_scale=True,
        peak_info=False,
        single_sided=True,
        peak_center=True,
        x_span=100e9,  # 频率范围
    ):
        """绘制频谱图"""
        if self.spectrum is None:
            self.calculate_spectrum(single_sided=single_sided)
        assert self.freqs is not None, "频率轴未计算"
        assert self.spectrum is not None, "频谱未计算"
        fig1, ax1 = mm.plt_ready(1, 1, figsize=(12, 6))
        assert fig1 is not None and ax1 is not None, "绘图环境未准备好"

        if log_scale:
            ax1[0].semilogy(self.freqs, self.spectrum, 'b-', label='Amplitude Spectrum')
        else:
            ax1[0].plot(self.freqs, self.spectrum, 'b-', label='Amplitude Spectrum')

        if peak_info:
            peak_freqs, peak_vals = self.find_peaks()
            for i, (freq, val) in enumerate(zip(peak_freqs, peak_vals)):
                ax1[0].plot(freq, val, 'ro')
                ax1[0].annotate(
                    f'{freq/1e6:.2f} MHz, {val:.2f}',
                    (freq, val),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9,
                )
        if peak_center:
            index = np.argmax(self.spectrum)
            ax1[0].set_xlim(
                -x_span / 2 + self.freqs[index],
                x_span / 2 + self.freqs[index],
            )
        ax1[0].set_title(title)
        ax1[0].set_xlabel('Frequency (Hz)')
        ax1[0].set_ylabel('Amplitude')
        ax1[0].grid(True)
        # ax1[0].xlim(0, self.freqs[-1])
        ax1[0].legend()
        fig1.tight_layout()

    def plot_time_domain(self, param):
        """绘制时间域信号"""
        fig2, ax2 = mm.plt_ready(1, 1, figsize=(12, 6))
        assert fig2 is not None and ax2 is not None, "绘图环境未准备好"

        if param == 'real':
            ax2[0].plot(self.t_values, self.signal.real, 'b', label='Real Part')
        elif param == 'imag':
            ax2[0].plot(self.t_values, self.signal.imag, 'r', label='Imaginary Part')
        elif param == 'abs':
            ax2[0].plot(
                self.t_values,
                AnalysisTools.intensity(self.signal),
                'g',
                label='Magnitude',
            )
        ax2[0].set_title('Time Domain Signal E(t)')
        ax2[0].set_xlabel('Time (s)')
        ax2[0].set_ylabel('Amplitude')
        ax2[0].grid(True)
        ax2[0].legend()
        fig2.tight_layout()


class AnalysisTools:
    @staticmethod
    def intensity(field):
        """计算光强 |E|^2"""
        return np.abs(field) ** 2

    @staticmethod
    def phase(field):
        """计算相位"""
        return np.angle(field)

    @staticmethod
    def unitformat(field):
        """将光场最大值归一化"""
        field -= np.max(field)  # 减去最大值以避免除零错误
        return field

    @staticmethod
    def plot_fields(t_values, e_i1, e_o2):
        """绘制输入和输出场"""
        plt.figure(figsize=(12, 8))

        # 输入场
        plt.subplot(2, 1, 1)
        plt.plot(
            t_values,
            AnalysisTools.intensity(e_i1),
            'b-',
            label='Input Intensity |E_i1|^2',
        )
        plt.plot(t_values, AnalysisTools.phase(e_i1), 'g--', label='Input Phase')
        plt.title('Input Field E_i1(t)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude/Phase')
        plt.legend()
        plt.grid(True)

        # 输出场
        plt.subplot(2, 1, 2)
        plt.plot(
            t_values,
            AnalysisTools.intensity(e_o2),
            'r-',
            label='Output Intensity |E_o2|^2',
        )
        plt.plot(t_values, AnalysisTools.phase(e_o2), 'm--', label='Output Phase')
        plt.title('Output Field E_o2(t)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude/Phase')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_transfer_function(t_values, e_i1, e_o2):
        """绘制系统的转移函数"""
        intensity_ratio = np.where(
            AnalysisTools.intensity(e_i1) > 1e-6,
            AnalysisTools.intensity(e_o2) / AnalysisTools.intensity(e_i1),
            0,
        )

        plt.figure(figsize=(10, 6))
        plt.plot(t_values, intensity_ratio, 'b-')
        plt.title('System Transfer Function |E_o2/E_i1|^2')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity Ratio')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.show()
