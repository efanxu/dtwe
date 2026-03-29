import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD, EEMD, CEEMDAN
from vmdpy import VMD
import pywt
from statsmodels.tsa.seasonal import STL


class SignalDecomposition:
    def __init__(self, args, data_path):
        self.args = args
        self.data_path = data_path

    def decompose(self, series, name):
        decomposition_methods = {
            'EMD': self.emd_decom,
            'EEMD': self.eemd_decom,
            'CEEMDAN': self.ceemdan_decom,
            'VMD': self.vmd_decom,
            'WD': self.wd_decom,
            'WPD': self.wpd_decom,
            'STL': self.stl_decom,
            'AVG': self.avg_decom,
        }
        if not isinstance(series, pd.Series):
            series = (series.values if isinstance(series, pd.DataFrame) else series).flatten()
            series = pd.Series(series)

        if self.args.dec_method in decomposition_methods and self.args.other_run:
            dec_df = decomposition_methods[self.args.dec_method](series, self.args.dec_k, self.args.dec_extra)
        else:
            dec_df = pd.read_csv(self.data_path + name + '.csv')

        if self.args.other_draw:
            self.draw_dec(series, dec_df, name, self.args.other_save)

        if self.args.other_save:
            pd.DataFrame.to_csv(dec_df, self.data_path + name + '.csv', index=False)
        return dec_df

    def draw_dec(self, data, S, name, other_save):
        series = data[-len(S):]
        S_num = int(len(S.columns))
        fig, axs = plt.subplots(S_num + 1, 1, figsize=(12, 10))
        for i in range(S_num + 1):
            ax = axs[i]
            ax.plot(series if i == 0 else S.iloc[:, i - 1])
            ax.set_ylabel('Original' if i == 0 else 'S' + str(i), fontweight='bold')
        axs[0].set_title(str.upper(name), fontweight='bold')
        fig.align_labels()
        plt.tight_layout()
        if other_save:
            plt.savefig(self.args.fig_path + name + '.png')
        plt.pause(2)
        plt.close()


    def emd_decom(self, series, k, extra=None):
        decom = EMD()
        imfs_emd = decom.emd(series.values, max_imf=k)
        imfs_num = np.shape(imfs_emd)[0]
        imfs_df = pd.DataFrame(imfs_emd.T)
        imfs_df.columns = ['S' + str(i + 1) for i in range(imfs_num)]
        return imfs_df

    def eemd_decom(self, series, k, extra=None):
        decom = EEMD()
        imfs_emd = decom.eemd(series.values, max_imf=k)
        imfs_num = np.shape(imfs_emd)[0]
        imfs_df = pd.DataFrame(imfs_emd.T)
        imfs_df.columns = ['S' + str(i + 1) for i in range(imfs_num)]
        return imfs_df

    def ceemdan_decom(self, series, k, extra=None):
        decom = CEEMDAN()
        imfs_emd = decom.ceemdan(series.values, max_imf=k)
        imfs_num = np.shape(imfs_emd)[0]
        imfs_df = pd.DataFrame(imfs_emd.T)
        imfs_df.columns = ['S' + str(i + 1) for i in range(imfs_num)]
        return imfs_df

    def vmd_decom(self, series, k=4, extra=None):
        alpha = 3000
        tau = 0
        DC = 0
        init = 1
        tol = 1e-7
        k = k+1
        u, u_hat, omega = VMD(series, alpha, tau, k, DC, init, tol)
        vmd_df = pd.DataFrame(u).iloc[::-1]
        vmd_df = np.array(vmd_df)
        vmd_df = pd.DataFrame(vmd_df.T)
        vmd_df.columns = ['S' + str(i + 1) for i in range(k)]
        return vmd_df

    def wd_decom(self, series, k=4, extra='coif4'):
        a = series
        ca = []
        cd = []
        for i in range(k):
            (a, d) = pywt.dwt(a, extra, 'smooth')
            ca.append(a)
            cd.append(d)
        rec_a = []
        rec_d = []
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, extra))
        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, extra))
        x1, x2 = pd.DataFrame(rec_a).T, pd.DataFrame(rec_d).T
        wd_s = pd.concat([x2, x1.iloc[:, -1]], axis=1)
        wd_s.columns = ['S' + str(i + 1) for i in range(len(x2.columns) + 1)]
        wd_s = wd_s.iloc[:len(series)]

        return wd_s

    def wpd_decom(self, series, k=3, extra='db10'):
        wp = pywt.WaveletPacket(data=series, wavelet=extra, mode='symmetric', maxlevel=k)
        node_name_list = [node.path for node in wp.get_level(k, 'freq')]
        rec_results = []
        for i in node_name_list:
            new_wp = pywt.WaveletPacket(data=np.zeros(len(series)), wavelet=extra, mode='symmetric')
            new_wp[i] = wp[i].data
            x_i = new_wp.reconstruct(update=True)
            rec_results.append(x_i)
        output = np.array(rec_results)
        wpd_df = pd.DataFrame(output).iloc[::-1].T
        wpd_df.columns = ['S' + str(i + 1) for i in range(len(wpd_df.columns))]
        return wpd_df

    def stl_decom(self, series, k, extra):
        extra = int(extra)
        stl = STL(series, period=extra)
        result = stl.fit()
        # 将 NumPy 数组转换为 Pandas Series
        trend_series = pd.Series(result.trend)
        seasonal_series = pd.Series(result.seasonal)
        resid_series = pd.Series(result.resid)
        # 将 Series 拼接成 DataFrame
        stl_df = pd.concat([resid_series, seasonal_series, trend_series], axis=1)
        stl_df.columns = ['S1', 'S2', 'S3']
        return stl_df.iloc[:, :]

    def avg_decom(self, series, k, extra):  # 增加步长参数
        window_size = int(extra)
        # 计算填充数量
        # pad_width = (window_size - 1) // 2
        #
        # # 在首尾填充数据
        # padded_data = np.pad(series, (pad_width, pad_width), mode='edge')
        #
        # # 计算滑动平均
        # weights = np.ones(window_size) / window_size
        # averages = np.convolve(padded_data, weights, mode='valid')

        # 无填充的计算滑动平均
        averages = series.rolling(window=window_size).mean()
        averages = averages.dropna()

        errors = series[-len(averages):] - averages
        print('滑动平均后比原数据短的长度:', len(series) - len(averages))

        result_df = pd.DataFrame({
            'Residual': errors,
            'Moving Mean': averages,
        })
        return result_df


