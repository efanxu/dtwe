from module.index.data_statistical import StatisticalTests
from module.matplotlib_config import set_matplotlib_params
from module.preprocessing import decomposition, data_processor
from module.index import evaluation, data_statistical
from module.index import evaluation
from minepy import MINE

from module import draw
# import cupy as cp
import re
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

from keras.models import Sequential
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Conv1D, LayerNormalization, Input, Reshape, Flatten
from keras.layers import TimeDistributed, MaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from mealpy.swarm_based import GWO, SSA
from module.models import elm
from module.index import evaluation
import time
import pickle
from sklearn.metrics import mean_squared_error

set_matplotlib_params()


def generate_filename(data_name='', season='', parameter_set='', model='', step=''):
    filename = f"{data_name}+{season}+{parameter_set}+{model}+{step}_step"
    return filename


def find_extremes(series):
    """检测时间序列中的局部极大值和极小值点，排除最后一个点"""
    extremes = []
    n = len(series)
    for i in range(1, n - 1):  # 不检查最后一个点
        prev_val = series[i - 1]
        curr_val = series[i]
        next_val = series[i + 1]
        if curr_val > prev_val and curr_val > next_val:
            extremes.append((i, 'max'))
        elif curr_val < prev_val and curr_val < next_val:
            extremes.append((i, 'min'))
    return extremes

def find_four_alternate_extremes(extremes):
    """从后往前寻找四个交替的极大极小值点，包含两个max和两个min"""
    if len(extremes) < 4:
        return None
    reversed_extremes = extremes[::-1]  # 反转以便从后往前处理
    first_type = reversed_extremes[0][1]
    # 确定目标类型序列
    if first_type == 'max':
        target_sequence = ['max', 'min', 'max', 'min']
    else:
        target_sequence = ['min', 'max', 'min', 'max']
    collected = []
    current_target_idx = 0
    for point in reversed_extremes:
        if point[1] == target_sequence[current_target_idx]:
            collected.append(point)
            current_target_idx += 1
            if current_target_idx == 4:
                break
    if len(collected) != 4:
        return None
    # 第四个点离端点最远
    return collected[3][0]

def find_segment(series):
    """主函数：找到符合条件的数据段"""
    if len(series) < 5:  # 至少需要5个点才能有四个极值点（不包含端点）
        return None
    extremes = find_extremes(series)
    if len(extremes) < 4:
        return None
    start_pos = find_four_alternate_extremes(extremes)
    if start_pos is None:
        return None
    # 返回从start_pos到末尾的数据（包括原始终点）
    return series[start_pos:]


def dtw_distance(X, Y):
    # 转换输入为numpy数组并确保二维形状
    X = np.array(X)
    Y = np.array(Y)

    # 处理一维情况
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    assert X.shape[1] == Y.shape[1], "X和Y的特征维度必须相同"

    n, m = X.shape[0], Y.shape[0]

    # 计算距离矩阵
    dist_matrix = np.sqrt(np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))

    # 初始化累积距离矩阵
    dtw = np.full((n, m), np.inf)
    dtw[0, 0] = dist_matrix[0, 0]

    # 填充第一行
    for j in range(1, m):
        dtw[0, j] = dtw[0, j - 1] + dist_matrix[0, j]

    # 填充第一列
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + dist_matrix[i, 0]

    # 填充剩余部分
    for i in range(1, n):
        for j in range(1, m):
            min_prev = min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
            dtw[i, j] = dist_matrix[i, j] + min_prev

    return dtw[-1, -1]


def find_similar_segment(X, Y, YY, extend_len):
    X_len = len(X)
    Y_len = len(Y)

    if X_len > Y_len - X_len:
        raise ValueError("X长度应小于Y长度的一半，以确保能找到两段不重叠的Y的子序列")

    min_distance = np.inf
    best_start_index = None

    # 遍历Y中的每个可能的起始点，计算与X的DTW距离
    for start in range(Y_len - X_len + 1):
        segment = Y[start:start + X_len]
        distance = dtw_distance(X, segment)

        if distance < min_distance:
            min_distance = distance
            best_start_index = start

    # 找到最相似的子序列及其后面的子序列
    similar_segment = Y[best_start_index:best_start_index + X_len]
    # next_segment = Y[best_start_index + X_len:best_start_index + 2 * X_len]
    next_segment = YY[best_start_index + X_len:best_start_index + X_len + extend_len]

    return similar_segment, next_segment


def prepare_time_map_data(feature_data1, feature_data2, target_data, feature_window_size, target_window_size, future_window_size):
    # 准备列表以存储特征和目标
    feature_windows = []  # 包含target的特征窗口
    feature_windows2 = []  # 不包含

    target_windows = []

    # 从数据中构建特征和目标, 每次向前滑动1个数据
    for i in range(len(feature_data1) - feature_window_size - target_window_size - future_window_size + 1):
        window_features = feature_data1[i + target_window_size:i + feature_window_size + target_window_size + future_window_size]
        window_features2 = feature_data2[i:i + feature_window_size]

        feature_windows.append(window_features.flatten())  # 将每个窗口的特征展平为一维数组
        feature_windows2.append(window_features2.flatten())  # 将每个窗口的特征展平为一维数组

        window_targets = target_data[i + feature_window_size:i + feature_window_size + target_window_size]
        target_windows.append(window_targets)  # 添加对应的目标值

    # 转换为 NumPy 数组
    feature_array1 = np.array(feature_windows).reshape(-1, feature_window_size + future_window_size)
    feature_array2 = np.array(feature_windows2).reshape(-1, feature_window_size)
    target_array = np.array(target_windows).reshape(-1, target_window_size)

    # print('映射对:', feature_array.shape, feature_array2.shape, target_array.shape)
    return feature_array1, feature_array2, target_array


class ModelTrainer:
    def __init__(self, args):
        self.args = args

    def _load_or_optimize_parameters(self, name, train_sc_x, train_sc_y):
        if self.args.is_opt == 1:
            return self._optimization(name, train_sc_x, train_sc_y)
        elif self.args.is_opt == 0:
            with open(os.path.join(self.args.args_path, f'{name}.pkl'), 'rb') as f:
                return pickle.load(f)
        return self.args.p  # 默认参数

    def _model_set(self, name, train_sc_x, train_sc_y, shape):
        model_builders = {
            'ELM': self._build_elm_model,
            'GBDT': self._build_gbdt_model,
            'TCN': self._build_tcn_model,
            'SVR': self._build_svr_model,
            'GPR': self._build_gpr_model,
            'LASSO': self._build_lasso_model,
        }

        if self.args.model_select not in model_builders:
            raise ValueError(f"Unsupported model type: {self.args.model_select}")

        return model_builders[self.args.model_select](name, train_sc_x, train_sc_y, shape)

    def _build_elm_model(self, name, train_sc_x, train_sc_y, shape):

        model = elm.elm_gpu(hidden_units=int(self.args.p['elm_filter']), activation_function='relu',
                            x=train_sc_x, y=train_sc_y, C2=self.args.p['C2'], elm_type=self.args.elm_type)
        model.fit(algorithm='solution2')
        joblib.dump(model, self.args.models_path + name + '.pkl')
        # model = joblib.load(self.args.models_path + name + '.pkl')
        return model
    def _build_gbdt_model(self, name, train_sc_x, train_sc_y, shape):
        base_model = GradientBoostingRegressor(
            n_estimators=self.args.p['n_estimators'],
            max_depth=self.args.p['max_depth'], random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(train_sc_x, train_sc_y)
        return model
    def _build_lasso_model(self, name, train_sc_x, train_sc_y, shape):
        model = Lasso(alpha=self.args.p['alpha'], max_iter=10000)
        model.fit(train_sc_x, train_sc_y)
        return model
    def _build_svr_model(self, name, train_sc_x, train_sc_y, shape):
        base_model = SVR(C=self.args.p['C'], kernel=self.args.p['kernel'], gamma=self.args.p['gamma'])
        model = MultiOutputRegressor(base_model)
        model.fit(train_sc_x, train_sc_y)
        return model
    def _build_gpr_model(self, name, train_sc_x, train_sc_y, shape):
        kernel = ConstantKernel() * DotProduct()
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        model.fit(train_sc_x, train_sc_y)
        return model
    def _build_tcn_model(self, name, train_sc_x, train_sc_y, shape):
        model = Sequential()
        model.add(Input(shape=(shape[1], shape[2])))

        for i in range(self.args.p['num_layers']):  # 根据层数动态构建
            model.add(Conv1D(self.args.p['tcn_filter1'], self.args.p['tcn_size1'],
                             activation='relu', padding='causal', dilation_rate=2**i))
            # model.add(LayerNormalization())
        model.add(Flatten())
        if self.args.tcn_out_map:
            model.add(Dense(15))
        else:
            model.add(Dense(self.args.label_len))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.p['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse')
        # model.compile(optimizer='adam', loss='mse')
        self._history_set(model, name, train_sc_x, train_sc_y)
        return model

    def _history_set(self, model, name, train_sc_x, train_sc_y):
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.p['patience'],
                                   verbose=self.args.p['verbose'], mode='auto')

        # 学习率衰减率
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.args.p['patience'],
                                      verbose=self.args.p['verbose'], cooldown=0, min_lr=1e-7, mode='auto')

        history = model.fit(train_sc_x, train_sc_y, epochs=self.args.p['epochs'],
                            batch_size=self.args.p['batch_size'], verbose=self.args.p['verbose'],
                            validation_split=self.args.p['validation_split'],
                            callbacks=[early_stop, reduce_lr])

        if self.args.is_save:
            model.save(os.path.join(self.args.models_path, f'{name}.h5'))


    def _optimization(self, name, train_sc_x, train_sc_y):
        X_train, X_val, y_train, y_val = train_test_split(train_sc_x, train_sc_y, test_size=self.args.val_len,
                                                          shuffle=False)

        # 定义模型相关属性
        model_parameters = {
            'ELM': {
                'LB': [8, 1],
                'UB': [1024, 2048],
                'decode': lambda solution: {
                    "elm_filter": int(solution[0]),
                    'C2': solution[1]
                }
            },
            'SVR': {
                'LB': [1, 0.1, -2],
                'UB': [100, 0.99, 0],
                'decode': lambda solution: {
                    'C': int(solution[0]),
                    'kernel': LabelEncoder().fit(['poly', 'rbf', 'linear']).inverse_transform([int(solution[1])])[0],
                    'gamma': 10 ** int(solution[2])
                }
            },

            'TCN': {
                'LB': [4, 2, 1, 0.00001],
                'UB': [8, 4, 3, 0.01],
                'decode': lambda solution: {
                    'tcn_filter1': 2 ** int(solution[0]),  # [16, 256]
                    'tcn_size1': 2 * int(solution[1]) - 1,  # {3, 5, 7}
                    'num_layers': int(solution[2]),  # {1, 2, 3}
                    'learning_rate': solution[3],
                }
            },
        }

        # 获取当前模型的参数
        params = model_parameters[self.args.opt_model]
        LB, UB = params['LB'], params['UB']

        # 定义适应度函数
        def fitness_function(solution):
            for key, value in params['decode'](solution).items():
                self.args.p[key] = value
            model = self._model_set(name, X_train, y_train, X_train.shape)
            y_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mdd = self.__mmd(y_val, y_pred, kernel='rbf', gamma=1.0)
            if self.args.elm_type == 'reg':
                fitness = rmse
            else:
                fitness = rmse + 0.1*mdd
            return fitness

        problem = {
            "fit_func": fitness_function,
            "lb": LB,
            "ub": UB,
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        # 运行SSA算法
        model = SSA.BaseSSA(epoch=20, pop_size=20)
        model.solve(problem)

        sol = params['decode'](model.solution[0])
        if self.args.is_save:
            with open(os.path.join(self.args.args_path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(sol, f)
        print('优化结束!', sol)
        return sol

    def __mmd(self, x, y, kernel='rbf', gamma=1.0):
        """
        Compute the Maximum Mean Discrepancy (MMD) between two distributions.
        :param x: numpy array, samples from distribution P.
        :param y: numpy array, samples from distribution Q.
        :param kernel: str, kernel type ('rbf' for Gaussian kernel).
        :param gamma: float, kernel bandwidth parameter.
        :return: float, MMD value.
        """
        if kernel == 'rbf':
            def kernel_func(u, v):
                # Compute pairwise squared Euclidean distances
                dist_matrix = np.sum(u ** 2, axis=1)[:, np.newaxis] + np.sum(v ** 2, axis=1) - 2 * np.dot(u, v.T)
                return np.exp(-gamma * dist_matrix)
        else:
            raise ValueError("Unsupported kernel type")

        m = x.shape[0]
        n = y.shape[0]

        k_xx = kernel_func(x, x)
        k_yy = kernel_func(y, y)
        k_xy = kernel_func(x, y)

        mmd_squared = (np.sum(k_xx) / (m * (m - 1)) +
                       np.sum(k_yy) / (n * (n - 1)) -
                       2 * np.sum(k_xy) / (m * n))

        return np.sqrt(mmd_squared + 1e-6)  # Add a small epsilon for numerical stability


class TimeSeriesForecasting:

    def __init__(self, args):
        self.args = args
        self.statistical = data_statistical.StatisticalTests(args)  # 初始化数据统计器
        self.processor = data_processor.DataProcessor(args)  # 初始化数据处理器
        self.processor.initialize_paths()  # 创建路径
        self.signal_decomp = decomposition.SignalDecomposition(args, data_path=self.args.local_data_path)  # 初始化信号分解器
        self.trainer = ModelTrainer(args)  # 初始化训练器

    def initialize(self):
        data = self.processor.read_data()  # 读取数据
        if self.args.target == '收盘价':
            data['时间索引'] = pd.to_datetime(data['日期'])
            data = data.drop(columns=['日期'])
            data.set_index('时间索引', inplace=True)
        else:
            data['时间索引'] = pd.to_datetime(data['日期'] + ' ' + data['时间'])
            data = data.drop(columns=['日期', '时间'])
            data.set_index('时间索引', inplace=True)

            if self.args.season == '春季':
                start_date = '2022-03-01'
                end_date = '2022-05-31'
            elif self.args.season == '夏季':
                start_date = '2022-06-01'
                end_date = '2022-08-31'
            elif self.args.season == '秋季':
                start_date = '2022-09-01'
                end_date = '2022-11-30'
            elif self.args.season == '冬季':
                start_date = '2022-12-01'
                end_date = '2023-02-28'
            data = data[(data.index >= start_date) & (data.index <= end_date)]

        if self.args.features == 'M':
            # cols_data = data.columns[1:]  # 所有特征列，第一列为时间列
            self.data_x = data
        elif self.args.features == 'S':
            self.data_x = data[[self.args.target]]  # 只选择目标列
            self.data_x = pd.DataFrame(self.data_x, columns=[self.args.target])

        self.data_y = data[[self.args.target]]  # (n, 1)
        self.series_y = self.data_x[self.args.target]  # (n, )

        # 确定验证集长度,测试集长度
        self.args.val_len = int(len(self.data_y) * self.args.test_rate)
        # self.args.pred_len = int(len(self.data_y) * self.args.test_rate)

        # 确定滞后长度
        self.args.seq_len = self.statistical.adf_test(self.series_y[:-self.args.pred_len])
        # 归一化
        if self.args.is_norm:
            # 对x进行归一化
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            train_val_data = self.data_x.iloc[:-self.args.pred_len, :]
            scaler_x.fit(train_val_data.values)
            self.data_sc_x = scaler_x.transform(self.data_x.values)
            # 对y进行归一化
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            train_val_data = self.data_y.iloc[:-self.args.pred_len, :]
            self.scaler.fit(train_val_data.values)
            self.data_sc_y = self.scaler.transform(self.data_y.values)

    def split_data(self, df_x, df_y, in_start=0):
        df_x = np.array(df_x)
        df_y = np.array(df_y)
        if df_x.ndim == 1:
            df_x = df_x.reshape(-1, 1)
        if df_y.ndim == 1:
            df_y = df_y.reshape(-1, 1)

        datax, datay = [], []
        for i in range(len(df_x) - self.args.label_len + 1):
            in_end = in_start
            out_end = int(in_end + self.args.label_len)+1
            if out_end < len(df_x) + 1:
                a = df_x[in_end]
                if df_y.ndim == 1:
                    b = df_y[in_end+1:out_end]
                else:
                    b = df_y[in_end+1:out_end, -1].reshape(-1, )
                datax.append(a)
                datay.append(b)
            in_start += 1

        datax, datay = np.array(datax), np.array(datay)
        print('相空间完成', datax.shape, datay.shape)
        return datax, datay

    def lag_len(self, series_y, max_lag=10):
        # 存储MIC值
        mic_values = []

        # 计算不同滞后长度的MIC
        for k in range(1, max_lag + 1):
            X_k = series_y[k:]  # 滞后版本
            X_original = series_y[:-k]  # 原始序列

            mine = MINE()
            mine.compute_score(X_original, X_k)
            mic_values.append(mine.mic())

        # 找到最佳滞后长度
        print(f"MIC值: {mic_values}")
        count_over_02 = sum(1 for value in mic_values if value > 0.2)  # 👈 直接遍历列表统计
        print(f"MIC值大于0.2的个数: {count_over_02}")
        return count_over_02

    def run(self):
        seasons = ['春季', '夏季', '秋季', '冬季']
        for i in seasons:
            self.args.season = i
            self.initialize()
            print('验证集长度:', self.args.val_len)
            print('测试集长度:', self.args.pred_len)

            self.args.elm_type = 'custom'
            self.args.extend_len = 15
            self.args.map_len = self.lag_len(self.data_sc_y.ravel())
            self.args.seq_len = self.args.map_len

            # 定义初始长度
            init = 500
            # dtw扩展原数据
            dtw_time1 = time.time()
            map_x1 = pd.DataFrame()
            # map_x2_list = []
            for i in range(init, len(self.data_sc_y)+1):
                cur_data = self.data_sc_y[:i].copy()

                result = find_segment(cur_data)  # 边界数据段

                # 寻找相似段
                similar, next_seg = find_similar_segment(X=result, Y=cur_data[:-self.args.extend_len], YY=cur_data, extend_len=self.args.extend_len)
                concat_data = np.concatenate((cur_data, next_seg), axis=0)  # 右扩充15个点

                map_x1 = pd.concat([map_x1, pd.DataFrame(concat_data[-self.args.extend_len-2*self.args.map_len:]).T], axis=0, ignore_index=True)

            dtw_time2 = time.time()
            dtw_time = dtw_time2 - dtw_time1
            print('dtw时间:', dtw_time)

            name = f"{self.args.filename[:2]} + {self.args.season}+ DTW_扩展数据"
            pd.DataFrame.to_csv(map_x1, self.args.global_data_path + name + '.csv', index=False)
            # # 读取扩展数据
            name = f"{self.args.filename[:2]} + {self.args.season}+ DTW_扩展数据"
            map_x1 = pd.read_csv(self.args.global_data_path + name + '.csv').values

            print('扩展数据完成map_x1:', map_x1.shape)



            if self.args.season == '春季': dtw_time = 253.8251
            elif self.args.season == '夏季': dtw_time = 179.6589
            elif self.args.season == '秋季': dtw_time = 231.4555
            elif self.args.season == '冬季': dtw_time = 205.8313


            # 获取映射的y
            self.args.model_select = 'ELM'
            self.args.opt_model = 'ELM'
            self.args.is_opt = 1

            map_time1 = time.time()
            dec_train_data = self.signal_decomp.decompose(self.data_sc_y[:-self.args.val_len], name='')

            map_y_list = []
            true_y_list = []
            for i in range(init, len(dec_train_data) + 1):
                update_data = dec_train_data.iloc[i - self.args.map_len:i, :]
                map_y_list.append(update_data.values)  # 将 DataFrame 转为 numpy 数组

            for i in range(init, len(self.data_sc_y) + 1):
                update_y = self.data_sc_y[i - self.args.map_len:i]
                true_y_list.append(update_y)
            true_y = np.array(true_y_list).reshape(-1, self.args.map_len)

            # 将列表转为三维 numpy 数组 (n_samples, seq_len, n_features)
            map_y = np.stack(map_y_list, axis=0)
            print('映射数据完成map_y:', map_y.shape)

            # 映射模型
            predictions = {}
            for s_num in range(1, 3):
                map_sy = map_y[:, :, s_num]

                # map_sx = map_x2[:, :, s_num]
                # map_x = np.concatenate((map_x1, map_sx), axis=1)

                train_map_x, test_map_x, train_map_y = map_x1[:-self.args.val_len], map_x1[-self.args.val_len:], map_sy

                map_name = generate_filename(
                    data_name=self.args.filename[:2],
                    season=self.args.season,
                    parameter_set=f"{self.args.dec_method}+S{s_num + 1}_Map_Single_pred",
                    model=self.args.model_select,
                    step=self.args.label_len,
                )
                optimized_params = self.trainer._load_or_optimize_parameters(map_name, train_map_x, train_map_y)
                for key, value in optimized_params.items():
                    self.args.p[key] = value
                # 映射拟合
                map_model = self.trainer._model_set(map_name, train_map_x, train_map_y, shape=(train_map_x.shape))

                predictions[f"S{s_num+1}"] = map_model.predict(map_x1)

                pd.DataFrame.to_csv(pd.DataFrame(predictions[f"S{s_num + 1}"]),
                                    self.args.local_data_path + map_name + '.csv', index=False)

            S2, S3 = predictions['S2'], predictions['S3']

            S1 = true_y - S2 - S3

            map_name1 = generate_filename(
                data_name=self.args.filename[:2],
                season=self.args.season,
                parameter_set=f"{self.args.dec_method}+S1_Map_Single_pred",
                model=self.args.model_select,
                step=self.args.label_len,
            )
            pd.DataFrame.to_csv(pd.DataFrame(S1), self.args.local_data_path + map_name1 + '.csv', index=False)

            S = np.concatenate((S1, S2, S3), axis=1)
            print('映射模型完成S:', S.shape)
            map_time2 = time.time()
            map_time = map_time2 - map_time1
            print('映射时间:', map_time)

            # 预测模型
            # self.args.model_select = 'SVR'
            # self.args.opt_model = 'SVR'
            # self.args.is_opt = 1
            self.args.elm_type = 'reg'


            pred_time1 = time.time()
            datax, datay = self.split_data(S, self.data_sc_y[-len(S):])
            train_x, test_x, train_y, test_y = train_test_split(datax, datay, test_size=self.args.val_len, shuffle=False)

            ensemble_name = generate_filename(
                data_name=self.args.filename[:2],
                season=self.args.season,
                parameter_set=f"{self.args.dec_method}+Ensemble_pred",
                model=self.args.model_select,
                step=self.args.label_len,
            )
            optimized_params = self.trainer._load_or_optimize_parameters(ensemble_name, train_x, train_y)
            for key, value in optimized_params.items():
                self.args.p[key] = value
            # 映射拟合
            ensemble_model = self.trainer._model_set(ensemble_name, train_x, train_y, shape=(test_x.shape))

            y_pred = ensemble_model.predict(test_x)

            pred_time2 = time.time()
            pred_time = pred_time2 - pred_time1
            print('预测时间:', pred_time)

            # 反归一化
            y_pred = self.scaler.inverse_transform(y_pred)
            test_y = self.scaler.inverse_transform(test_y)

            pd.DataFrame.to_csv(pd.DataFrame(y_pred), self.args.local_data_path + ensemble_name + '.csv', index=False)

            # 评估

            evaluation_metrics = evaluation.EvaluationMetrics(self.args.base_path, name=ensemble_name)
            evaluation_metrics.deter_metrices(y_pred, test_y, run_time=dtw_time+map_time+pred_time)
            # plt.plot(y_pred[:, -1], label='pred')
            # plt.plot(test_y[:, -1], label='true')
            # plt.legend()
            # plt.show()

import random
import argparse
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    # torch.manual_seed(fix_seed)

    parser = argparse.ArgumentParser(description='Paper_test')
    # 基本配置参数
    parser.add_argument('--is_run', type=int, required=False, default=1, help='是否训练模型，1表示训练，0表示读取预测结果')  # 模型方面
    parser.add_argument('--is_save', type=int, required=False, default=1, help='是否保存预测结果和预测图，1表示保存，0表示不保存')
    parser.add_argument('--is_draw', type=int, required=False, default=1, help='是否展示结果图，1表示训练，0表示测试')
    parser.add_argument('--other_run', type=int, required=False, default=1, help='是否运行预处理方法，1表示运行，0表示读取')  # 预处理方面
    parser.add_argument('--other_save', type=int, required=False, default=0, help='是否保存预处理结结果和图，1表示保存，0表示不保存')
    parser.add_argument('--other_draw', type=int, required=False, default=0, help='是否展示预处理的结果图，1表示训练，0表示测试')

    # 数据集参数
    parser.add_argument('--base_path', type=str, default='D:/Codes/project/paper4/', help='项目路径')
    parser.add_argument('--data_path', type=str, default='D:/Codes/dataset/wind/陆上风/', help='数据集路径')
    parser.add_argument('--filename', type=str, required=False, default='广东阳江.csv', help='浙江六横, 广东阳江，内蒙古')
    parser.add_argument('--target', type=str, default='风速(m/s)', help='预测的列名')
    # parser.add_argument('--data_path', type=str, default='D:/Codes/dataset/carbon/', help='数据集路径')
    # parser.add_argument('--filename', type=str, required=False, default='GD.xlsx', help='数据集文件名')
    # parser.add_argument('--target', type=str, default='收盘价', help='预测的列名')

    parser.add_argument('--exp_path', type=str, required=False, default='exp_其他数据集', help='一个类别的实验名称')
    parser.add_argument('--features', type=str, default='S',
                        help='预测任务, 选项:[M, S]; M:多因素预测, S:单变量预测')
    parser.add_argument('--is_norm', type=int, default=1, help='是否使用标准化；真 1 假 0')
    parser.add_argument('--test_rate', type=float, default=0.2, help='测试集比率')

    # 预测任务参数
    parser.add_argument('--seq_len', type=int, default=20, help='滞后')
    parser.add_argument('--label_len', type=int, default=1, help='预测步长')
    parser.add_argument('--pred_len', type=int, default=1, help='测试集长度')
    parser.add_argument('--pred_mode', type=str, default='Direct', help='MIMO:多输入多输出, RecMo:递归多输出, Direct:直接预测')
    parser.add_argument('--total_len', type=int, default=12, help='多个预测步长的滚动窗口长度')
    parser.add_argument('--model_select', type=str, required=False, default='ELM',
                        help='可供选择的模型: [ELM, SVR, LASSO, LSSVR, BPNN, GRU, TCN, GBDT, GPR,]')
    parser.add_argument('--opt_model', type=str, required=False, default='ELM',
                        help='需要优化的名字: [ELM, SVR, LASSO, LSSVR, BPNN, GRU, TCN, GBDT]')
    parser.add_argument('--is_opt', type=str, required=False, default=1,
                        help='是否优化模型，1表示优化，0表示读取优化后的参数，-1表示默认参数')
    # 分解参数
    parser.add_argument('--dec_method', type=str, default='STL',
                        help='分解方法：EMD, EEMD, CEEMDAN, VMD, WD, WPD, STL, AVG')
    parser.add_argument('--dec_k', type=int, default=3, help='分解层数')
    parser.add_argument('--dec_extra', type=str, default='7', help='用于指定小波或STL周期等其他参数,小波：字符串，STL：4')

    # 模型参数
    parser.add_argument('--p', type=float, default={'elm_filter': 32, "C2": 1,  # elm
                                                    'C': 1, 'gamma': 0.1, 'kernel': 'rbf',  # svr
                                                    'C1': 1, 'gamma1': 0.1, 'kernel1': 'rbf',  # lssvr
                                                    'alpha': 0.05,  # lasso
                                                    'gru_filter1': 32,   # gru
                                                    'bpnn_filter1': 32,
                                                    'tcn_filter1': 32, 'tcn_size1': 3, 'tcn_rate1': 2,  # tcn
                                                    'n_estimators': 100, 'max_depth': 5,  # gbdt

                                                    'learning_rate': 0.01, 'num_layers': 1,

                                                    'Dropout': 0.1, 'verbose': 0, 'epochs': 100,
                                                    'batch_size': 32, 'validation_split': 0.1, 'patience': 20,

                                                    'w1': 50, 'w2': 1,
                                                    }, help='模型超参数')

    args = parser.parse_args()

    forecasting = TimeSeriesForecasting(args)
    forecasting.run()




