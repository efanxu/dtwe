import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import math
import os


class EvaluationMetrics:
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def save_to_excel(self, eval_results):
        file_name = self.path + 'evaluation_results.xlsx'
        # 检查文件是否存在
        if os.path.exists(file_name):
            # 如果文件存在，则读取现有数据
            existing_df = pd.read_excel(file_name)
            # 将新的评估结果添加到现有数据的下一行
            combined_df = pd.concat([existing_df, eval_results], ignore_index=True)
        else:
            # 如果文件不存在，则直接使用新数据
            combined_df = eval_results
        # 保存到Excel文件
        combined_df.to_excel(file_name, index=False)

    def smape(self, y_true, y_pred):
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2
        # 避免除以0
        denom = np.where(denom == 0, 1, denom)
        return np.mean(np.abs(y_pred - y_true) / denom)

    def deter_metrices(self, y_test, y_pred, run_time):  # 确定性评估指标
        y_test, y_pred = np.array(y_test).ravel(), np.array(y_pred).ravel()  # 多维数据扁平化
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        smape = self.smape(y_test, y_pred)
        nrmse = (rmse / (np.max(y_test) - np.min(y_test)))
        ia = self.calculate_ia(y_test, y_pred)
        r2, rmse, nrmse, mae, mape, smape, ia = [round(value, 3) for value in [r2, rmse, nrmse, mae, mape, smape,  ia]]

        print(f'---- {self.name} Deterministic Evaluation ----')
        print('MAE\tRMSE\tSMAPE\tIA')
        print(mae, '\t', rmse, '\t', smape, '\t', ia)

        # 构建 DataFrame
        eval_results = pd.DataFrame({'Model': [self.name],
                                     'MAE': [mae],
                                     'RMSE': [rmse],
                                     'NRMSE': [nrmse],
                                     'MAPE': [mape],
                                     'IA': [ia],
                                     'R2': [r2],
                                     'Time': [run_time]})

        print('-' * 80)
        # self.save_to_excel(eval_results)
        return eval_results

    def inter_metrices(self, y_test, y_max, y_min):  # 区间性评估指标
        picp = self._PICP(y_test, y_max, y_min)
        pinaw = self._PINAW(y_test, y_max, y_min)
        cwc = self._CWC(picp, pinaw)
        ais = self._AIS(y_test, y_max, y_min)
        cpia = self._calculate_cpia(y_test, y_max, y_min, pinaw)
        picp, pinaw, cwc, ais, cpia = [round(value, 4) for value in [picp, pinaw, cwc, ais, cpia]]

        print(f'---- {self.name} Interval Evaluation ----')
        print(f'{"PICP":^10}\t{"PINAW":^10}\t{"CWC":^10}\t{"AIS":^10}\t{"CPIA":^10}')
        print(f'{picp:^10.3f}\t{pinaw:^10.3f}\t{cwc:^10.3f}\t{ais:^10.3f}\t{cpia:^10.3f}')

        # 构建 DataFrame
        eval_results = pd.DataFrame({'Model': [self.name],
                                     'PICP': [picp],
                                     'PINAW': [pinaw],
                                     'CWC': [cwc],
                                     'AIS': [ais],
                                     'CPIA': [cpia]})

        print('-' * 80)
        # self.save_to_excel(eval_results)
        return eval_results

    def calculate_ia(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        y_mean = np.mean(y_true)

        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum((np.abs(y_pred - y_mean) + np.abs(y_true - y_mean)) ** 2)

        ia = 1 - (numerator / denominator)
        return ia

    def calculate_ct(self, y_hat, U, L, d):
        T = len(y_hat)
        ct = np.zeros(T)

        for t in range(T):
            if U[t] <= y_hat[t] <= U[t] + d[t] / 2:
                ct[t] = 1 - (y_hat[t] - U[t]) / d[t]
            elif L[t] <= y_hat[t] <= U[t]:
                ct[t] = 1
            elif L[t] - d[t] / 2 <= y_hat[t] <= L[t]:
                ct[t] = 1 - (L[t] - y_hat[t]) / d[t]
            else:
                ct[t] = 0
        return ct

    def _calculate_cpia(self, y_hat, U, L, PINAW):
        d = U - L
        ct = self.calculate_ct(y_hat, U, L, d)
        cpia = np.mean((1 - PINAW) * ct)
        return cpia

    def _PICP(self, y_true, y_pred_max, y_pred_min):
        y_true, y_pred_max, y_pred_min = np.array(y_true), np.array(y_pred_max), np.array(y_pred_min)
        count = 0
        for i in range(len(y_true)):
            if y_pred_min[i] <= y_true[i] <= y_pred_max[i]:
                count += 1
        PICP = count / len(y_true)
        return PICP

    def _PINAW(self, y_true, y_pred_max, y_pred_min):
        y_true, y_pred_max, y_pred_min = np.array(y_true), np.array(y_pred_max), np.array(y_pred_min)
        R = np.max(y_true) - np.min(y_true)
        R1 = [y_pred_max[i] - y_pred_min[i] for i in range(len(y_true))]
        PINAW = np.mean(R1) / R
        return PINAW

    def _CWC(self, PICP, PINAW, v=0.9, n=10):
        if PICP <= v:
            CWC = PINAW * (1 + math.exp(n * (v - PICP)))
        else:
            CWC = PINAW
        return CWC

    def _AIS(self, y_true, y_pred_max, y_pred_min, alpha=0.1):
        y_true, y_pred_max, y_pred_min = np.array(y_true), np.array(y_pred_max), np.array(y_pred_min)
        S1 = []
        for i in range(len(y_true)):
            if y_pred_min[i] <= y_true[i] <= y_pred_max[i]:
                S1.append(-2 * alpha * (y_pred_max[i] - y_pred_min[i]))
            elif y_true[i] < y_pred_min[i]:
                S1.append(-2 * alpha * (y_pred_max[i] - y_pred_min[i]) - 4 * (y_pred_min[i] - y_true[i]))
            else:
                S1.append(-2 * alpha * (y_pred_max[i] - y_pred_min[i]) - 4 * (y_true[i] - y_pred_max[i]))
        AIS = np.mean(S1)
        return AIS


'''
# 生成一些随机数据
np.random.seed(0)  # 设置随机种子以保证结果可重复
y_test = np.random.rand(100) * 100  # 真实值
y_pred = y_test + (np.random.rand(100) * 20 - 10)  # 预测值（添加一些噪声）

# 创建评估指标实例
evaluator = EvaluationMetrics(path='./', name='随机模型')

# 计算确定性评估指标并保存结果
eval_results_deter = evaluator.deter_metrices(y_test, y_pred)

# 生成一些区间评估的预测最大值和最小值
y_max = y_pred + 10  # 预测最大值
y_min = y_pred - 10  # 预测最小值

# 计算区间性评估指标并保存结果
eval_results_inter = evaluator.inter_metrices(y_test, y_max, y_min)
'''