import pandas as pd
import numpy as np
import os


class DataProcessor:
    def __init__(self, args):
        self.base_path = args.base_path + 'checkpoint/'  # 默认路径
        self.exp_path = self.base_path + args.exp_path + '/'  # 每个实验路径
        self.args = args  # 所有参数

    def initialize_paths(self):
        """
        初始化路径
        """
        sub_paths = {
            'global_data_path': os.path.join(self.base_path, 'global_data_path/'),
            'local_data_path': os.path.join(self.exp_path, '1.local_data_path/'),
            'args_path': os.path.join(self.exp_path, '2.args_path/'),
            'models_path': os.path.join(self.exp_path, '3.models_path/'),
            'fig_path': os.path.join(self.exp_path, '4.fig_path/')
        }

        # 创建所有必要的路径
        for key, sub_path in sub_paths.items():
            os.makedirs(sub_path, exist_ok=True)  # 创建路径

        # 使用元组解包方式赋值
        self.args.global_data_path, self.args.local_data_path, self.args.args_path, \
            self.args.models_path, self.args.fig_path = tuple(sub_paths.values())

    def read_data(self):
        """读取指定文件中的数据"""
        file_path = os.path.join(self.args.data_path, self.args.filename)

        # 根据文件后缀读取不同格式的数据
        if self.args.filename.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif self.args.filename.endswith('.xlsx'):
            data = pd.read_excel(file_path)
        elif self.args.filename.endswith('.txt'):
            data = pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("不支持的文件格式: 请使用 .csv, .xlsx 或 .txt 文件.")
        print(f"成功读取文件: {file_path}")

        return data

    def split_data(self, df_x, df_y, in_start=0):
        df_x = np.array(df_x)
        df_y = np.array(df_y)
        if df_x.ndim == 1:
            df_x = df_x.reshape(-1, 1)
        if df_y.ndim == 1:
            df_y = df_y.reshape(-1, 1)

        datax, datay = [], []
        for i in range(len(df_x)-self.args.seq_len-self.args.label_len+1):
            in_end = int(in_start + self.args.seq_len)
            out_end = int(in_end + self.args.label_len)
            if out_end < len(df_x) + 1:
                a = df_x[in_start:in_end]
                if df_y.ndim == 1:
                    b = df_y[in_end:out_end]
                else:
                    b = df_y[in_end:out_end, -1].reshape(-1, )
                datax.append(a)
                datay.append(b)
            in_start += 1

        # # 补齐最后一个样本,用于预测未来
        # for i in range(self.args.label_len):
        #     a = df_x[in_start: in_start + self.args.seq_len]
        #     in_start += 1
        # datax.append(a)
        # datay.append([np.nan for i in range(self.args.label_len)])

        datax, datay = np.array(datax), np.array(datay)
        print('相空间完成', datax.shape, datay.shape)
        return datax, datay


'''
# 示例用法
    parser = argparse.ArgumentParser(description='Paper')

    parser.add_argument('--base_path', type=str, default='D:/Codes/project/other/', help='项目路径')
    parser.add_argument('--data_path', type=str, default='D:/Codes/dataset/carbon/', help='数据集路径')
    parser.add_argument('--exp_path', type=str, required=False, default='实验一/', help='一个类别的实验名称')
    parser.add_argument('--filename', type=str, required=False, default='GD.xlsx', help='数据集文件名')
    parser.add_argument('--features', type=str, default='S',
                        help='预测任务, 选项:[M, S]; M:多因素预测, S:单变量预测')
    parser.add_argument('--target', type=str, default='收盘价', help='预测的列名')
    
    from module.preprocessing import decomposition, data_processor
    processor = data_processor.DataProcessor(args)
    global_data_path, local_data_path, args_path, models_path, fig_path = processor.initialize_paths()
    data = processor.read_data()
    series = data['收盘价']
'''
