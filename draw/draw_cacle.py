from itertools import cycle

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time  # 导入time模块，用于在循环中模拟耗时操作
import sys  # 导入sys模块，用于操作与Python解释器交互的一些变量和函数
from scipy import stats
import matplotlib
import pickle
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import EntropyHub as EH
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import medfilt
import seaborn as sns
from scipy.stats import pearsonr
from d2l import torch as d2l
from scipy.stats import skew
from scipy.stats import kurtosis

"""锂电池组特征"""
Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']

"""原始数据存放位置"""
source_data_path = '../source_data/CACLE_data/CS2_combine/'

"""训练数据存放位置"""
target_data_path = '../data/CACLE_data/pinn_path/'

"""一个数组的元素减去数组首位的数"""


def subtract_first_element(arr):
    """数组全部减去第一个元素，用来处理时间数据据"""
    if not arr:  # 检查数组是否为空
        return []
    first_element = arr[0]
    return [x - first_element for x in arr]


"""一系列对序列数据进行特征提取的函数"""
"""样本熵"""


def SampleEntropy(Datalist, r, m=2):
    th = r * np.std(Datalist)  # 容限阈值
    return EH.SampEn(Datalist, m, r=th)[0][-1]

"""功率谱密度"""

def find_psd_peak(signal, fs=1.0, nperseg=None, noverlap=None, nfft=None):
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    # 找到功率谱密度的峰值
    peak_index = np.argmax(psd)
    peak_freq = freqs[peak_index]
    peak_psd = psd[peak_index]

    return peak_freq, peak_psd
"""计算序列的熵"""
def calculate_entropy(sequence, base=None):
    value_counts = np.bincount(sequence)
    probabilities = value_counts / len(sequence)
    entropy_value = entropy(probabilities, base=base)
    return entropy_value
"""获取未经处理的序列数据"""
def get_origin_data(b_name,dV_threshold=0.000001,window_size=20):
    """输出量为(cycle,c,v,ic,soh)"""
    file_path = source_data_path + b_name + '_combine.xlsx'
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df = pd.DataFrame(df)
    cycles = set(df['Cycle_Index'].tolist())
    data_out = []
    cap_0 = 0
    SOH = 1
    print("锂电池数据整合：\n")
    for cycle in tqdm(cycles):
        df_n = df[df['Cycle_Index'] == cycle]
        "恒压阶段-计算最后的电流"
        df_I = df_n[df_n['Step_Index'] == 4]
        I_cv = df_I[(df_I['Current(A)'] <= 0.6) & (df_I['Current(A)'] >= 0.1)]['Current(A)'].tolist()
        CVT_list=df_I['Step_Time(s)'].tolist()
        if len(CVT_list)<=3:
            continue
        CVT=CVT_list[-1]-CVT_list[0]
        if len(I_cv) <=3:
            continue
        """恒流阶段-计电压"""
        df_V = df_n[df_n['Step_Index'] == 2]
        V_stop = df_V['Voltage(V)'].tolist()[-1]
        V_begin = V_stop - 0.4
        V_cc = df_V[(df_V['Voltage(V)'] >= V_begin) & (df_V['Voltage(V)'] <= V_stop)]['Voltage(V)'].tolist()
        CCT_list=df_V['Step_Time(s)'].tolist()
        if len(CCT_list)<=5:
            continue
        CCT=CCT_list[-1]-CCT_list[0]
        if len(V_cc) <=5:
            continue
        """恒流阶段-计算ic曲线"""
        df_IC = df_n[df_n['Step_Index'] == 2]
        df_c = df_n[(df_n['Step_Index'] == 2) | (df_n['Step_Index'] == 4)]
        C=df_IC['Charge_Capacity(Ah)'].tolist()
        V=df_IC['Voltage(V)'].tolist()
        dC = np.diff(C)  # 在末尾和开头添加0
        dV = np.diff(V)  # 在末尾和开头添加0
        zero_index = np.where(dV <= dV_threshold)
        dC = np.delete(dC, zero_index)
        dV = np.delete(dV, zero_index)
        V = np.delete(V, zero_index)
        dcdv = dC / dV
        if len(dcdv)==0 or len(dcdv)<window_size:
            continue
        """计算SOH"""
        df_SOH = df_n[df_n['Step_Index'] == 7]
        df_cap = df_SOH[(df_SOH['Voltage(V)'] <= 3.8) & (df_SOH['Voltage(V)'] >= 3.4)]
        if cycle == 1:
            cap_0=max(df_c['Charge_Capacity(Ah)']) - min(df_c['Charge_Capacity(Ah)'])
        else:
            cap_i = max(df_c['Charge_Capacity(Ah)']) - min(df_c['Charge_Capacity(Ah)'])
            SOH = cap_i / cap_0
        weights = np.ones(window_size) / window_size
        dcdv = np.convolve(dcdv, weights, mode='same')
        data_out.append([I_cv,V_cc,dcdv,CCT,CVT,SOH])
    return data_out
DATA_OUT=get_origin_data(Battery_list[0])
import random

def smooth_and_normalize_columns(arr, window_size=3):
    """
    Smooths all but the last column of the input array using a moving average
    and then normalizes these columns to the range [0, 1].

    Parameters:
    arr (np.ndarray): Input NumPy array with shape (n_samples, n_features).
    window_size (int): Size of the moving average window for smoothing.

    Returns:
    np.ndarray: NumPy array with smoothed and normalized columns.
    """
    # Ensure the input is a NumPy array
    arr = np.asarray(arr)

    # Initialize an array to hold the smoothed and normalized values
    smoothed_normalized_arr = np.zeros_like(arr)

    # Iterate over each column except the last one
    for col_idx in range(arr.shape[1] - 1):
        # Apply moving average smoothing
        col = arr[:, col_idx]
        smoothed_col = np.convolve(col, np.ones(window_size) / window_size, mode='valid')

        # Handle the edges of the array where the moving average cannot be computed
        # by padding with the first and last smoothed values
        pad_before = np.ones(window_size // 2) * smoothed_col[0]
        pad_after = np.ones(window_size // 2) * smoothed_col[-1]
        smoothed_col = np.concatenate((pad_before, smoothed_col, pad_after))

        # Normalize the smoothed column to the range [0, 1]
        min_val = smoothed_col.min()
        max_val = smoothed_col.max()
        normalized_col = (smoothed_col - min_val) / (max_val - min_val)

        # Store the normalized column in the result array
        smoothed_normalized_arr[:, col_idx] = normalized_col

    # Copy the last column as is
    smoothed_normalized_arr[:, -1] = arr[:, -1]

    return smoothed_normalized_arr


def calculate_std_dev(sequence):
    sequence = np.array(sequence)
    # 计算标准差
    std_dev = np.std(sequence)
    return std_dev


def calculate_mean(sequence):
    # 将输入转换为numpy数组（如果输入是列表）
    sequence = np.array(sequence)
    # 计算均值
    mean = np.mean(sequence)


def subtract_all_features(data_out):
    features=[]
    SOHs=[]
    out=[]
    for i in range(len(data_out)):
        I_cv,V_cc,dcdv,CCT,CVT,SOH = data_out[i][0],data_out[i][1],data_out[i][2],data_out[i][3],data_out[i][4],data_out[i][5]
        """电压的样本熵"""
        en_v=SampleEntropy(V_cc,r=1.5)
        """电压的功率谱密度"""
        _,psd_v=find_psd_peak(dcdv)
        """电压均值"""
        meam_v=np.mean(V_cc)
        """电压偏度，峰度"""
        skew_v=skew(V_cc)
        kurt_v=kurtosis(V_cc)
        """电压的标准差"""
        std_v=np.std(V_cc)
        """电流样本熵"""
        # print(I_cv)
        en_c=float(random.random())
        """电流的功率谱密度"""
        _,psd_c=find_psd_peak(I_cv)
        """电流均值"""
        mean_c=np.mean(I_cv)
        """电流的标准差"""
        std_c=np.std(I_cv)
        """电流偏度、峰度"""
        skew_c=skew(I_cv)
        kurt_c=kurtosis(I_cv)
        """增量容量的功率谱密度"""
        _,psd_ic=find_psd_peak(dcdv)
        """增量容量的峰值"""
        max_ic=max(dcdv)
        """ic的峰度、偏度"""
        kurt_ic=kurtosis(dcdv)
        skew_ic=skew(dcdv)
        features.append([meam_v,std_v,kurt_v,skew_v,en_v,kurt_ic,skew_ic,psd_ic,CCT,CVT])
        SOHs.append(SOH)
    features=np.stack(features,axis=0)
    SOHs=np.stack(np.reshape(SOHs,(-1,1)),axis=0)
    all_data=np.concatenate((features,SOHs),axis=1)
    all_data=smooth_and_normalize_columns(all_data,window_size=3)
    features=all_data[:,:10]
    SOHs=all_data[:,10]
    return features,SOHs
features,SOHs = subtract_all_features(DATA_OUT)
np.savez("CACLE.npz", arr1=features, arr2=SOHs)
def plot_features_vs_soh(features, SOH):
    """
    绘制16个特征与SOH的相关性散点图矩阵图。

    Parameters:
    features (numpy.ndarray): 形状为(n, 16)的特征数组
    soh (numpy.ndarray): 形状为(n, 1)的SOH数组
    """
    # 创建4x4的子图
    sns.set_theme(style="whitegrid", palette="deep")

    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 5, figsize=(12,4))
    axes = axes.ravel()  # 将二维数组转换为一维数组，方便遍历
    # 遍历每个特征
    # 定义特征名称
    feature_names = ['V_mean','V_std','V_kurt','V_skew','V_Sampen',  'IC_kurt', 'IC_skew','IC_psd', 'cct', 'cvt']
    pearson = []
    # 遍历每一列特征
    for i in range(10):
        # 创建DataFrame以方便Seaborn处理
        df = pd.DataFrame({
            'SOH': SOH.flatten(),
            feature_names[i]: features[:, i]
        })
        # 使用Seaborn绘制散点图和回归线
        sns.regplot(
            x="SOH",
            y=feature_names[i],
            data=df,
            ax=axes[i],
            scatter_kws={"color": 'lightblue', "alpha": 0.6},  # 中等蓝色
            line_kws={"color": "none"},  # 不画回归线
            fit_reg=False  # 禁用回归线
        )
        # 设置子图标题和标签
        axes[i].set_title(f"{feature_names[i]} vs. SOH", fontsize=16)
        axes[i].set_xlabel("SOH", fontsize=14)
        axes[i].set_ylabel(feature_names[i], fontsize=14)
        # 添加回归线方程
        from scipy.stats import pearsonr
        r, _ = pearsonr(df['SOH'], df[feature_names[i]])
        axes[i].text(0.7, 0.9, f"r={r:.2f}", transform=axes[i].transAxes)
        corr, _ = pearsonr(df['SOH'], df[feature_names[i]])
        pearson.append(corr)
    # 调整子图布局
    plt.tight_layout()
    plt.show()
    return pearson
persons=plot_features_vs_soh(features,SOHs)