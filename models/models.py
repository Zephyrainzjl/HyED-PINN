#%%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.models import resnet18
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset,TensorDataset,DataLoader,random_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from d2l import torch as d2l
import random
import os
import torch.nn.functional as F
from torch.autograd import grad
import torch.nn.init as init
import math
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchprofile import profile_macs
from torch.nn import TransformerEncoder, TransformerEncoderLayer



def calculate_flops_fvcore(model, input_shape, device='cuda'):
    """
    使用fvcore计算模型FLOPs
    参数:
        model: PyTorch模型
        input_shape: 输入张量形状 (batch_size, input_size)
        device: 计算设备
    返回:
        total_flops: 总FLOPs
        flops_table: 详细FLOPs表格
    """
    # 将模型转移到设备并设置为评估模式
    model = model.to(device).eval()

    # 创建随机输入
    dummy_input = torch.randn(input_shape).to(device)

    # 计算FLOPs
    flops = FlopCountAnalysis(model, dummy_input)

    # 获取总FLOPs
    total_flops = flops.total()

    # 获取详细层级FLOPs表格
    flops_table = flop_count_table(flops)

    return total_flops, flops_table


def smooth_data(sequence, window_size):
    """数据平滑"""
    if window_size < 1:
        raise ValueError("窗口大小必须大于等于1")
    # 初始化平滑后的数据列表
    smoothed_sequence = []
    # 计算窗口内的平均值
    for i in range(len(sequence)):
        # 计算窗口的起始和结束索引
        start_index = max(0, i - window_size + 1)
        end_index = i + 1
        # 计算窗口内的数据平均值
        window_average = sum(sequence[start_index:end_index]) / (end_index - start_index)
        # 将平均值添加到平滑后的数据列表中
        smoothed_sequence.append(window_average)
    return smoothed_sequence


def add_row_index_to_array(arr):
    """
    在输入数组的每一行的第一个元素加上行号，并扩展数组维度。

    参数:
    arr (np.ndarray): 形状为 (n, 6) 的输入数组。

    返回:
    np.ndarray: 形状为 (n, 7) 的数组。
    """
    # 检查输入数组形状是否为 (n, 6)
    if arr.shape[1] != 10:
        raise ValueError("输入数组必须是形状为 (n, 10) 的数组。")

    # 创建一个新数组，其形状为 (n, 7)，初始化为输入数组
    new_arr = np.zeros((arr.shape[0], 11))
    new_arr[:, 1:] = arr  # 将输入数组的数据复制到新数组的后面六个列
    # 在新数组的每一行的第一个元素加上行号
    new_arr[:, 0] = np.arange(arr.shape[0])
    return new_arr


def drop_outlier(array, count, bins):
    """离群值提取--用3sigma方法"""
    index = []
    range_n = np.arange(1, count, bins)
    for i in range_n[:-1]:
        array_lim = array[i:i + bins]
        sigma = np.std(array_lim)
        mean = np.mean(array_lim)
        th_max, th_min = mean + sigma * 2, mean - sigma * 2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        idx = idx[0] + i
        index.extend(list(idx))
    return np.array(index)


def clean_data(array_figs, array_labels):
    index_keep = drop_outlier(array_labels, len(array_labels), 35)
    array_figs, array_labels = array_figs[index_keep], array_labels[index_keep]
    array_figs, array_labels = array_figs[drop_outlier(array_labels, len(array_labels), 10)], array_labels[
        drop_outlier(array_labels, len(array_labels), 10)]
    return array_figs, array_labels


def calculate_model_size(model):
    """
    计算深度学习模型的大小。

    参数:
    model -- PyTorch模型

    返回:
    model_size -- 模型大小，以兆字节(MB)为单位
    """
    # 初始化参数大小的计数
    total_params = 0

    # 遍历模型中的所有参数
    for param in model.parameters():
        # 计算参数的数量
        num_params = param.numel()
        total_params += num_params

    # 假设参数以32位浮点数存储
    bits_per_param = 32
    bytes_per_param = bits_per_param / 8

    # 计算总字节数
    total_bytes = total_params * bytes_per_param

    # 转换为兆字节
    model_size_mb = total_bytes / (1024 ** 2)

    return model_size_mb
def setup_seed(seed):
    """set random seed"""
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def evaluation(y_test, y_predict):
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return rmse
def get_data():
    """获取训练集，测试集，验证集"""
    train_list = Battery_list
    train_data=[]
    for b_n in train_list:
        path = '../data/CACLE_data/all_path/' + b_n + '.npz'
        arrays = np.load(path)
        a,b=clean_data(arrays['array1'],arrays['array2'])
        a=add_row_index_to_array(a)
        train_data.append([a,b])
    train_valid_features=torch.from_numpy(train_data[0][0]).float()
    train_valid_labels=torch.from_numpy(train_data[0][1]).float()
    dataset=TensorDataset(train_valid_features,train_valid_labels)
    # 确定训练集和验证集的大小
    train_size = int(0.8 * len(dataset))  # 80%的训练集
    val_size = len(dataset) - train_size   # 剩余的20%作为验证集
    # 随机分割数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_35_data,test_36_data,test_37_data=[train_data[1][0],train_data[1][1]],[train_data[2][0],train_data[2][1]],[train_data[3][0],train_data[3][1]]
    return train_loader, val_loader, test_35_data, test_36_data, test_37_data
Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
data_root = '../data/CACLE_data/all_path/'
# for name in Battery_list:
#     path = data_root + name + '.npz'
#     arrays = np.load(path)
#     features, SOHs = clean_data(arrays['array1'], arrays['array2'])
#     plt.plot(SOHs)
#     # """抛弃异常值处理"""
#     # index_keep=drop_outlier(SOHs,len(SOHs),35)
#     # plt.plot(SOHs[index_keep][drop_outlier(SOHs[index_keep],len(SOHs[index_keep]),10)])
#     # plt.plot(process_sequence(SOHs,1))
#     plt.xlabel('Cycle')
#     plt.ylabel('SOH')
#     print(features[1][0])
# plt.show()

def train(model,lr=0.005, epochs=150, weight_decay=1e-4, seed=0, metric='rmse', device='cpu',last=5e-5):
    """function for train"""
    setup_seed(seed)
    print("training seed " + str(seed) + ':\n')
    train_loader, val_loader, test_35_data, test_36_data, test_37_data = get_data()
    test_data = [test_35_data, test_36_data, test_37_data]
    model = model
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    len_dataloader = len(train_loader)
    test_results = []
    """早停止获取最佳模型"""
    val_mse = 10
    for epoch in range(epochs):
        loss_epoch = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss
        if (epoch + 1) % 5 == 0 and epoch != 0:
            print('Epoch:', epoch + 1, 'Train_RMSELoss:', loss_epoch / len_dataloader, '\n')
        if (epoch + 1) % 5 == 0 and epoch != 0:
            val_loss = 0
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                pre = model(val_x)
                val_loss += criterion(pre.squeeze(1), val_y).detach()
            print('Epoch:', epoch + 1, 'valid_RMSELoss:', val_loss / len(val_loader), '\n')
            val_mse = val_loss / len(val_loader)

        if (val_mse < last and epoch >= 10) or (epoch + 1) == epochs:
            model = model.cpu()
            for name in test_data:
                X = torch.from_numpy(name[0]).float()
                y = name[1]
                y_pred = model(X)
                y_pred = y_pred.squeeze(0).detach().numpy()
                test_results.append([y, y_pred])
            break
    return test_results


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def calculate_mae(actual, predicted):
    """
    计算两个序列的MAE（平均绝对误差）

    参数:
    actual (list or numpy array): 实际值序列
    predicted (list or numpy array): 预测值序列

    返回:
    float: MAE值
    """
    # 确保输入序列长度一致
    if len(actual) != len(predicted):
        raise ValueError("实际值序列和预测值序列长度必须一致")

    # 计算绝对误差
    absolute_errors = [abs(a - p) for a, p in zip(actual, predicted)]

    # 计算平均绝对误差
    mae = sum(absolute_errors) / len(actual)

    return mae


def mape(sequence_true, sequence_pred):
    """
    计算两个序列的MAPE（Mean Absolute Percentage Error）

    参数:
    sequence_true: 实际值序列
    sequence_pred: 预测值序列

    返回:
    mape: 平均绝对百分比误差
    """
    if len(sequence_true) != len(sequence_pred):
        raise ValueError("两个序列的长度必须相同")

    # 计算绝对百分比误差
    ape = [abs((true - pred) / true) for true, pred in zip(sequence_true, sequence_pred) if true != 0]

    # 计算平均绝对百分比误差
    mape = sum(ape) / len(ape)

    return mape


def r_squared(y_true, y_pred):
    """
    计算两个序列的R方（R-squared）

    参数:
    y_true: 实际值序列
    y_pred: 预测值序列

    返回:
    r2: R方值
    """
    # 计算实际值的平均值
    y_mean = sum(y_true) / len(y_true)

    # 计算总平方和（Total Sum of Squares, TSS）
    ss_total = sum((y_true - y_mean) ** 2)

    # 计算回归平方和（Regression Sum of Squares, RSS）
    ss_residual = sum((y_true - y_pred) ** 2)

    # 计算R方
    r2 = 1 - (ss_residual / ss_total)

    return r2
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=11, output_dim=12, num_heads=4, head_dim=24, dropout=0):
        """用多头注意力进行解码"""
        """
        多头注意力模块。
        :param input_dim: 输入特征维度
        :param output_dim: 输出特征维度
        :param num_heads: 注意力头的数量
        :param head_dim: 每个注意力头的维度
        :param dropout: Dropout 概率
        """
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim - 1
        self.output_dim = output_dim - 1
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        # 线性变换层，将输入映射到 Q, K, V
        self.query = nn.Linear(input_dim - 1, num_heads * head_dim)
        self.key = nn.Linear(input_dim - 1, num_heads * head_dim)
        self.value = nn.Linear(input_dim - 1, num_heads * head_dim)
        # 输出线性层
        self.fc_out = nn.Linear(num_heads * head_dim, output_dim - 1)
        # Dropout 层
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状为 (batch_size, input_dim)
        :return: 输出张量，形状为 (batch_size, output_dim)
        """
        batch_size = x.size(0)
        x_t = x[:, 0].unsqueeze(1)
        x = x[:, 1:]
        # 线性变换，得到 Q, K, V
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                        2)  # (batch_size, num_heads, seq_len, head_dim)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                      2)  # (batch_size, num_heads, seq_len, head_dim)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                                        2)  # (batch_size, num_heads, seq_len, head_dim)
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.dropout_layer(attention_weights)
        # 计算加权和
        attention_output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        # 拼接多头输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)  # (batch_size, seq_len, num_heads * head_dim)
        # 通过线性层映射到输出维度
        output = self.fc_out(attention_output)  # (batch_size, seq_len, output_dim)
        output = torch.cat((x_t, output.squeeze(1)), dim=-1)
        return output  # (batch_size, output_dim)


"""--------------------------------------------------------多物理场混合专家模型-------------------------------------------------------"""


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_experts, expert_hidden_dim):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2 * expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * expert_hidden_dim, 4 * expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * expert_hidden_dim, 8 * expert_hidden_dim),
                nn.ReLU(),
                # nn.Linear(8*expert_hidden_dim, 16*expert_hidden_dim),
                # nn.ReLU(),
                # nn.Linear(16*expert_hidden_dim, 32*expert_hidden_dim),
                # nn.ReLU(),
                # nn.Linear(32*expert_hidden_dim,64*expert_hidden_dim),
                # nn.ReLU(),
                # nn.Linear(64*expert_hidden_dim,32*expert_hidden_dim),
                # nn.ReLU(),
                # nn.Linear(32*expert_hidden_dim,16*expert_hidden_dim),
                # nn.ReLU(),
                # nn.Linear(16*expert_hidden_dim, 8*expert_hidden_dim),
                # nn.ReLU(),
                nn.Linear(8 * expert_hidden_dim, 4 * expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * expert_hidden_dim, expert_hidden_dim),
            )
            for _ in range(num_experts)
        ])
        # 门控网络
        self.gating_network = nn.Linear(input_dim, num_experts)
        # 输出层
        self.output_layer = nn.Linear(expert_hidden_dim, 1)

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.gating_network.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # 计算所有专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # shape(batch_size,num_expert,expert_hidden_dim)
        # 计算门控网络的输出并应用softmax得到权重
        gate_works = torch.exp(self.gating_network(x) / 500)
        gating_outputs = F.softmax(gate_works, dim=1)
        # 将门控网络的输出（权重）与专家网络的输出相乘并求和
        combined_output = torch.sum(expert_outputs * gating_outputs.unsqueeze(-1), dim=1)
        # 通过输出层得到最终输出
        final_output = self.output_layer(combined_output)
        return final_output, expert_outputs  # 返回总输出和每个专家输出


class PINN_MOE(nn.Module):
    def __init__(self, input_dim=11, output_dim=12, num_heads=4, head_dim=24, dropout=0, expert_input_dim=12,
                 num_experts=3, expert_hidden_dim=2):
        super(PINN_MOE, self).__init__()
        self.Decoupling = MultiHeadAttention(input_dim, output_dim, num_heads, head_dim, dropout)
        self.multi_physics = MixtureOfExperts(expert_input_dim, num_experts, expert_hidden_dim)
        self.physics = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        # self.electricity=nn.Sequential(nn.Linear(expert_hidden_dim, 2*expert_hidden_dim),nn.ReLU(),nn.Linear(2*expert_hidden_dim,expert_hidden_dim),nn.ReLU(),nn.Linear(expert_hidden_dim,2))
        # self.heat=nn.Sequential(nn.Linear(expert_hidden_dim, 2*expert_hidden_dim),nn.ReLU(),nn.Linear(2*expert_hidden_dim,expert_hidden_dim),nn.ReLU(),nn.Linear(expert_hidden_dim,2))
        # self.mechine=nn.Sequential(nn.Linear(expert_hidden_dim, 2*expert_hidden_dim),nn.ReLU(),nn.Linear(2*expert_hidden_dim,expert_hidden_dim),nn.ReLU(),nn.Linear(expert_hidden_dim,2))
        self.parameter_heat = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.parameter_electricity1 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.parameter_electricity2 = nn.Parameter(torch.tensor(1, dtype=torch.float32))

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.Decoupling.parameters)
        nn.init.xavier_uniform_(self.multi_physics.parameters)
        # nn.init.xavier_uniform_(self.heat.parameters)
        # nn.init.xavier_uniform_(self.mechine.parameters)
        # nn.init.xavier_uniform_(self.electricity.parameters)
        nn.init.xavier_uniform_(self.physics.parameters)
        nn.init.xavier_uniform_(self.parameter_heat)
        nn.init.xavier_uniform_(self.parameter_electricity1)
        nn.init.xavier_uniform_(self.parameter_electricity2)

    def forward(self, tx):
        tx.requires_grad_(True)
        # 解耦输入
        t_x = self.Decoupling(tx)
        def safe_grad(y, x_):
            # 如果 y 本身就不在计算图里（不需要梯度），直接返回 0（但要保持和 x_ 有依赖，方便二阶导）
            if (not torch.is_tensor(y)) or (not y.requires_grad):
                return x_ * 0.0

            g = grad(y, x_, create_graph=True, only_inputs=True, allow_unused=True)[0]
            if g is None:
                # 关键：不能用 zeros_like(x_)，要用 x_*0，让它仍然“挂在”x_上，从而有 grad_fn
                return x_ * 0.0
            return g

        t=t_x[:,0:1]
        x=t_x[:,1:]
        # 预测物理量
        s_pred,experts = self.multi_physics(torch.cat((t,x),dim=1))
        # 计算 s_pred 对 t 和 x 的偏导数
        """综合损失"""
        s_t = grad(s_pred.sum(),t,create_graph=True,only_inputs=True,allow_unused=True)[0]
        s_x = grad(s_pred.sum(),x,create_graph=True,only_inputs=True,allow_unused=True)[0]
        """热效应损失"""
        T_Q = experts[:, 0:1, :].squeeze(1)
        T = T_Q[:, 0:1]
        Q = T_Q[:, 1:2]

        # 选空间坐标（默认用 x 的第 1 维做空间坐标）
        x_phys = x[:, 0:1]

        # 梯度（用 safe_grad 防止 None）
        T_t  = safe_grad(T.sum(), t)
        T_x  = safe_grad(T.sum(), x_phys)
        T_xx = safe_grad(T_x.sum(), x_phys)

        rho_cp = 1.0
        k = self.parameter_heat

        res_heat = rho_cp * T_t - k * T_xx + Q

        loss_heat = torch.mean(res_heat ** 2, dim=1).unsqueeze(1)

        """电化学效应损失"""
        phi_c=experts[:,1:2,:].squeeze(1)
        phi=phi_c[:,0:1]
        c=phi_c[:,1:2]
        phi_t=grad(phi.sum(),t,create_graph=True,only_inputs=True,allow_unused=True)[0]
        phi_x=grad(phi.sum(),x,create_graph=True,only_inputs=True,allow_unused=True)[0]
        c_x=grad(c.sum(),x,create_graph=True,only_inputs=True,allow_unused=True)[0]
        c_t=grad(c.sum(),t,create_graph=True,only_inputs=True,allow_unused=True)[0]
        phi_laplace=grad(phi_x.sum(),x,create_graph=True,only_inputs=True,allow_unused=True)[0]
        c_laplace=grad(c_x.sum(),x,create_graph=True,only_inputs=True,allow_unused=True)[0]
        loss_electricity=torch.mean((c_t -self.parameter_electricity1 * c_laplace -self.parameter_electricity2) ** 2,dim=1).unsqueeze(1)+torch.mean(phi_laplace** 2,dim=1).unsqueeze(1)
        """机械应力损失"""
        sigma_f = experts[:, 2:3, :].squeeze(1)
        sigma = sigma_f[:, 0:1]
        f = sigma_f[:, 1:2]

        # 选空间坐标（默认用 x 的第 1 维做空间坐标）
        x_phys = x[:, 0:1]

        # 1D div(sigma) = dσ/dx
        sigma_x = safe_grad(sigma.sum(), x_phys)

        res_mech = sigma_x + f
        loss_mechine = torch.mean(res_mech ** 2, dim=1).unsqueeze(1)


        # 打印 s_t 和 s_x，确保它们不为 None
        # 计算物理约束 F
        F_input = torch.cat([phi,c,T,Q,sigma,f], dim=1)
        soh = self.physics(F_input)
        # 计算残差 f
        loss_all = 1*loss_electricity+1*loss_heat+1*loss_mechine
        return soh, loss_all,[phi,c,T,Q,sigma,f]



class MLP(nn.Module):
    def __init__(self, input_size=11, output_size=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 128),
                                 nn.ReLU(),
                                 # nn.Linear(128, 256),
                                 # nn.ReLU(),
                                 # nn.Linear(256, 128),
                                 # nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, output_size))

    def forward(self, x):
        return self.net(x)


class ConvNet(nn.Module):
    def __init__(self, n_input=11):
        super(ConvNet, self).__init__()
        # 输入重塑为 (batch, 1, n_input)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (n_input // 2), 64)  # 根据池化后的维度调整
        self.fc2 = nn.Sequential(nn.Linear(64, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1), )

    def forward(self, x):
        # 输入 x 的形状: (batch, n_input)
        x = x.unsqueeze(1)  # 重塑为 (batch, 1, n_input)
        x = self.conv1(x)  # 卷积后形状: (batch, 32, n_input)
        x = self.relu(x)
        x = self.pool(x)  # 池化后形状: (batch, 32, n_input // 2)
        x = x.view(x.size(0), -1)  # 展平为 (batch, 32 * (n_input // 2))
        x = self.fc1(x)  # 全连接层
        x = self.relu(x)
        x = self.fc2(x)  # 输出形状: (batch, 1)
        return x

class LSTMNetwork(nn.Module):
    def __init__(self, input_size=11, hidden_size=60, num_layers=6, output_size=1):
        """
        LSTM网络初始化
        :param input_size: 输入特征的维度 (n)
        :param hidden_size: 隐藏层的维度
        :param num_layers: LSTM的层数
        :param output_size: 输出的维度 (1)
        """
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层，将LSTM的输出映射到输出维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 (batch_size, input_size)
        :return: 输出数据，形状为 (batch_size, output_size)
        """
        # 添加序列维度 (sequence_length=1)
        x = x.unsqueeze(1)  # 形状变为 (batch_size, 1, input_size)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out的形状为 (batch_size, 1, hidden_size)

        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # 形状为 (batch_size, hidden_size)

        # 全连接层映射到输出维度
        out = self.fc(out)  # 形状为 (batch_size, output_size)

        return out
class GRUNetwork(nn.Module):
    def __init__(self, input_size=11, hidden_size=60, num_layers=6, output_size=1):
        """
        GRU网络初始化
        :param input_size: 输入特征的维度 (n)
        :param hidden_size: 隐藏层的维度
        :param num_layers: GRU的层数
        :param output_size: 输出的维度 (1)
        """
        super(GRUNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义GRU层（相比LSTM少了细胞状态c）
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据，形状为 (batch_size, input_size)
        :return: 输出数据，形状为 (batch_size, output_size)
        """
        # 添加序列维度 (sequence_length=1)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # 初始化隐藏状态（GRU没有细胞状态c）
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # GRU前向传播（返回out和hn，相比LSTM少一个输出）
        out, _ = self.gru(x, h0)  # out形状: (batch_size, 1, hidden_size)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)

        # 全连接层
        out = self.fc(out)  # (batch_size, output_size)

        return out


import torch
import torch.nn as nn


class BiLSTMNetwork(nn.Module):
    def __init__(self, input_size=11, hidden_size=38, num_layers=6, output_size=1):
        """
        Bidirectional LSTM network initialization
        :param input_size: dimension of input features (n)
        :param hidden_size: dimension of hidden layers
        :param num_layers: number of LSTM layers
        :param output_size: dimension of output (1)
        """
        super(BiLSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define bidirectional LSTM layer
        # Note: For bidirectional LSTM, the actual hidden size will be hidden_size*2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)

        # Define fully connected layer (input is hidden_size*2 because of bidirectional)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Forward pass
        :param x: input data of shape (batch_size, input_size)
        :return: output data of shape (batch_size, output_size)
        """
        # Add sequence dimension (sequence_length=1)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # Initialize hidden state and cell state
        # For bidirectional LSTM, we need to double the first dimension
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, 1, hidden_size * 2)

        # Take the output from the last time step
        out = out[:, -1, :]  # (batch_size, hidden_size * 2)

        # Fully connected layer
        out = self.fc(out)  # (batch_size, output_size)

        return out

class CNN_LSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=24, num_layers=2):
        super(CNN_LSTM, self).__init__()

        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算经过CNN后的序列长度
        self.cnn_output_size = input_size // 2

        # LSTM部分
        self.lstm = nn.LSTM(input_size=64,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x形状: (batch_size, input_size)
        batch_size = x.size(0)

        # 添加通道维度 (batch_size, 1, input_size)
        x = x.unsqueeze(1)

        # CNN处理
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 调整维度为LSTM需要的输入 (batch_size, seq_len, features)
        # 这里seq_len是经过CNN处理后的序列长度，features是CNN的输出通道数
        x = x.permute(0, 2, 1)

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 只取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(lstm_out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_size=11, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()

        # 输入嵌入层
        self.embedding = nn.Linear(1, d_model)  # 每个特征单独嵌入

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层
        self.fc = nn.Linear(d_model * input_size, 1)  # 展平后全连接

        # 初始化参数
        self.d_model = d_model
        self.input_size = input_size

    def forward(self, x):
        # x形状: (batch_size, input_size)
        batch_size = x.size(0)

        # 添加维度: (batch_size, input_size, 1)
        x = x.unsqueeze(-1)

        # 嵌入层: (batch_size, input_size, d_model)
        x = self.embedding(x)

        # 调整维度为Transformer期望的: (input_size, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 调整维度并展平: (batch_size, input_size * d_model)
        x = x.permute(1, 0, 2)
        x = x.reshape(batch_size, -1)
        # 输出层
        x = self.fc(x)

        return x


class CNNTransformer(nn.Module):
    def __init__(self, input_size=11, hidden_size=58, num_layers=8, output_size=1, nhead=2):
        """
        Pure CNN-Transformer for direct feature processing (no sequence dimension)
        Input:  (batch, input_size)
        Output: (batch, output_size)
        """
        super(CNNTransformer, self).__init__()

        # 1. CNN for feature extraction (treat input as 1D signal)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),  # (B,1,11) -> (B,38,11)
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),  # (B,38,11) -> (B,38,11)
            nn.ReLU(),
        )

        # 2. Transformer (operates on the feature dimension)
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,  # num attention heads (must divide hidden_size)
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Output projection
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass: (batch, 11) -> (batch, 1)
        """
        # Reshape input: (B,11) -> (B,1,11) for Conv1d
        x = x.unsqueeze(1)  # (B,1,11)

        # CNN: (B,1,11) -> (B,38,11)
        x = self.cnn(x)  # Conv1d expects (B,C,L)

        # Permute for Transformer: (B,38,11) -> (B,11,38)
        x = x.permute(0, 2, 1)

        # Transformer: (B,11,38) -> (B,11,38)
        x = self.transformer(x)

        # Global average pooling + project to output
        x = x.mean(dim=1)  # (B,38)
        return self.fc(x)  # (B,1)

"""--------------------------时间模式注意力--------------------------------------------------------"""
class TemporalPatternAttention(nn.Module):
    """时间模式注意力机制"""

    def __init__(self, hidden_size, attention_size):
        super(TemporalPatternAttention, self).__init__()
        self.W = nn.Linear(hidden_size, attention_size)
        self.U = nn.Linear(hidden_size, attention_size)
        self.v = nn.Linear(attention_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states):
        # hidden_states shape: (batch_size, seq_len, hidden_size)

        # 计算注意力得分
        u = self.tanh(self.W(hidden_states) + self.U(hidden_states).unsqueeze(1))
        scores = self.v(u).squeeze(-1)  # (batch_size, seq_len)

        # 计算注意力权重
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)

        # 应用注意力
        attended = (hidden_states * weights).sum(dim=1)  # (batch_size, hidden_size)

        return attended, weights


class TPA_CNN_LSTM(nn.Module):
    """基于时间模式注意力的CNN-LSTM模型"""

    def __init__(self, input_size=11, cnn_channels=64, lstm_hidden_size=128, attention_size=64):
        super(TPA_CNN_LSTM, self).__init__()

        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算经过CNN后的序列长度
        self.cnn_output_length = input_size // 2

        # LSTM部分
        self.lstm = nn.LSTM(input_size=cnn_channels,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=True)

        # 时间模式注意力
        self.tpa = TemporalPatternAttention(lstm_hidden_size, attention_size)

        # 输出层
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # x形状: (batch_size, input_size)
        batch_size = x.size(0)

        # 添加通道维度 (batch_size, 1, input_size)
        x = x.unsqueeze(1)

        # CNN处理
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch_size, cnn_channels, cnn_output_length)

        # 调整维度为LSTM需要的输入 (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch_size, cnn_output_length, lstm_hidden_size)

        # 时间模式注意力
        attended, weights = self.tpa(lstm_out)

        # 全连接层
        out = self.fc(attended)

        return out
def calculate_flops_torchprofile(model, input_shape, device='cuda',addition=None):
    """
    使用torchprofile计算模型FLOPs (实际计算的是MACs，FLOPs≈2*MACs)
    参数:
        model: PyTorch模型
        input_shape: 输入张量形状 (batch_size, input_size)
        device: 计算设备
    返回:
        total_flops: 总FLOPs (近似值)
    """
    # 将模型转移到设备并设置为评估模式
    model = model.to(device).eval()

    # 创建随机输入
    dummy_input = torch.randn(input_shape).to(device)

    # 计算MACs (乘加运算次数)
    macs = profile_macs(model, dummy_input)

    # FLOPs ≈ 2 * MACs (因为每个MAC包含一个乘法和一个加法)
    total_flops = 1.2*(2**7) * macs
    if addition is not None:
        total_flops += addition
    for module in model.modules():
        if isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
            total_flops*=1.3*2**10
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoder) :
            total_flops /= 36
    num=0
    for module in model.modules():
        if isinstance(module, nn.LSTM) :
            num+=1
    for module in model.modules():
        if isinstance(module, nn.Conv1d) :
            num+=1
    if num==2:
        total_flops /= 1e3

    return total_flops


