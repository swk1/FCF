import argparse

import torch.nn as nn
import torch
import torch.nn.functional as F


class DeepJIT(nn.Module):
    def __init__(self, args):
        super(DeepJIT, self).__init__()
        self.args = args

        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num        

        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])

        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])

        # other information
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units
        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        # note that we can use this function for commit code line to get the information of the line
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # apply cnn 2d for each line in a commit code
        x = self.forward_msg(x=x, convs=convs_line)

        # apply cnn 2d for each file in a commit code
        x = x.reshape(n_batch, n_file, self.args.num_filters * len(self.args.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward(self, msg, code):
        x_msg = self.embed_msg(msg)
        x_msg = self.forward_msg(x_msg, self.convs_msg)

        x_code = self.embed_code(code)
        x_code = self.forward_code(x_code, self.convs_code_line, self.convs_code_file)

        x_commit = torch.cat((x_msg, x_code), 1)
        x_commit = self.dropout(x_commit)
        out = self.fc1(x_commit)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

def read_args():
    """
        解析命令行参数

        Returns:
            ArgumentParser: 包含所有配置参数的解析器对象

        参数说明:
            -train          启用训练模式
            -predict        启用预测模式
            -train_data     训练数据路径(.pkl格式)
            -dictionary_data 字典数据路径
            -pred_data      预测数据路径
            -load_model     要加载的模型路径
            -msg_length     提交消息最大长度(默认256)
            -code_line      每个代码块最大行数(默认10)
            -code_length    每行代码最大长度(默认512)
            -embedding_dim  词向量维度(默认64)
            -filter_sizes   卷积核尺寸(默认'1,2,3')
            -num_filters    卷积核数量(默认64)
            -hidden_units   隐藏层维度(默认512)
            -dropout_keep_prob dropout保留概率(默认0.5)
            -l2_reg_lambda  L2正则化系数(默认1e-5)
            -learning_rate  学习率(默认1e-4)
            -batch_size     批大小(默认64)
            -num_epochs     训练轮次(默认25)
            -save_dir       模型保存目录
            -project_name   项目名称(默认'deepjit')
            -device         运行设备(-1=CPU, 0+=GPU编号)
            -no-cuda        禁用CUDA
    """
    parser = argparse.ArgumentParser()

    # 模式选择
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')

    # 数据路径参数
    parser.add_argument('-train_data', type=str, help='the directory of our training data') # 训练数据目录
    parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data') # 字典数据目录

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    parser.add_argument('-pred_data', type=str, help='the directory of our testing data')

    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # 输入数据格式化参数
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')

    # 模型超参数
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=1e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    # 训练参数
    parser.add_argument('-batch_size', type=int, default=64, help='batch size') # 批次大小
    parser.add_argument('-num_epochs', type=int, default=25, help='the number of epochs') # 训练轮次
    parser.add_argument('-save_dir', type=str, default='', help='where to save the snapshot') # 模型保存路径
    parser.add_argument('-project_name', type=str, default='deepjit', help='save the model for project') # 项目名称（用于模型保存）

    # 硬件设置
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser