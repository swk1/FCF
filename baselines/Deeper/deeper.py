import torch
import os
from baselines.utils.performance_measure import PerformanceMeasure
from baselines.utils.results_writer import ResultWriter
from sklearn import preprocessing
import time
from baselines.Deeper.LR import LR
from baselines.Deeper.DBN import DBN # 深度信念网络模型
import numpy as np
import math
import random
import torch.nn as nn
from baselines.utils.preprocess_data import load_data, load_test_dataframe
import warnings

warnings.filterwarnings("ignore") # 忽略警告信息

# 数据集的列名定义
colomn_names = ['project', 'parent_hashes', 'commit_hash', 'author_name',
                'author_email', 'author_date', 'author_date_unix_timestamp',
                'commit_message', 'la', 'ld', 'fileschanged', 'nf', 'ns', 'nd',
                'entropy', 'ndev', 'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp',
                'classification', 'fix', 'is_buggy_commit']
# 使用的特征名称列表
feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
# 标签名称
label_name = ["is_buggy_commit"]

# 设计随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:  # 如果有GPU设备
        torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子


def mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
       生成普通的小批量数据
       参数:
           X: 特征数据
           Y: 标签数据
           mini_batch_size: 批量大小(默认64)
           seed: 随机种子
       返回:
           小批量数据列表
       """

    m = X.shape[0]  # 样本数量
    mini_batches = list()
    # np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y  # 不进行洗牌

    # Step 2: Partition (X, Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitioning
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) # 计算完整的小批量数量

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        else:
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # # 处理剩余不足一个小批量的样本
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        if len(Y.shape) == 1:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m]
        else:
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def mini_batches_update(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = list()
    # np.random.seed(seed)

    # Step 1: No shuffle (X, Y)
    shuffled_X, shuffled_Y = X, Y

    Y = Y.tolist()
    Y_pos = [i for i in range(len(Y)) if Y[i] == 1.0]
    Y_neg = [i for i in range(len(Y)) if Y[i] == 0.0]

    # Step 2: Randomly pick mini_batch_size / 2 from each of positive and negative labels
    num_complete_minibatches = int(math.floor(m / float(mini_batch_size))) + 1
    for k in range(0, num_complete_minibatches):
        indexes = sorted(
            random.sample(Y_pos, int(mini_batch_size / 2)) + random.sample(Y_neg, int(mini_batch_size / 2)))
        mini_batch_X, mini_batch_Y = shuffled_X[indexes], shuffled_Y[indexes]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def DBN_JIT(train_features, train_labels, test_features, test_labels, hidden_units=[20, 12, 12], num_epochs_LR=200):
    """
       深度信念网络(DBN)与逻辑回归(LR)联合训练和预测
       参数:
           train_features: 训练特征
           train_labels: 训练标签
           test_features: 测试特征
           test_labels: 测试标签
           hidden_units: DBN隐藏层单元数(默认[20,12,12])
           num_epochs_LR: LR训练轮数(默认200)
       返回:
           测试集的预测概率
    """
    # starttime = time.time()
    dbn_model = DBN(visible_units=train_features.shape[1], # 输入单元数
                    hidden_units=hidden_units, # 隐藏层结构
                    use_gpu=False) # 不使用GPU
    # 静态训练DBN模型(10轮)
    dbn_model.train_static(train_features, train_labels, num_epochs=20)

    # 2. 使用DBN提取特征
    DBN_train_features, _ = dbn_model.forward(train_features) # 训练集特征提取
    DBN_test_features, _ = dbn_model.forward(test_features) # 测试集特征提取
    DBN_train_features = DBN_train_features.numpy()
    DBN_test_features = DBN_test_features.numpy()

    # 3. 合并原始特征和DBN提取特征
    train_features = np.hstack((train_features, DBN_train_features))
    test_features = np.hstack((test_features, DBN_test_features))

    # 4. 准备逻辑回归模型
    if len(train_labels.shape) == 1:
        num_classes = 1 # 二分类问题
    else:
        num_classes = train_labels.shape[1]
    # lr_model = LR(input_size=hidden_units, num_classes=num_classes)
    lr_model = LR(input_size=train_features.shape[1], num_classes=num_classes) # 不清楚
    optimizer = torch.optim.Adam(lr_model.parameters(), lr=0.00001) # Adam优化器

    # 6. 训练逻辑回归模型
    steps = 0
    batches_test = mini_batches(X=test_features, Y=test_labels)
    for epoch in range(1, num_epochs_LR + 1): # Epoch：整个训练集被训练几次
        # building batches for training model
        batches_train = mini_batches_update(X=train_features, Y=train_labels) # 平衡批次
        for batch in batches_train:
            x_batch, y_batch = batch
            x_batch, y_batch = torch.tensor(x_batch).float(), torch.tensor(y_batch).float()

            # 前向传播
            optimizer.zero_grad()
            predict = lr_model.forward(x_batch)
            # 计算损失
            loss = nn.BCELoss()
            loss = loss(predict, y_batch)
            # 反向传播
            loss.backward()
            optimizer.step()
            # 打印训练信息
            steps += 1 # 小批次里的数据
            if steps % 50 == 0:
                print('\rEpoch: {} step: {} - loss: {:.6f}'.format(epoch, steps, loss.item()))
    y_pred_prob, lables = lr_model.predict(data=batches_test)

    return y_pred_prob


def DBN_train_and_eval(baseline_name: str):
    """
       训练和评估DBN模型
       参数:
           baseline_name: 基准模型名称
    """
    # 1. 设置随机种子并加载数据
    set_seed(seed=39)
    X_train, y_train, X_test, y_test = load_data(base_path, baseline_name)
    # 2. 数据标准化
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    # 3. 构建并训练模型
    print(f"building model {baseline_name}")
    y_pred_prob = DBN_JIT(X_train, y_train, X_test, y_test)
    # 4. 准备结果数据
    result_df = load_test_dataframe(base_path, baseline_name)
    result_df["defective_commit_prob"] = y_pred_prob # 缺陷概率
    result_df["defective_commit_pred"] = [1.0 if p >= 0.5 else 0.0 for p in y_pred_prob] # 预测标签
    # 5. 评估性能并保存结果
    presults = PerformanceMeasure().eval_metrics(result_df=result_df)
    # print(presults)
    ResultWriter().write_result(result_path=result_path, method_name="Deeper",
                                presults=presults)

if __name__ == "__main__":
    print("Running deeper model")
    base_path = "data/"
    result_path = os.path.dirname(os.path.dirname(__file__)) + '\\results\\'
    DBN_train_and_eval('deeper')
