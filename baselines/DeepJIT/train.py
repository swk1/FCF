import pickle

import numpy as np

from baselines.DeepJIT.model import DeepJIT, read_args
import torch 
from tqdm import tqdm
from baselines.DeepJIT.padding import padding_data
from baselines.DeepJIT.utils import mini_batches_train, save, set_seed
import torch.nn as nn
import os, datetime
from torch.utils.tensorboard import SummaryWriter

from baselines.config import config

log_writer = None

def train_model(data, params):

    log_writer = SummaryWriter(params.project_name+"_log_runs")

    data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data
    
    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    # params.save_dir = os.path.join(params.save_dir, params.project_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    params.save_dir = os.path.join(params.save_dir, params.project_name)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    print("building model")
    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()
    for epoch in range(1, params.num_epochs + 1):
        total_loss = 0
        # building batches for training model
        batches = mini_batches_train(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
        for i, (batch) in enumerate(tqdm(batches)):
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:            
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print('Epoch %i / %i -- Total loss: %f' % (epoch, params.num_epochs, total_loss)) 
        log_writer.add_scalar("Train/Loss",float(total_loss), epoch)

        save(model, params.save_dir, 'epoch', epoch)

# 开始训练
def train (params):
    params.train_data =  config.get_current_directory() + "data/deepjit/features_train.pkl"
    params.dictionary_data = config.get_current_directory() + "data/deepjit/dataset_dict.pkl"
    params.device = 0
    params.save_dir = config.get_current_directory() + "model"
    # 加载训练数据
    data = pickle.load(open(params.train_data, 'rb'))
    ids, labels, msgs, codes = data
    labels = np.array(labels)
    # 加载字典
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary
    # 数据填充处理
    pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')
    pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
    # 重组数据并开始训练
    data = (pad_msg, pad_code, labels, dict_msg, dict_code)
    train_model(data=data, params=params)
    print("训练完成")

if __name__ == '__main__':
    # 解析参数并设置随机种子
    params = read_args().parse_args()
    set_seed(seed=42)
    params.device = 0
    train(params)
    print("训练完成")