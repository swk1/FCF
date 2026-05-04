import numpy as np
import pickle

from baselines.DeepJIT.evaluation import evaluation_model
from baselines.DeepJIT.model import read_args
from baselines.DeepJIT.padding import padding_data
from baselines.DeepJIT.utils import set_seed

log_writer = None

# 开始预测
def predict(params):
    params.pred_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/deepjit/features_test.pkl"
    params.load_model = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/model/deepjit/deepjit/epoch_25.pt"
    params.dictionary_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/deepjit/dataset_dict.pkl"
    params.device = 0
    # 加载测试数据
    data = pickle.load(open(params.pred_data, 'rb'))
    ids, labels, msgs, codes = data
    labels = np.array(labels)
    ids = np.array(ids)
    # 加载字典
    dictionary = pickle.load(open(params.dictionary_data, 'rb'))
    dict_msg, dict_code = dictionary
    # 数据填充处理
    pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')
    pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
    # 重组数据并开始评估
    data = (ids, pad_msg, pad_code, labels, dict_msg, dict_code)
    evaluation_model(data=data, params=params)
    print("训练完成")


if __name__ == '__main__':
    # 解析参数并设置随机种子
    params = read_args().parse_args()
    set_seed(seed=42)
    params.device = 0
    predict(params)
    print("预测完成")