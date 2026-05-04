import os

import transformers

from baselines.CC2Vec.cc2vecTransfor import read_args, set_seed, cc2ftr_train, cc2ftr_predict
from baselines.CC2Vec.jit_DExtended import Dextended_read_args, Dextended_predict, Dextended_train

cc2ftr_train1 = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_cc2ftr -train -train_data data/cc2vec/features_train.pkl  -dictionary_data data/cc2vec/dataset_dict.pkl -save-dir model/cc2vec/snapshot/cc2vec/ftr  '

cc2ftr_predict_train1 = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_cc2ftr -batch_size 256 -predict -predict_data  data/cc2vec/features_train.pkl -dictionary_data  data/cc2vec/dataset_dict.pkl -load_model model/cc2vec/snapshot/cc2vec/ftr/cc2vec/epoch_50.pt -name  train_cc2ftr.pkl '

cc2ftr_predict_test1 = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_cc2ftr -batch_size 256 -predict -predict_data  data/cc2vec/features_test.pkl -dictionary_data  data/cc2vec/dataset_dict.pkl -load_model model/cc2vec/snapshot/cc2vec/ftr/cc2vec/epoch_50.pt -name  test_cc2ftr.pkl'

deepjit_train1 = 'CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_DExtended -train -train_data data/deepjit/features_train.pkl -train_data_cc2ftr data/cc2vec/train_cc2ftr.pkl -dictionary_data data/deepjit/dataset_dict.pkl -save-dir model/cc2vec/snapshot/cc2vec/model'

deepjit_predict1 = "CUDA_VISIBLE_DEVICES={} python3  -m baselines.CC2Vec_Modified.jit_DExtended -predict -pred_data data/deepjit/features_test.pkl -pred_data_cc2ftr data/cc2vec/test_cc2ftr.pkl -dictionary_data data/deepjit/dataset_dict.pkl -load_model model/cc2vec/snapshot/cc2vec/model/epoch_50.pt "


def CC2Vec_train_and_eval():
    visible_device = 0
    cmd1 = cc2ftr_train.format(visible_device)

    print(cmd1)
    print("<<<<<<<<<<<<<<<<<<<< Step 1: training cc2vec>>>>>>>>>>>>>>>>>>>")
    os.system(cmd1)

    cmd2 = cc2ftr_predict_train1.format(visible_device)
    print(cmd2)
    print("<<<<<<<<<<<<<<<<<<<< Step 2: get cc2vec's representation for deepjit train data>>>>>>>>>>>>>>>>>>>")
    os.system(cmd2)

    cmd3 = cc2ftr_predict_test1.format(visible_device)
    print(cmd3)
    print("<<<<<<<<<<<<<<<<<<<< Step 3: get cc2vec's representation for deepjit test data>>>>>>>>>>>>>>>>>>>")
    os.system(cmd3)

    cmd4 = deepjit_train1.format(visible_device)
    print(cmd4)
    print("<<<<<<<<<<<<<<<<<<<< Step 4: training deepjit combined cc2vec representation>>>>>>>>>>>>>>>>>>>")
    os.system(cmd4)

    cmd5 = deepjit_predict1.format(visible_device)
    print(cmd5)
    print("<<<<<<<<<<<<<<<<<<<< Step 5: evaluating model>>>>>>>>>>>>>>>>>>>")
    os.system(cmd5)


def CC2Vec_train_and_eval2():
    params = read_args().parse_args()
    set_seed(seed=42)

    train_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/features_train.pkl"
    dictionary_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/dataset_dict.pkl"
    save_dir = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/snapshot/cc2vec/ftr"
    visible_device = 0 # 可用GPU
    print("<<<<<<<<<<<<<<<<<<<< Step 1: training cc2vec>>>>>>>>>>>>>>>>>>>")
    cc2ftr_train(params,train_data,dictionary_data,save_dir,visible_device)

    batch_size = 256
    predict_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/features_train.pkl"
    load_model = 'C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/snapshot/cc2vec/ftr/cc2vec/epoch_50.pt'
    name = "train_cc2ftr.pkl"
    print("<<<<<<<<<<<<<<<<<<<< Step 2: get cc2vec's representation for deepjit train data>>>>>>>>>>>>>>>>>>>")
    cc2ftr_predict(params,batch_size,predict_data,dictionary_data,load_model,name)

    predict_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/features_test.pkl"
    name ="test_cc2ftr.pkl"
    print("<<<<<<<<<<<<<<<<<<<< Step 3: get cc2vec's representation for deepjit test data>>>>>>>>>>>>>>>>>>>")
    cc2ftr_predict(params, batch_size, predict_data, dictionary_data, load_model, name)

    params = Dextended_read_args().parse_args()
    train_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/deepjit/features_train.pkl"
    train_data_cc2ftr = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/train_cc2ftr.pkl"
    dictionary_data ="C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/deepjit/dataset_dict.pkl"
    save_dir = 'C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/model/cc2vec/snapshot/cc2vec/model'
    print("<<<<<<<<<<<<<<<<<<<< Step 4: training deepjit combined cc2vec representation>>>>>>>>>>>>>>>>>>>")
    Dextended_train(params,train_data,train_data_cc2ftr,dictionary_data,save_dir)

    pred_data = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/deepjit/features_test.pkl"
    pred_data_cc2ftr = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/data/cc2vec/test_cc2ftr.pkl"
    load_model = "C:/Users/Administrator/Desktop/毕业论文/论文代码/JIT-BiCC-main/JIT-BiCC-main/model/cc2vec/snapshot/cc2vec/model/epoch_50.pt"
    print("<<<<<<<<<<<<<<<<<<<< Step 5: evaluating model>>>>>>>>>>>>>>>>>>>")
    Dextended_predict(params,pred_data,pred_data_cc2ftr,load_model)
if __name__ == "__main__":
    print("Runing CC2Vec model")
    CC2Vec_train_and_eval2()
