import os


def DeepJIT_train_and_eval(baseline_name: str = None):
    raw_train = 'python3 -m baselines.DeepJIT.main -train -train_data data/deepjit/features_train.pkl -dictionary_data data/deepjit/dataset_dict.pkl  -device 0 -save_dir model/deepjit'

    raw_test = 'python3 -m baselines.DeepJIT.main -predict -pred_data data/deepjit/features_test.pkl -dictionary_data data/deepjit/dataset_dict.pkl -load_model model/deepjit/epoch_25.pt  -device 0'

    # train
    print("**********************Training*********************")
    return_code = os.system(raw_train)
    print(f"命令执行状态码: {return_code}")  # 正常应为0，非0表示出错
    print("**********************Evaluating*********************")
    return_code = os.system(raw_test)
    print(f"命令执行状态码: {return_code}")  # 正常应为0，非0表示出错


if __name__ == "__main__":
    print("Running deepjit model")
    DeepJIT_train_and_eval('deepjit')
    # pass
