from sklearn.linear_model import LogisticRegression
import os
from baselines.utils.performance_measure import PerformanceMeasure
from baselines.utils.results_writer import ResultWriter
from sklearn import preprocessing
from baselines.utils.preprocess_data import load_data, load_test_dataframe

# 特征和标签列名定义
feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp", "sexp"]
label_name = ["is_buggy_commit"]

def LA_train_and_eval(baseline_name: str = None):
    # 数据加载部分（baseline_name = la）
    X_train, y_train, X_test, y_test = load_data(base_path, baseline_name)
    # 数据标准化处理
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    # ()模型训练
    model = LogisticRegression(max_iter=7000).fit(X_train, y_train)
    # 预测结果
    y_pred_prob = model.predict_proba(X_test)[:, 1] #是类别 1（缺陷）的概率
    y_pred = model.predict(X_test) # 直接返回模型预测的 类别标签（0 或 1）


    result_df = load_test_dataframe(base_path, baseline_name)
    result_df['defective_commit_pred'] = y_pred
    result_df['defective_commit_prob'] = y_pred_prob

    presults = PerformanceMeasure().eval_metrics(result_df=result_df)
    #print(presults)
    ResultWriter().write_result(result_path=result_path, method_name="LApredict",presults=presults)


def test():
    load_data(base_path, 'la')


if __name__ == "__main__":
    print("Running LA model")
    base_path = "data/"
    result_path = os.path.dirname(os.path.dirname(__file__)) + '\\results\\'
    print(result_path)
    LA_train_and_eval('la')
