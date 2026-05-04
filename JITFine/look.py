import pickle
import pandas as pd

if __name__ == "__main__":
    with open('C:/Users/Administrator/Desktop/毕业论文/服务器版本代码/JIT-BiCC-main/data/jitfine/features_test.pkl', 'rb') as f:  # 'rb' 表示二进制只读模式
        data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        print("\n" + "=" * 50)
        print("DataFrame detected! Detailed info:")
        print("=" * 50)
        print(f"Shape: {data.shape}")  # 显示数据维度 (行数, 列数)
        print("\nFirst 10 rows:")
        print(data.head(10))  # 显示前10行
        print("\nColumns:")
        print(data.columns.tolist())  # 显示所有列名
        print("\nData types:")
        print(data.dtypes)  # 显示每列的数据类型
    else:
        # 如果不是DataFrame，尝试获取类型和其他信息
        print(f"\nData type: {type(data)}")
        # 如果是字典，尝试打印其键
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
