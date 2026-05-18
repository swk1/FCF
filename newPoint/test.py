from __future__ import absolute_import, division, print_function

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import csv
from torch.utils.data import DataLoader,  SequentialSampler
from tqdm import tqdm

from JITFine.concat.run import parse_args, commit_with_codes, deal_with_attns
from JITFine.config import config

# 添加自定义模块路径
sys.path.append("/mnt/sda/jyz/Code_Change/CommitMaster/")  # Change to root directory of your project


from JITFine.my_util import  TextDataset

# 初始化日志记录器
logger = logging.getLogger(__name__)

def test_configs(modelName):
    """初始化模型训练配置参数

        本函数创建并设置所有必要的训练参数，
        避免了每次运行时都需手动输入长命令行的繁琐。
        """
    # 创建命令行参数解析器对象
    parser = parse_args()

    # 新增不平衡学习配置
    parser.use_logits = True  # False = 原始逻辑, True = 代价敏感学习逻辑
    parser.use_sampler = None  # None / 'undersample' / 'oversample'
    parser.use_pos_weight = False  # 是否启用 pos_weight

    # 设置项目根目录路径（根据用户实际位置调整）
    route = config.get_config().get("current_directory")
    # 模型相关配置
    parser.output_dir = route + "JITFine/model/saved_models_"+modelName+"/checkpoints"  # 模型保存路径
    parser.model_name_or_path = "C:/Users/ZETTAKIT/PycharmProjects/JIT-BICC/local_roberta"  # 指向目录
    parser.tokenizer_name = "C:/Users/ZETTAKIT/PycharmProjects/JIT-BICC/local_tokenizer"  # 同样指向目录
    parser.do_test = True  # 测试模式
    # 数据文件路径配置
    parser.train_data_file = [
        route + "data/jitfine/changes_train.pkl",  # 训练数据文件：代码变更记录
        route + "data/jitfine/features_train.pkl"  # 训练数据文件：提取的特征
    ]
    parser.eval_data_file = [
        route + "data/jitfine/changes_valid.pkl",  # 验证数据文件：代码变更记录
        route + "data/jitfine/features_valid.pkl"  # 验证数据文件：提取的特征
    ]
    parser.test_data_file = [
        route + "data/jitfine/changes_test.pkl",  # 测试数据文件：代码变更记录
        route + "data/jitfine/features_test.pkl"  # 测试数据文件：提取的特征
    ]

    # 训练超参数设置
    parser.epochs = 50  # 训练轮数（epoch）
    parser.max_seq_length = 512  # 模型最大输入序列长度
    parser.max_msg_length = 64  # 提交信息部分的最大长度
    parser.train_batch_size = 64  # 训练批次大小 与训练时不同
    parser.eval_batch_size = 4  # 验证/测试批次大小，减小该值
    parser.learning_rate = 2e-5  # 学习率
    parser.max_grad_norm = 1.0  # 梯度裁剪的最大范数
    parser.evaluate_during_training = True
    parser.only_adds = True  # 仅针对新加入的行进行评估
    parser.buggy_line_filepath = route + "data/jitfine/changes_complete_buggy_line_level.pkl"
    parser.seed = 42  # 随机种子(确保结果可复现)

    return parser  # 返回配置好的参数解析器


def test2(args, model, tokenizer, best_threshold=0.5, save_preds=True, collect_attn=False):
    # 1) dataloader：Windows 建议 num_workers=0
    cache_dataset = os.path.dirname(args.test_data_file[0]) + f'/test_set_cache_msg{args.max_msg_length}.pkl'
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl'
    logger.info("Cache Dataset file at %s ", cache_dataset)
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0,                # ★ Windows: 0
        pin_memory=False,
        persistent_workers=False
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size   = %d", args.eval_batch_size)

    # 2) 在线指标（不攒全量）
    tp = fp = fn = tn = 0
    total_loss = 0.0
    steps = 0

    # 3) 结果流式落盘（可选）
    pred_path = os.path.join(args.output_dir, "predictions.csv")
    writer = None
    if save_preds:
        f_csv = open(pred_path, "w", newline='', encoding="utf-8")
        writer = csv.writer(f_csv, delimiter='\t')
        writer.writerow(["commit_id", "prob", "pred", "label"])

    bar = tqdm(eval_dataloader, total=len(eval_dataloader), desc="Testing", ncols=100)

    for batch in bar:
        inputs_ids, attn_masks, manual_features, labels = [x.to(args.device) for x in batch]
        with torch.no_grad():
            # ★ 默认不收集注意力，省内存
            loss, prob, attn_weights = model(
                inputs_ids, attn_masks, manual_features, labels,
                output_attentions=bool(collect_attn)
            )
            total_loss += loss.mean().item()
            steps += 1

            # prob: [B,1] -> numpy
            p = prob.squeeze(1).detach().cpu().numpy()
            y = labels.detach().cpu().numpy()
            pred = (p > best_threshold).astype(int)

        # 在线更新混淆矩阵
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())

        # 结果边算边写，不攒内存
        if save_preds:
            # 取 commit_id：从 dataset 读取当前 batch 的样本索引
            # 假设 DataLoader 没打乱（SequentialSampler），可以用一个游标
            # 更稳的做法：在 TextDataset __getitem__ 返回 commit_id，一并打包进 batch
            start = (steps-1) * args.eval_batch_size
            for i in range(len(p)):
                example = eval_dataset.examples[start + i]
                writer.writerow([example.commit_id, float(p[i]), int(pred[i]), int(y[i])])

        # 实时进度
        bar.set_postfix(avg_loss=round(total_loss / steps, 5))

        # 释放
        del prob, p, y, pred
        torch.cuda.empty_cache()

    if save_preds:
        f_csv.close()
        logger.info("Predictions saved to %s", pred_path)

    # 4) 计算指标（无需再拼接全量数组）
    precision = tp / (tp + fp + 1e-12) if (tp + fp) else 0.0
    recall    = tp / (tp + fn + 1e-12) if (tp + fn) else 0.0
    f1        = (2*precision*recall)/(precision+recall+1e-12) if (precision+recall) else 0.0

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
        "avg_loss": round(total_loss / max(1, steps), 6),
    }
    logger.info("***** Eval results *****")
    for k in sorted(result.keys()):
        logger.info("  %s = %s", k, str(round(result[k], 4)))



def test(args, model, tokenizer, modelName,best_threshold=0.5):
    # build dataloader
    cache_dataset = os.path.dirname(args.test_data_file[0]) + f'/test_set_cache_msg{args.max_msg_length}.pkl'
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl'
    logger.info("Cache Dataset file at %s ", cache_dataset)
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=args.eval_batch_size, num_workers=4
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    logits, y_trues, attns = [], [], []

    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit, attn_weights = model(
                inputs_ids, attn_masks, manual_features, labels, output_attentions=True
            )
            last_layer_attn_weights = attn_weights

            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            attns.append(last_layer_attn_weights.cpu().numpy())
        nb_eval_steps += 1

    # concat
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    attns = np.concatenate(attns, 0)

    y_probs = logits[:, -1]
    y_preds = y_probs > best_threshold

    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
    recall = recall_score(y_trues, y_preds, average='binary')
    precision = precision_score(y_trues, y_preds, average='binary')
    f1 = f1_score(y_trues, y_preds, average='binary')
    try:
        auc = roc_auc_score(y_trues, y_probs)
    except Exception:
        auc = float("nan")

    summary = {
        "eval_loss": float(eval_loss / max(nb_eval_steps, 1)),
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_auc": float(auc) if auc == auc else auc,
        "eval_threshold": float(best_threshold),
    }
    logger.info("***** Eval results (Overall) *****")
    for key in sorted(summary.keys()):
        v = summary[key]
        logger.info("  %s = %s", key, str(round(v, 6)) if isinstance(v, float) else str(v))

    # ========= 1) 构建 commit_hash -> project 映射（来自 features_test.pkl） =========
    # args.test_data_file 通常是 [changes_test.pkl, features_test.pkl]
    feat_file = None
    for fp in args.test_data_file:
        if isinstance(fp, str) and fp.endswith(".pkl") and ("features" in os.path.basename(fp).lower()):
            feat_file = fp
            break
    if feat_file is None:
        # 兜底：把最后一个当 features（你的数据就是这样）
        feat_file = args.test_data_file[-1]

    commit2proj = {}
    try:
        feat_df = pd.read_pickle(feat_file)
        if ("commit_hash" in feat_df.columns) and ("project" in feat_df.columns):
            commit2proj = dict(
                zip(
                    feat_df["commit_hash"].astype(str).str.strip().str.lower(),
                    feat_df["project"].astype(str).str.strip()
                )
            )
            logger.info("Loaded commit->project mapping from %s (n=%d, projects=%d)",
                        feat_file, len(commit2proj), feat_df["project"].nunique())
        else:
            logger.warning("features file %s lacks columns: commit_hash/project, project export will be skipped.", feat_file)
    except Exception as e:
        logger.warning("failed to load features file for project mapping: %s", str(e))

    # ========= 2) 保留你原有的 buggy-line cache / effort-aware 计算逻辑 =========
    result_rows_legacy = []   # 原 predictions.csv：4列
    result_rows_proj = []     # 新增：5列含 project

    cache_buggy_line = os.path.join(
        os.path.dirname(args.buggy_line_filepath),
        'changes_complete_buggy_line_level_cache.pkl'
    )
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    unknown_proj_cnt = 0

    for example, pred, prob, attn in zip(eval_dataset.examples, y_preds, y_probs, attns):
        cid = str(example.commit_id).strip()
        cid_key = cid.lower()

        # 原输出（兼容你现有逻辑）：commit_id / prob / pred / label
        result_rows_legacy.append([cid, float(prob), int(pred), int(example.label)])

        # 新输出（用于方案A按项目统计）
        proj = commit2proj.get(cid_key, "UNKNOWN") if commit2proj else "UNKNOWN"
        if proj == "UNKNOWN":
            unknown_proj_cnt += 1
        result_rows_proj.append([cid, proj, float(prob), int(pred), int(example.label)])

        # 你原来的行级分析逻辑（保持不变）
        if int(example.label) == 1 and int(pred) == 1 and '[ADD]' in example.input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_id]
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                example, attn, pred, cur_codes, cur_labels, args.only_adds
            )
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    if len(top_10_acc) > 0:
        logger.info(
            'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
                round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
                round(np.mean(top_20_percent_LOC_recall), 4),
                round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
        )

    if unknown_proj_cnt > 0:
        logger.warning("Project mapping missing for %d commits (project=UNKNOWN).", unknown_proj_cnt)

    # ========= 3) 输出文件 =========
    # (a) 保持原 predictions.csv（无表头，4列）
    pd.DataFrame(result_rows_legacy).to_csv(
        os.path.join(args.output_dir, "predictions.csv"),
        sep='\t', index=None, header=None
    )

    # (b) 新增：带 project 的预测明细（有表头，5列）
    df_pred = pd.DataFrame(
        result_rows_proj,
        columns=["commit_id", "project", "prob", "pred", "label"]
    )
    df_pred.to_csv(
        os.path.join(args.output_dir, "predictions_with_project.csv"),
        sep='\t', index=False
    )

    # (c) 新增：按项目汇总指标（方案A：以 project 为重复单元）
    proj_metrics = []
    if "project" in df_pred.columns and df_pred["project"].nunique() > 1:
        for proj, g in df_pred.groupby("project"):
            y = g["label"].astype(int).values
            yhat = g["pred"].astype(int).values
            p = g["prob"].astype(float).values

            r = recall_score(y, yhat, average='binary')
            pr = precision_score(y, yhat, average='binary')
            f = f1_score(y, yhat, average='binary')
            pos = int(np.sum(y))

            # AUC：若该项目测试子集只有单类，则无法计算 -> NaN
            try:
                a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            except Exception:
                a = float("nan")

            proj_metrics.append([proj, int(len(g)), pos, float(f), float(a) if a == a else a, float(r), float(pr)])

        df_proj = pd.DataFrame(
            proj_metrics,
            columns=["project", "n", "pos", "f1", "auc", "recall", "precision"]
        ).sort_values("project")

        df_proj.to_csv(
            os.path.join(args.output_dir, modelName+"_metrics.csv"),
            sep='\t', index=False
        )
        logger.info("Saved per-project metrics to %s", os.path.join(args.output_dir, "project_metrics.csv"))
    else:
        logger.warning("Per-project metrics skipped (project mapping missing or only one project).")