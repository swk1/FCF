from __future__ import absolute_import, division, print_function
import multiprocessing as mp
import logging
import os
import pickle
import random
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader,  SequentialSampler, RandomSampler
from transformers import ( get_linear_schedule_with_warmup,RobertaConfig,  RobertaTokenizer, RobertaModel)
from tqdm import tqdm

from JITFine.newPointModel.Enum.Model_Enum import ModelType
from JITFine.newPointModel.model.FCFModel import FCFModel
from JITFine.newPointModel.model.FilmModel import FilmModel
from JITFine.newPointModel.model.FilmModel2 import FilmModel2
from JITFine.newPointModel.model.MultiheadAttentionModel import CroAttModel
from JITFine.config import parse_args
from JITFine.config import config
from baselines.utils.results_writer import ResultWriter

# 添加自定义模块路径
sys.path.append("/mnt/sda/jyz/Code_Change/CommitMaster/")  # Change to root directory of your project

from JITFine.my_util import TextDataset, eval_result, preprocess_code_line, \
    get_line_level_metrics, create_path_if_not_exist
from sklearn.metrics import recall_score, precision_score, f1_score

# 初始化日志记录器
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)

    # Todo: recover evaluation strategy after development
    args.save_steps = len(train_dataloader) // 5
    # args.save_steps = len(train_dataloader)

    args.warmup_steps = 0
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    base_lr = args.learning_rate  # e.g. 1e-5
    head_lr = getattr(args, "head_lr", 1e-4)

    no_decay = ['bias', 'LayerNorm.weight']

    encoder_params = list(model.encoder.named_parameters())
    head_named_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("encoder.")]

    optimizer_grouped_parameters = [
        # encoder decay / no_decay
        {"params": [p for n, p in encoder_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": base_lr},
        {"params": [p for n, p in encoder_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": base_lr},

        # head decay / no_decay
        {"params": [p for n, p in head_named_params if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, "lr": head_lr},
        {"params": [p for n, p in head_named_params if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": head_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)


    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    best_f1 = 0
    patience = 0
    model.zero_grad()

    global_step = 0

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits, _ = model(inputs_ids, attn_masks, manual_features, labels)
            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % args.save_steps == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            if (step + 1) % args.save_steps == 0:
                results = evaluate(args, model, tokenizer, eval_when_training=True)
                logger.info("Saving epoch %d step %d model, patience %d", idx, global_step, patience)
                # Save model checkpoint
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1']
                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_f1, 4))
                    logger.info("  " + "*" * 20)

                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                    patience = 0
                    torch.save({
                        'epoch': idx,
                        'step': step,
                        'patience': patience,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                else:
                    patience += 1
                    if patience > args.patience:
                        logger.info('patience greater than {}, early stop!'.format(args.patience))
                        return


def evaluate(args, model, tokenizer, eval_when_training=False):
    """模型评估函数"""
    # 1. 构建测试数据集的缓存路径
    cache_dataset = os.path.dirname(args.eval_data_file[0]) + f'/valid_set_cache_msg{args.max_msg_length}.pkl'
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl' # 完整的缓存文件路径
    logger.info("Cache Dataset file at %s ", cache_dataset)
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb')) # 从缓存加载预处理过的数据集
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file, mode='valid')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    # 3. 创建测试数据加载器
    eval_sampler = SequentialSampler(eval_dataset) # 顺序采样器（确保数据顺序不变）
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4) # 使用4个进程加载数据

    # 4. 多GPU设置
    # 如果有多块GPU，将模型包装为DataParallel进行并行推理
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # 5. 打印测试信息
    logger.info("***** 开始预测 *****")
    logger.info("  验证集样本数量：%d", len(eval_dataset))
    logger.info("  评估批次大小：%d", args.eval_batch_size)

    # 6. 初始化评估指标
    eval_loss = 0.0 # 累计损失
    nb_eval_steps = 0 # 评估步数计数器
    logits = [] # 存储模型输出的原始预测值
    y_trues = [] # 存储真实标签

    total_batches = len(eval_dataloader)
    progress_bar = tqdm(
        total=total_batches,
        desc="评估进度",
        unit="batch",
        bar_format="{l_bar}{bar:40}{r_bar}",
        position=0,
        leave=True,
        disable=False,  # 可以根据args参数控制是否显示
        colour='green'
    )

    # 7. 开始评估循环
    model.eval()
    for step, batch in enumerate(eval_dataloader):

        # 将批数据移动到指定设备（GPU/CPU）
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit, _ = model(inputs_ids, attn_masks, manual_features, labels)
            torch.cuda.empty_cache()
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
        progress_bar.update(1)  # 更新進度條
        # 更新进度条信息
        if step % max(1, total_batches // 10) == 0:  # 每处理10%的批次更新一次
            avg_loss = eval_loss / (step + 1)
            progress_bar.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "steps": f"{step + 1}/{total_batches}"
            })
    progress_bar.close() # 清理资源
    # 8. 合并所有批次的评估结果
    logits = np.concatenate(logits, 0)  # - 按样本维度（第0维）拼接logits
    y_trues = np.concatenate(y_trues, 0) # - 按样本维度拼接真实标签
    best_threshold = 0.5 # 设置二分类阈值（默认0.5）

    # 9. 计算评估指标
    #修改一下
    y_preds = logits[:, -1] > best_threshold # 将模型输出转换为二分类预测（大于阈值视为正类）
    recall = recall_score(y_trues, y_preds, average='binary') # 计算召回率（真正例比例）
    precision = precision_score(y_trues, y_preds, average='binary') # 计算精确率（预测为正例中的真实正例比例）
    f1 = f1_score(y_trues, y_preds, average='binary') # 计算F1分数（精确率和召回率的调和平均）
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** 预测结果 *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test2(args, model, tokenizer, best_threshold=0.5):
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
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    attns = []
    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit, attn_weights = model(inputs_ids, attn_masks, manual_features, labels, output_attentions=True)
            last_layer_attn_weights = attn_weights
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
            attns.append(last_layer_attn_weights.cpu().numpy())

        nb_eval_steps += 1
    # output result
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    attns = np.concatenate(attns, 0)

    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    result = []

    cache_buggy_line = os.path.join(os.path.dirname(args.buggy_line_filepath),
                                    'changes_complete_buggy_line_level_cache.pkl')
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    for example, pred, prob, attn in zip(eval_dataset.examples, y_preds, logits[:, -1], attns):
        result.append([example.commit_id, prob, pred, example.label])

        # calculate
        if int(example.label) == 1 and int(pred) == 1 and '[ADD]' in example.input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_id]
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = deal_with_attns(
                example, attn,
                pred, cur_codes,
                cur_labels, args.only_adds)
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    logger.info(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )
    RF_result = pd.DataFrame(result)
    RF_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)

def test(args, model, tokenizer, modelName, best_threshold=0.5):
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
    # Windows/稳定性：建议 num_workers=0（你 test2 已这么做）
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=args.eval_batch_size, num_workers=0
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
    feat_file = None
    for fp in args.test_data_file:
        if isinstance(fp, str) and fp.endswith(".pkl") and ("features" in os.path.basename(fp).lower()):
            feat_file = fp
            break
    if feat_file is None:
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

    # ========= 2) buggy-line cache / effort-aware 计算逻辑 =========
    result_rows_legacy = []   # predictions.csv：4列
    result_rows_proj = []     # predictions_with_project.csv：5列
    effort_rows = []          # ★新增：commit 级 effort 指标（用于按项目聚合）

    cache_buggy_line = os.path.join(
        os.path.dirname(args.buggy_line_filepath),
        'changes_complete_buggy_line_level_cache.pkl'
    )
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall = [], [], []
    top_10_acc, top_5_acc = [], []
    unknown_proj_cnt = 0

    for example, pred, prob, attn in zip(eval_dataset.examples, y_preds, y_probs, attns):
        cid = str(example.commit_id).strip()
        cid_key = cid.lower()

        # 原输出（兼容你现有逻辑）：commit_id / prob / pred / label
        result_rows_legacy.append([cid, float(prob), int(pred), int(example.label)])

        # 新输出：含 project
        proj = commit2proj.get(cid_key, "UNKNOWN") if commit2proj else "UNKNOWN"
        if proj == "UNKNOWN":
            unknown_proj_cnt += 1
        result_rows_proj.append([cid, proj, float(prob), int(pred), int(example.label)])

        # effort-aware（保持你原有触发条件）
        if int(example.label) == 1 and int(pred) == 1 and '[ADD]' in example.input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_id]
            cur_IFA, cur_recall20, cur_effort20, cur_top10, cur_top5 = deal_with_attns(
                example, attn, pred, cur_codes, cur_labels, args.only_adds
            )
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_recall20)
            effort_at_20_percent_LOC_recall.append(cur_effort20)
            top_10_acc.append(cur_top10)
            top_5_acc.append(cur_top5)

            # ★新增：写入 commit 级 effort 记录，用于项目聚合
            effort_rows.append([
                cid, proj,
                float(cur_IFA),
                float(cur_recall20),
                float(cur_effort20)
            ])

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
    # (a) 原 predictions.csv（无表头，4列）
    pd.DataFrame(result_rows_legacy).to_csv(
        os.path.join(args.output_dir, "predictions.csv"),
        sep='\t', index=None, header=None
    )

    # (b) 带 project 的预测明细（有表头，5列）
    df_pred = pd.DataFrame(
        result_rows_proj,
        columns=["commit_id", "project", "prob", "pred", "label"]
    )
    df_pred.to_csv(
        os.path.join(args.output_dir, "predictions_with_project.csv"),
        sep='\t', index=False
    )

    # (c) ★可选：保存 commit 级 effort 指标明细（方便你排查/画图）
    if len(effort_rows) > 0:
        df_effort = pd.DataFrame(
            effort_rows,
            columns=[
                "commit_id", "project",
                "ifa",
                "recall_at_20_percent_effort",
                "effort_at_20_percent_loc_recall"
            ]
        )
        df_effort.to_csv(
            os.path.join(args.output_dir, "effort_with_project.csv"),
            sep='\t', index=False
        )
    else:
        df_effort = None

    # (d) 按项目汇总：分类指标 + ★新增 3 个 effort-aware 指标
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

            try:
                a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
            except Exception:
                a = float("nan")

            # ★新增：该项目的 effort 指标（对齐你已有逻辑：只统计能计算出来的 commit）
            if df_effort is not None:
                eg = df_effort[df_effort["project"] == proj]
                effort_n = int(len(eg))
                if effort_n > 0:
                    ifa_mean = float(np.nanmean(eg["ifa"].values))
                    recall20_mean = float(np.nanmean(eg["recall_at_20_percent_effort"].values))
                    effort20_mean = float(np.nanmean(eg["effort_at_20_percent_loc_recall"].values))
                else:
                    ifa_mean = float("nan")
                    recall20_mean = float("nan")
                    effort20_mean = float("nan")
            else:
                effort_n = 0
                ifa_mean = float("nan")
                recall20_mean = float("nan")
                effort20_mean = float("nan")

            proj_metrics.append([
                proj, int(len(g)), pos,
                float(f), float(a) if a == a else a, float(r), float(pr),
                effort_n, ifa_mean, recall20_mean, effort20_mean
            ])

        df_proj = pd.DataFrame(
            proj_metrics,
            columns=[
                "project", "n", "pos",
                "f1", "auc", "recall", "precision",
                "effort_n",
                "ifa",
                "recall_at_20_percent_effort",
                "effort_at_20_percent_loc_recall"
            ]
        ).sort_values("project")

        out_path = os.path.join(args.output_dir, modelName + "_metrics.csv")
        df_proj.to_csv(out_path, sep='\t', index=False)
        logger.info("Saved per-project metrics to %s", out_path)
    else:
        logger.warning("Per-project metrics skipped (project mapping missing or only one project).")




def commit_with_codes(filepath, tokenizer):
    data = pd.read_pickle(filepath)
    commit2codes = []
    idx2label = []
    for _, item in data.iterrows():
        commit_id, idx, changed_type, label, raw_changed_line, changed_line = item
        line_tokens = [token.replace('\u0120', '') for token in tokenizer.tokenize(changed_line)]
        for token in line_tokens:
            commit2codes.append([commit_id, idx, changed_type, token])
        idx2label.append([commit_id, idx, label])
    commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token'])
    idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
    return commit2codes, idx2label


def deal_with_attns(item, attns, pred, commit2codes, idx2label, only_adds=False):
    '''
    score for each token
    :param item:
    :param attns:
    :param pred:
    :param commit2codes:
    :param idx2label:
    :return:
    '''
    commit_id = item.commit_id
    input_tokens = item.input_tokens
    commit_label = item.label

    # remove msg,cls,eos,del
    begin_pos = input_tokens.index('[ADD]')
    end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1

    attn_df = pd.DataFrame()
    attn_df['token'] = [token.replace('\u0120', '') for token in
                        input_tokens[begin_pos:end_pos]]
    # average score for multi-heads
    attns = attns.mean(axis=0)[begin_pos:end_pos]
    attn_df['score'] = attns
    attn_df = attn_df.sort_values(by='score', ascending=False)
    attn_df = attn_df.groupby('token').sum()
    attn_df['token'] = attn_df.index
    attn_df = attn_df.reset_index(drop=True)

    # calculate score for each line in commit
    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # only count for added lines
    commit2codes = commit2codes.drop('commit_id', axis=1)
    commit2codes = commit2codes.drop('changed_type', axis=1)

    result_df = pd.merge(commit2codes, attn_df, how='left', on='token')
    result_df = result_df.groupby(['idx']).sum()
    result_df = result_df.reset_index(drop=False)

    result_df = pd.merge(result_df, idx2label, how='inner', on='idx')
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        result_df['score'].tolist(), result_df['label'].tolist())
    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc


def main(args,modelName):
    # 检查CUDA是否可用，优先使用GPU加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # 设置日志格式：时间戳 - 日志级别 - 日志名称 - 日志信息
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 记录硬件信息（设备类型和GPU数量）
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # 3. 设置随机种子保证结果可复现
    set_seed(args)
    # 4. 加载模型配置
    # 根据参数加载RoBERTa模型配置：
    # - 如果指定了config_name参数，则使用该名称的配置
    # - 否则使用model_name_or_path指定的模型默认配置
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.num_labels # 设置模型标签数量（二分类任务设置为1）
    config.feature_size = args.feature_size   # 设置特征向量大小（人工提取的特征维度）

    # 5. 加载分词器并添加特殊标记
    # ------------------------------------------------------------
    # 加载RoBERTa分词器（需要访问Hugging Face模型仓库）
    # 注意：需要网络连接，国内可能需要代理
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]} # 定义项目中需要添加的特殊标记
    tokenizer.add_special_tokens(special_tokens_dict) # 将特殊标记添加到分词器词汇表中

    # 初始化基础RoBERTa模型
    # ----------------------------------------------------
    # 从预训练路径加载RoBERTa基础模型
    # 注意：args.model_name_or_path 应为有效的模型标识符或本地路径
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Training/evaluation parameters %s", args)

    # 构建自定义模型
    if modelName == ModelType.CroAttModel.value:
        model = CroAttModel(model, config, tokenizer, args)
        logger.info("Training/evaluation model %s", modelName)
    elif modelName == ModelType.FiLM.value:
        model = FilmModel(model, config, tokenizer, args)
        logger.info("Training/evaluation model %s", modelName)
    elif modelName == ModelType.FCF.value:
        model = FCFModel(model, config, tokenizer, args)
        logger.info("Training/evaluation model %s", modelName)
    elif modelName == ModelType.FiLM2.value:
        model = FilmModel2(model, config, tokenizer, args)

    # 训练模式: 启动训练
    if args.do_train:
        # 加载预训练检查点（如果指定）
        if args.semantic_checkpoint:
            semantic_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.semantic_checkpoint, '{}'.format(semantic_checkpoint_prefix))
            logger.info("Loading semantic checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if args.manual_checkpoint:
            manual_checkpoint_prefix = 'checkpoint-best-f1/model.bin'
            output_dir = os.path.join(args.manual_checkpoint, '{}'.format(manual_checkpoint_prefix))
            logger.info("Loading manual checkpoint from {}".format(output_dir))
            checkpoint = torch.load(output_dir)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # 准备训练数据集
        logger.info("准备训练数据集")
        train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
        # 启动训练流程
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {"experiment_name": "JITDP"}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)
        result = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = '/checkpoint-best-f1/model.bin'
        output_dir = args.output_dir + checkpoint_prefix
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 显示成功加载的模型对应的训练轮次
        logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        model.to(args.device)
        test(args, model, tokenizer,modelName, best_threshold=0.5)
        evalResults = eval_result(os.path.join(args.output_dir, "predictions.csv"), args.test_data_file[-1])
        ResultWriter().write_result(result_path=output_dir, method_name="concat", presults=evalResults)
    return results

def configs(modelName):
    """初始化模型训练配置参数

        本函数创建并设置所有必要的训练参数，
        避免了每次运行时都需手动输入长命令行的繁琐。
        """

    # 创建命令行参数解析器对象
    parser = parse_args()

    # 设置项目根目录路径（根据用户实际位置调整）
    route = config.get_config().get("current_directory")

    # 模型相关配置
    parser.output_dir = route + "JITFine/model/saved_models_concat"+modelName+"/checkpoints"  # 模型保存路径
    parser.result_dir = route + "JITFine/model/saved_models_concat/result"
    parser.model_name_or_path = "C:/Users/ZETTAKIT/PycharmProjects/JIT-BICC/local_roberta"  # 指向目录
    parser.tokenizer_name = "C:/Users/ZETTAKIT/PycharmProjects/JIT-BICC/local_tokenizer"  # 同样指向目录
    parser.do_train = True # 训练模式
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
    parser.epochs = 50  # 训练轮数（epochs）
    parser.max_seq_length = 512  # 模型最大输入序列长度
    parser.max_msg_length = 64  # 提交信息部分的最大长度
    parser.train_batch_size = 24  # 训练批次大小
    parser.eval_batch_size = 128  # 验证/测试批次大小
    parser.learning_rate = 1e-5  # 学习率
    parser.max_grad_norm = 1.0  # 梯度裁剪的最大范数
    parser.evaluate_during_training = True
    parser.feature_size = 14  # 手动特征向量的维度大小
    parser.patience = 10  # 早停(early stopping)的耐心值(patience)
    parser.seed = 42  # 随机种子(确保结果可复现)
    parser.head_lr = 2e-4

    #film2相关开关
    parser.film_on = "token" # "token" 或 "cls"
    parser.film_scale = 0.1

    return parser  # 返回配置好的参数解析器

def eval_configs(modelName):
    """初始化模型预测配置参数
       本函数创建并设置所有必要的训练参数，
       避免了每次运行时都需手动输入长命令行的繁琐。
    """
    # 创建命令行参数解析器对象
    parser = parse_args()

    # 设置项目根目录路径（根据用户实际位置调整）
    route = config.get_config().get("current_directory")
    # 模型相关配置
    parser.output_dir = route + "JITFine/model/saved_models_concat"+modelName+"/checkpoints"  # 模型保存路径
    parser.model_name_or_path = "C:/Users/ZETTAKIT/PycharmProjects/JIT-BICC/local_roberta"  # 指向目录
    parser.tokenizer_name = "C:/Users/ZETTAKIT/PycharmProjects/JIT-BICC/local_tokenizer"  # 同样指向目录
    parser.do_test = True # 测试模式
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
    parser.only_adds = True # 仅针对新加入的行进行评估
    parser.buggy_line_filepath = route + "data/jitfine/changes_complete_buggy_line_level.pkl"
    parser.seed = 42  # 随机种子(确保结果可复现)

    return parser  # 返回配置好的参数解析器


if __name__ == "__main__":
    mp.freeze_support()  # Windows 兼容
    modelName = ModelType.FCF.value
    # cur_args = configs(modelName) # 训练配置
    cur_args = eval_configs(modelName) # 测试配置
    create_path_if_not_exist(cur_args.output_dir)
    main(cur_args,modelName)
