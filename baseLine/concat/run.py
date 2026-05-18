from __future__ import absolute_import, division, print_function

import argparse
import gc
import glob
import logging
import os
import pickle
import random
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel)
from tqdm import tqdm, trange

from baselines.config import config

# 添加自定义模块路径
sys.path.append("/mnt/sda/jyz/Code_Change/CommitMaster/")  # Change to root directory of your project

from JITFine.concat.model import Model
from JITFine.my_util import convert_examples_to_features, TextDataset, eval_result, preprocess_code_line, \
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

    # 构建数据加载器
    train_sampler = RandomSampler(train_dataset)  # 随机采样
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    args.max_steps = args.epochs * len(train_dataloader)

    # 计算训练参数,计算1/5报告一次
    args.save_steps = len(train_dataloader) // 5
    # args.save_steps = len(train_dataloader)
    args.warmup_steps = 0
    model.to(args.device)

    # 准备优化器和学习率调度器
    no_decay = ['bias', 'LayerNorm.weight'] # 不进行权重衰减的参数
    # 将模型参数分为两组
    # 第一组：普通权重（应用权重衰减防止过拟合）
    # 第二组：特殊参数（如bias、LayerNorm参数，不应用权重衰减）
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    # 构建AdamW优化器，用于指导模型参数应该往哪个方向改？改多少？
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    # 多GPU训练支持
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("开始训练")
    logger.info("  样本总量：%d", len(train_dataset))
    logger.info("  计划训练次数：%d", args.epochs)
    logger.info("  每个设备每次实际处理的批量大小：%d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  全局批量大小：%d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  每次参数更新前积累的小批量数量：%d", args.gradient_accumulation_steps)
    logger.info("  记录整个训练过程中将执行的参数更新次数：%d", args.max_steps)

    # 训练循环
    best_f1 = 0
    global_step = 0
    model.zero_grad()
    patience = 0

    for idx in range(args.epochs):
        # 使用tqdm创建进度条，直观显示训练进度
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        # 初始化当前epoch的损失统计
        tr_loss = 0 # 累计损失值
        tr_num = 0  # 处理批次计数（用于平均损失计算）
        # 遍历训练数据加载器中的所有批次
        for step, batch in enumerate(bar):
            # 准备输入数据：将每个批次张量移到指定设备(CPU/GPU)
            (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
            # 设置模型为训练模式（启用dropout和batch normalization）
            model.train()
            # 前向传播：模型计算损失和logits
            loss, logits, _ = model(inputs_ids, attn_masks, manual_features, labels)
            # 多GPU损失平均
            if args.n_gpu > 1:
                loss = loss.mean()
            # 梯度累积处理：如果设置了梯度累积，按累积步数缩放损失值
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 记录当前损失值并更新统计
            tr_loss += loss.item()
            tr_num += 1
            # 定期报告训练进度：每到指定的保存步数时记录日志
            if (step + 1) % args.save_steps == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step + 1, round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0

            # 反向传播和参数更新
            loss.backward()
            # 梯度裁剪：防止梯度爆炸（限制梯度范围在[-max_grad_norm, max_grad_norm]内）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # 梯度累积结束后执行参数更新
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            global_step += 1

            # 定期报告训练进度：每到指定的保存步数时记录日志
            if (step + 1) % args.save_steps == 0:
                # 在验证集上评估当前模型性能
                results = evaluate(args, model, tokenizer, eval_when_training=True)
                # 记录每一批的模型参数
                # checkpoint_prefix = f'epoch_{idx}_step_{step}'
                # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                # if not os.path.exists(output_dir):
                #     os.makedirs(output_dir)
                # model_to_save = model.module if hasattr(model, 'module') else model
                # output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                # torch.save({
                #     'epoch': idx,
                #     'step': step,
                #     'patience': patience,
                #     'model_state_dict': model_to_save.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler': scheduler.state_dict()}, output_dir)
                logger.info("Saving epoch %d step %d model, patience %d", idx, global_step, patience)
                # 早停机制
                if results['eval_f1'] > best_f1:
                    best_f1 = results['eval_f1'] # 更新最佳F1分数
                    # 打印当前最佳F1分数
                    logger.info("  " + "*" * 20)
                    logger.info("  Best f1:%s", round(best_f1, 4)) # 四舍五入
                    # 准备保存最佳模型的目录结构
                    checkpoint_prefix = 'checkpoint-best-f1'
                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # 处理多GPU训练时的模型状态（如果是DataParallel）
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                    patience = 0
                    # 保存最佳模型
                    torch.save({
                        'epoch': idx,
                        'step': step,
                        'patience': patience,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}, output_dir)
                    logger.info("模型保存路径为：%s", output_dir)
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


def test(args, model, tokenizer, best_threshold=0.5):
    """模型测试函数"""
    # 构建缓存数据集文件的路径，使用提交消息的最大长度作为文件名的一部分
    cache_dataset = os.path.dirname(args.test_data_file[0]) + f'/test_set_cache_msg{args.max_msg_length}.pkl'
    # 如果不需要抽象处理（使用原始代码表示），修改缓存文件名
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl'
    logger.info("Cache Dataset file at %s ", cache_dataset) # 记录缓存文件路径
    # 检查缓存数据集是否存在
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        # 创建新的文本数据集对象
        eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    # 创建顺序采样器（保持数据顺序）
    eval_sampler = SequentialSampler(eval_dataset)
    # 创建数据加载器，设置批次大小和工作进程数
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # 多GPU评估支持
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 评估开始日志
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 初始化评估指标
    eval_loss = 0.0
    nb_eval_steps = 0
    # 将模型设为评估模式（禁用dropout等）
    model.eval()
    logits = []  # 模型输出logits
    y_trues = [] # 真实标签
    attns = []   # 注意力权重（用于后续定位分析）

    progress_bar = tqdm(
        total=len(eval_dataloader),
        desc="评估进度",
        unit="batch",
        bar_format="{l_bar}{bar:40}{r_bar}",
        position=0,
        leave=True,
        disable=False,  # 可以根据args参数控制是否显示
        colour='green'
    )

    for step, batch in enumerate(eval_dataloader):
        # 将数据移至指定设备（CPU/GPU）
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        # 禁用梯度计算（节省内存）
        with torch.no_grad():
            # 模型前向传播（特别注意获取注意力权重）
            loss, logit, attn_weights = model(inputs_ids, attn_masks, manual_features, labels, output_attentions=True)
            last_layer_attn_weights = attn_weights # 提取最后一层的注意力权重
            eval_loss += loss.mean().item() # 累加损失值
            logits.append(logit.cpu().numpy()) # 存储当前批次的logits（移至CPU）
            y_trues.append(labels.cpu().numpy()) # 存储当前批次的真实标签（移至CPU）
            attns.append(last_layer_attn_weights.cpu().numpy()) # 存储当前批次的注意力权重（移至CPU）
        nb_eval_steps += 1

        progress_bar.update(1)  # 更新進度條
        # 更新进度条信息
        if step % max(1, len(eval_dataloader) // 10) == 0:  # 每处理10%的批次更新一次
            avg_loss = eval_loss / (step + 1)
            progress_bar.set_postfix({
                "avg_loss": f"{avg_loss:.4f}",
                "steps": f"{step + 1}/{len(eval_dataloader)}"
            })
        # 每次迭代后强制垃圾回收
        del last_layer_attn_weights
        torch.cuda.empty_cache()
        gc.collect()
    progress_bar.close()  # 清理资源

    # 合并所有批次的结果
    logits = np.concatenate(logits, 0)   # 拼接所有logits
    y_trues = np.concatenate(y_trues, 0) # 拼接所有真实标签
    attns = np.concatenate(attns, 0)     # 拼接所有注意力权重

    # 根据阈值生成预测标签（二分类决策）
    y_preds = logits[:, -1] > best_threshold
    # 计算评估指标
    recall = recall_score(y_trues, y_preds, average='binary')
    precision = precision_score(y_trues, y_preds, average='binary')
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


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", nargs=2, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", nargs=2, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_seed', type=int, default=123456,
                        help="random seed for data order initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument('--feature_size', type=int, default=14,
                        help="Number of features")
    parser.add_argument('--num_labels', type=int, default=2,
                        help="Number of labels")
    parser.add_argument('--semantic_checkpoint', type=str, default=None,
                        help="Best checkpoint for semantic feature")
    parser.add_argument('--manual_checkpoint', type=str, default=None,
                        help="Best checkpoint for manual feature")
    parser.add_argument('--max_msg_length', type=int, default=64,
                        help="Number of labels")
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument("--only_adds", action='store_true',
                        help="Whether to run eval on the only added lines.")
    parser.add_argument("--buggy_line_filepath", type=str,
                        help="complete buggy line-level  data file for RQ3")

    args = parser.parse_args()
    args.no_abstraction = True
    return args

def main(args):
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
    # config.hidden_dropout_prob = args.head_dropout_prob # 注释掉的可选配置：设置隐藏层dropout概率

    # 5. 加载分词器并添加特殊标记
    # ------------------------------------------------------------
    # 加载RoBERTa分词器（需要访问Hugging Face模型仓库）
    # 注意：需要网络连接，国内可能需要代理
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]} # 定义项目中需要添加的特殊标记
    tokenizer.add_special_tokens(special_tokens_dict) # 将特殊标记添加到分词器词汇表中

    # TODO: 多任务与单任务预训练加载机制的切换逻辑
    # 目的：实现在多任务预训练和单任务精调之间灵活切换模型加载方式
    # ----------------------------------------------------
    # 方案1：多任务预训练模型的加载路径
    # model_dir = os.path.join(args.model_name_or_path, "multitask_model")
    # 加载保存的多任务模型
    # with open(os.path.join(model_dir, "saved_model.pkl"), "rb") as dump_file:
    #     model = pickle.load(dump_file)
    # 从多任务模型中提取MLM任务的RoBERTa子模型
    # 注意：这里假设任务名"MLM"（Masked Language Modeling）是存在的
    # model = model.taskmodels_dict["MLM"].roberta

    # TODO: 以下代码仅用于调试目的 - 正式运行时应注释掉
    # 创建文本数据集实例用于调试
    # train_dataset = TextDataset(tokenizer, args, file_path=args.train_data_file)
    # # 遍历数据集的前3个样本并打印详细信息
    # for idx, example in enumerate(train_dataset.examples[:3]):
    #     logger.info("*** Example ***")
    #     logger.info("idx: {}".format(idx))
    #     logger.info("label: {}".format(example.label))
    #     logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
    #     logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    # 初始化基础RoBERTa模型
    # ----------------------------------------------------
    # 从预训练路径加载RoBERTa基础模型
    # 注意：args.model_name_or_path 应为有效的模型标识符或本地路径
    model = RobertaModel.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Training/evaluation parameters %s", args)



    # 构建自定义模型
    model = Model(model, config, tokenizer, args)

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

        # 打印示例数据（调试用）
        # for idx, example in enumerate(train_dataset.examples[:1]):
        #     logger.info("*** 示例 ***")
        #     logger.info("索引: {}".format(idx))
        #     logger.info("标签: {}".format(example.label))
        #     logger.info("输入token: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
        #     logger.info("输入ID: {}".format(' '.join(map(str, example.input_ids))))
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
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        checkpoint = torch.load(output_dir)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 显示成功加载的模型对应的训练轮次
        logger.info("Successfully load epoch {}'s model checkpoint".format(checkpoint['epoch']))
        model.to(args.device)
        test(args, model, tokenizer, best_threshold=0.5)
        eval_result(os.path.join(args.output_dir, "predictions.csv"), args.test_data_file[-1])

    return results

def configs():
    """初始化模型训练配置参数

        本函数创建并设置所有必要的训练参数，
        避免了每次运行时都需手动输入长命令行的繁琐。
        """

    # 创建命令行参数解析器对象
    parser = parse_args()

    # 设置项目根目录路径（根据用户实际位置调整）
    route = "C:/Users/Administrator/Desktop/JIT-BiCC-main/JIT-BiCC-main/"

    # 模型相关配置
    parser.output_dir = route + "jitfine/saved_models_concat/checkpoints"  # 模型保存路径
    parser.config_name = "microsoft/codebert-base"  # 使用的预训练模型配置名称
    parser.model_name_or_path = "microsoft/codebert-base"  # 预训练模型路径或标识符
    parser.tokenizer_name = "microsoft/codebert-base"  # 分词器名称
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
    parser.epoch = 50  # 训练轮数（epoch）
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
    return parser  # 返回配置好的参数解析器

def eval_configs():
    """初始化模型预测配置参数
       本函数创建并设置所有必要的训练参数，
       避免了每次运行时都需手动输入长命令行的繁琐。
    """
    # 创建命令行参数解析器对象
    parser = parse_args()

    # 设置项目根目录路径（根据用户实际位置调整）
    route = config.get_config().get("current_directory")
    # 模型相关配置
    parser.output_dir = route + "jitfine/saved_models_concat/checkpoints"  # 模型保存路径
    parser.config_name = "microsoft/codebert-base"  # 使用的预训练模型配置名称
    parser.model_name_or_path = "microsoft/codebert-base"  # 预训练模型路径或标识符
    parser.tokenizer_name = "microsoft/codebert-base"  # 分词器名称
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
    parser.epoch = 50  # 训练轮数（epoch）
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
    cur_args = configs() # 训练配置
    # cur_args = eval_configs() # 测试配置
    # cur_args = parse_args() # 服务器输入参数
    create_path_if_not_exist(cur_args.output_dir)
    main(cur_args)

