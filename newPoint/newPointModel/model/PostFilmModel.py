# -*- coding: utf-8 -*-
"""
Post-FiLM 版本：在编码器输出端进行分类前的“后置”FiLM 调制（仅调制 CLS 表示）。

说明：
- 你现有的 Middle-FiLM 是对全部 token hidden states 做 FiLM；
- 这里的 Post-FiLM 更“保守”，只在进入分类头之前对 CLS 向量做 FiLM，
  用于模拟“交互之后再注入条件”的后置插入（在仅 Film-only 结构下的可实现版本）。

接口与 Middle-FiLM 保持一致：
    forward(inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None)

使用方式（示例）：
    from PostFilmModel import FilmModel   # FilmModel 是别名，等价于 PostFiLMModel
"""

import torch
import torch.nn as nn
from torch.nn import BCELoss


class FiLMConditioner(nn.Module):
    """
    从手工度量特征生成 FiLM 参数 (gamma, beta)，并对 hidden states 做逐通道缩放/平移。
    hidden_states: (B, L, H) 或 (B, 1, H)
    manual_features: (B, F)
    """
    def __init__(self, config):
        super().__init__()
        self.feature_size = getattr(config, "feature_size", None)
        self.hidden_size = getattr(config, "hidden_size", None)
        if self.feature_size is None or self.hidden_size is None:
            raise ValueError("config must provide `feature_size` and `hidden_size` for FiLMConditioner.")
        self.film = nn.Linear(self.feature_size, 2 * self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, manual_features: torch.Tensor) -> torch.Tensor:
        gb = self.film(manual_features)  # (B, 2H)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, H), (B, H)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * hidden_states + beta


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features: torch.Tensor, manual_features=None) -> torch.Tensor:
        x = features[:, 0, :]  # CLS token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class PostFiLMModel(nn.Module):
    """
    Post-FiLM：只对 CLS token 做 FiLM，再送入分类头。
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.film = FiLMConditioner(config)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None):
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions
        )

        hidden_states = outputs[0]  # (B, L, H)

        # attention 返回逻辑：与 Middle-FiLM 一致
        last_layer_attn_weights = None
        if output_attentions:
            num_layers = getattr(self.config, "num_hidden_layers", None)
            if num_layers is None:
                last_layer_attn_weights = outputs.attentions[-1][:, :, 0].detach()
            else:
                last_layer_attn_weights = outputs.attentions[num_layers - 1][:, :, 0].detach()

        # ====== 后置插入：仅调制 CLS 表示 ======
        if manual_features is not None:
            cls_token = hidden_states[:, 0:1, :]              # (B, 1, H)
            cls_token = self.film(cls_token, manual_features) # (B, 1, H)
            # 写回（不改变其他 token）
            hidden_states = hidden_states.clone()
            hidden_states[:, 0:1, :] = cls_token

        logits = self.classifier(hidden_states, manual_features=None)
        prob = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob


# 兼容你现有的构建脚本：from PostFilmModel import FilmModel
FilmModel = PostFiLMModel
