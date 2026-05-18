# -*- coding: utf-8 -*-
"""
Input-FiLM 版本：在编码器输入端（token embedding 级）注入 FiLM 条件调制。

对齐你现有的 Middle-FiLM (FilmModel.py) 的接口：
    forward(inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None)

使用方式（示例）：
    from InputFilmModel import FilmModel   # FilmModel 是别名，等价于 InputFiLMModel
"""

import warnings
import torch
import torch.nn as nn
from torch.nn import BCELoss


class FiLMConditioner(nn.Module):
    """
    从手工度量特征生成 FiLM 参数 (gamma, beta)，并对 token hidden states 做逐通道缩放/平移。
    hidden_states: (B, L, H)
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
        # broadcast to (B, L, H)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * hidden_states + beta


class RobertaClassificationHead(nn.Module):
    """
    与你 Middle-FiLM 文件保持一致：使用 CLS 向量做二分类（sigmoid 由外层完成）。
    """
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


class InputFiLMModel(nn.Module):
    """
    Input-FiLM：在 encoder 之前拿到 embeddings，并在 embedding 级执行 FiLM(hidden_states, manual_features)。
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.film = FiLMConditioner(config)
        self.classifier = RobertaClassificationHead(config)

    def _get_full_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        尽量复用 transformer 模型自身的 embedding 逻辑（包含 position/token_type 等）。
        - 优先：encoder.embeddings(input_ids=...)
        - 兜底：encoder.get_input_embeddings()(input_ids)（仅词嵌入，不含位置嵌入）
        """
        if hasattr(self.encoder, "embeddings") and callable(getattr(self.encoder, "embeddings")):
            try:
                return self.encoder.embeddings(input_ids=input_ids)
            except TypeError:
                # 有些实现 embeddings 不接受关键字 input_ids
                return self.encoder.embeddings(input_ids)
        if hasattr(self.encoder, "get_input_embeddings") and callable(getattr(self.encoder, "get_input_embeddings")):
            warnings.warn(
                "encoder has no `.embeddings`; falling back to word embeddings only via `get_input_embeddings()`.",
                RuntimeWarning,
            )
            return self.encoder.get_input_embeddings()(input_ids)
        raise AttributeError("encoder must provide `.embeddings` or `.get_input_embeddings()` to support Input-FiLM.")

    def forward(self, inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None):
        # 1) 取 embeddings
        if manual_features is not None:
            embeds = self._get_full_embeddings(inputs_ids)  # (B, L, H)
            embeds = self.film(embeds, manual_features)
            outputs = self.encoder(
                inputs_embeds=embeds,
                attention_mask=attn_masks,
                output_attentions=output_attentions
            )
        else:
            # 无手工特征时，保持与基线一致，直接走 input_ids
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
                # 兜底：取最后一层 attentions
                last_layer_attn_weights = outputs.attentions[-1][:, :, 0].detach()
            else:
                last_layer_attn_weights = outputs.attentions[num_layers - 1][:, :, 0].detach()

        # 2) 分类
        logits = self.classifier(hidden_states, manual_features=None)
        prob = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob


# 兼容你现有的构建脚本：from InputFilmModel import FilmModel
FilmModel = InputFiLMModel
