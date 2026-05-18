# model_mlb.py
# ---------------------------------------------------------
# BERT + 手工特征 的 Low-rank Bilinear Fusion (MLB) 模型
# 接口基本兼容你原来的 Model：
#   Model(encoder, config, tokenizer, args)
#   forward(inputs_ids, attn_masks, manual_features=None,
#           labels=None, output_attentions=None)
# ---------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import BCELoss

from JITFine.improveModel.ImbalanceHandel.imbalanced_handel import get_loss_function


class MLBFusion(nn.Module):
    """
    Low-rank Bilinear (MLB) fusion:
    - sem: [B, hidden_size]  （语义向量，例如 BERT CLS）
    - man: [B, feature_size] （手工特征）
    通过低秩投影 + 逐元素乘法建模二阶交互，再映射回 hidden_size。
    """

    def __init__(self, hidden_size: int, feature_size: int,
                 proj_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        # 低秩维度 k，默认取 hidden_size//2，至少 64
        if proj_dim is None:
            proj_dim = max(64, hidden_size // 2)
        self.proj_dim = proj_dim

        # 语义特征与手工特征分别映射到同一低秩空间 R^k
        self.proj_sem = nn.Linear(hidden_size, proj_dim)
        self.proj_man = nn.Linear(feature_size, proj_dim)

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # 从低秩空间再映射回 hidden_size，供后续分类使用
        self.out_proj = nn.Linear(proj_dim, hidden_size)

        # 初始化
        nn.init.xavier_uniform_(self.proj_sem.weight)
        nn.init.xavier_uniform_(self.proj_man.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, sem: torch.Tensor, man: torch.Tensor) -> torch.Tensor:
        """
        sem: [B, hidden_size]
        man: [B, feature_size]
        return: [B, hidden_size]
        """
        # 低秩投影
        sem_p = self.proj_sem(sem)              # [B, k]
        man_p = self.proj_man(man.float())      # [B, k]

        # MLB 核心：逐元素乘法 -> 低秩双线性交互
        fused_low = sem_p * man_p               # [B, k]
        fused_low = self.activation(fused_low)
        fused_low = self.dropout(fused_low)

        # 映射回 hidden_size
        fused = self.out_proj(fused_low)        # [B, hidden_size]
        fused = torch.tanh(fused)
        return fused


class RobertaClassificationHead(nn.Module):
    """
    Classification Head with MLB fusion.
    - 从 encoder 输出中取 CLS 向量
    - 与手工特征通过 MLB 进行双线性交互
    - 再做二分类
    """

    def __init__(self, config, mlb_proj_dim: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        # MLB 融合模块
        self.mlb = MLBFusion(
            hidden_size=self.hidden_size,
            feature_size=self.feature_size,
            proj_dim=mlb_proj_dim,
            dropout=getattr(config, "hidden_dropout_prob", 0.1),
        )

        # 最终分类层：hidden -> 1
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features: torch.Tensor,
                manual_features: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """
        features: encoder 输出的 last_hidden_state [B, L, H]
        manual_features: 手工特征 [B, feature_size]
        """
        if manual_features is None:
            raise ValueError("manual_features must be provided for MLB fusion.")

        # 取 CLS token 的语义表示
        cls_vec = features[:, 0, :]              # [B, H]

        # MLB 融合
        fused = self.mlb(cls_vec, manual_features)  # [B, H]

        # 分类
        fused = self.dropout(fused)
        logits = self.out_proj(fused)            # [B, 1]
        return logits


class MLBModel(nn.Module):
    """
    主模型：Encoder (如 RoBERTa/CodeBERT) + MLB 融合分类头
    使用方式基本与原来的 Model 一致：
        model = Model(encoder, config, tokenizer, args)
        loss, prob, attn = model(inputs_ids, attn_masks, manual_features, labels, output_attentions)
    """

    def __init__(self, encoder, config, tokenizer=None, args=None,
                 mlb_proj_dim: int = None):
        super(MLBModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer

        self.args = args
        self.use_logits = getattr(args, "use_logits", False)  # 是否使用损失逻辑
        self.pos_weight = getattr(args, "pos_weight", None)  # 损失权重
        self.loss_fn = get_loss_function(
            use_logits=self.use_logits,
            pos_weight=self.pos_weight.to(args.device) if self.pos_weight is not None else None
        )
        self.classifier = RobertaClassificationHead(config, mlb_proj_dim=mlb_proj_dim)

    def forward(self,
                inputs_ids: torch.Tensor,
                attn_masks: torch.Tensor,
                manual_features: torch.Tensor = None,
                labels: torch.Tensor = None,
                output_attentions: bool = False):
        """
        inputs_ids: [B, L]
        attn_masks: [B, L]
        manual_features: [B, feature_size]
        labels: [B] or [B,1]
        """
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions
        )

        # last_hidden_state 通常在 outputs[0]
        last_hidden_state = outputs[0]  # [B, L, H]

        # 取最后一层注意力（可选，用于可视化）
        last_layer_attn_weights = None
        if output_attentions and hasattr(outputs, "attentions") and outputs.attentions is not None:
            num_layers = getattr(self.config, "num_hidden_layers", len(outputs.attentions))
            last_layer_attn_weights = outputs.attentions[num_layers - 1][:, :, 0].detach()

        # 分类头（带 MLB 融合）
        logits = self.classifier(last_hidden_state, manual_features)  # [B, 1]
        prob = torch.sigmoid(logits)                                 # [B, 1]

        if labels is not None:
            if self.use_logits: # 使用逻辑损失
                loss = self.loss_fn(logits.view(-1), labels.float())
            else:
                labels = labels.unsqueeze(1).float() if labels.dim() == 1 else labels.float()
                loss = self.loss_fn(prob, labels)
            return loss, prob, last_layer_attn_weights
        else:
            return prob
