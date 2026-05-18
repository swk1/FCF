import torch
import torch.nn as nn
from torch.nn import BCELoss

from JITFine.improveModel.ImbalanceHandel.imbalanced_handel import get_loss_function


class RobertaClassificationHead(nn.Module):
    """
    Improved classification head with gated feature fusion.
    支持BERT编码语义特征 + 手工特征的自适应融合。
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 手工特征 -> 映射到相同维度空间
        self.manual_fc = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.hidden_dropout_prob)
        )

        # 门控机制（决定语义特征与手工特征的融合比例）
        self.gate_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )

        # 最终分类层
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, features, manual_features=None, **kwargs):
        # 1. 取RoBERTa的CLS语义表示
        x = features[:, 0, :]  # [batch, hidden]

        # 2. 线性映射手工特征
        y = self.manual_fc(manual_features.float())  # [batch, hidden]

        # 3. 门控融合
        fusion_gate = self.gate_fc(torch.cat((x, y), dim=-1))  # [batch, hidden]
        fused = fusion_gate * x + (1 - fusion_gate) * y  # 加权融合

        # 4. 分类
        fused = self.dropout(fused)
        logits = self.out_proj(fused)
        return logits


class Model(nn.Module):
    """
    主模型：编码器 + 融合分类器
    """

    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.use_logits = getattr(args, "use_logits", False) # 是否使用损失逻辑
        self.pos_weight = getattr(args, "pos_weight", None) # 损失权重
        self.loss_fn = get_loss_function(
            use_logits=self.use_logits,
            pos_weight=self.pos_weight.to(args.device) if self.pos_weight is not None else None
        )

    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None):
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions
        )

        last_layer_attn_weights = None
        if output_attentions:
            last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0].detach()

        logits = self.classifier(outputs[0], manual_features)
        prob = torch.sigmoid(logits)

        if labels is not None:
            if self.use_logits: # 使用逻辑损失
                loss = self.loss_fn(logits.view(-1), labels.float())
                return loss, torch.sigmoid(logits), last_layer_attn_weights  # 推理时仍返回概率
            else:
                loss = self.loss_fn(prob.view(-1), labels.float())
                return loss, prob, last_layer_attn_weights
        else:
            return prob
