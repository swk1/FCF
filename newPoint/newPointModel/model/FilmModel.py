import torch
import torch.nn as nn
from torch.nn import BCELoss


class FiLMConditioner(nn.Module):
    """
    用手工特征 m 生成 FiLM 参数 (gamma, beta)，对 hidden states 做条件调制：
        H' = gamma(m) ⊙ H + beta(m)
    gamma/beta: (B, H)，会 broadcast 到 (B, L, H)
    """
    def __init__(self, config):
        super().__init__()
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size

        # 小型 MLP：m -> hidden -> 2*hidden (gamma,beta)
        mid = getattr(config, "film_mid_size", self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_size, mid),
            nn.ReLU(),
            nn.Linear(mid, self.hidden_size * 2)
        )
        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, manual_features: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (B, L, H)
        manual_features: (B, F)
        """
        m = manual_features.float()
        gb = self.mlp(m)  # (B, 2H)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, H), (B, H)

        # 让 gamma 初始更接近 1，训练更稳定：gamma = 1 + tanh(gamma)
        gamma = 1.0 + torch.tanh(gamma)
        beta = torch.tanh(beta)

        # broadcast 到 token 维度
        gamma = gamma.unsqueeze(1)  # (B, 1, H)
        beta = beta.unsqueeze(1)    # (B, 1, H)

        out = gamma * hidden_states + beta
        out = self.ln(out)
        return out


class RobertaClassificationHead(nn.Module):
    """
    FiLM 后只用 CLS 表示做分类（manual 已通过 FiLM 进入编码过程）
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # CLS
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits


class FilmModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(FilmModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # ====== FiLM 条件调制模块（创新点3核心）======
        self.film = FiLMConditioner(config)

        # 分类头
        self.classifier = RobertaClassificationHead(config)

    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None):

        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions
        )

        hidden_states = outputs[0]  # (B, L, H)

        # 保持你原来的 attention 返回逻辑（用于 test/定位流程）
        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0].detach() \
            if output_attentions else None

        # ====== 中间层融合：FiLM 调制 token hidden states ======
        if manual_features is not None:
            hidden_states = self.film(hidden_states, manual_features)

        # 分类
        logits = self.classifier(hidden_states, manual_features=None)
        prob = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob
