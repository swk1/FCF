import torch
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """
    Cross-Attention Fusion Head (M→S, Vector-level)
    - Manual features are compressed into a single vector (B, H) and used as Query
    - Cross-Attn: Query = manual vector (B,1,H), Key/Value = semantic tokens (B, L, H)
    - Transformer-block style on query vector
    - Final: concat([semantic_CLS, fused_query]) -> classifier
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ========= 0) Stabilization / Norms =========
        self.manual_ln = nn.LayerNorm(self.feature_size)
        self.manual_vec_proj = nn.Linear(self.feature_size, self.hidden_size)
        self.manual_vec_ln = nn.LayerNorm(self.hidden_size)

        self.q_ln = nn.LayerNorm(self.hidden_size)
        self.kv_ln = nn.LayerNorm(self.hidden_size)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # ========= 1) Cross-Attention =========
        num_heads = getattr(config, "cross_attn_heads", 8)
        if self.hidden_size % num_heads != 0:
            num_heads = 1

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        self.post_attn_ln = nn.LayerNorm(self.hidden_size)

        # ========= 2) FFN =========
        ffn_mult = getattr(config, "ffn_mult", 4)
        ffn_hidden = self.hidden_size * ffn_mult

        self.ffn_ln = nn.LayerNorm(self.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(ffn_hidden, self.hidden_size),
        )
        self.post_ffn_ln = nn.LayerNorm(self.hidden_size)

        # ========= 3) Classifier =========
        self.out_proj = nn.Linear(self.hidden_size * 2, 1)

    def _build_manual_vector(self, manual_features: torch.Tensor) -> torch.Tensor:
        m = manual_features.float()
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = self.manual_ln(m)

        if m.shape[1] != self.feature_size:
            raise ValueError(f"manual feature dim mismatch: got {m.shape[1]}, expect {self.feature_size}")

        v = torch.tanh(self.manual_vec_proj(m))
        v = self.manual_vec_ln(v)
        return v  # (B,H)

    def forward(self, features, manual_features=None, **kwargs):
        x0 = features[:, 0, :]  # semantic CLS (B,H)

        if manual_features is None:
            z = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            z = self.dropout(z)
            return self.out_proj(z)

        q0 = self._build_manual_vector(manual_features)  # (B,H)

        kv = self.kv_ln(features)                        # (B,L,H)
        q = self.q_ln(q0).unsqueeze(1)                   # (B,1,H)

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B,1,H)
        attn_out = attn_out.squeeze(1)  # (B,H)

        x1 = self.post_attn_ln(q0 + self.resid_dropout(attn_out))  # residual on query vector

        ff_in = self.ffn_ln(x1)
        ff_out = self.ffn(ff_in)
        x2 = self.post_ffn_ln(x1 + self.resid_dropout(ff_out))     # fused query vector

        z = torch.cat([x0, x2], dim=-1)
        z = self.dropout(z)
        return self.out_proj(z)

class CroAttModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CroAttModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None):
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions
        )

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0].detach() \
            if output_attentions else None

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states, manual_features)

        if labels is not None:
            pos_weight = getattr(self.args, "pos_weight", None)
            if pos_weight is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=logits.device))
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.unsqueeze(1).float())
            prob = torch.sigmoid(logits)
            return loss, prob, last_layer_attn_weights
        else:
            prob = torch.sigmoid(logits)
            return prob
