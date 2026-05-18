import torch
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """
    Transformer-block style Cross-Attention Fusion Head
    - Manual features are tokenized into (B, F, H)
    - Cross-Attn: Query = CLS, Key/Value = manual tokens
    - Residual + LN
    - FFN + Residual + LN  (Transformer block)
    - Final: concat([original_CLS, fused_CLS]) -> classifier
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # ========= 0) Stabilization / Norms =========
        # manual feature normalization (across F)
        self.manual_ln = nn.LayerNorm(self.feature_size)

        # Pre-Norm for query and key/value
        self.q_ln = nn.LayerNorm(self.hidden_size)
        self.kv_ln = nn.LayerNorm(self.hidden_size)

        # Dropout for residual paths
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # ========= 1) Manual features tokenization =========
        # scalar feature value -> hidden token
        self.feat_value_proj = nn.Linear(1, self.hidden_size)
        # feature id embedding
        self.feat_id_embed = nn.Embedding(self.feature_size, self.hidden_size)
        # token-level norm
        self.feat_ln = nn.LayerNorm(self.hidden_size)

        # ========= 2) Cross-Attention =========
        num_heads = getattr(config, "cross_attn_heads", 8)
        if self.hidden_size % num_heads != 0:
            num_heads = 1  # safe fallback

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )

        # Post-attn LayerNorm (after residual)
        self.post_attn_ln = nn.LayerNorm(self.hidden_size)

        # ========= 3) FFN (Transformer block) =========
        ffn_mult = getattr(config, "ffn_mult", 4)
        ffn_hidden = self.hidden_size * ffn_mult

        self.ffn_ln = nn.LayerNorm(self.hidden_size)  # Pre-Norm for FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, ffn_hidden),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(ffn_hidden, self.hidden_size),
        )
        self.post_ffn_ln = nn.LayerNorm(self.hidden_size)

        # ========= 4) Classifier =========
        # concat(original CLS, fused CLS)
        self.out_proj = nn.Linear(self.hidden_size * 2, 1)

    def _build_manual_tokens(self, manual_features: torch.Tensor) -> torch.Tensor:
        """
        manual_features: (B, F)
        return: manual tokens (B, F, H)
        """
        m = manual_features.float()  # (B, F)
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = self.manual_ln(m)

        bsz, fsz = m.shape
        if fsz != self.feature_size:
            raise ValueError(f"manual feature dim mismatch: got {fsz}, expect {self.feature_size}")

        # value embedding: (B,F,1)->(B,F,H)
        v = m.unsqueeze(-1)
        v = torch.tanh(self.feat_value_proj(v))

        # id embedding: (B,F)->(B,F,H)
        ids = torch.arange(fsz, device=m.device).unsqueeze(0).expand(bsz, fsz)
        e = self.feat_id_embed(ids)

        tokens = self.feat_ln(v + e)  # (B,F,H)
        return tokens

    def forward(self, features, manual_features=None, **kwargs):
        """
        features: (B, L, H) = encoder last_hidden_state
        manual_features: (B, F)
        return logits: (B, 1)
        """
        # original CLS semantic embedding
        x0 = features[:, 0, :]  # (B, H)

        # degrade gracefully if no manual features
        if manual_features is None:
            z = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            z = self.dropout(z)
            return self.out_proj(z)

        # manual tokens
        y_tokens = self._build_manual_tokens(manual_features)  # (B, F, H)

        # ===== Transformer Block: Cross-Attn =====
        # Pre-Norm
        q = self.q_ln(x0).unsqueeze(1)   # (B,1,H)
        kv = self.kv_ln(y_tokens)        # (B,F,H)

        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B,1,H)
        attn_out = attn_out.squeeze(1)  # (B,H)

        # Residual + LN
        x1 = self.post_attn_ln(x0 + self.resid_dropout(attn_out))     # (B,H)

        # ===== Transformer Block: FFN =====
        ff_in = self.ffn_ln(x1)                 # Pre-Norm for FFN
        ff_out = self.ffn(ff_in)                # (B,H)
        x2 = self.post_ffn_ln(x1 + self.resid_dropout(ff_out))        # (B,H)

        # Final fusion (keep original CLS + fused CLS)
        z = torch.cat([x0, x2], dim=-1)         # (B, 2H)
        z = self.dropout(z)
        logits = self.out_proj(z)               # (B,1)
        return logits

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

        # 保持你原来的“返回最后一层 CLS attention”逻辑不变（用于你现有 test/定位流程）
        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0].detach() \
            if output_attentions else None

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states, manual_features)  # or self.classifier(outputs[0], manual_features)

        if labels is not None:
            # pos_weight 可选：例如 args.pos_weight 或动态计算
            pos_weight = getattr(self.args, "pos_weight", None)
            if pos_weight is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=logits.device))
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.unsqueeze(1).float())
            prob = torch.sigmoid(logits)  # 用于评估输出
            return loss, prob, last_layer_attn_weights
        else:
            prob = torch.sigmoid(logits)
            return prob
