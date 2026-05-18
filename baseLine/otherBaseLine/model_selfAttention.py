import torch
import torch.nn as nn
from torch.nn import BCELoss


def _pick_num_heads(hidden_size: int, preferred: int = 8) -> int:
    """Pick a valid nhead that divides hidden_size."""
    for h in [preferred, 16, 12, 10, 8, 6, 4, 3, 2, 1]:
        if hidden_size % h == 0:
            return h
    return 1


class TransformerSelfAttentionFusionHead(nn.Module):
    """Transformer Encoder based self-attention fusion head."""

    def __init__(self, config):
        super().__init__()

        self.hidden_size = int(getattr(config, "hidden_size"))
        self.feature_size = int(getattr(config, "feature_size"))
        self.dropout_p = float(getattr(config, "hidden_dropout_prob", 0.1))

        # Fusion hyper-params (can be set into config in run.py)
        self.fusion_mode = str(getattr(config, "sa_fusion_mode", "2token"))  # "2token" | "fine"
        self.use_fuse_token = bool(getattr(config, "sa_use_fuse_token", True))
        self.num_layers = int(getattr(config, "sa_num_layers", 1))
        self.nhead = int(getattr(config, "sa_nhead", _pick_num_heads(self.hidden_size, preferred=8)))
        self.ffn_dim = int(getattr(config, "sa_ffn_dim", self.hidden_size * 4))
        self.attn_dropout = float(getattr(config, "sa_attn_dropout", self.dropout_p))

        if self.hidden_size % self.nhead != 0:
            # last-resort fallback: pick a compatible head count
            self.nhead = _pick_num_heads(self.hidden_size, preferred=self.nhead)

        # 1) Build tokens
        # semantic CLS projection
        self.proj_sem = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout_p),
        )

        # manual feature projection (for 2token)
        self.proj_manual = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout_p),
        )

        # fine-grained manual tokens: scalar -> hidden
        # shared linear + feature-type embedding (so F_i are distinguishable)
        self.manual_scalar_fc = nn.Linear(1, self.hidden_size)
        self.manual_type_emb = nn.Parameter(torch.zeros(self.feature_size, self.hidden_size))
        nn.init.normal_(self.manual_type_emb, mean=0.0, std=0.02)

        # optional [FUSE] token
        if self.use_fuse_token:
            self.fuse_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            nn.init.normal_(self.fuse_token, mean=0.0, std=0.02)
        else:
            self.fuse_token = None

        # positional embedding (learnable), enough for: [FUSE] + [SEM] + (max) features
        max_tokens = 1 + 1 + self.feature_size  # fuse + sem + features
        self.pos_emb = nn.Parameter(torch.zeros(1, max_tokens, self.hidden_size))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # 2) Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=self.ffn_dim,
            dropout=self.attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)

        # 3) Readout & classifier
        self.dropout = nn.Dropout(self.dropout_p)
        self.out_proj = nn.Linear(self.hidden_size, 1)

    def _build_token_sequence(self, sem_vec: torch.Tensor, manual_features: torch.Tensor) -> torch.Tensor:
        """Return tokens [B, T, H]"""
        B = sem_vec.size(0)

        if self.fusion_mode.lower() == "2token":
            man_vec = self.proj_manual(manual_features.float())  # [B, H]
            tokens = torch.stack([sem_vec, man_vec], dim=1)  # [B, 2, H]
        elif self.fusion_mode.lower() == "fine":
            # manual_features: [B, F] -> [B, F, 1] -> [B, F, H]
            mf = manual_features.float().unsqueeze(-1)
            man_tokens = self.manual_scalar_fc(mf)  # [B, F, H]
            man_tokens = man_tokens + self.manual_type_emb.unsqueeze(0)  # [B, F, H]
            tokens = torch.cat([sem_vec.unsqueeze(1), man_tokens], dim=1)  # [B, 1+F, H]
        else:
            raise ValueError(f"Unknown sa_fusion_mode={self.fusion_mode}. Use '2token' or 'fine'.")

        if self.use_fuse_token:
            fuse = self.fuse_token.expand(B, -1, -1)  # [B, 1, H]
            tokens = torch.cat([fuse, tokens], dim=1)  # [B, 1+T, H]

        # add positional embedding (slice to current length)
        T = tokens.size(1)
        tokens = tokens + self.pos_emb[:, :T, :]
        return tokens

    def forward(self, features: torch.Tensor, manual_features: torch.Tensor) -> torch.Tensor:
        """features: RoBERTa last_hidden_state [B, L, H]"""
        # CLS
        cls = features[:, 0, :]  # [B, H]
        sem_vec = self.proj_sem(cls)  # [B, H]

        if manual_features is None:
            raise ValueError("manual_features must be provided for self-attention fusion.")

        tokens = self._build_token_sequence(sem_vec, manual_features)  # [B, T, H]
        fused_tokens = self.encoder(tokens)  # [B, T, H]

        # readout
        if self.use_fuse_token:
            fused = fused_tokens[:, 0, :]  # [B, H]
        else:
            fused = fused_tokens.mean(dim=1)  # [B, H]

        fused = self.dropout(fused)
        logits = self.out_proj(fused)  # [B, 1]
        return logits


class SAFusionModel(nn.Module):
    """Main model: RoBERTa encoder + Transformer self-attention fusion head."""

    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.classifier = TransformerSelfAttentionFusionHead(config)

    def forward(
        self,
        inputs_ids,
        attn_masks,
        manual_features=None,
        labels=None,
        output_attentions=None,
    ):
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=output_attentions,
        )

        # keep the same behavior as your existing Model: return last-layer attn (CLS row)
        last_layer_attn_weights = None
        if output_attentions and getattr(outputs, "attentions", None) is not None:
            # shape: [B, heads, L, L] -> keep CLS query (index 0)
            last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0].detach()

        logits = self.classifier(outputs[0], manual_features)
        prob = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights
        else:
            return prob
