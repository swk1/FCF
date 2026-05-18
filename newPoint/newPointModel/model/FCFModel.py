# FCFModel_Stronger.py
import torch
import torch.nn as nn


# =========================
# Feature Gate: soft feature selection
# =========================
class FeatureGate(nn.Module):
    """
    Produce per-feature weights in (0,1) to suppress noisy/redundant manual dims.
    """
    def __init__(self, feature_size: int, gate_hidden: int = 0, init_bias: float = 0.0):
        super().__init__()
        if gate_hidden and gate_hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(feature_size, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, feature_size),
                nn.Sigmoid()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.Sigmoid()
            )

        # initialize bias so gate starts near ~0.5 (or other)
        if isinstance(self.net[0], nn.Linear) and self.net[0].bias is not None:
            nn.init.constant_(self.net[0].bias, init_bias)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.net(m)  # (B, F)


# =========================
# FiLM (Middle Fusion) with alpha gate
# =========================
class FiLMConditioner(nn.Module):
    """
    FiLM: H' = gamma(m) ⊙ H + beta(m)
    Add alpha gate to control modulation strength:
        H_out = H + alpha * (H' - H)

    alpha is learned from manual features, init small to avoid destabilizing.
    """
    def __init__(self, config):
        super().__init__()
        self.feature_size = config.feature_size
        self.hidden_size = config.hidden_size
        self.dropout_p = getattr(config, "hidden_dropout_prob", 0.1)

        mid = getattr(config, "film_mid_size", self.hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_size, mid),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(mid, self.hidden_size * 2),
        )

        # zero-init last layer -> gamma,beta near 0 at start (stable)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # alpha gate (scalar per sample)
        self.alpha_proj = nn.Linear(self.feature_size, 1)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -2.0)  # alpha ≈ sigmoid(-2)=0.12 initially

        self.ln = nn.LayerNorm(self.hidden_size)

    def forward(self, hidden_states: torch.Tensor, manual_features: torch.Tensor):
        """
        hidden_states: (B, L, H)
        manual_features: (B, F)
        """
        m = manual_features.float()
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

        gb = self.mlp(m)                    # (B, 2H)
        gamma, beta = gb.chunk(2, dim=-1)   # (B,H), (B,H)

        gamma = 1.0 + torch.tanh(gamma)
        beta = torch.tanh(beta)

        gamma = gamma.unsqueeze(1)  # (B,1,H)
        beta = beta.unsqueeze(1)    # (B,1,H)

        film_out = gamma * hidden_states + beta
        film_out = self.ln(film_out)

        alpha = torch.sigmoid(self.alpha_proj(m))  # (B,1)
        alpha = alpha.unsqueeze(1)                 # (B,1,1) broadcast to (B,L,H)

        out = hidden_states + alpha * (film_out - hidden_states)
        return out, alpha.squeeze(1)  # out:(B,L,H), alpha:(B,1)


# =========================
# Cross-Attn Fusion Head (Transformer-block style) + stronger classifier
# =========================
class CrossAttnFusionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # manual feature normalization
        self.manual_ln = nn.LayerNorm(self.feature_size)
        self.feature_dropout = nn.Dropout(getattr(config, "feature_dropout_prob", 0.0))

        # manual tokens
        self.feat_value_proj = nn.Linear(1, self.hidden_size)
        self.feat_id_embed = nn.Embedding(self.feature_size, self.hidden_size)
        self.feat_ln = nn.LayerNorm(self.hidden_size)

        # Cross-Attn
        num_heads = getattr(config, "cross_attn_heads", 8)
        if self.hidden_size % num_heads != 0:
            num_heads = 1

        self.q_ln = nn.LayerNorm(self.hidden_size)
        self.kv_ln = nn.LayerNorm(self.hidden_size)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=config.hidden_dropout_prob,
            batch_first=True
        )
        self.post_attn_ln = nn.LayerNorm(self.hidden_size)

        # FFN
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

        # Stronger classifier: MLP instead of single Linear
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.hidden_size, 1)
        )

    def _build_manual_tokens(self, manual_features: torch.Tensor) -> torch.Tensor:
        m = manual_features.float()
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        m = self.manual_ln(m)

        if self.training:
            m = self.feature_dropout(m)

        bsz, fsz = m.shape
        if fsz != self.feature_size:
            raise ValueError(f"manual feature dim mismatch: got {fsz}, expect {self.feature_size}")

        v = torch.tanh(self.feat_value_proj(m.unsqueeze(-1)))  # (B,F,H)
        ids = torch.arange(fsz, device=m.device).unsqueeze(0).expand(bsz, fsz)
        e = self.feat_id_embed(ids)
        return self.feat_ln(v + e)

    def forward(self, hidden_states: torch.Tensor, manual_features: torch.Tensor = None):
        x0 = hidden_states[:, 0, :]  # (B,H)

        if manual_features is None:
            z = torch.cat([x0, torch.zeros_like(x0)], dim=-1)
            z = self.dropout(z)
            return self.out_proj(z)

        y_tokens = self._build_manual_tokens(manual_features)

        q = self.q_ln(x0).unsqueeze(1)  # (B,1,H)
        kv = self.kv_ln(y_tokens)       # (B,F,H)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        attn_out = attn_out.squeeze(1)

        x1 = self.post_attn_ln(x0 + self.resid_dropout(attn_out))

        ff_out = self.ffn(self.ffn_ln(x1))
        x2 = self.post_ffn_ln(x1 + self.resid_dropout(ff_out))

        z = torch.cat([x0, x2], dim=-1)
        z = self.dropout(z)
        return self.out_proj(z)


# =========================
# Stronger FCF Model
# =========================
class FCFModel(nn.Module):
    """
    Same interface as your project:
      forward(inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None)
    returns:
      if labels is not None: (loss, prob, last_layer_attn_weights)
      else: prob
    """
    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # feature gate shared by FiLM & Cross-Attn
        gate_hidden = getattr(config, "feature_gate_hidden", 0)
        self.feature_gate = FeatureGate(config.feature_size, gate_hidden=gate_hidden, init_bias=0.0)

        self.film = FiLMConditioner(config)
        self.classifier = CrossAttnFusionHead(config)

    def forward(self, inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None):
        out_attn = bool(output_attentions)
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=out_attn,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state  # (B,L,H)

        last_layer_attn_weights = None
        if out_attn and getattr(outputs, "attentions", None) is not None:
            last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :, 0, :].detach()

        mf = None
        if manual_features is not None:
            mf = manual_features.float()
            mf = torch.nan_to_num(mf, nan=0.0, posinf=0.0, neginf=0.0)

            # shared feature gate
            g = self.feature_gate(mf)  # (B,F)
            mf = mf * g

            # FiLM with alpha gate (residual blend)
            hidden_states, _alpha = self.film(hidden_states, mf)

        logits = self.classifier(hidden_states, mf)  # (B,1)
        prob = torch.sigmoid(logits)

        if labels is not None:
            pos_weight = getattr(self.args, "pos_weight", None)
            if pos_weight is not None:
                loss_fct = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([float(pos_weight)], device=logits.device)
                )
            else:
                loss_fct = nn.BCEWithLogitsLoss()

            loss = loss_fct(logits, labels.unsqueeze(1).float())
            return loss, prob, last_layer_attn_weights

        return prob
