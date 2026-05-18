import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """
    scores: [..., L]
    mask:   same broadcastable shape as scores (bool or 0/1), True means keep.
    Returns probs with zeros on masked positions and rows renormalized.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    scores = scores.masked_fill(~mask, -1e9)
    probs = torch.softmax(scores, dim=dim)
    probs = probs * mask.to(dtype=probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True) + eps
    return probs / denom


class AttentionPooling(nn.Module):
    """Learned attention pooling over a masked sequence."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B,L,H], mask: [B,L] -> [B,H]"""
        if mask.dtype != torch.bool:
            mask = mask.bool()
        scores = self.proj(x).squeeze(-1)  # [B,L]
        probs = masked_softmax(scores, mask, dim=1)  # [B,L]
        return torch.bmm(probs.unsqueeze(1), x).squeeze(1)


class MultiHeadCoAttention(nn.Module):
    """
    Bidirectional co-attention using Q/K/V projections (multi-head, masked).
    We keep the packed sequence [B,L,H] but apply key-masks to realize:
      - msg -> code
      - code -> msg

    Output:
      msg_tokens: [B,L,H] (message-aware representations)
      code_tokens:[B,L,H] (code-aware representations)
      attn_mc:    [B,L,L] (message->code attention weights for visualization)
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size({hidden_size}) must be divisible by num_heads({num_heads})")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Shared projections (lightweight and stable).
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)

        self.drop = nn.Dropout(dropout)
        self.ln_msg = nn.LayerNorm(hidden_size)
        self.ln_code = nn.LayerNorm(hidden_size)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        # [B,L,H] -> [B,heads,L,head_dim]
        B, L, H = x.shape
        x = x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def _unreshape(self, x: torch.Tensor) -> torch.Tensor:
        # [B,heads,L,head_dim] -> [B,L,H]
        B, heads, L, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, heads * hd)

    def forward(
        self,
        sequence_output: torch.Tensor,  # [B,L,H]
        attn_mask: torch.Tensor,        # [B,L] 0/1
        msg_mask: torch.Tensor,         # [B,L] bool
        code_mask: torch.Tensor,        # [B,L] bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = sequence_output.shape
        attn_mask = attn_mask.bool()

        q = self._reshape(self.wq(sequence_output))  # [B,h,L,d]
        k = self._reshape(self.wk(sequence_output))
        v = self._reshape(self.wv(sequence_output))

        # scores: [B,h,L(query),L(key)]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # ---- msg -> code (keys restricted to code positions) ----
        key_mask_code = (code_mask & attn_mask).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        mask_mc = key_mask_code.expand(B, self.num_heads, L, L)
        attn_mc_h = masked_softmax(scores, mask_mc, dim=-1)  # [B,h,L,L]
        attn_mc_h = self.drop(attn_mc_h)
        ctx_mc = torch.matmul(attn_mc_h, v)  # [B,h,L,d]
        ctx_mc = self._unreshape(ctx_mc)  # [B,L,H]
        msg_tokens = self.ln_msg(sequence_output + ctx_mc)

        # ---- code -> msg (keys restricted to message positions) ----
        key_mask_msg = (msg_mask & attn_mask).unsqueeze(1).unsqueeze(2)
        mask_cm = key_mask_msg.expand(B, self.num_heads, L, L)
        attn_cm_h = masked_softmax(scores, mask_cm, dim=-1)
        attn_cm_h = self.drop(attn_cm_h)
        ctx_cm = torch.matmul(attn_cm_h, v)
        ctx_cm = self._unreshape(ctx_cm)
        code_tokens = self.ln_code(sequence_output + ctx_cm)

        # For visualization keep a single-head view: average across heads.
        attn_mc = attn_mc_h.mean(dim=1)  # [B,L,L]
        return msg_tokens, code_tokens, attn_mc


class CoAttnClassificationHead(nn.Module):
    """Co-attn (msg<->code) + gated fusion with manual features."""

    def __init__(self, config, num_heads: int = 4):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.feature_size = config.feature_size
        drop_p = getattr(config, "hidden_dropout_prob", 0.1)

        self.coattn = MultiHeadCoAttention(self.hidden_size, num_heads=num_heads, dropout=drop_p)

        self.pool_msg = AttentionPooling(self.hidden_size, dropout=drop_p)
        self.pool_code = AttentionPooling(self.hidden_size, dropout=drop_p)

        # semantic pair features: [m, c, |m-c|, m*c] -> hidden
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(drop_p),
        )

        # manual features -> hidden
        self.manual_fc = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(drop_p),
        )

        # gate(semantic, manual) -> fused hidden
        self.gate_fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(drop_p)

    def forward(
        self,
        sequence_output: torch.Tensor,  # [B,L,H]
        attn_mask: torch.Tensor,        # [B,L]
        msg_mask: torch.Tensor,         # [B,L] bool
        code_mask: torch.Tensor,        # [B,L] bool
        manual_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        msg_tokens, code_tokens, attn_mc = self.coattn(sequence_output, attn_mask, msg_mask, code_mask)

        valid = attn_mask.bool()
        msg_valid = msg_mask & valid
        code_valid = code_mask & valid

        # Learned pooling (more stable than mean when code is long).
        m_vec = self.pool_msg(msg_tokens, msg_valid)
        c_vec = self.pool_code(code_tokens, code_valid)

        # Fallback to CLS if a segment is empty (can happen after truncation)
        cls_vec = sequence_output[:, 0, :]
        m_cnt = msg_valid.sum(dim=1)
        c_cnt = code_valid.sum(dim=1)
        m_vec = torch.where(m_cnt.unsqueeze(1) > 0, m_vec, cls_vec)
        c_vec = torch.where(c_cnt.unsqueeze(1) > 0, c_vec, cls_vec)

        sem_pair = torch.cat([m_vec, c_vec, torch.abs(m_vec - c_vec), m_vec * c_vec], dim=-1)
        x = self.semantic_proj(sem_pair)  # [B,H]

        if manual_features is None:
            fused = x
        else:
            y = self.manual_fc(manual_features.float())
            gate = self.gate_fc(torch.cat([x, y], dim=-1))
            fused = gate * x + (1.0 - gate) * y

        fused = self.dropout(fused)
        logits = self.out_proj(fused)  # [B,1]
        return logits, attn_mc


class CoAttnModel(nn.Module):
    """
    Drop-in replacement for your current `Model`:
      forward(...) returns (loss, prob, attn_weights) when labels is not None, else returns prob.

    Key fixes vs old version:
      - Loss supports BCEWithLogitsLoss (+pos_weight) when args.use_logits=True.
      - [ADD]/[DEL] segmentation robust: run.py should add special tokens and resize embeddings.
      - Stronger co-attention: Q/K/V multi-head + learned pooling.
    """

    def __init__(self, encoder, config, tokenizer, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.use_logits = bool(getattr(args, "use_logits", True))
        self.use_pos_weight = bool(getattr(args, "use_pos_weight", False))

        # will be set by train.py (global neg/pos). Keep as Tensor on device when available.
        self.pos_weight: Optional[torch.Tensor] = None

        self.add_id = tokenizer.convert_tokens_to_ids("[ADD]")
        self.del_id = tokenizer.convert_tokens_to_ids("[DEL]")
        self.sep_id = tokenizer.sep_token_id
        self.unk_id = tokenizer.unk_token_id

        # Heads: prefer args.coattn_heads, else fall back.
        num_heads = int(getattr(args, "coattn_heads", 4))
        self.classifier = CoAttnClassificationHead(config, num_heads=num_heads)

        # Warn if special tokens not properly registered.
        if self.add_id == self.unk_id or self.del_id == self.unk_id:
            logger.warning(
                "[ADD]/[DEL] map to UNK. Please call tokenizer.add_special_tokens({additional_special_tokens:[...]}) "
                "and encoder.resize_token_embeddings(len(tokenizer)) in run.py before training."
            )

    @torch.no_grad()
    def _first_pos(self, x: torch.Tensor, token_id: int) -> torch.Tensor:
        """x:[B,L] -> first position of token_id per row, or -1."""
        match = (x == token_id)
        pos = match.float().argmax(dim=1)
        has = match.any(dim=1)
        return torch.where(has, pos, torch.full_like(pos, -1))

    def _build_segment_masks(self, input_ids: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Packed format:
          [CLS] msg_tokens [ADD] added_tokens [DEL] removed_tokens [SEP] PAD...

        We build:
          msg_mask: positions between CLS and [ADD]
          code_mask: positions between [ADD] and [SEP], excluding special tokens
        """
        B, L = input_ids.size()
        device = input_ids.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        add_pos = self._first_pos(input_ids, self.add_id)
        sep_pos = self._first_pos(input_ids, self.sep_id)

        # fallback: if [ADD] missing, msg empty and code uses everything after CLS
        add_pos_safe = torch.where(add_pos >= 0, add_pos, torch.zeros_like(add_pos))
        # fallback: if [SEP] missing, use last valid
        last_valid = (attn_mask.long().sum(dim=1) - 1).clamp(min=1)
        sep_pos_safe = torch.where(sep_pos >= 0, sep_pos, last_valid)

        msg_mask = (positions > 0) & (positions < add_pos_safe.unsqueeze(1))

        code_mask = (positions > add_pos_safe.unsqueeze(1)) & (positions < sep_pos_safe.unsqueeze(1))
        code_mask = code_mask & (input_ids != self.add_id) & (input_ids != self.del_id) & (input_ids != self.sep_id)

        valid = attn_mask.bool()
        msg_mask = msg_mask & valid
        code_mask = code_mask & valid
        return msg_mask, code_mask

    def _get_pos_weight(self, device: torch.device) -> Optional[torch.Tensor]:
        """Resolve pos_weight precedence: args.pos_weight > model.pos_weight > None."""
        if not self.use_pos_weight:
            return None

        # manual override
        pw = getattr(self.args, "pos_weight", None)
        if pw is not None:
            return torch.tensor(float(pw), dtype=torch.float, device=device)

        if self.pos_weight is not None:
            return self.pos_weight.to(device=device, dtype=torch.float)

        return None

    def forward(
        self,
        inputs_ids: torch.Tensor,
        attn_masks: torch.Tensor,
        manual_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        outputs = self.encoder(
            input_ids=inputs_ids,
            attention_mask=attn_masks,
            output_attentions=False,
        )
        sequence_output = outputs[0]  # [B,L,H]

        msg_mask, code_mask = self._build_segment_masks(inputs_ids, attn_masks)
        logits, attn_mc = self.classifier(sequence_output, attn_masks, msg_mask, code_mask, manual_features)

        prob = torch.sigmoid(logits)

        # keep compatibility with your code that expects "CLS attention": [B, heads?, L]
        coattn_cls = None
        if output_attentions:
            coattn_cls = attn_mc[:, 0, :].unsqueeze(1).detach()

        if labels is not None:
            labels_ = labels.unsqueeze(1).float()
            if self.use_logits:
                pos_weight = self._get_pos_weight(device=logits.device)
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = loss_fct(logits, labels_)
            else:
                loss_fct = nn.BCELoss()
                loss = loss_fct(prob, labels_)
            return loss, prob, coattn_cls

        return prob

