import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

from JITFine.improveModel.ImbalanceHandel.imbalanced_handel import get_loss_function

# =========================
# Try to import PyG's GATConv. If unavailable, fall back to a pure-PyTorch GAT.
# =========================
USE_PYG = True
try:
    from torch_geometric.nn import GATConv  # type: ignore
except Exception:
    USE_PYG = False


class SimpleGATLayer(nn.Module):
    """
    Lightweight multi-head GAT layer in pure PyTorch.
    Supports batched inputs: h [B, N, Din]  -> out [B, N, Dout]
    For our case N=2 (semantic node, manual node), but it works for small N in general.
    """
    def __init__(self, in_dim, out_dim, heads=2, concat=False, negative_slope=0.2, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.drop = nn.Dropout(dropout)

        # W: [H, Din, Dout], a: [H, 2*Dout]
        self.W = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.a = nn.Parameter(torch.empty(heads, 2 * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, h):
        # h: [B, N, Din]
        B, N, _ = h.size()
        head_outs = []
        for k in range(self.heads):
            Wh = torch.matmul(h, self.W[k])                      # [B, N, Dout]
            Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)         # [B, N, N, Dout]
            Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)         # [B, N, N, Dout]
            cat_ij = torch.cat([Wh_i, Wh_j], dim=-1)             # [B, N, N, 2*Dout]
            e = F.leaky_relu(torch.matmul(cat_ij, self.a[k].unsqueeze(-1)).squeeze(-1),
                             negative_slope=self.negative_slope)  # [B, N, N]
            alpha = torch.softmax(e, dim=-1)                      # attention over neighbors j
            alpha = self.drop(alpha)
            out = torch.matmul(alpha, Wh)                         # [B, N, Dout]
            head_outs.append(out)
        if self.concat:
            out = torch.cat(head_outs, dim=-1)                    # [B, N, H*Dout]
        else:
            out = torch.mean(torch.stack(head_outs, dim=0), dim=0)# [B, N, Dout]
        return out


class GraphFusionBlock(nn.Module):
    """
    A small stack of GAT layers (either PyG GATConv or SimpleGAT fallback), with LN/Dropout.
    Operates on a 2-node graph per sample: node0 = semantic, node1 = manual.
    """
    def __init__(self, hidden_size, num_layers=1, heads=2, dropout=0.1, use_pyg=USE_PYG):
        super().__init__()
        self.use_pyg = use_pyg
        self.layers = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.drop = nn.Dropout(dropout)

        for _ in range(num_layers):
            if self.use_pyg:
                # PyG GATConv expects per-graph edge_index; we'll process per-sample (N=2) in a loop
                self.layers.append(GATConv(in_channels=hidden_size,
                                           out_channels=hidden_size,
                                           heads=heads,
                                           concat=False,
                                           dropout=dropout))
            else:
                self.layers.append(SimpleGATLayer(in_dim=hidden_size,
                                                  out_dim=hidden_size,
                                                  heads=heads,
                                                  concat=False,
                                                  dropout=dropout))
            self.lns.append(nn.LayerNorm(hidden_size))

    def forward(self, node_feats):
        """
        node_feats: [B, 2, H]
        Returns:    [B, 2, H]
        """
        B, N, H = node_feats.size()
        x = node_feats
        if self.use_pyg:
            # static 2-node bidirectional edges for each sample
            base_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=x.device)
            for gat, ln in zip(self.layers, self.lns):
                outs = []
                for i in range(B):
                    h = x[i]                            # [2, H]
                    out_i = gat(h, base_edge_index)     # [2, H]
                    outs.append(out_i)
                x_new = torch.stack(outs, dim=0)        # [B, 2, H]
                x = ln(x + self.drop(x_new))            # residual + LN
            return x
        else:
            # batched GAT (pure torch)
            for gat, ln in zip(self.layers, self.lns):
                x_new = gat(x)                          # [B, 2, H]
                x = ln(x + self.drop(x_new))
            return x


class GraphFusionModel(nn.Module):
    """
    Graph-based Feature Fusion for JIT Defect Prediction (no gating).
    - Binary classification by default: BCELoss over sigmoid probability.
    - API compatible with your current training/eval pipeline:
      forward(inputs_ids, attn_masks, manual_features=None, labels=None, output_attentions=None)
      returns (loss, prob, None)  when labels is not None
              prob               otherwise
    """

    def __init__(self, config, args,roberta_model=None, gnn_layers=1, gnn_heads=2, graph_dropout=0.1):
        super().__init__()

        # 1) Encoder

        self.roberta = roberta_model if roberta_model else RobertaModel.from_pretrained(config.model_name_or_path)
        self.hidden_size = getattr(config, "hidden_size", self.roberta.config.hidden_size)
        self.feature_size = config.feature_size
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        # 类不平衡
        self.args = args
        self.use_logits = getattr(args, "use_logits", False)  # 是否使用损失逻辑
        self.pos_weight = getattr(args, "pos_weight", None)  # 损失权重
        self.loss_fn = get_loss_function(
            use_logits=self.use_logits,
            pos_weight=self.pos_weight.to(args.device) if self.pos_weight is not None else None
        )

        # 2) Feature projections
        self.proj_sem = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        )
        self.proj_manual = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        )

        # 3) Graph fusion block (2 nodes: semantic & manual)
        self.graph = GraphFusionBlock(hidden_size=self.hidden_size,
                                      num_layers=gnn_layers,
                                      heads=gnn_heads,
                                      dropout=graph_dropout,
                                      use_pyg=USE_PYG)

        # 4) Readout & classifier (binary -> one logit)
        self.readout = nn.Identity()  # mean over nodes is done right before classifier for clarity
        self.classifier = nn.Linear(self.hidden_size, 1)

        # 5) Loss
        #self.loss_fct = nn.BCELoss()


    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None):
        # Encoder: get semantic CLS
        enc_out = self.roberta(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)
        cls = enc_out.last_hidden_state[:, 0, :]                  # [B, H]
        sem_vec = self.proj_sem(cls)                              # [B, H]

        # Project manual features
        if manual_features is None:
            raise ValueError("manual_features must be provided for fusion.")
        man_vec = self.proj_manual(manual_features.float())       # [B, H]

        # Build per-sample 2-node graph: node0=sem, node1=manual
        nodes = torch.stack([sem_vec, man_vec], dim=1)            # [B, 2, H]
        fused_nodes = self.graph(nodes)                           # [B, 2, H]

        # Readout (mean over the 2 nodes)
        fused = fused_nodes.mean(dim=1)                           # [B, H]
        fused = self.dropout(fused)


        # Binary probability
        logits = self.classifier(fused)  # [B, 1]
        prob = torch.sigmoid(logits)  # [B, 1]

        last_layer_attn = None
        if output_attentions and enc_out.attentions is not None:
            last_layer_attn = enc_out.attentions[self.roberta.config.num_hidden_layers - 1]

        if labels is not None:
            # Expect labels shape [B] or [B, 1], use float for BCE
            # loss = self.loss_fct(prob, labels.unsqueeze(1).float())
            if self.use_logits: # 使用逻辑损失
                loss = self.loss_fn(logits.view(-1), labels.float())
                return loss, torch.sigmoid(logits), last_layer_attn  # 推理时仍返回概率
            else:
                loss = self.loss_fct(prob, labels.unsqueeze(1).float())
                return loss, prob, last_layer_attn
        else:
            return prob
