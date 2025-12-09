# -*- coding: UTF-8 -*-
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from models.sequential.PAtt import LayerNorm, ACT2FN


# =========================================================
#  DPAtt2 用到的 2×2 determinant（显式公式，比 linalg.det 快）
# =========================================================
def det2(a, b, c, d):
    # det([[a,b],[c,d]])
    return a * d - b * c


# =========================================================
#  DPAtt2 注意力：与 PAtt2 基本一致，但 kernel = L * C^{-1}
# =========================================================
class DPAtt2Attention(nn.Module):
    """
    DPAtt2: 用 item 多样性核 C 作为先验。
    kernel = L(S) @ C^{-1}_sub
    """

    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.emb_size
        self.num_heads = args.num_heads
        self.attention_head_size = self.hidden_size // self.num_heads
        self.dropout_prob = args.dropout
        self.max_seq_len = args.history_max

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size)
        self.out_dropout = nn.Dropout(self.dropout_prob)

    # -----------------------------------------------------
    # C-sub extraction: C_ij from global kernel C
    # -----------------------------------------------------
    def extract_C_sub(self, seq_ids, C):
        """
        seq_ids: [B, L]
        C: [item_num, item_num] 多样性核矩阵
        输出 size: [B, L, L]
        """
        B, L = seq_ids.shape
        C_sub = []

        for b in range(B):
            ids = seq_ids[b]  # [L]
            sub = C[ids][:, ids]  # [L,L]
            C_sub.append(sub)

        C_sub = torch.stack(C_sub, dim=0)  # [B, L, L]
        return C_sub

    def forward(self, input_tensor, attention_mask, seq_item_ids=None, C_kernel=None):
        """
        input_tensor: [B,L,D]
        seq_item_ids: [B,L] item id，用于从 C-kernel 中抽取子矩阵
        """
        if attention_mask.dim() == 4:
            attention_mask = attention_mask.squeeze(1)

        B, L, D = input_tensor.shape

        # ----- S = sampler -----
        S = self.query(input_tensor)
        S2 = S * S  # element-wise square
        L_kernel = torch.bmm(S2, S2.transpose(1, 2))  # [B,L,L]

        # ----- 如果无 C-kernel，则行为与 PAtt2 完全一致 -----
        if C_kernel is None or seq_item_ids is None:
            T = L_kernel
        else:
            # ----- 提取 C_sub -----
            C_sub = self.extract_C_sub(seq_item_ids, C_kernel)  # [B,L,L]
            # add epsilon & invert
            eye = torch.eye(L, device=input_tensor.device).unsqueeze(0)
            C_inv = torch.inverse(C_sub + 1e-4 * eye)
            # T = L @ C^{-1}
            T = torch.bmm(L_kernel, C_inv)

        # ----- 计算所有 2×2 子式 -----
        idx = torch.arange(L, device=input_tensor.device)
        ii = idx.view(-1, 1).repeat(1, L).reshape(-1)
        jj = idx.view(1, -1).repeat(L, 1).reshape(-1)

        T_ii = T[:, ii, ii]
        T_ij = T[:, ii, jj]
        T_ji = T[:, jj, ii]
        T_jj = T[:, jj, jj]

        subdet = det2(T_ii, T_ij, T_ji, T_jj)  # [B,L*L]
        subdet = subdet.reshape(B, L, L)

        # ----- 归一化 -----
        denom = torch.sum(torch.triu(subdet, diagonal=1), dim=(1, 2), keepdim=True)
        denom = torch.clamp(denom, min=1e-9)
        subprob = subdet / denom

        # ----- PAtt 风格：增加 diag 后取负 -----
        diag_T = torch.diagonal(T, dim1=1, dim2=2)
        diag_matrix = torch.diag_embed(diag_T)

        score = -(subprob + diag_matrix) / math.sqrt(self.attention_head_size)
        score = score + attention_mask

        attn = nn.Softmax(dim=-1)(score)
        attn = self.attn_dropout(attn)

        V = self.value(input_tensor)
        ctx = torch.matmul(attn, V)

        out = self.dense(ctx)
        out = self.out_dropout(out)
        out = self.LayerNorm(out + input_tensor)

        return out


# =========================================================
#  DPPLayer & Encoder（直接复用 PAtt2 的结构）
# =========================================================
class DPAtt2Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = DPAtt2Attention(args)
        self.dense1 = nn.Linear(args.emb_size, args.emb_size * 4)
        self.dense2 = nn.Linear(args.emb_size * 4, args.emb_size)
        self.LayerNorm = LayerNorm(args.emb_size)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states, attn_mask, seq_item_ids=None, C_kernel=None):
        att_out = self.attention(
            hidden_states, attn_mask,
            seq_item_ids=seq_item_ids, C_kernel=C_kernel
        )

        out = self.dense1(att_out)
        out = self.act_fn(out)
        out = self.dense2(out)
        out = self.dropout(out)
        out = self.LayerNorm(out + att_out)
        return out


class DPAtt2Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        layer = DPAtt2Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_layers)])

    def forward(self, hidden_states, attn_mask, seq_item_ids, C_kernel):
        for blk in self.layer:
            hidden_states = blk(
                hidden_states, attn_mask,
                seq_item_ids=seq_item_ids,
                C_kernel=C_kernel
            )
        return hidden_states


# =========================================================
#  DPAtt2 主模型（完全复刻你 PAtt2 的 forward 结构）
# =========================================================
class DPAtt2(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--num_heads', type=int, default=4)
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max

        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)

        self.encoder = DPAtt2Encoder(args)
        self.LayerNorm = LayerNorm(self.emb_size)
        self.dropout = nn.Dropout(args.dropout)

        self.apply(self.init_weights)

        # 由 main.py 注入：model.C_kernel = ...
        self.C_kernel = None

    def forward(self, feed_dict):
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths']
        user_ids = feed_dict['user_id']

        B, L = history.shape

        valid = (history > 0).long()

        # embeddings
        his_vec = self.i_embeddings(history)
        pos_idx = (lengths[:, None] - torch.arange(L, device=self.device)).clamp(min=0)
        pos_idx = pos_idx * valid
        pos_vec = self.p_embeddings(pos_idx)
        user_vec = self.u_embeddings(user_ids).unsqueeze(1)

        seq_emb = his_vec + pos_vec + user_vec
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # mask
        causal = torch.triu(torch.ones(L, L, device=self.device), diagonal=1)
        attn_mask = causal * -1e9
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)

        pad = (history == 0)
        pad_col = pad.unsqueeze(1).expand(B, L, L)
        pad_row = pad.unsqueeze(2).expand(B, L, L)
        attn_mask = attn_mask + (pad_col | pad_row).float() * -1e9

        # encode
        out = self.encoder(seq_emb, attn_mask, seq_item_ids=history, C_kernel=self.C_kernel)

        # last hidden
        last_idx = (lengths - 1).clamp(min=0)
        last_vec = out[torch.arange(B), last_idx]

        tgt = self.i_embeddings(i_ids)
        pred = torch.einsum('bd,bnd->bn', last_vec, tgt)
        return {'prediction': pred}
