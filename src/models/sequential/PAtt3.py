# -*- coding: UTF-8 -*-
# @Author  : Optimized PAtt3 & DPAtt3
# @Email   :

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseModel import SequentialModel


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class DPPAttention3(nn.Module):
    def __init__(self, args):
        super(DPPAttention3, self).__init__()
        self.hidden_size = args.emb_size
        self.dropout_prob = args.dropout

        # Sampler S
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        # Value (Standard transformer setup, though PAtt often uses identity)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(self.dropout_prob)

        # 核心优化：可学习的缩放因子 lambda
        # 3阶行列式的值域很大，需要更强的缩放
        self.scale_factor = nn.Parameter(torch.tensor(1.0 / math.sqrt(self.hidden_size)))

    def _compute_det_3x3(self, matrix_3x3):
        """
        手动计算 3x3 矩阵的行列式，比 torch.linalg.det 在小矩阵上梯度更稳定。
        matrix_3x3: [..., 3, 3]
        Formula: a(ei - fh) - b(di - fg) + c(dh - eg)
        """
        # Unpack elements
        # [..., 3, 3] -> separate components
        m00 = matrix_3x3[..., 0, 0]
        m01 = matrix_3x3[..., 0, 1]
        m02 = matrix_3x3[..., 0, 2]
        m10 = matrix_3x3[..., 1, 0]
        m11 = matrix_3x3[..., 1, 1]
        m12 = matrix_3x3[..., 1, 2]
        m20 = matrix_3x3[..., 2, 0]
        m21 = matrix_3x3[..., 2, 1]
        m22 = matrix_3x3[..., 2, 2]

        term1 = m00 * (m11 * m22 - m12 * m21)
        term2 = m01 * (m10 * m22 - m12 * m20)
        term3 = m02 * (m10 * m21 - m11 * m20)

        det = term1 - term2 + term3
        return det

    def forward(self, input_tensor, attention_mask, diversity_kernel=None):
        batch_len, seq_len, _ = input_tensor.shape

        # 1. Prepare Kernel
        if diversity_kernel is not None:
            kernel = diversity_kernel  # DPAtt3 传入 T kernel
        else:
            # PAtt3 标准逻辑: L = S * S^T
            sampler = self.query(input_tensor)
            # 缩放 Sampler 防止 L 值过大
            sampler = sampler / math.sqrt(math.sqrt(self.hidden_size))
            s_sq = torch.mul(sampler, sampler)
            kernel = torch.bmm(s_sq, s_sq.transpose(1, 2))  # [B, L, L]

        # 2. Construct 3x3 Sub-matrices [B, L, L, L, 3, 3]
        # ReChorus 的 seq_len 通常较短 (20-50)，这种内存开销是可以接受的
        # 如果 OOM，需要改写为循环分块计算

        # Expand indices
        indices = torch.arange(seq_len, device=input_tensor.device)
        idx_i = indices.view(-1, 1, 1).expand(seq_len, seq_len, seq_len).reshape(-1)
        idx_j = indices.view(1, -1, 1).expand(seq_len, seq_len, seq_len).reshape(-1)
        idx_k = indices.view(1, 1, -1).expand(seq_len, seq_len, seq_len).reshape(-1)

        # Flatten Kernel for gathering
        kernel_flat = kernel.view(batch_len, -1)

        def gather_val(r, c):
            # global index in flattened [B, L*L]
            g_idx = r * seq_len + c
            # [B, L*L*L]
            return kernel_flat.gather(1, g_idx.unsqueeze(0).expand(batch_len, -1))

        # Gather all 9 components
        # Row 0
        k_ii = gather_val(idx_i, idx_i)
        k_ij = gather_val(idx_i, idx_j)
        k_ik = gather_val(idx_i, idx_k)
        # Row 1
        k_ji = gather_val(idx_j, idx_i)
        k_jj = gather_val(idx_j, idx_j)
        k_jk = gather_val(idx_j, idx_k)
        # Row 2
        k_ki = gather_val(idx_k, idx_i)
        k_kj = gather_val(idx_k, idx_j)
        k_kk = gather_val(idx_k, idx_k)

        # Stack into [B, L, L, L, 3, 3]
        # 优化：不显式 stack 整个大矩阵，直接用公式计算 det
        # 这样节省显存并加快速度

        # term1 = m00 * (m11 * m22 - m12 * m21)
        term1 = k_ii * (k_jj * k_kk - k_jk * k_kj)
        # term2 = m01 * (m10 * m22 - m12 * m20)
        term2 = k_ij * (k_ji * k_kk - k_jk * k_ki)
        # term3 = m02 * (m10 * m21 - m11 * m20)
        term3 = k_ik * (k_ji * k_kj - k_jj * k_ki)

        det_values = term1 - term2 + term3  # [B, L*L*L]

        # Reshape back to [B, L, L, L]
        det_values = det_values.view(batch_len, seq_len, seq_len, seq_len)

        # 3. Marginalize over k (Summation)
        # 必须 Mask 掉 padding 的 k，否则会引入噪声
        # attention_mask: [B, L, L], 0 为有效, -inf 为 mask
        # 取 attention_mask 的一行判断有效长度
        valid_mask_k = (attention_mask[:, 0, :] > -1e4).float()  # [B, L]
        valid_mask_k = valid_mask_k.view(batch_len, 1, 1, seq_len)

        # 核心：PAtt 原理是 Dependecy = Negation of Probability
        # Probability 正比于 det。如果 det 很大(diversity高)，则 dependency 小。
        # 因此 Attention Score 正比于 -det

        # 对 k 求和得到 pairwise score
        # [B, L, L]
        marginal_det = (det_values * valid_mask_k).sum(dim=3)

        # 4. Normalization & Logits
        # PAtt 使用 - (Prob + Diag)
        one_diag = torch.diagonal(kernel, dim1=1, dim2=2)  # [B, L]
        one_diag_matrix = torch.diag_embed(one_diag)

        # 注意：这里我们使用 learnable scale factor 代替固定的 sqrt(d)
        # 因为 det(3x3) 的量级是 L^3，变化范围极大
        score = -(marginal_det + one_diag_matrix) * self.scale_factor

        # 5. Final Mask & Softmax
        score = score + attention_mask
        attn_probs = nn.Softmax(dim=-1)(score)
        attn_probs = self.attn_dropout(attn_probs)

        # 6. Output
        value_layer = self.value(input_tensor)  # Explicit value projection
        context = torch.matmul(attn_probs, value_layer)

        output = self.dense(context)
        output = self.out_dropout(output)
        output = self.LayerNorm(output + input_tensor)

        return output


class DPPLayer3(nn.Module):
    def __init__(self, args):
        super(DPPLayer3, self).__init__()
        self.attention = DPPAttention3(args)
        self.dense_1 = nn.Linear(args.emb_size, args.emb_size * 4)
        self.dense_2 = nn.Linear(args.emb_size * 4, args.emb_size)
        self.LayerNorm = LayerNorm(args.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = F.gelu

    def forward(self, hidden_states, attention_mask, diversity_kernel=None):
        attention_output = self.attention(hidden_states, attention_mask, diversity_kernel)
        output = self.dense_1(attention_output)
        output = self.act_fn(output)
        output = self.dense_2(output)
        output = self.dropout(output)
        output = self.LayerNorm(output + attention_output)
        return output


class DPPEncoder3(nn.Module):
    def __init__(self, args):
        super(DPPEncoder3, self).__init__()
        layer = DPPLayer3(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_layers)])

    def forward(self, hidden_states, attention_mask, diversity_kernel=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, diversity_kernel)
        return hidden_states


class PAtt3(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Embedding size.')
        parser.add_argument('--num_layers', type=int, default=1, help='Number of DPP layers.')
        parser.add_argument('--num_heads', type=int, default=4, help='Number of heads.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max

        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)

        self.item_encoder = DPPEncoder3(args)
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, feed_dict):
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths']
        user_ids = feed_dict['user_id']
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)
        len_range = torch.arange(self.max_his, device=self.device)
        pos_idx = (lengths[:, None] - len_range[None, :seq_len]).clamp(min=0)
        pos_idx = pos_idx * valid_his
        pos_vectors = self.p_embeddings(pos_idx)
        user_vectors = self.u_embeddings(user_ids).unsqueeze(1)

        sequence_emb = his_vectors + pos_vectors + user_vectors
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # Masking
        causal = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
        attn_mask = (causal * -1e9).unsqueeze(0).expand(batch_size, -1, -1)
        pad = (history == 0)
        pad_col = pad.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        pad_row = pad.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        final_mask = attn_mask + (pad_col | pad_row).float() * -1e9

        # PAtt3 不需要 diversity_kernel
        output_features = self.item_encoder(sequence_emb, final_mask, diversity_kernel=None)

        seq_indices = (lengths - 1).clamp(min=0)
        idx = seq_indices.view(-1, 1, 1).expand(-1, 1, self.emb_size)
        his_vector = output_features.gather(1, idx).squeeze(1)

        target_vectors = self.i_embeddings(i_ids)
        prediction = torch.einsum('bd,bnd->bn', his_vector, target_vectors)
        return {'prediction': prediction.view(batch_size, -1)}