# -*- coding: UTF-8 -*-
# @Author  : DPAtt3 implementation for ReChorus
# @Email   :

import torch
import torch.nn as nn
from models.sequential.PAtt3 import PAtt3

class DPAtt3(PAtt3):
    """
    DPAtt3: Diversity-aware Probabilistic Attention (Triple Dependency)
    继承自 PAtt3，在 Attention 计算中引入多样性核 T = L * C^(-1)
    """

    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers']

    def __init__(self, args, corpus):
        # 初始化父类 PAtt3 (构建 item_encoder, embeddings 等)
        super().__init__(args, corpus)

        # Diversity Representation V
        # 用于生成类别/多样性核 C = V^T * V
        # 我们使用独立的线性映射层将 Item Embedding 映射到多样性空间
        self.div_embeddings = nn.Linear(self.emb_size, self.emb_size)

        # 初始化该层权重
        self.div_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.div_embeddings.bias.data.zero_()

    def forward(self, feed_dict):
        i_ids = feed_dict['item_id']
        history = feed_dict['history_items']
        lengths = feed_dict['lengths']
        user_ids = feed_dict['user_id']

        batch_size, seq_len = history.shape

        # 1. Embeddings (与 PAtt3 保持一致)
        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)  # [B, L, D]

        len_range = torch.arange(self.max_his, device=self.device)
        pos_idx = (lengths[:, None] - len_range[None, :seq_len]).clamp(min=0)
        pos_idx = pos_idx * valid_his
        pos_vectors = self.p_embeddings(pos_idx)
        user_vectors = self.u_embeddings(user_ids).unsqueeze(1)

        sequence_emb = his_vectors + pos_vectors + user_vectors
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # =========================================================
        # DPAtt3 核心逻辑：构建 Integrated Kernel T
        # =========================================================

        # A. 计算 Relevance Kernel L = S * S^T
        # 为了保证关联性建模的一致性，我们复用 Encoder 第一层 Attention 的 Query 投影参数作为 Sampler
        # 注意：这里假设 Encoder 是 DPPEncoder3 类型
        sampler = self.item_encoder.layer[0].attention.query(sequence_emb)
        s_layer = torch.mul(sampler, sampler)
        L_kernel = torch.bmm(s_layer, s_layer.transpose(1, 2))  # [B, L, L]

        # B. 计算 Diversity Kernel C = V * V^T
        # V 是基于历史序列的多样性特征表示
        V_seq = self.div_embeddings(his_vectors)  # [B, L, D]
        C_kernel = torch.bmm(V_seq, V_seq.transpose(1, 2))  # [B, L, L]

        # 加上 identity 矩阵防止 C 不可逆或数值不稳定
        eye_matrix = torch.eye(seq_len, device=self.device).unsqueeze(0)
        C_kernel = C_kernel + eye_matrix * 1e-5

        # C. 计算 Integrated Kernel T = L * C^(-1)
        # 使用 solve 求解线性方程组比直接求逆更稳定
        # 我们求解 X 使得 X * C = L => C^T * X^T = L^T
        # 由于 C 是对称阵 (V*V^T)，C^T = C
        # torch.linalg.solve(A, B) 解 AX = B
        # 所以我们计算 solve(C, L^T) 得到 T^T
        T_transposed = torch.linalg.solve(C_kernel, L_kernel.transpose(1, 2))
        T_kernel = T_transposed.transpose(1, 2)  # [B, L, L]

        # =========================================================
        # 结束核心逻辑，继续标准流程
        # =========================================================

        # 2. Attention Mask (Causal + Padding)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
        attn_mask = (causal * -1e9).unsqueeze(0).expand(batch_size, -1, -1)

        pad = (history == 0)
        pad_col = pad.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        pad_row = pad.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        final_mask = attn_mask + (pad_col | pad_row).float() * -1e9

        # 3. Encoder (传入计算好的 Diversity Kernel T)
        # Encoder 内部会优先使用传入的 kernel 进行行列式计算
        output_features = self.item_encoder(sequence_emb, final_mask, diversity_kernel=T_kernel)

        # 4. Gather Last Item & Prediction
        seq_indices = (lengths - 1).clamp(min=0)
        idx = seq_indices.view(-1, 1, 1).expand(-1, 1, self.emb_size)
        his_vector = output_features.gather(1, idx).squeeze(1)

        target_vectors = self.i_embeddings(i_ids)
        prediction = torch.einsum('bd,bnd->bn', his_vector, target_vectors)

        return {'prediction': prediction.view(batch_size, -1)}