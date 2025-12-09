# -*- coding: UTF-8 -*-
# @Author  : PAtt reproduction on ReChorus
# @Email   :

import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel


# ============================================================================
#  一些基础组件（与 PAtt 原代码保持一致）
# ============================================================================

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {
    "gelu": gelu,
    "relu": F.relu,
    "swish": swish
}


class LayerNorm(nn.Module):
    """BERT 风格的 LayerNorm（与原仓库实现一致）"""

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


# ============================================================================
#  PAtt 的 DPP 注意力模块（根据 l-lyl/PAtt 的思想改写，适配 ReChorus）
# ============================================================================

class DPPAttention(nn.Module):
    """
    这里只实现论文中的 PAtt2（k=2），即只考虑二元依赖。
    """

    def __init__(self, args):
        super(DPPAttention, self).__init__()

        self.hidden_size = args.emb_size
        self.num_attention_heads = args.num_heads
        self.dropout_prob = args.dropout
        self.max_seq_len = args.history_max

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (self.hidden_size, self.num_attention_heads)
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # PAtt 中只需要 Sampler（记为 S），这里用 query 线性层产生
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        # 论文里 Value 直接用输入也可以，这里保留一层方便扩展
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(self.dropout_prob)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(self.dropout_prob)

        # 2x2 子式的对角稳定项
        K = 2
        a_diag = torch.eye(K) * 1e-5
        a_diag = a_diag.reshape((1, K, K))
        # 对于每个序列，我们会用到所有 2 子集，最多 L^2 个
        self.register_buffer(
            "sub_diag",
            a_diag.repeat(self.max_seq_len * self.max_seq_len, 1, 1)
        )

    def forward(self, input_tensor, attention_mask):
        """
        input_tensor: [B, L, D]
        attention_mask: [B, L, L] 或 [B,1,L,L]
        这里我们统一处理成 [B, L, L]
        """

        # ---- 统一 mask 形状 ----
        if attention_mask.dim() == 4:
            # [B,1,L,L] -> [B,L,L]
            attention_mask = attention_mask.squeeze(1)
        elif attention_mask.dim() != 3:
            raise ValueError(
                f"attention_mask must be [B,L,L] or [B,1,L,L], but got {attention_mask.shape}"
            )

        batch_len, seq_len, _ = input_tensor.shape

        # ---- 1. 生成 Sampler S ----
        # 这里没有显式 multi-head 拆分，和原 PAtt 的实现类似，
        # 直接把 all_head_size 视作一个大的维度（ReChorus 里也允许这样）
        query_layer = self.query(input_tensor)      # [B, L, D]
        value_layer = input_tensor                  # 直接用输入作为 value

        # ---- 2. 构建 DPP Kernel L = S S^T ----
        # 参考原实现，对 query 先进行逐元素平方增强表示
        q_layer = torch.mul(query_layer, query_layer)         # [B, L, D]
        qk_kernel = torch.bmm(q_layer, q_layer.transpose(1, 2))  # [B, L, L]

        # ---- 3. 计算所有 2x2 子式的行列式作为未归一化概率 ----
        # 构造所有 (i, j) 的下标组合
        indices = torch.arange(seq_len, device=input_tensor.device)
        idx_i = indices.view(-1, 1).repeat(1, seq_len).view(-1)  # 0,0,0,...,1,1,1,...
        idx_j = indices.view(1, -1).repeat(seq_len, 1).view(-1)  # 0,1,2,...,0,1,2,...

        # 提取子矩阵元素
        k_ii = qk_kernel[:, idx_i, idx_i]  # [B, L*L]
        k_ij = qk_kernel[:, idx_i, idx_j]  # [B, L*L]
        k_ji = qk_kernel[:, idx_j, idx_i]  # [B, L*L]
        k_jj = qk_kernel[:, idx_j, idx_j]  # [B, L*L]

        # 组装成 [B, L*L, 2, 2]
        tuple_subkernel = torch.stack(
            [
                torch.stack([k_ii, k_ij], dim=-1),
                torch.stack([k_ji, k_jj], dim=-1)
            ],
            dim=-2
        )

        num_elements = tuple_subkernel.shape[1]
        current_sub_diag = self.sub_diag[:num_elements, :, :].to(input_tensor.device)
        current_sub_diag = current_sub_diag.unsqueeze(0).expand(batch_len, -1, -1, -1)

        # 行列式 -> [B, L*L]
        tuple_subdet = torch.linalg.det(tuple_subkernel + current_sub_diag)
        tuple_subdet = tuple_subdet.view(batch_len, seq_len, seq_len)  # [B, L, L]

        # ---- 4. 归一化，得到 2-DPP 概率 ----
        # 只对上三角（i < j）求和作为分母
        denominator = torch.sum(torch.triu(tuple_subdet, diagonal=1), dim=(1, 2), keepdim=True)
        denominator = torch.clamp(denominator, min=1e-9)
        tuple_subprob = tuple_subdet / denominator  # [B, L, L]

        # ---- 5. 转成 Attention Score：注意 PAtt 的“取负 + 加 diag” 逻辑 ----
        one_diag = torch.diagonal(qk_kernel, dim1=1, dim2=2)      # [B, L]
        one_diag_matrix = torch.diag_embed(one_diag)              # [B, L, L]
        # 注意这里是概率 + diag，再整体取负
        tuple_subprob = -(tuple_subprob + one_diag_matrix)        # [B, L, L]

        # scale
        tuple_subprob = tuple_subprob / math.sqrt(self.attention_head_size)

        # ---- 6. 加上 mask，softmax 得到注意力权重 ----
        attention_scores = tuple_subprob + attention_mask         # [B, L, L]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # ---- 7. 加权求和 + 残差层归一化 ----
        context_layer = torch.matmul(attention_probs, value_layer)  # [B, L, D]

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DPPLayer(nn.Module):
    """一层 DPP 注意力 + FFN（Transformer Block）"""

    def __init__(self, args):
        super(DPPLayer, self).__init__()
        self.attention = DPPAttention(args)

        self.dense_1 = nn.Linear(args.emb_size, args.emb_size * 4)
        self.dense_2 = nn.Linear(args.emb_size * 4, args.emb_size)
        self.LayerNorm = LayerNorm(args.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)  # [B, L, D]

        output = self.dense_1(attention_output)
        output = self.act_fn(output)
        output = self.dense_2(output)
        output = self.dropout(output)
        output = self.LayerNorm(output + attention_output)

        return output  # [B, L, D]


class DPPEncoder(nn.Module):
    """堆叠多层 DPPLayer"""

    def __init__(self, args):
        super(DPPEncoder, self).__init__()
        layer = DPPLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_layers)])

    def forward(self, hidden_states, attention_mask):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


# ============================================================================
#  ReChorus 中的模型定义：PAtt（继承 SequentialModel）
# ============================================================================

class PAtt(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'num_layers', 'num_heads']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Embedding size.')
        parser.add_argument('--num_layers', type=int, default=1, help='Number of DPP layers.')
        parser.add_argument('--num_heads', type=int, default=4, help='Number of heads (kept for compatibility).')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.max_his = args.history_max

        # 基本统计来自 BaseModel：self.item_num, self.user_num, self.device 等

        # 1) Item / Position / User Embeddings
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        # position embedding 长度 = max_his + 1（含位置 0）
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)

        # 2) DPP Encoder
        self.item_encoder = DPPEncoder(args)
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)

        self.apply(self.init_weights)

    # ---- 参数初始化，与 ReChorus 其他模型保持一致 ----
    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    # ---- 前向传播：核心逻辑 ----
    def forward(self, feed_dict):
        """
        feed_dict keys:
            'item_id'       : [B, N] 候选 item（第 0 维是正例，之后是负例）
            'history_items' : [B, L] 历史序列
            'lengths'       : [B]     每个序列的有效长度
            'user_id'       : [B]     用户 id
        """
        i_ids = feed_dict['item_id']          # [B, N]
        history = feed_dict['history_items']  # [B, L]
        lengths = feed_dict['lengths']        # [B]
        user_ids = feed_dict['user_id']       # [B]

        batch_size, seq_len = history.shape

        # ---- 1) Embedding ----
        # 有效历史 mask
        valid_his = (history > 0).long()                        # [B, L]

        # item embedding
        his_vectors = self.i_embeddings(history)                # [B, L, D]

        # position embedding（倒序编码 + clamp 防止负索引）
        len_range = torch.arange(self.max_his, device=self.device)  # [max_his]
        # lengths[:,None] - [0..L-1]，可能为负，先 clamp，再乘有效 mask
        pos_idx = (lengths[:, None] - len_range[None, :seq_len]).clamp(min=0)
        pos_idx = pos_idx * valid_his                            # [B, L]
        pos_vectors = self.p_embeddings(pos_idx)                 # [B, L, D]

        # user embedding
        user_vectors = self.u_embeddings(user_ids).unsqueeze(1)  # [B, 1, D]

        # 融合
        sequence_emb = his_vectors + pos_vectors + user_vectors  # [B, L, D]
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        # ---- 2) 构造 3D Attention Mask: [B, L, L] ----
        # 2.1 因果 mask：只允许看当前位置及之前
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.float32),
            diagonal=1
        )  # 上三角（严格）为 1
        attn_mask = (causal * -1e9).unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]

        # 2.2 padding mask：屏蔽 padding 的行和列
        pad = (history == 0)                                     # [B, L]
        pad_col = pad.unsqueeze(1).expand(batch_size, seq_len, seq_len)  # key 方向
        pad_row = pad.unsqueeze(2).expand(batch_size, seq_len, seq_len)  # query 方向
        final_mask = attn_mask + (pad_col | pad_row).float() * -1e9      # [B, L, L]

        # ---- 3) DPP Encoder ----
        output_features = self.item_encoder(sequence_emb, final_mask)    # [B, L, D]

        # ---- 4) 取每个样本的最后一个有效位置表示 ----
        # lengths 是有效长度，最后一个有效位置 index = lengths - 1
        seq_indices = (lengths - 1).clamp(min=0)                 # [B]
        idx = seq_indices.view(-1, 1, 1).expand(-1, 1, self.emb_size)  # [B,1,D]
        his_vector = output_features.gather(1, idx).squeeze(1)   # [B, D]

        # ---- 5) 与候选 item 做内积打分 ----
        target_vectors = self.i_embeddings(i_ids)                # [B, N, D]

        # [B,D] · [B,N,D] -> [B,N]
        prediction = torch.einsum('bd,bnd->bn', his_vector, target_vectors)

        return {'prediction': prediction.view(batch_size, -1)}