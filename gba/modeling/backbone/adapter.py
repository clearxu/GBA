import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F
from timm.models.layers import DropPath


from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1,
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def pcnorm(self,residual, bn):
        norm = F.batch_norm(
            residual, bn.running_mean, bn.running_var,
            None, None, bn.training, bn.momentum, bn.eps)

        phase, _ = decompose(residual, 'phase')
        _, norm_amp = decompose(norm, 'amp')

        residual = compose(
            phase,
            norm_amp
        )

        residual = residual * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
        return residual

    def replace_denormals(self,x, threshold=1e-5):
        y = x.clone()
        y[(x < threshold) & (x > -1.0 * threshold)] = threshold
        return y

    def decompose(self,x, mode='all'):
        fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
        if mode == 'all' or mode == 'amp':
            fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
            fft_amp = torch.sqrt(replace_denormals(fft_amp))
        else:
            fft_amp = None

        if mode == 'all' or mode == 'phase':
            fft_pha = torch.atan2(fft_im[..., 1], replace_denormals(fft_im[..., 0]))
        else:
            fft_pha = None
        return fft_pha, fft_amp

    def compose(self,phase, amp):
        x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
        x = x / math.sqrt(x.shape[2] * x.shape[3])
        x = torch.view_as_complex(x)
        return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')

    def forward(self, q, k, v, attention_mask=None, return_attention=True):
        if len(q.size()) == 3:
            bsz, tgt_len, embed_dim = q.size()
        elif len(q.size()) == 2:
            tgt_len, embed_dim = q.size()
            bsz = k.shape[0]
            q = q.expand(bsz, tgt_len, embed_dim)

        query_states = self.q_proj(q) * self.scale

        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        # print(query_states.shape)
        # exit(0)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))


        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # max_weights, _ = attn_weights.max(dim=-1)  # 取每个头对所有键的最大值
        # max_weights_mean = max_weights.mean(dim=0)  # 求头之间的平均值，得到每个查询的平均最大权重
        #
        # # 确定屏蔽阈值
        # threshold = 0.5  # 这是一个示例值，需要根据实际情况调整
        #
        # # 构建屏蔽矩阵
        # mask = max_weights_mean > threshold  # 超过阈值的特征将被屏蔽
        #
        # # 应用屏蔽
        # # 假设 visual_features 是形状为 [num_queries, feature_dim] 的视觉特征矩阵
        # mask_expanded = mask.unsqueeze(-1).expand_as(attn_weights)
        # attn_weights = attn_weights * mask_expanded.float()
        # print(visual_features_masked.shape,value_states.shape)
        # torch.Size([48, 65536, 255])
        # torch.Size([48, 255, 96]
        # exit(0)


        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)


        attn_output = torch.bmm(attn_probs, value_states)
        # print(attn_output.shape,attn_probs.shape,value_states.shape)
        # torch.Size([48, 65536, 96])
        # torch.Size([48, 65536, 255])
        # torch.Size([48, 255, 96])



        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        # w = attn_probs.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # print(w.shape) 6 8 65536 255 -->6 65536 8 255 --> 6 65536 2040
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # print(attn_output.shape) torch.Size([6, 8, 65536, 96])
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        # print(attn_output.shape) torch.Size([6, 65536, 768])
        attn_output = self.out_proj(attn_output)


        return attn_output, attn_weights



class ContextInteraction(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-1, use_layer_scale = False,
                 clamp_min_for_underflow = False, clamp_max_for_overflow = False):

        super(ContextInteraction, self).__init__()

        # pre_layer norm
        self.layer_norm_q_1 = nn.LayerNorm(q_dim)
        self.layer_norm_k_1 = nn.LayerNorm(k_dim)
        self.attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       clamp_min_for_underflow=clamp_min_for_underflow,
                                       clamp_max_for_overflow=clamp_max_for_overflow)

        # add layer scale for training stability
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma = nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True)

        self.bn = nn.BatchNorm2d(q_dim)


    def replace_denormals(self,x, threshold=1e-5):
        y = x.clone()
        y[(x < threshold) & (x > -1.0 * threshold)] = threshold
        return y

    def decompose(self,x, mode='all'):
        fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
        if mode == 'all' or mode == 'amp':
            fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
            fft_amp = torch.sqrt(self.replace_denormals(fft_amp))
        else:
            fft_amp = None

        if mode == 'all' or mode == 'phase':
            fft_pha = torch.atan2(fft_im[..., 1], self.replace_denormals(fft_im[..., 0]))
        else:
            fft_pha = None
        return fft_pha, fft_amp

    def compose(self,phase, amp):
        x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
        x = x / math.sqrt(x.shape[2] * x.shape[3])
        x = torch.view_as_complex(x)
        return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')

    def forward(self, multi_scale_features, text_feat, attention_mask=None):

        q= multi_scale_features
        k = text_feat
        v = text_feat
        # print(q.shape,k.shape)
        output = []
        bs, _, h, w = q.shape
        k = k.expand(bs, k.shape[0], k.shape[1])
        v = v.expand(bs, v.shape[0], v.shape[1])
        # for q_index, q in enumerate([q0, q1, q2, q3]):
        bs, _, h, w = q.shape
        q = q.flatten(2).transpose(1, 2)
        q = self.layer_norm_q_1(q)
        k, v = self.layer_norm_k_1(k), self.layer_norm_k_1(v)
        delta_q,attn = self.attn(q, k, v)
        delta_q = delta_q.to(dtype=torch.float32)

        q_p = delta_q.transpose(1, 2).contiguous().view(bs, -1, h, w)
        norm = F.batch_norm(
            q_p, self.bn.running_mean, self.bn.running_var,
            None, None, self.bn.training, self.bn.momentum, self.bn.eps)

        phase, _ = self.decompose(q_p, 'phase')
        _, norm_amp = self.decompose(norm, 'amp')

        residual = self.compose(
            phase,
            norm_amp
        )

        q_pc = residual * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)


        delta_q = q_pc.reshape(bs,-1,h*w).permute(0,2,1)

        if self.use_layer_scale:
            q = q + self.drop_path(self.gamma * delta_q)
        else:
            q = q + delta_q
        q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)

        output.append(q)

        return q





class StyleHallucination(nn.Module):

    def __init__(self,  concentration_coeff=0.0156, base_style_num=64):
        super().__init__()
        # self.args = args
        style_dim = 192
        self.concentration = torch.tensor([concentration_coeff] * base_style_num, device='cuda')
        self._dirichlet = tdist.dirichlet.Dirichlet(concentration=self.concentration)

        self.register_buffer("proto_mean", torch.zeros((base_style_num, style_dim), requires_grad=False))
        self.register_buffer("proto_std", torch.zeros((base_style_num, style_dim), requires_grad=False))
        self.bn = nn.BatchNorm2d(style_dim)

    def pcnorm(self, residual):
        residual = residual.to(dtype=torch.float32)
        norm = F.batch_norm(
            residual, self.bn.running_mean, self.bn.running_var,
            None, None, self.bn.training, self.bn.momentum, self.bn.eps)

        phase, _ = self.decompose(residual, 'phase')
        _, norm_amp = self.decompose(norm, 'amp')

        residual = self.compose(phase, norm_amp)

        residual = residual * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)
        return residual

    def replace_denormals(self, x, threshold=1e-5):
        y = x.clone()
        y[(x < threshold) & (x > -1.0 * threshold)] = threshold
        return y

    def decompose(self, x, mode='all'):
        fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
        if mode == 'all' or mode == 'amp':
            fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
            fft_amp = torch.sqrt(self.replace_denormals(fft_amp))
        # else:
        #     fft_amp = None

        if mode == 'all' or mode == 'phase':
            fft_pha = torch.atan2(fft_im[..., 1], self.replace_denormals(fft_im[..., 0]))
        # else:
        #     fft_pha = None
        return fft_pha, fft_amp

    def compose(self, phase, amp):
        x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
        x = x / math.sqrt(x.shape[2] * x.shape[3])
        x = torch.view_as_complex(x)
        return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')

    def forward(self, x):
        B, C, H, W = x.size()

        residual = x.to(dtype=torch.float32)
        norm = F.batch_norm(
            residual, self.bn.running_mean, self.bn.running_var,
            None, None, self.bn.training, self.bn.momentum, self.bn.eps)

        phase, norm_amp = self.decompose(norm, 'all')

        combine_weights = self._dirichlet.sample((B,))  # B,C
        combine_weights = combine_weights.detach()
        new_mean = combine_weights @ self.proto_mean.data  # B,C
        new_std = combine_weights @ self.proto_std.data


        norm_amp = norm_amp * new_std.unsqueeze(-1).unsqueeze(-1) + new_mean.unsqueeze(-1).unsqueeze(-1)

        residual = self.compose(phase, norm_amp)
        x_new = residual * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)

        return x_new


class MonaOp_st(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

        self.prompt = torch.nn.parameter.Parameter(torch.randn(in_features, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(in_features), requires_grad=True)

        self.relu = nn.SiLU(inplace=True)
        # self.bn = nn.BatchNorm2d(in_features)

        # self.cluster = ContentAttention(dim=192, window_size=8, num_heads=8)
        self.st =StyleHallucination()

    def forward(self, x):
        # h, w = hw
        b, c, h, w = x.shape
        # x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        identity = x
        # conv1_x = self.drop_area1(self.conv1(x))
        # conv2_x = self.drop_area2(self.conv2(x))
        # conv3_x = self.drop_area3(self.conv3(x))

        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = self.st(conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)
        x = self.relu(x)

        x = x.reshape(b, c, -1).permute(0, 2, 1)

        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        # B, N, 1
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = x @ self.top_down_transform

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2) + identity
        # x = x.reshape(b, c, -1).permute(0, 2, 1)
        return x

class MonaOp_sem(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

        self.prompt = torch.nn.parameter.Parameter(torch.randn(in_features, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(in_features), requires_grad=True)

        self.relu = nn.SiLU(inplace=True)
        # self.bn = nn.BatchNorm2d(in_features)
        self.ci = ContextInteraction( q_dim=in_features, k_dim=768, embed_dim=in_features, num_heads=1)
        # self.cluster = ContentAttention(dim=192, window_size=8, num_heads=8)

    def pcnorm(self, residual, bn):
        residual = residual.to(dtype=torch.float32)
        norm = F.batch_norm(
            residual, bn.running_mean, bn.running_var,
            None, None, bn.training, bn.momentum, bn.eps)

        phase, _ = self.decompose(residual, 'phase')
        _, norm_amp = self.decompose(norm, 'amp')

        residual = self.compose(
            phase,
            norm_amp
        )

        residual = residual * bn.weight.view(1, -1, 1, 1) + bn.bias.view(1, -1, 1, 1)
        return residual

    def replace_denormals(self, x, threshold=1e-5):
        y = x.clone()
        y[(x < threshold) & (x > -1.0 * threshold)] = threshold
        return y

    def decompose(self, x, mode='all'):
        fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
        if mode == 'all' or mode == 'amp':
            fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
            fft_amp = torch.sqrt(self.replace_denormals(fft_amp))
        else:
            fft_amp = None

        if mode == 'all' or mode == 'phase':
            fft_pha = torch.atan2(fft_im[..., 1], self.replace_denormals(fft_im[..., 0]))
        else:
            fft_pha = None
        return fft_pha, fft_amp

    def compose(self, phase, amp):
        x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1)
        x = x / math.sqrt(x.shape[2] * x.shape[3])
        x = torch.view_as_complex(x)
        return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')

    def forward(self, x,text_feature):
        # h, w = hw
        b, c, h, w = x.shape


        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        x = self.ci((conv1_x + conv2_x + conv3_x)/3.0, text_feature)

        identity = x

        x = self.projector(x)
        x = self.relu(x)

        x = x.reshape(b, c, -1).permute(0, 2, 1)

        cos_sim = F.normalize(x, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)
        # B, N, 1
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = x @ self.top_down_transform

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2) + identity
        # x = x.reshape(b, c, -1).permute(0, 2, 1)
        return x