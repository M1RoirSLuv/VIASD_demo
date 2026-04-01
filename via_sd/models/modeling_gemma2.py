# coding=utf-8
"""
Swift-style Gemma2 with layer skipping support.

对应 modeling_llama.py 为 LLaMA 所做的修改, 本文件为 Gemma2 提供相同接口:
  - LlamaForCausalLM.self_draft()    →  Gemma2ForCausalLM.self_draft()
  - LlamaForCausalLM.set_skip_layers →  Gemma2ForCausalLM.set_skip_layers()
  - LlamaForCausalLM.get_skip_layers →  Gemma2ForCausalLM.get_skip_layers()

Gemma2 与 LLaMA 的关键架构差异:
  1. 每个 DecoderLayer 有 4 个 LayerNorm (pre/post attn, pre/post mlp)
  2. 注意力有 logit softcapping (attn_logit_softcapping)
  3. final logits 有 final_logit_softcapping
  4. 交替使用 sliding_window 注意力和全局注意力
  5. GQA (Grouped Query Attention)

KV Cache 策略:
  与 modeling_llama.py 相同: 使用 kv_cache.py 的自定义 KVCache,
  通过 initialize_past_key_values() 预分配连续张量,
  在 Gemma2Attention.forward() 内通过 past_key_value[k].cat() 追加。

参考:
  transformers.models.gemma2.modeling_gemma2
  model/swift/modeling_llama.py
  model/swift/kv_cache.py
"""

import math
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config

# transformers 版本兼容: 不同版本的 Gemma2 导出名可能不同
from transformers.models.gemma2 import modeling_gemma2 as _gemma2_module
Gemma2RMSNorm = getattr(_gemma2_module, "Gemma2RMSNorm",
                        getattr(_gemma2_module, "GemmaRMSNorm", None))
Gemma2MLP = _gemma2_module.Gemma2MLP
_Gemma2Model = _gemma2_module.Gemma2Model
_Gemma2ForCausalLM = _gemma2_module.Gemma2ForCausalLM

# RoPE: 新版可能叫 Gemma2RotaryEmbedding 或 GemmaRotaryEmbedding
Gemma2RotaryEmbedding = getattr(_gemma2_module, "Gemma2RotaryEmbedding",
                                getattr(_gemma2_module, "GemmaRotaryEmbedding", None))

# 这两个函数在新版 transformers 中可能从 llama 或 gemma2 模块导入
apply_rotary_pos_emb = getattr(_gemma2_module, "apply_rotary_pos_emb", None)
if apply_rotary_pos_emb is None:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
repeat_kv = getattr(_gemma2_module, "repeat_kv", None)
if repeat_kv is None:
    from transformers.models.llama.modeling_llama import repeat_kv

logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# 全局跳层状态 (与 modeling_llama.py 相同的机制)
# ---------------------------------------------------------------------------
_enabled_draft = False
_attn_skip_layer_id_set = []
_mlp_skip_layer_id_set  = []


# ---------------------------------------------------------------------------
# Gemma2Attention with custom KVCache support
# ---------------------------------------------------------------------------

class Gemma2Attention(nn.Module):
    """
    Gemma2 多头注意力, 支持自定义 KVCache.

    与 modeling_llama.py 的 LlamaAttention 相同的修改点:
      past_key_value[0].cat(key_states, dim=2)   # 追加 key
      past_key_value[1].cat(value_states, dim=2) # 追加 value

    Gemma2 特有:
      - attn_logit_softcapping: 注意力权重 softcapping
      - sliding_window: 部分层使用滑动窗口 (layer_idx % 2 == 0 为滑动窗口层)
      - query_pre_attn_scalar: 缩放因子
    """

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size        = config.hidden_size
        self.num_heads          = config.num_attention_heads
        self.head_dim           = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Gemma2 特有: 注意力 softcapping 和缩放
        self.attn_logit_softcapping = config.attn_logit_softcapping
        self.scaling = config.query_pre_attn_scalar ** -0.5

        # Sliding window (偶数层用滑动窗口, 奇数层用全局注意力)
        self.sliding_window = (
            config.sliding_window
            if (layer_idx % 2 == 0) and hasattr(config, "sliding_window")
            else None
        )

        # RoPE 初始化: 新版 transformers 用 config= 参数, 旧版用位置参数
        # Gemma2Config 可能没有 rope_theta (默认 10000.0)
        _rope_theta = getattr(config, 'rope_theta', 10000.0)
        try:
            self.rotary_emb = Gemma2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=_rope_theta,
            )
        except TypeError:
            self.rotary_emb = Gemma2RotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,       # KVCache pair [key_cache, value_cache]
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,          self.head_dim).transpose(1, 2)
        key_states   = key_states.view  (bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # 新版 transformers: rotary_emb(x, position_ids) → cos/sin 已按 position 索引
        # 旧版 transformers: rotary_emb(x, seq_len=) → cos/sin 需要 position_ids 索引
        _new_rope_api = False
        try:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        except TypeError:
            _new_rope_api = True
            if position_ids is None:
                past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
                position_ids = torch.arange(past_len, past_len + q_len, device=hidden_states.device).unsqueeze(0)
            cos, sin = self.rotary_emb(value_states, position_ids)

        if _new_rope_api:
            # 新版: cos/sin 已索引, apply_rotary_pos_emb 不需要 position_ids
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # KV Cache 追加 (与 modeling_llama.py L169-170 相同)
        if past_key_value is not None:
            key_states   = past_key_value[0].cat(key_states,   dim=2)
            value_states = past_key_value[1].cat(value_states, dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # GQA: 扩展 KV heads
        key_states   = repeat_kv(key_states,   self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 注意力权重 (含 Gemma2 的 softcapping)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 滑动窗口 mask (推理时, 若 kv_seq_len > sliding_window, 需要遮盖旧位置)
        if self.sliding_window is not None and kv_seq_len > self.sliding_window:
            sw = self.sliding_window
            # 仅保留最近 sw 个位置
            sliding_mask = torch.zeros_like(attn_weights)
            sliding_mask[..., :-sw] = torch.finfo(attn_weights.dtype).min
            # 只对新 token (query) 应用
            if q_len == 1:
                attn_weights = attn_weights + sliding_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output  = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value


# ---------------------------------------------------------------------------
# Gemma2DecoderLayer with layer skipping
# ---------------------------------------------------------------------------

class Gemma2DecoderLayer(nn.Module):
    """
    Gemma2 解码器层, 支持跳层 (对应 modeling_llama.py 的 LlamaDecoderLayer).

    Gemma2 与 LLaMA 的结构差异:
      LLaMA:  pre_norm → attn → residual, pre_norm → mlp → residual
      Gemma2: pre_norm → attn → post_norm → residual, pre_norm → mlp → post_norm → residual

    跳层逻辑 (与 modeling_llama.py 完全对应):
      推理模式 (not training):
        if _enabled_draft and self.layer_id in _attn_skip_layer_id_set:
            hidden_states = residual  # 跳过 attn, 保持残差
        if _enabled_draft and self.layer_id in _mlp_skip_layer_id_set:
            hidden_states = residual  # 跳过 mlp, 保持残差
    """

    def __init__(self, config: Gemma2Config, layer_id: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id    = layer_id

        self.self_attn             = Gemma2Attention(config, layer_idx=layer_id)
        self.mlp                   = Gemma2MLP(config)
        self.input_layernorm       = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm  = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple:
        # ── Attention 块 ──────────────────────────────────────────────────
        residual = hidden_states

        if _enabled_draft and self.layer_id in _attn_skip_layer_id_set:
            # 跳过 attention: 直接传递残差, KV cache 不更新
            present_key_value = None
        else:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_states, _, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            # Gemma2 特有: post-attention layernorm (在残差加之前)
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

        # ── MLP 块 ────────────────────────────────────────────────────────
        residual = hidden_states

        if _enabled_draft and self.layer_id in _mlp_skip_layer_id_set:
            # 跳过 MLP: 直接传递残差
            pass
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            # Gemma2 特有: post-feedforward layernorm
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ---------------------------------------------------------------------------
# Gemma2Model with custom decoder layers
# ---------------------------------------------------------------------------

class Gemma2Model(_Gemma2Model):
    """
    替换所有 DecoderLayer 为支持跳层的 Gemma2DecoderLayer.
    """

    def __init__(self, config: Gemma2Config):
        super(_Gemma2Model, self).__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size  = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            Gemma2DecoderLayer(config, layer_id=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        else:
            batch_size, seq_length, _ = inputs_embeds.shape

        past_key_values_length = 0
        if past_key_values is not None:
            for pkv in past_key_values:
                if pkv is not None:
                    past_key_values_length = pkv[0].shape[2]
                    break

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Gemma2 normalizes embeddings
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds * normalizer

        # 构建 causal attention mask
        attention_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, past_key_values_length, past_key_values
        )

        hidden_states = inputs_embeds
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
        )

    def _update_causal_mask(self, attention_mask, inputs_embeds, past_key_values_length, past_key_values):
        """构建 Gemma2 causal mask (适配自定义 KVCache 的 shape 读取方式)."""
        bsz, seq_len, _ = inputs_embeds.shape
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        # 计算 kv 总长度
        kv_len = seq_len + past_key_values_length

        # 标准 causal mask: query positions [0..seq_len-1] 对应绝对位置 [past..past+seq_len-1]
        causal_mask = torch.full(
            (seq_len, kv_len), fill_value=torch.finfo(dtype).min, dtype=dtype, device=device
        )
        # 每个 query 位置 i 可以看到 kv 位置 [0 .. past_key_values_length + i]
        rows = torch.arange(seq_len, device=device)
        cols = torch.arange(kv_len, device=device)
        causal_mask.masked_fill_(
            cols[None, :] <= (rows[:, None] + past_key_values_length), 0
        )
        # 扩展为 [bsz, 1, seq_len, kv_len]
        causal_mask = causal_mask[None, None, :, :].expand(bsz, 1, seq_len, kv_len)

        if attention_mask is not None and attention_mask.dim() == 2:
            expanded = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
            causal_mask = causal_mask + expanded.to(device)

        return causal_mask


# ---------------------------------------------------------------------------
# Gemma2ForCausalLM with Swift interface
# ---------------------------------------------------------------------------

class Gemma2ForCausalLM(_Gemma2ForCausalLM):
    """
    Gemma2 因果语言模型, 添加 Swift 风格的跳层推理接口.

    新增接口 (与 modeling_llama.py 的 LlamaForCausalLM 完全对应):
      - self_draft(enabled=True)     上下文管理器, 激活跳层模式
      - set_skip_layers(attn, mlp)   设置跳层配置
      - get_skip_layers()            获取当前跳层配置

    用法:
        model = Gemma2ForCausalLM.from_pretrained("google/gemma-2-9b")
        attn_skip = list(range(1, model.config.num_hidden_layers-1, 2))
        mlp_skip  = list(range(1, model.config.num_hidden_layers-1, 2))
        model.set_skip_layers(attn_skip, mlp_skip)
        with model.self_draft():
            outputs = model.model(input_ids=..., past_key_values=...)
    """

    def __init__(self, config: Gemma2Config):
        super(_Gemma2ForCausalLM, self).__init__(config)
        self.model = Gemma2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @contextmanager
    def self_draft(self, enabled: bool = True):
        """激活跳层推理模式 (对应 modeling_llama.py 的 self_draft)."""
        global _enabled_draft
        _enabled_draft = enabled
        try:
            yield None
        finally:
            _enabled_draft = False

    def set_skip_layers(
        self,
        attn_skip_layer_id_set=None,
        mlp_skip_layer_id_set=None,
    ):
        """设置跳层配置 (对应 modeling_llama.py 的 set_skip_layers)."""
        global _attn_skip_layer_id_set, _mlp_skip_layer_id_set
        if attn_skip_layer_id_set is not None:
            _attn_skip_layer_id_set = list(attn_skip_layer_id_set)
        if mlp_skip_layer_id_set is not None:
            _mlp_skip_layer_id_set = list(mlp_skip_layer_id_set)

    def get_skip_layers(self):
        """获取当前跳层配置 (对应 modeling_llama.py 的 get_skip_layers)."""
        return _attn_skip_layer_id_set, _mlp_skip_layer_id_set

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Gemma2 特有: final logit softcapping
        if hasattr(self.config, "final_logit_softcapping") and self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )


# ---------------------------------------------------------------------------
# initialize_past_key_values for Gemma2 (对应 kv_cache.py 的 LLaMA 版本)
# ---------------------------------------------------------------------------

def initialize_past_key_values_gemma2(model: Gemma2ForCausalLM):
    """
    为 Gemma2 初始化自定义 KVCache (对应 kv_cache.py: initialize_past_key_values).

    Gemma2 与 LLaMA 的差异:
      - head_dim 由 config.head_dim 直接提供 (LLaMA 需要 hidden_size // num_heads 计算)
      - num_key_value_heads 用于 GQA

    返回与 kv_cache.py 相同的三元组:
      (past_key_values, past_key_values_data_list, current_length_data)
    """
    from .kv_cache import KVCache

    config = model.config
    batch_size = 1
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    # 限制预分配长度, 避免 OOM (Gemma2 的 max_position_embeddings 可能非常大)
    max_len = min(getattr(config, 'max_position_embeddings', 4096), 4096)

    # 获取各层设备 (支持模型并行)
    devices = []
    for i in range(num_layers):
        try:
            devices.append(model.model.layers[i].self_attn.q_proj.weight.device)
        except Exception:
            devices.append(torch.device("cpu"))

    # 按设备分组预分配 KV Cache 张量
    past_key_values_data_list = []
    startnum, startdevice = 0, devices[0]
    for id, dev in enumerate(devices):
        if dev != startdevice:
            past_key_values_data_list.append(
                torch.zeros(startnum * 2, batch_size, num_kv_heads, max_len, head_dim,
                            device=startdevice, dtype=model.dtype)
            )
            startdevice = dev
            startnum = 0
        startnum += 1
    past_key_values_data_list.append(
        torch.zeros(startnum * 2, batch_size, num_kv_heads, max_len, head_dim,
                    device=startdevice, dtype=model.dtype)
    )

    current_length_data = torch.zeros(num_layers * 2, dtype=torch.long, device="cpu")

    past_key_values = []
    bias = 0
    start_dev = devices[0]
    for i in range(num_layers):
        dev = devices[i]
        if dev != start_dev:
            bias = 0
            start_dev = dev
        dev_idx = next(
            j for j, d in enumerate(past_key_values_data_list)
            if d.device == dev
        )
        past_key_values.append([
            KVCache(past_key_values_data_list[dev_idx][2 * bias + j],
                    current_length_data[i * 2 + j])
            for j in range(2)
        ])
        bias += 1

    return past_key_values, past_key_values_data_list, current_length_data
