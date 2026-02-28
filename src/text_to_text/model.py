"""
Text-to-Text Module — LLaMA-style Decoder-Only Transformer

This module implements a small LLaMA-style transformer that can be:
  1. Pre-trained from scratch on text corpora
  2. Fine-tuned from an existing checkpoint (recommended)
  3. Used for inference with quantization

For production, we recommend starting from a pre-trained checkpoint
(TinyLlama-1.1B or Pythia-1.4B) and fine-tuning with LoRA.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLMConfig:
    """Configuration for the LLM."""

    vocab_size: int = 32000
    hidden_dim: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 4  # Grouped-Query Attention (GQA)
    intermediate_dim: int = 5504  # SwiGLU FFN
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 10000.0

    @classmethod
    def tiny(cls) -> "LLMConfig":
        """~125M param config for testing."""
        return cls(
            hidden_dim=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=4,
            intermediate_dim=2048,
        )

    @classmethod
    def small(cls) -> "LLMConfig":
        """~1.3B param config for Phase 1."""
        return cls()

    @classmethod
    def medium(cls) -> "LLMConfig":
        """~7B param config for Phase 5."""
        return cls(
            hidden_dim=4096,
            num_layers=32,
            num_heads=32,
            num_kv_heads=8,
            intermediate_dim=11008,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm — no mean subtraction or bias.
    Used in LLaMA, Mistral, and most modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE).

    Encodes position information by rotating query/key vectors.
    Allows extending context length after training via frequency scaling.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.freqs.dtype)
        freqs = torch.outer(t, self.freqs)
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # x shape: (batch, heads, seq_len, head_dim)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA).

    Uses fewer KV heads than query heads, reducing KV-cache memory
    by a factor of (num_heads / num_kv_heads) during inference.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.kv_group_size = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(seq_len, x.device)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV cache for inference
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match query heads (GQA)
        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(mask is None and kv_cache is None)
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output), new_kv_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    SwiGLU = Swish(xW_gate) * (xW_up), followed by down projection.
    ~1-2% better than ReLU FFN at similar parameter count.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_dim)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm + attention + residual
        h, new_kv_cache = self.attention(self.attention_norm(x), mask, kv_cache)
        x = x + h
        # Pre-norm + FFN + residual
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv_cache


class CustomLLM(nn.Module):
    """Complete LLaMA-style Language Model.

    Usage:
        config = LLMConfig.tiny()  # For testing
        model = CustomLLM(config)

        # Training
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        logits, _ = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))

        # Generation
        generated = model.generate(prompt_ids, max_new_tokens=100)
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying (embed and lm_head share weights)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embed(input_ids)
        new_kv_caches = []

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, mask, cache)
            new_kv_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV-cache.

        Args:
            input_ids: Prompt token IDs, shape (1, prompt_len).
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_p: Nucleus sampling threshold.
            top_k: Top-K sampling threshold.

        Returns:
            Generated token IDs including the prompt.
        """
        kv_caches = None

        for _ in range(max_new_tokens):
            # Only feed the last token if we have KV cache
            if kv_caches is not None:
                model_input = input_ids[:, -1:]
            else:
                model_input = input_ids

            logits, kv_caches = self(model_input, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature

            # Top-K filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[:, -1:]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- Convenience: Load from HuggingFace pre-trained checkpoints ---

def load_pretrained_llm(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    lora_path: str = None,
    quantization: str = "int4",
):
    """Load a pre-trained LLM from HuggingFace, optionally with LoRA weights.

    This is the RECOMMENDED starting point. Pre-training from scratch
    is expensive and unnecessary for most use cases.

    Args:
        model_name: HuggingFace model ID. Good options:
            - "microsoft/Phi-3-mini-4k-instruct" (3.8B, excellent reasoning)
            - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (1.1B, fast)
            - "mistralai/Mistral-7B-Instruct-v0.2" (7B, high quality)
        lora_path: Path to a LoRA checkpoint directory. If provided,
            the LoRA adapter is loaded and merged into the base model.
        quantization: Quantization mode — "int4", "int8", or None for full precision.
            Use "int4" for GPUs with 8GB VRAM or less.

    Returns:
        (model, tokenizer) tuple ready for fine-tuning or inference.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        lora_path if lora_path else model_name,
        trust_remote_code=True,
    )

    model = _load_model_with_fallback(model_name, quantization)

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    return model, tokenizer


def _load_model_with_fallback(model_name: str, quantization: str):
    """Try to load with quantization, fall back to float16 + CPU offload on failure."""
    from transformers import AutoModelForCausalLM

    # Attempt 1: Try quantization if requested
    if quantization in ("int4", "int8"):
        try:
            from transformers import BitsAndBytesConfig

            if quantization == "int4":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            print(f"[LLM] Loading with {quantization} quantization...")
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[LLM] Quantization failed ({e}), falling back to float16 with CPU offload...")

    # Attempt 2: float16 with auto device map (splits between GPU and CPU)
    print("[LLM] Loading in float16 with automatic GPU/CPU split...")
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
