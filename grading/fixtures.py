"""Small deterministic fixtures: no download, token, or prior-homework solution."""

import torch
from torch import nn


def model():
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))


def batches(device="cpu"):
    generator = torch.Generator().manual_seed(987)
    return [
        (
            torch.randn(8, 4, generator=generator).to(device),
            torch.randn(8, 2, generator=generator).to(device),
        )
        for _ in range(3)
    ]


def create_lm(path):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    path.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3}
    vocab.update(
        {
            word: i + 4
            for i, word in enumerate(
                "hello world red green blue one two three small model system training test alpha beta gamma".split()
            )
        }
    )
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        padding_side="left",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(path)
    torch.manual_seed(719)
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=3,
    )
    LlamaForCausalLM(config).save_pretrained(path)
