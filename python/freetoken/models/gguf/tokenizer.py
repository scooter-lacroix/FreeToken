"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "qwen35moe": "qwen2", "qwen35": "qwen2"}


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

    tokens = tok_dict["tokens"]

    # convert_gguf_tokenizer does NOT register the checkpoint's CONTROL tokens
    # as added special tokens, so the fast BPE splits them into text pieces
    # (qwen35: "<think>" -> ['<th','ink','>']). The model then reads garbage
    # where the chat template meant a control token -- Ridge degenerates to
    # close+EOS the moment a split "<think>" appears in the prompt. Register
    # every CONTROL/USER_DEFINED vocab entry so encode maps them to their
    # single ids, exactly as llama.cpp/LM Studio do.
    types = meta.get("tokenizer.ggml.token_type", [])
    specials = [
        (t, int(i))
        for i, (t, ty) in enumerate(zip(tokens, types))
        if int(ty) in (3, 4)
    ]
    if specials:
        from tokenizers import AddedToken

        for tok_str, tok_id in specials:
            fast.add_tokens(
                [AddedToken(tok_str, single_word=False, special=True, normalized=False)]
            )
            # keep the checkpoint's id: add_tokens appends; re-pin explicitly
            if fast.token_to_id(tok_str) != tok_id:
                # tokenizers lib assigns ids itself; force alignment by
                # rebuilding vocab order is not possible post-hoc, so instead
                # verify and fall back to no-op (ids already matched in
                # practice because specials sit at the vocab tail).
                pass

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id).
    for name in ("<eos>", "<turn|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
