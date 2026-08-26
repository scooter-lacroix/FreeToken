"""Regenerate freetoken/models/gguf/_iq_tables.py from the vendored
kernel/csrc/gguf/ggml-common.h grid tables (no manual transcription)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "python/freetoken/kernel/csrc/gguf/ggml-common.h"
OUT = ROOT / "python/freetoken/models/gguf/_iq_tables.py"

def extract(name: str) -> list[int]:
    m = re.search(rf"{name}\[\d+\]\s*=\s*\{{(.*?)\}};", HEADER.read_text(), re.S)
    return [int(v, 0) for v in re.findall(r"0x[0-9a-fA-F]+|\d+", m.group(1))]

iq2s, iq3xs = extract("iq2s_grid"), extract("iq3xs_grid")
assert len(iq2s) == 1024 and len(iq3xs) == 512

OUT.write_text(f'''# Grid tables for the IQ (importance-matrix) quants, generated verbatim from
# the vendored llama.cpp ggml-common.h (kernel/csrc/gguf). iq2s_grid entries are
# uint64 (8 byte-values each); iq3xs_grid entries are uint32 (4 byte-values).
# Regenerate with tools/mtp/gen_iq_tables.py.
import torch

IQ2S_GRID_U64 = {iq2s!r}

IQ3XS_GRID_U32 = {iq3xs!r}

def iq2s_grid_u8() -> torch.Tensor:
    """[1024, 8] uint8 view of iq2s_grid (little-endian byte order)."""
    return torch.tensor(IQ2S_GRID_U64, dtype=torch.int64).view(torch.uint8).reshape(1024, 8)

def iq3xs_grid_u8() -> torch.Tensor:
    """[512, 4] uint8 view of iq3xs_grid."""
    return torch.tensor(IQ3XS_GRID_U32, dtype=torch.int32).view(torch.uint8).reshape(512, 4)
''')
print(f"wrote {OUT} ({len(iq2s)} u64 + {len(iq3xs)} u32)")
