"""
tq1ir.frontend.gguf_frontend - Frontend GGUF -> TQ1-IR.

Le um ficheiro GGUF (BitNet b1.58 ou MOM.gguf) e produz um Module TQ1-IR
com a tabela de pesos preenchida e as funcoes do transformer construidas.

O frontend nao carrega os dados dos pesos em memoria - apenas regista
os offsets e tamanhos. E o backend que decide quando streamar os pesos.

Arquitectura assumida (inferida do GGUF ou das constantes padrao BitNet):
    n_layers = 30, d_model = 2048, n_heads = 20, n_kv_heads = 5,
    head_dim = 128, d_ffn = 5632, vocab = 128256
"""

import struct
import mmap
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from ..types import TernaryTensorType, F32TensorType, I8TensorType
from ..ir import Module, WeightEntry, Function, Block
from ..ops import (Value, EmbedOp, TernaryLinearOp, RMSNormOp,
                   RoPEOp, SiLUMulOp, AttentionOp, ResidualAddOp, ReturnOp)

# Constantes GGUF
GGUF_MAGIC    = b"GGUF"
GGUF_VERSION  = 3
GGML_F32      = 0
GGML_F16      = 1
GGML_Q8_0     = 8
GGML_I2_S     = 9    # formato customizado BitNet (i2 com escala)
GGML_T_BITNET = 36   # tipo ternario nativo BitNet b1.58 2B4T
GGML_I8       = 24   # int8 (usado no MOM.gguf para embeddings)

_GGML_BYTES = {GGML_F32: 4, GGML_F16: 2, GGML_Q8_0: 1, GGML_I8: 1}


class _GGUFReader:
    """Le o cabecalho e a tabela de tensores de um GGUF (mmap, lazy)."""

    GGUF_METADATA_VALUE_TYPE = {
        0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
        4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
        8: None, 9: None,
        10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
    }

    def __init__(self, path: str):
        self.path = path
        self.kv: Dict[str, Any] = {}
        self.tensors: Dict[str, Dict] = {}
        self._data_offset = 0
        self._parse()

    def _parse(self):
        with open(self.path, "rb") as f:
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        pos = 0
        magic = data[pos:pos+4]; pos += 4
        if magic != GGUF_MAGIC:
            raise ValueError(f"Nao e um ficheiro GGUF (magic={magic})")

        version = struct.unpack_from("<I", data, pos)[0]; pos += 4
        n_tensors = struct.unpack_from("<Q", data, pos)[0]; pos += 8
        n_kv      = struct.unpack_from("<Q", data, pos)[0]; pos += 8

        for _ in range(n_kv):
            key, pos = self._read_str(data, pos)
            vtype = struct.unpack_from("<I", data, pos)[0]; pos += 4
            value, pos = self._read_value(data, pos, vtype)
            self.kv[key] = value

        tensor_infos = []
        for _ in range(n_tensors):
            name, pos = self._read_str(data, pos)
            n_dims = struct.unpack_from("<I", data, pos)[0]; pos += 4
            dims = struct.unpack_from(f"<{n_dims}Q", data, pos); pos += n_dims * 8
            ggml_type = struct.unpack_from("<I", data, pos)[0]; pos += 4
            offset = struct.unpack_from("<Q", data, pos)[0]; pos += 8
            tensor_infos.append((name, dims, ggml_type, offset))

        alignment = self.kv.get("general.alignment", 32)
        remainder = pos % alignment
        if remainder:
            pos += alignment - remainder
        self._data_offset = pos

        for name, dims, ggml_type, rel_offset in tensor_infos:
            n_elems = 1
            for d in dims: n_elems *= d
            byte_size = self._calc_size(ggml_type, n_elems)
            self.tensors[name] = {
                "dims": dims,
                "ggml_type": ggml_type,
                "offset": self._data_offset + rel_offset,
                "byte_size": byte_size,
                "n_elems": n_elems,
            }

        data.close()

    def _calc_size(self, ggml_type: int, n_elems: int) -> int:
        if ggml_type in (GGML_I2_S, GGML_T_BITNET):
            # n_elems//4 bytes packed + 32 bytes de cauda
            # (primeiro fp32 = escala global, resto = padding).
            return n_elems // 4 + 32
        bpe = _GGML_BYTES.get(ggml_type, 2)
        return n_elems * bpe

    def _read_str(self, data, pos) -> Tuple[str, int]:
        length = struct.unpack_from("<Q", data, pos)[0]; pos += 8
        s = data[pos:pos+length].decode("utf-8", errors="replace"); pos += length
        return s, pos

    def _read_value(self, data, pos, vtype):
        if vtype == 8:
            return self._read_str(data, pos)
        if vtype == 9:
            elem_type = struct.unpack_from("<I", data, pos)[0]; pos += 4
            count     = struct.unpack_from("<Q", data, pos)[0]; pos += 8
            values = []
            for _ in range(count):
                v, pos = self._read_value(data, pos, elem_type)
                values.append(v)
            return values, pos
        fmt_entry = self.GGUF_METADATA_VALUE_TYPE.get(vtype)
        if fmt_entry is None:
            return None, pos
        fmt, size = fmt_entry
        value = struct.unpack_from(f"<{fmt}", data, pos)[0]; pos += size
        return value, pos


def _tq1_type(dims: tuple, ggml_type: int) -> Any:
    shape = tuple(reversed(dims))
    if ggml_type in (GGML_I2_S, GGML_T_BITNET):
        return TernaryTensorType(shape=shape, group_size=128)
    elif ggml_type == GGML_I8:
        return I8TensorType(shape=shape)
    else:
        return F32TensorType(shape=shape)


def load_gguf(gguf_path: str) -> Module:
    """Le um GGUF BitNet e produz um Module TQ1-IR (com residuals)."""
    path = Path(gguf_path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF nao encontrado: {gguf_path}")

    reader = _GGUFReader(str(path))

    kv = reader.kv
    _pfx = next((p for p in ("llama.", "bitnet-b1.58.", "")
                 if f"{p}block_count" in kv), "llama.")
    def _kv(key, default):
        return kv.get(f"{_pfx}{key}", kv.get(f"llama.{key}", default))
    n_layers   = int(_kv("block_count", 30))
    d_model    = int(_kv("embedding_length", 2048))
    n_heads    = int(_kv("attention.head_count", 20))
    n_kv_heads = int(_kv("attention.head_count_kv", 5))
    d_ffn      = int(_kv("feed_forward_length", 5632))
    vocab_size = int(kv.get("llama.vocab_size",
                    kv.get("tokenizer.ggml.tokens", 128256)
                    if not isinstance(kv.get("tokenizer.ggml.tokens"), list)
                    else len(kv["tokenizer.ggml.tokens"])))
    head_dim   = d_model // n_heads
    model_name = kv.get("general.name", path.stem)

    mod = Module(
        name=model_name.replace(" ", "_").replace("-", "_").lower(),
        source_gguf=str(path),
        metadata={
            "n_layers": n_layers, "d_model": d_model,
            "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "d_ffn": d_ffn, "vocab_size": vocab_size, "head_dim": head_dim,
        }
    )

    for tname, info in reader.tensors.items():
        entry = WeightEntry(
            name=tname,
            type=_tq1_type(info["dims"], info["ggml_type"]),
            byte_offset=info["offset"],
            byte_size=info["byte_size"],
            ggml_type=info["ggml_type"],
        )
        mod.add_weight(entry)

    for i in range(n_layers):
        fn = _build_layer_function(mod, i, d_model, n_heads, n_kv_heads, head_dim, d_ffn)
        mod.add_function(fn)

    return mod


def _build_layer_function(mod: Module, layer_idx: int,
                           d_model: int, n_heads: int, n_kv_heads: int,
                           head_dim: int, d_ffn: int) -> Function:
    """Constroi o IR de uma transformer layer BitNet (com residuals).

    Topologia (corrige o bug do paper Sec 3.2 / Sec 7):
        x -> rms_norm(attn_norm) -> Q,K,V -> RoPE -> attention -> o_proj
        h_attn = ResidualAdd(x, o_proj)            <- skip 1
        rms_norm(ffn_norm) sobre h_attn -> gate,up -> silu_mul -> ffn_down
        h_out = ResidualAdd(h_attn, ffn_out)       <- skip 2
        return h_out
    """
    p = f"blk.{layer_idx}"
    act_t = F32TensorType(shape=(-1, d_model))

    x = Value(type=act_t, name=f"x{layer_idx}")
    pos = Value(type=F32TensorType(shape=(-1,)), name=f"pos{layer_idx}")

    fn = Function(name=f"layer_{layer_idx}", inputs=[x, pos], outputs=[])
    b = fn.body

    def w(suffix):
        candidates = [f"{p}.{suffix}.weight", f"{p}.{suffix}"]
        for c in candidates:
            if c in mod.weights:
                return c
        return candidates[0]

    def scale(suffix):
        return w(suffix)

    # Atencao
    x_norm = b.append(RMSNormOp(
        result=Value(type=act_t, name=f"attn_norm_{layer_idx}"),
        input=x, weight_ref=w("attn_norm"),
    )).result

    q_t = F32TensorType(shape=(-1, n_heads * head_dim))
    k_t = F32TensorType(shape=(-1, n_kv_heads * head_dim))
    v_t = F32TensorType(shape=(-1, n_kv_heads * head_dim))

    q = b.append(TernaryLinearOp(
        result=Value(type=q_t, name=f"q_{layer_idx}"),
        input=x_norm, weight_ref=w("attn_q"), scale_ref=scale("attn_q"),
    )).result
    k = b.append(TernaryLinearOp(
        result=Value(type=k_t, name=f"k_{layer_idx}"),
        input=x_norm, weight_ref=w("attn_k"), scale_ref=scale("attn_k"),
    )).result
    v = b.append(TernaryLinearOp(
        result=Value(type=v_t, name=f"v_{layer_idx}"),
        input=x_norm, weight_ref=w("attn_v"), scale_ref=scale("attn_v"),
    )).result

    q_rope = b.append(RoPEOp(
        result=Value(type=q_t, name=f"q_rope_{layer_idx}"),
        input=q, positions=pos,
    )).result
    k_rope = b.append(RoPEOp(
        result=Value(type=k_t, name=f"k_rope_{layer_idx}"),
        input=k, positions=pos,
    )).result

    attn_out = b.append(AttentionOp(
        result=Value(type=act_t, name=f"attn_out_{layer_idx}"),
        q=q_rope, k=k_rope, v=v,
        n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
    )).result

    o_proj = b.append(TernaryLinearOp(
        result=Value(type=act_t, name=f"o_proj_{layer_idx}"),
        input=attn_out, weight_ref=w("attn_output"), scale_ref=scale("attn_output"),
    )).result

    # Skip connection da atencao: h_attn = x + o_proj
    # (espelha mom/engine "h = h + tmm(attn_output, a)")
    h_attn = b.append(ResidualAddOp(
        result=Value(type=act_t, name=f"h_attn_{layer_idx}"),
        lhs=x, rhs=o_proj,
    )).result

    # FFN
    ffn_act_t = F32TensorType(shape=(-1, d_model))
    ffn_inner_t = F32TensorType(shape=(-1, d_ffn))

    ffn_norm = b.append(RMSNormOp(
        result=Value(type=ffn_act_t, name=f"ffn_norm_{layer_idx}"),
        input=h_attn, weight_ref=w("ffn_norm"),
    )).result

    gate = b.append(TernaryLinearOp(
        result=Value(type=ffn_inner_t, name=f"gate_{layer_idx}"),
        input=ffn_norm, weight_ref=w("ffn_gate"), scale_ref=scale("ffn_gate"),
    )).result
    up = b.append(TernaryLinearOp(
        result=Value(type=ffn_inner_t, name=f"up_{layer_idx}"),
        input=ffn_norm, weight_ref=w("ffn_up"), scale_ref=scale("ffn_up"),
    )).result

    act = b.append(SiLUMulOp(
        result=Value(type=ffn_inner_t, name=f"act_{layer_idx}"),
        gate=gate, up=up,
    )).result

    ffn_out = b.append(TernaryLinearOp(
        result=Value(type=act_t, name=f"ffn_out_{layer_idx}"),
        input=act, weight_ref=w("ffn_down"), scale_ref=scale("ffn_down"),
    )).result

    # Skip connection do FFN: h_out = h_attn + ffn_out
    out = b.append(ResidualAddOp(
        result=Value(type=act_t, name=f"out_{layer_idx}"),
        lhs=h_attn, rhs=ffn_out,
    )).result

    b.append(ReturnOp(values=[out]))
    fn.outputs = [out]
    return fn
