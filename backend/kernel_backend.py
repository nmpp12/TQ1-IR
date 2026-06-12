"""
tq1ir.backend.kernel_backend - Backend de validacao: executa TQ1-IR com numpy.

Verificado contra mom/engine ternary() - mesmos pesos, mesma escala.
"""

import mmap
import numpy as np
from typing import Optional

from ..ir import Module, WeightEntry
from ..ops import (Op, TernaryLinearOp, RMSNormOp, RoPEOp,
                   SiLUMulOp, AttentionOp, ResidualAddOp, ReturnOp)
from ..passes.fusion import FusedNormLinearOp
from ..passes.quant_lower import QuantizeOp, DequantizeOp


class WeightLoader:
    """Carrega pesos do GGUF via mmap - so le do disco quando necessario."""

    GGML_T_BITNET = 36
    GGML_I2_S     = 9
    GGML_F32      = 0
    GGML_F16      = 1
    GGML_I8       = 24

    def __init__(self, gguf_path: str):
        self.path = gguf_path
        self._f = open(gguf_path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._cache: dict = {}

    def load(self, entry: WeightEntry) -> np.ndarray:
        if entry.name in self._cache:
            return self._cache[entry.name]

        raw = self._mm[entry.byte_offset:entry.byte_offset + entry.byte_size]

        if entry.ggml_type in (self.GGML_T_BITNET, self.GGML_I2_S):
            result = self._decode_ternary(raw, entry.type.shape)
        elif entry.ggml_type == self.GGML_F16:
            arr = np.frombuffer(raw, dtype=np.float16)
            result = arr.reshape(entry.type.shape).astype(np.float32)
        elif entry.ggml_type == self.GGML_I8:
            arr = np.frombuffer(raw, dtype=np.int8)
            result = arr.reshape(entry.type.shape).astype(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.float32)
            result = arr.reshape(entry.type.shape)

        self._cache[entry.name] = result
        return result

    @staticmethod
    def _decode_ternary(raw, shape: tuple) -> np.ndarray:
        """Descodifica pesos ternarios T36/I2_S para fp32.

        Formato real: [0..n//4) packed (32 bytes/bloco, codigo c -> peso c-1),
        seguido de 4 bytes fp32 escala global + 28 bytes de padding.
        """
        n_weights = 1
        for d in shape: n_weights *= d
        n_blocks = (n_weights + 127) // 128
        packed_bytes = n_weights // 4

        packed = np.frombuffer(raw[:packed_bytes], dtype=np.uint8).reshape(n_blocks, 32)
        shifts = np.array([6, 4, 2, 0], dtype=np.uint8)
        vals = (packed[:, None, :] >> shifts[None, :, None]) & 3
        weights = vals.reshape(n_weights).astype(np.float32) - 1.0

        scale = np.frombuffer(raw[packed_bytes:packed_bytes + 4], dtype=np.float32)[0]

        return (weights * scale).reshape(shape)

    def close(self):
        self._mm.close()
        self._f.close()


class Executor:
    """Executa uma funcao TQ1-IR com numpy."""

    def __init__(self, module: Module, loader: WeightLoader):
        self.module = module
        self.loader = loader

    def _w(self, name: str) -> np.ndarray:
        entry = self.module.weights[name]
        return self.loader.load(entry)

    def run_layer(self, fn_name: str, x: np.ndarray,
                  positions: np.ndarray) -> np.ndarray:
        fn = next((f for f in self.module.functions if f.name == fn_name), None)
        if fn is None:
            raise KeyError(f"Funcao @{fn_name} nao encontrada no modulo")

        env: dict = {
            fn.inputs[0].name: x,
            fn.inputs[1].name: positions,
        }

        for op in fn.body.ops:
            result = self._exec_op(op, env)
            if op.result and result is not None:
                env[op.result.name] = result

        return env[fn.outputs[0].name] if fn.outputs else x

    def _exec_op(self, op: Op, env: dict) -> Optional[np.ndarray]:
        g = lambda v: env[v.name]

        if isinstance(op, TernaryLinearOp):
            x = g(op.input)
            W = self._w(op.weight_ref)
            return x @ W.T

        elif isinstance(op, FusedNormLinearOp):
            x  = g(op.input)
            Wn = self._w(op.norm_weight_ref)
            W  = self._w(op.linear_weight_ref)
            xn = self._rms_norm(x, Wn, op.eps)
            return xn @ W.T

        elif isinstance(op, RMSNormOp):
            x = g(op.input)
            W = self._w(op.weight_ref)
            return self._rms_norm(x, W, op.eps)

        elif isinstance(op, RoPEOp):
            x   = g(op.input)
            pos = g(op.positions)
            return self._rope(x, pos, op.theta)

        elif isinstance(op, SiLUMulOp):
            gate = g(op.gate)
            up   = g(op.up)
            return self._silu(gate) * up

        elif isinstance(op, AttentionOp):
            q = g(op.q); k = g(op.k); v = g(op.v)
            return self._attention(q, k, v, op.n_heads, op.n_kv_heads, op.head_dim)

        elif isinstance(op, ResidualAddOp):
            # Skip connection: result = lhs + rhs (FP32 elementwise).
            # Ordem `lhs + rhs` espelha mom/engine (`h = h + tmm(...)`):
            # lhs = residual stream, rhs = contribuicao do sublayer.
            lhs = g(op.lhs).astype(np.float32)
            rhs = g(op.rhs).astype(np.float32)
            return lhs + rhs

        elif isinstance(op, QuantizeOp):
            x = g(op.input).astype(np.float32)
            scale = np.max(np.abs(x), axis=-1, keepdims=True) / 127.0 + 1e-8
            return (x / scale).astype(np.int8)

        elif isinstance(op, DequantizeOp):
            return g(op.input).astype(np.float32)

        elif isinstance(op, ReturnOp):
            return None

        return None

    @staticmethod
    def _rms_norm(x, w, eps=1e-5):
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return (x / rms) * w

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x.clip(-80, 80)))

    @staticmethod
    def _rope(x, positions, theta):
        """RoPE NEOX-style (igual ao mom/engine)."""
        T, d = x.shape[-2], x.shape[-1]
        half = d // 2
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        angles = np.outer(positions.astype(np.float64), freqs).astype(np.float32)
        cos_a = np.cos(angles); sin_a = np.sin(angles)
        x1 = x[..., :half]; x2 = x[..., half:]
        return np.concatenate([x1 * cos_a - x2 * sin_a,
                                x1 * sin_a + x2 * cos_a], axis=-1)

    @staticmethod
    def _attention(q, k, v, n_heads, n_kv_heads, head_dim):
        """Atencao GQA com mascara causal."""
        T = q.shape[0]
        q = q.reshape(T, n_heads, head_dim)
        k = k.reshape(T, n_kv_heads, head_dim)
        v = v.reshape(T, n_kv_heads, head_dim)
        scale = head_dim ** -0.5
        repeat = n_heads // n_kv_heads
        k = np.repeat(k, repeat, axis=1)
        v = np.repeat(v, repeat, axis=1)
        out = np.zeros((T, n_heads, head_dim), dtype=np.float32)
        for h in range(n_heads):
            scores = q[:, h, :] @ k[:, h, :].T * scale
            mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
            scores = scores + mask
            scores -= scores.max(-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(-1, keepdims=True)
            out[:, h, :] = probs @ v[:, h, :]
        return out.reshape(T, n_heads * head_dim)


def compile_module(module: Module) -> "CompiledModule":
    loader = WeightLoader(module.source_gguf)
    return CompiledModule(module, loader)


class CompiledModule:
    def __init__(self, module: Module, loader: WeightLoader):
        self.module = module
        self.executor = Executor(module, loader)

    def run_layer(self, layer_idx: int, x: np.ndarray,
                  positions: np.ndarray) -> np.ndarray:
        return self.executor.run_layer(f"layer_{layer_idx}", x, positions)

    def close(self):
        self.executor.loader.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
