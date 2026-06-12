"""
tq1ir.backend.isa_backend - Emite a ISA do acelerador a partir do TQ1-IR.

Instrucoes:
  EMBED, TGEMV, TGEMV_N (norm fundido), RMSNORM, ROPE, ATTN, SILU_MUL,
  RESID (FP32 skip add, ALU escalar), QUANT_I8, DEQUANT, RET.
"""

from ..ir import Module, Function
from ..ops import (Op, EmbedOp, TernaryLinearOp, RMSNormOp,
                   RoPEOp, SiLUMulOp, AttentionOp, ResidualAddOp, ReturnOp)
try:
    from ..passes.fusion import FusedNormLinearOp
except ImportError:
    FusedNormLinearOp = None
try:
    from ..passes.quant_lower import QuantizeOp, DequantizeOp
except ImportError:
    QuantizeOp = DequantizeOp = None


def emit_isa(module: Module, functions=None) -> str:
    lines = [
        f"; TQ1 ISA - {module.name}",
        f"; fonte: {module.source_gguf}",
        f"; {len(module.weights)} tensores | {len(module.functions)} funcoes",
        "",
        "; === MAPA DE MEMORIA DE PESOS (lazy - offsets no GGUF) ===",
    ]

    ternary_count = sum(1 for w in module.weights.values()
                        if w.ggml_type in (9, 36))
    non_ternary = len(module.weights) - ternary_count
    total_tern_mb = sum(
        w.byte_size for w in module.weights.values() if w.ggml_type in (9, 36)
    ) / (1024*1024)
    lines.append(f"; {ternary_count} tensores ternarios ({total_tern_mb:.0f} MB)")
    lines.append(f"; {non_ternary} tensores densos (F32/F16/I8)")
    lines.append("")

    fns_to_emit = [
        fn for fn in module.functions
        if functions is None or fn.name in functions
    ]

    for fn in fns_to_emit:
        lines.append(f"; === FUNCAO @{fn.name} ===")
        ins_str  = ", ".join(f"%{v.name}" for v in fn.inputs)
        outs_str = ", ".join(f"%{v.name}" for v in fn.outputs)
        lines.append(f"; entrada: ({ins_str})  saida: ({outs_str})")
        lines.append("")

        for op in fn.body.ops:
            isa_line = _emit_op(op)
            if isa_line:
                lines.append(f"    {isa_line}")

        lines.append("")

    return "\n".join(lines)


def _emit_op(op: Op):
    if isinstance(op, EmbedOp):
        dst = f"%{op.result.name}" if op.result else "_"
        scale = f", @{op.scale_ref}" if op.scale_ref else ""
        return f"EMBED     {dst}, %{op.token_ids.name}, @{op.weight_ref}{scale}"

    elif isinstance(op, TernaryLinearOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return f"TGEMV     {dst}, %{op.input.name}, @{op.weight_ref}"

    elif isinstance(op, RMSNormOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return f"RMSNORM   {dst}, %{op.input.name}, @{op.weight_ref}"

    elif isinstance(op, RoPEOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return (f"ROPE      {dst}, %{op.input.name}, "
                f"%{op.positions.name}, {{theta={op.theta}}}")

    elif isinstance(op, SiLUMulOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return f"SILU_MUL  {dst}, %{op.gate.name}, %{op.up.name}"

    elif isinstance(op, AttentionOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return (f"ATTN      {dst}, %{op.q.name}, %{op.k.name}, %{op.v.name}, "
                f"{{nh={op.n_heads},nkv={op.n_kv_heads},hd={op.head_dim}}}")

    elif isinstance(op, ResidualAddOp):
        # FP32 elementwise feita pela ALU escalar (NAO pelo systolic array).
        dst = f"%{op.result.name}" if op.result else "_"
        return f"RESID     {dst}, %{op.lhs.name}, %{op.rhs.name}             ; FP32 skip add"

    elif isinstance(op, ReturnOp):
        vals = ", ".join(f"%{v.name}" for v in op.values)
        return f"RET       {vals}"

    elif FusedNormLinearOp and isinstance(op, FusedNormLinearOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return (f"TGEMV_N   {dst}, %{op.input.name}, "
                f"@{op.norm_weight_ref}, @{op.linear_weight_ref}  ; fused RMSNorm+TGEMV")

    elif QuantizeOp and isinstance(op, QuantizeOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return f"QUANT_I8  {dst}, %{op.input.name}             ; F32->I8"

    elif DequantizeOp and isinstance(op, DequantizeOp):
        dst = f"%{op.result.name}" if op.result else "_"
        return f"DEQUANT   {dst}, %{op.input.name}             ; I8->F32"

    return f"; [desconhecido: {type(op).__name__}]"


def count_instructions(module: Module) -> dict:
    counts = {}
    for fn in module.functions:
        for op in fn.body.ops:
            name = type(op).__name__
            counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    counts["TOTAL"] = total
    return counts
