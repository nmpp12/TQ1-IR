"""
tq1ir.passes.quant_lower - Lowering F32 -> I8 das activacoes.

Estrategia "i8_after_linear":
  Apos cada TernaryLinear/FusedNormLinear, insere Quantize(F32->I8).
  Antes de cada RMSNorm/RoPE/ResidualAdd que receba I8, insere Dequantize.

ResidualAddOp e F32-only por design: a soma do residual stream tem de ficar
bit-exact contra mom/engine, int8 partia-o. Por isso lhs/rhs sao
forcosamente dequantizados.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from ..ir import Module, Function
from ..ops import (Op, Value, TernaryLinearOp, RMSNormOp, RoPEOp,
                   AttentionOp, ResidualAddOp, ReturnOp)
from ..types import F32TensorType, I8TensorType
from .fusion import FusedNormLinearOp


@dataclass
class QuantizeOp(Op):
    """F32 -> I8 (absmax por linha)."""
    input: Optional[Value] = None

    def operands(self): return [self.input] if self.input else []

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.quantize %{self.input.name}  // F32->I8"


@dataclass
class DequantizeOp(Op):
    """I8 -> F32."""
    input: Optional[Value] = None

    def operands(self): return [self.input] if self.input else []

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.dequantize %{self.input.name}  // I8->F32"


def lower_activations(module: Module,
                      strategy: str = "i8_after_linear") -> Tuple[Module, int]:
    if strategy != "i8_after_linear":
        raise ValueError(f"Estrategia desconhecida: {strategy}")
    total = 0
    for fn in module.functions:
        total += _lower_function(fn)
    return module, total


def _lower_function(fn: Function) -> int:
    new_ops: List[Op] = []
    inserted = 0
    quant_map: dict = {}

    for op in fn.body.ops:
        _patch_inputs(op, quant_map, new_ops)
        new_ops.append(op)

        if isinstance(op, (TernaryLinearOp, FusedNormLinearOp)) and op.result:
            orig = op.result
            if isinstance(orig.type, F32TensorType):
                i8_type = I8TensorType(shape=orig.type.shape)
                q_val = Value(type=i8_type, name=f"{orig.name}_q8")
                q_op = QuantizeOp(result=q_val, input=orig)
                new_ops.append(q_op)
                quant_map[orig.name] = q_val
                inserted += 1

    fn.body.ops = new_ops
    return inserted


def _patch_inputs(op: Op, quant_map: dict, new_ops: list):
    """Insere DequantizeOp quando um operando F32-required foi quantizado
    a montante. Inclui ResidualAddOp (lhs e rhs)."""
    needs_f32 = {
        RMSNormOp: ("input",),
        RoPEOp: ("input",),
        ResidualAddOp: ("lhs", "rhs"),
    }

    for op_cls, attrs in needs_f32.items():
        if isinstance(op, op_cls):
            for attr in attrs:
                val: Optional[Value] = getattr(op, attr, None)
                if val and val.name in quant_map:
                    q_val = quant_map[val.name]
                    f32_type = F32TensorType(shape=q_val.type.shape)
                    dq_val = Value(type=f32_type, name=f"{val.name}_dq")
                    dq_op = DequantizeOp(result=dq_val, input=q_val)
                    new_ops.append(dq_op)
                    setattr(op, attr, dq_val)
            break

    if isinstance(op, ReturnOp):
        new_vals = []
        for v in op.values:
            if v.name in quant_map:
                q_val = quant_map[v.name]
                f32_type = F32TensorType(shape=q_val.type.shape)
                dq_val = Value(type=f32_type, name=f"{v.name}_ret_dq")
                dq_op = DequantizeOp(result=dq_val, input=q_val)
                new_ops.append(dq_op)
                new_vals.append(dq_val)
            else:
                new_vals.append(v)
        op.values = new_vals
