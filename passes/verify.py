"""
tq1ir.passes.verify - Pass de verificacao de tipos do dialect TQ1.

Pass de analise pura: le o IR, nao o modifica, reporta erros.

Regras:
  1. WEIGHT REFS - todos os @tensores referenciados existem.
  2. TIPOS DE ENTRADA:
       TernaryLinearOp : input deve ser F32TensorType ou I8TensorType
       RMSNormOp       : input deve ser F32TensorType
       RoPEOp          : input deve ser F32TensorType
       SiLUMulOp       : gate e up devem ter o mesmo tipo
       AttentionOp     : q, k, v devem ser F32TensorType
       ResidualAddOp   : lhs e rhs devem ser F32TensorType e mesma shape
  3. TIPOS DE PESOS:
       TernaryLinearOp : weight deve ser TernaryTensorType
       RMSNormOp       : weight deve ser F32TensorType
  4. CONSISTENCIA DE DIMENSOES (quando conhecidas).
  5. TIPOS DE SAIDA.
"""

from dataclasses import dataclass
from typing import List, Tuple

from ..ir import Module, Function, WeightEntry
from ..types import (TernaryTensorType, F32TensorType, I8TensorType,
                     fmt_type, AnyType)
from ..ops import (Op, Value, EmbedOp, TernaryLinearOp, RMSNormOp,
                   RoPEOp, SiLUMulOp, AttentionOp, ResidualAddOp, ReturnOp)


@dataclass
class Diagnostic:
    """Um erro ou aviso do verificador de tipos."""
    level: str
    fn_name: str
    op_type: str
    result_name: str
    message: str

    def __str__(self):
        loc = f"@{self.fn_name} / {self.op_type}"
        if self.result_name:
            loc += f" -> %{self.result_name}"
        return f"[{self.level}] {loc}: {self.message}"


def _is_activation(t: AnyType) -> bool:
    return isinstance(t, (F32TensorType, I8TensorType))

def _dims_compatible(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    return all(da == db or da == -1 or db == -1 for da, db in zip(a, b))


def _check_ternary_linear(op: TernaryLinearOp, mod: Module,
                           fn_name: str) -> List[Diagnostic]:
    diags = []
    res = op.result.name if op.result else "?"

    if op.input is None:
        diags.append(Diagnostic("ERRO", fn_name, "TernaryLinearOp", res,
                                 "input e None"))
        return diags

    if not _is_activation(op.input.type):
        diags.append(Diagnostic("ERRO", fn_name, "TernaryLinearOp", res,
                                 f"input deve ser F32 ou I8, "
                                 f"recebeu {fmt_type(op.input.type)}"))

    if op.weight_ref not in mod.weights:
        diags.append(Diagnostic("ERRO", fn_name, "TernaryLinearOp", res,
                                 f"@{op.weight_ref} nao encontrado na tabela de pesos"))
        return diags

    w: WeightEntry = mod.weights[op.weight_ref]

    if not isinstance(w.type, TernaryTensorType):
        diags.append(Diagnostic("AVISO", fn_name, "TernaryLinearOp", res,
                                 f"@{op.weight_ref} tem tipo {fmt_type(w.type)}, "
                                 f"esperado TernaryTensorType"))

    if (len(w.type.shape) == 2 and
            len(op.input.type.shape) >= 1 and
            op.input.type.shape[-1] != -1 and
            w.type.shape[-1] != -1):
        d_in_weight = w.type.shape[-1]
        d_in_act    = op.input.type.shape[-1]
        if d_in_weight != d_in_act:
            diags.append(Diagnostic("ERRO", fn_name, "TernaryLinearOp", res,
                                     f"dimensao de entrada incompativel: "
                                     f"activacao d_in={d_in_act}, "
                                     f"peso d_in={d_in_weight} "
                                     f"(@{op.weight_ref})"))

    if op.result and not isinstance(op.result.type, F32TensorType):
        diags.append(Diagnostic("AVISO", fn_name, "TernaryLinearOp", res,
                                 f"resultado deveria ser F32, "
                                 f"tem {fmt_type(op.result.type)}"))

    return diags


def _check_rms_norm(op: RMSNormOp, mod: Module, fn_name: str) -> List[Diagnostic]:
    diags = []
    res = op.result.name if op.result else "?"

    if op.input is None:
        diags.append(Diagnostic("ERRO", fn_name, "RMSNormOp", res, "input e None"))
        return diags

    if not isinstance(op.input.type, F32TensorType):
        diags.append(Diagnostic("ERRO", fn_name, "RMSNormOp", res,
                                 f"input deve ser F32, "
                                 f"recebeu {fmt_type(op.input.type)}"))

    if op.weight_ref not in mod.weights:
        diags.append(Diagnostic("ERRO", fn_name, "RMSNormOp", res,
                                 f"@{op.weight_ref} nao encontrado"))
        return diags

    w = mod.weights[op.weight_ref]
    if not isinstance(w.type, F32TensorType):
        diags.append(Diagnostic("AVISO", fn_name, "RMSNormOp", res,
                                 f"peso @{op.weight_ref} deveria ser F32, "
                                 f"tem {fmt_type(w.type)}"))

    if (len(w.type.shape) == 1 and
            len(op.input.type.shape) >= 1 and
            w.type.shape[0] != -1 and
            op.input.type.shape[-1] != -1):
        if w.type.shape[0] != op.input.type.shape[-1]:
            diags.append(Diagnostic("ERRO", fn_name, "RMSNormOp", res,
                                     f"peso shape={w.type.shape[0]} != "
                                     f"activacao d_model={op.input.type.shape[-1]}"))

    return diags


def _check_rope(op: RoPEOp, fn_name: str) -> List[Diagnostic]:
    diags = []
    res = op.result.name if op.result else "?"

    if op.input is None:
        diags.append(Diagnostic("ERRO", fn_name, "RoPEOp", res, "input e None"))
    elif not isinstance(op.input.type, F32TensorType):
        diags.append(Diagnostic("ERRO", fn_name, "RoPEOp", res,
                                 f"input deve ser F32, recebeu {fmt_type(op.input.type)}"))

    if op.positions is None:
        diags.append(Diagnostic("ERRO", fn_name, "RoPEOp", res, "positions e None"))

    if op.theta <= 0:
        diags.append(Diagnostic("ERRO", fn_name, "RoPEOp", res,
                                 f"theta deve ser positivo, tem {op.theta}"))

    return diags


def _check_silu_mul(op: SiLUMulOp, fn_name: str) -> List[Diagnostic]:
    diags = []
    res = op.result.name if op.result else "?"

    if op.gate is None or op.up is None:
        diags.append(Diagnostic("ERRO", fn_name, "SiLUMulOp", res,
                                 "gate ou up e None"))
        return diags

    if op.gate.type != op.up.type:
        diags.append(Diagnostic("ERRO", fn_name, "SiLUMulOp", res,
                                 f"gate ({fmt_type(op.gate.type)}) e "
                                 f"up ({fmt_type(op.up.type)}) tem tipos diferentes"))

    return diags


def _check_attention(op: AttentionOp, fn_name: str) -> List[Diagnostic]:
    diags = []
    res = op.result.name if op.result else "?"

    for name, val in [("q", op.q), ("k", op.k), ("v", op.v)]:
        if val is None:
            diags.append(Diagnostic("ERRO", fn_name, "AttentionOp", res,
                                     f"{name} e None"))
        elif not isinstance(val.type, F32TensorType):
            diags.append(Diagnostic("AVISO", fn_name, "AttentionOp", res,
                                     f"{name} deveria ser F32, tem {fmt_type(val.type)}"))

    if op.n_heads <= 0 or op.n_kv_heads <= 0 or op.head_dim <= 0:
        diags.append(Diagnostic("ERRO", fn_name, "AttentionOp", res,
                                 f"parametros invalidos: n_heads={op.n_heads}, "
                                 f"n_kv={op.n_kv_heads}, hd={op.head_dim}"))

    if op.n_heads % op.n_kv_heads != 0:
        diags.append(Diagnostic("ERRO", fn_name, "AttentionOp", res,
                                 f"n_heads={op.n_heads} nao e multiplo de "
                                 f"n_kv_heads={op.n_kv_heads}"))

    return diags


def _check_residual_add(op: ResidualAddOp, fn_name: str) -> List[Diagnostic]:
    """Residuals sao FP32-only (manter bit-exact contra mom/engine) e
    sem broadcast - lhs e rhs tem de partilhar a shape."""
    diags = []
    res = op.result.name if op.result else "?"

    if op.lhs is None or op.rhs is None:
        diags.append(Diagnostic("ERRO", fn_name, "ResidualAddOp", res,
                                 "lhs ou rhs e None"))
        return diags

    if not isinstance(op.lhs.type, F32TensorType):
        diags.append(Diagnostic("ERRO", fn_name, "ResidualAddOp", res,
                                 f"lhs deve ser F32, recebeu {fmt_type(op.lhs.type)}"))
    if not isinstance(op.rhs.type, F32TensorType):
        diags.append(Diagnostic("ERRO", fn_name, "ResidualAddOp", res,
                                 f"rhs deve ser F32, recebeu {fmt_type(op.rhs.type)}"))

    if (isinstance(op.lhs.type, F32TensorType) and
            isinstance(op.rhs.type, F32TensorType)):
        if not _dims_compatible(op.lhs.type.shape, op.rhs.type.shape):
            diags.append(Diagnostic("ERRO", fn_name, "ResidualAddOp", res,
                                     f"shapes incompativeis: lhs={op.lhs.type.shape} "
                                     f"vs rhs={op.rhs.type.shape}"))

    if op.result and not isinstance(op.result.type, F32TensorType):
        diags.append(Diagnostic("AVISO", fn_name, "ResidualAddOp", res,
                                 f"resultado deveria ser F32, tem {fmt_type(op.result.type)}"))

    return diags


def _check_return(op: ReturnOp, fn, fn_name: str) -> List[Diagnostic]:
    diags = []

    if len(op.values) != len(fn.outputs):
        diags.append(Diagnostic("ERRO", fn_name, "ReturnOp", "",
                                 f"devolve {len(op.values)} valores, "
                                 f"funcao declara {len(fn.outputs)} outputs"))
        return diags

    for i, (ret_val, out_val) in enumerate(zip(op.values, fn.outputs)):
        if not _dims_compatible(ret_val.type.shape, out_val.type.shape):
            diags.append(Diagnostic("AVISO", fn_name, "ReturnOp", "",
                                     f"output {i}: shape {ret_val.type.shape} != "
                                     f"{out_val.type.shape}"))

    return diags


def verify(module: Module) -> Tuple[List[Diagnostic], List[Diagnostic]]:
    """Verifica o modulo inteiro. Retorna (erros, avisos)."""
    erros: List[Diagnostic] = []
    avisos: List[Diagnostic] = []

    def _add(diags):
        for d in diags:
            (erros if d.level == "ERRO" else avisos).append(d)

    for fn in module.functions:
        for op in fn.body.ops:
            if isinstance(op, TernaryLinearOp):
                _add(_check_ternary_linear(op, module, fn.name))
            elif isinstance(op, RMSNormOp):
                _add(_check_rms_norm(op, module, fn.name))
            elif isinstance(op, RoPEOp):
                _add(_check_rope(op, fn.name))
            elif isinstance(op, SiLUMulOp):
                _add(_check_silu_mul(op, fn.name))
            elif isinstance(op, AttentionOp):
                _add(_check_attention(op, fn.name))
            elif isinstance(op, ResidualAddOp):
                _add(_check_residual_add(op, fn.name))
            elif isinstance(op, ReturnOp):
                _add(_check_return(op, fn, fn.name))

    return erros, avisos


def verify_or_raise(module: Module) -> None:
    erros, avisos = verify(module)
    for a in avisos:
        print(f"  {a}")
    if erros:
        print()
        for e in erros:
            print(f"  {e}")
        raise TypeError(
            f"Verificacao do IR falhou: {len(erros)} erro(s), "
            f"{len(avisos)} aviso(s)"
        )
