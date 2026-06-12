"""
tq1ir.passes.fusion — Pass de fusão de operações.

Detecta padrões no IR e substitui sequências de Ops por versões fundidas.
Ops fundidas fazem a mesma computação mas eliminam buffers intermédios,
reduzindo a pressão sobre a largura de banda de memória — o gargalo
central do "memory wall" que este projecto endereça.

Padrão implementado: RMSNorm → TernaryLinear
─────────────────────────────────────────────
  %n = tq1.rms_norm   %x, @W_norm
  %y = tq1.ternary_linear %n, @W, @scale

  →  %y = tq1.fused_norm_linear %x, @W_norm, @W, @scale

Benefício em hardware:
  A activação normalizada (%n, 2560 floats = 10KB) nunca precisa de ser
  escrita/lida da memória — sai da unidade de norma directamente para o
  array sistólico. Num sistema com 34 GB/s de largura de banda, eliminar
  este buffer poupa ~0.6 µs por layer por token (estimativa para 30 layers).

Fusão multi-consumidor (ex: attn_norm → Q, K, V):
  Se %n tem N consumidores TernaryLinear, cada um é fundido individualmente:
    %q = fused_norm_linear %x, @attn_norm, @q_proj, @q_proj
    %k = fused_norm_linear %x, @attn_norm, @k_proj, @k_proj
    %v = fused_norm_linear %x, @attn_norm, @v_proj, @v_proj
  A norma é recomputada N vezes — mais barato que bufferizar o resultado.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from ..ir import Module, Function, Block
from ..ops import (Op, Value, RMSNormOp, TernaryLinearOp, ReturnOp,
                   EmbedOp, RoPEOp, SiLUMulOp, AttentionOp)
from ..types import F32TensorType


# ─── Nova Op fundida ──────────────────────────────────────────────────────────

@dataclass
class FusedNormLinearOp(Op):
    """
    RMSNorm + TernaryLinear fundidos numa única operação.

    Semântica: result = ternary_linear(rms_norm(input, W_norm, eps), W, scale)

    Em hardware: a unidade de norma alimenta directamente o array sistólico,
    sem buffer intermédio.
    """
    input: Optional[Value] = None
    norm_weight_ref: str = ""
    linear_weight_ref: str = ""
    scale_ref: str = ""
    eps: float = 1e-5

    def operands(self): return [self.input] if self.input else []

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return (f"{r} = tq1.fused_norm_linear %{self.input.name}, "
                f"@{self.norm_weight_ref}, @{self.linear_weight_ref} "
                f"{{eps={self.eps}}}")


# ─── Utilitários de análise de uso ────────────────────────────────────────────

def _build_use_map(fn: Function) -> Dict[str, List[Op]]:
    """
    Para cada Value (por nome), lista as Ops que o usam como input.
    Necessário para decidir se é seguro fundir (ex: se %norm é só
    usado por TernaryLinear, podemos fundir sem copiar o valor).
    """
    uses: Dict[str, List[Op]] = {}
    for op in fn.body.ops:
        for v in op.operands():
            uses.setdefault(v.name, []).append(op)
    return uses


# ─── Pass principal ───────────────────────────────────────────────────────────

def fuse_norm_linear(module: Module) -> Tuple[Module, int]:
    """
    Aplica a fusão RMSNorm → TernaryLinear a todas as funções do módulo.

    Retorna:
        (módulo modificado in-place, número de fusões realizadas)

    Não modifica funções onde o padrão não aparece.
    """
    total_fused = 0

    for fn in module.functions:
        fused = _fuse_function(fn)
        total_fused += fused

    return module, total_fused


def _fuse_function(fn: Function) -> int:
    """Aplica fusões numa única função. Retorna o número de fusões."""
    use_map = _build_use_map(fn)
    new_ops: List[Op] = []
    fused_count = 0
    skip = set()   # índices de Ops que foram absorvidas numa fusão

    ops = fn.body.ops

    for i, op in enumerate(ops):
        if i in skip:
            continue

        # Verifica se esta Op é um RMSNorm cujos consumidores são todos TernaryLinear
        if isinstance(op, RMSNormOp) and op.result:
            norm_result_name = op.result.name
            consumers = use_map.get(norm_result_name, [])

            # Todos os consumidores devem ser TernaryLinear
            all_linear = consumers and all(
                isinstance(c, TernaryLinearOp) for c in consumers
            )

            if all_linear:
                # Encontra os índices dos TernaryLinear consumidores
                consumer_indices = []
                for j, other_op in enumerate(ops):
                    if other_op in consumers:
                        consumer_indices.append(j)

                # Substitui cada TernaryLinear por FusedNormLinear
                for j in consumer_indices:
                    lin: TernaryLinearOp = ops[j]
                    fused_op = FusedNormLinearOp(
                        result=lin.result,
                        input=op.input,
                        norm_weight_ref=op.weight_ref,
                        linear_weight_ref=lin.weight_ref,
                        scale_ref=lin.scale_ref,
                        eps=op.eps,
                    )
                    new_ops.append(fused_op)
                    skip.add(j)
                    fused_count += 1

                skip.add(i)   # o RMSNorm original é eliminado
                continue

        new_ops.append(op)

    fn.body.ops = new_ops
    return fused_count


def fusion_stats(module: Module) -> Dict[str, int]:
    """Conta operações fundidas vs originais (útil para o paper)."""
    counts: Dict[str, int] = {}
    for fn in module.functions:
        for op in fn.body.ops:
            name = type(op).__name__
            counts[name] = counts.get(name, 0) + 1
    counts["TOTAL"] = sum(v for k, v in counts.items() if k != "TOTAL")
    return counts
