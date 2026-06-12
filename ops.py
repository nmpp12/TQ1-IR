"""
tq1ir.ops - Operacoes do dialect TQ1.

Cada Op recebe Values de entrada (SSA) e produz um Value de saida.
O IR e Static Single Assignment: cada Value e definido exactamente uma vez.

Operacoes:
  EmbedOp          - lookup de embeddings (int8 no MOM.gguf)
  TernaryLinearOp  - matmul com pesos ternarios (o coracao do acelerador)
  RMSNormOp        - normalizacao RMS
  RoPEOp           - embeddings posicionais rotativos (NEOX, theta=5e5)
  SiLUMulOp        - activacao FFN: SiLU(gate) * up
  AttentionOp      - atencao agrupada (GQA 20:5)
  ResidualAddOp    - adicao elementwise FP32 (skip connection)
  ReturnOp         - termina uma funcao
"""

from dataclasses import dataclass, field
from typing import Optional, List
from .types import AnyType, fmt_type

_counter = 0

def _uid() -> str:
    global _counter
    _counter += 1
    return f"v{_counter}"


@dataclass
class Value:
    """Valor SSA - resultado de uma operacao no IR."""
    type: AnyType
    name: str = field(default="")

    def __post_init__(self):
        if not self.name:
            self.name = _uid()

    def __repr__(self):
        return f"%{self.name}: {fmt_type(self.type)}"


@dataclass
class Op:
    """Base de todas as operacoes TQ1."""
    result: Optional[Value] = field(default=None, repr=False)

    def operands(self) -> List[Value]:
        raise NotImplementedError

    def fmt(self) -> str:
        raise NotImplementedError


# Operacoes concretas

@dataclass
class EmbedOp(Op):
    """Lookup de embeddings: token_ids (T) -> activacoes (T x d_model).

    No MOM.gguf os embeddings estao em int8 + escala fp32 por linha.
    """
    token_ids: Optional[Value] = None
    weight_ref: str = ""
    scale_ref: str = ""

    def operands(self): return [self.token_ids] if self.token_ids else []

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        suffix = f", @{self.scale_ref}" if self.scale_ref else ""
        return f"{r} = tq1.embed %{self.token_ids.name}, @{self.weight_ref}{suffix}"


@dataclass
class TernaryLinearOp(Op):
    """Multiplicacao matricial com pesos ternarios.

        result = input @ weight.T * scale_per_group

    Operacao central do acelerador bit-serial: pesos {-1, 0, +1} transformam
    multiplicacoes em adicoes/subtracoes simples.
    """
    input: Optional[Value] = None
    weight_ref: str = ""
    scale_ref: str = ""

    def operands(self): return [self.input] if self.input else []

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.ternary_linear %{self.input.name}, @{self.weight_ref}, @{self.scale_ref}"


@dataclass
class RMSNormOp(Op):
    """RMS Layer Normalization.

        x_norm = x / sqrt(mean(x^2) + eps) * weight
    """
    input: Optional[Value] = None
    weight_ref: str = ""
    eps: float = 1e-5

    def operands(self): return [self.input] if self.input else []

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.rms_norm %{self.input.name}, @{self.weight_ref} {{eps={self.eps}}}"


@dataclass
class RoPEOp(Op):
    """Rotary Position Embedding - estilo NEOX, theta = 5e5.

    Aplicada separadamente a Q e K antes da atencao.
    """
    input: Optional[Value] = None
    positions: Optional[Value] = None
    theta: float = 500_000.0

    def operands(self):
        return [x for x in [self.input, self.positions] if x is not None]

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.rope %{self.input.name}, %{self.positions.name} {{theta={self.theta}}}"


@dataclass
class SiLUMulOp(Op):
    """Activacao FFN: SiLU(gate) * up (elementwise)."""
    gate: Optional[Value] = None
    up: Optional[Value] = None

    def operands(self): return [x for x in [self.gate, self.up] if x is not None]

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.silu_mul %{self.gate.name}, %{self.up.name}"


@dataclass
class AttentionOp(Op):
    """Grouped Query Attention (GQA). BitNet 2B4T: 20 Q, 5 KV, head_dim=128."""
    q: Optional[Value] = None
    k: Optional[Value] = None
    v: Optional[Value] = None
    n_heads: int = 20
    n_kv_heads: int = 5
    head_dim: int = 128

    def operands(self): return [x for x in [self.q, self.k, self.v] if x is not None]

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return (f"{r} = tq1.attention %{self.q.name}, %{self.k.name}, %{self.v.name} "
                f"{{n_heads={self.n_heads}, n_kv={self.n_kv_heads}, hd={self.head_dim}}}")


@dataclass
class ResidualAddOp(Op):
    """Adicao residual elementwise - o skip connection do transformer.

        result = lhs + rhs        # FP32, mesma shape, sem broadcast

    Semantica identica a do mom/engine: `h = h + tmm(...)`. O `lhs` e o
    residual stream a montante (`h`); o `rhs` e a contribuicao do sublayer
    (saida de attention ou FFN, ja projectada).

    Tipo FP32 forcado: int8 partia a equivalencia bit-a-bit contra o engine
    (a adicao acumula-se atraves de 30 layers, ruido de quant compoe-se).
    O hardware fa-la-a numa ALU escalar - nao passa pelo systolic array.

    Nao e absorvida pelo pass de fusao (que so consome RMSNorm->TernaryLinear).
    O quant_lower insere Dequantize antes dos operandos quantizados.
    """
    lhs: Optional[Value] = None
    rhs: Optional[Value] = None

    def operands(self):
        return [x for x in [self.lhs, self.rhs] if x is not None]

    def fmt(self):
        r = f"%{self.result.name}" if self.result else "_"
        return f"{r} = tq1.residual_add %{self.lhs.name}, %{self.rhs.name}"


@dataclass
class ReturnOp(Op):
    """Termina uma funcao e devolve valores."""
    values: List[Value] = field(default_factory=list)

    def operands(self): return self.values

    def fmt(self):
        vals = ", ".join(f"%{v.name}" for v in self.values)
        return f"tq1.return {vals}"
