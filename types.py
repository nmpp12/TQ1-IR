"""
tq1ir.types — Sistema de tipos do dialect TQ1.

Um "tipo" descreve a forma e a representação de dados que fluem
entre operações. Três tipos fundamentais:

  TernaryTensorType : pesos {-1, 0, +1} compactados em formato i2_s
  F32TensorType     : activações em ponto flutuante 32-bit
  I8TensorType      : activações quantizadas em int8 (absmax por linha)

Todos são frozen (imutáveis) — tipos são valores, não objectos.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TernaryTensorType:
    """
    Tensor de pesos ternários em formato i2_s.

    Armazenamento físico: 4 pesos por byte, em pares de bits alto→baixo.
    Cada grupo de `group_size` pesos tem uma escala fp32 associada.

    Mapeamento dos códigos de 2 bits:
        00 → 0
        01 → +1
        10 → -1
    """
    shape: Tuple[int, ...]   # dimensões lógicas (linhas, colunas)
    group_size: int = 128    # pesos por grupo de escala (padrão BitNet)

    def n_weights(self) -> int:
        n = 1
        for d in self.shape: n *= d
        return n

    def nbytes_packed(self) -> int:
        """Bytes dos pesos compactados (4 pesos/byte)."""
        return (self.n_weights() + 3) // 4

    def nscales(self) -> int:
        """Número de escalas fp32 necessárias."""
        return (self.n_weights() + self.group_size - 1) // self.group_size


@dataclass(frozen=True)
class F32TensorType:
    """Tensor de activações em ponto flutuante de 32 bits."""
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class I8TensorType:
    """Tensor de activações quantizadas em int8, escala por linha."""
    shape: Tuple[int, ...]


AnyType = TernaryTensorType | F32TensorType | I8TensorType


def fmt_type(t: AnyType) -> str:
    """Representação textual de um tipo para o IR printer."""
    dims = "x".join(str(d) if d >= 0 else "?" for d in t.shape)
    if isinstance(t, TernaryTensorType):
        return f"!tq1.ternary<{dims}, g{t.group_size}>"
    elif isinstance(t, I8TensorType):
        return f"!tq1.i8<{dims}>"
    return f"!tq1.f32<{dims}>"
