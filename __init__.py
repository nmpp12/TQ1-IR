# tq1ir - Dialect MLIR-inspired para inferencia de LLMs com pesos ternarios
#
# Estrutura:
#   tq1ir.types    - sistema de tipos
#   tq1ir.ops      - operacoes SSA
#   tq1ir.ir       - modulo/funcao/bloco + pretty-printer
#   tq1ir.frontend - GGUF -> TQ1-IR
#   tq1ir.passes   - verify, fusion, quant_lower
#   tq1ir.backend  - ISA emitter, kernel backend

from .types import TernaryTensorType, F32TensorType, I8TensorType, fmt_type
from .ops   import (Value, TernaryLinearOp, RMSNormOp, RoPEOp,
                    SiLUMulOp, AttentionOp, EmbedOp,
                    ResidualAddOp, ReturnOp)
from .ir    import Module, Function, Block, WeightEntry

__version__ = "0.2.1"
