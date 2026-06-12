"""
tq1ir.ir — Estrutura do módulo TQ1-IR.

Hierarquia:
    Module
      ├── WeightTable  (referências lazy ao GGUF — não os dados)
      └── Function[]
            └── Block
                  └── Op[]

O IR é "lazy" em relação aos pesos: guarda apenas o offset e tamanho
no ficheiro GGUF. O backend decide quando carregar os dados.
Princípio: os pesos vivem em memória, a computação vai até eles.
(Mesmo design que o BitROM e o nosso OwnBrain.)
"""

from dataclasses import dataclass, field
from typing import List, Dict
from .types import AnyType, fmt_type
from .ops import Op, Value


@dataclass
class WeightEntry:
    """
    Entrada na tabela de pesos — referência lazy a um tensor no GGUF.
    Não guarda dados, apenas a localização.
    """
    name: str
    type: AnyType
    byte_offset: int
    byte_size: int
    ggml_type: int = 0   # código GGML: 0=F32, 1=F16, 9=I2_S, 24=I8, ...

    GGML_TYPE_NAMES = {0: "F32", 1: "F16", 8: "Q8_0", 9: "I2_S", 24: "I8"}

    def fmt(self) -> str:
        tname = self.GGML_TYPE_NAMES.get(self.ggml_type, f"T{self.ggml_type}")
        return (f"@{self.name} : {fmt_type(self.type)} "
                f"// {tname}, offset={self.byte_offset:#010x}, {self.byte_size:,} bytes")


@dataclass
class Block:
    """
    Bloco básico — sequência linear de Ops.
    O dialect TQ1 é dataflow puro: sem saltos no IR.
    """
    ops: List[Op] = field(default_factory=list)

    def append(self, op: Op) -> Op:
        self.ops.append(op)
        return op

    def __len__(self): return len(self.ops)
    def __iter__(self): return iter(self.ops)


@dataclass
class Function:
    """
    Função no módulo TQ1-IR.

    Exemplos típicos:
        @layer(i)   — uma transformer layer desdobrada
        @forward    — forward pass completo
    """
    name: str
    inputs: List[Value] = field(default_factory=list)
    outputs: List[Value] = field(default_factory=list)
    body: Block = field(default_factory=Block)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Module:
    """
    Módulo TQ1-IR — nível de topo do compilador.

    Contém:
      - tabela de pesos (referências lazy ao GGUF)
      - funções (o grafo de computação)
      - metadados do modelo (arquitectura, config)
    """
    name: str
    source_gguf: str
    weights: Dict[str, WeightEntry] = field(default_factory=dict)
    functions: List[Function] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def add_weight(self, entry: WeightEntry) -> WeightEntry:
        self.weights[entry.name] = entry
        return entry

    def add_function(self, fn: Function) -> Function:
        self.functions.append(fn)
        return fn

    def dump(self, max_weights: int = 12) -> str:
        """
        Imprime o módulo em formato textual legível (o "assembly" do compilador).

        Exemplo:
            tq1.module @bitnet_2b4t {
              // fonte: .../ggml-model-i2_s.gguf
              // 331 tensores | 2 funções

              @token_embd : !tq1.i8<128256x2048, g128> // I8, offset=0x00007a80
              ...

              tq1.func @layer_0(%x: !tq1.f32<?x2048>) -> (!tq1.f32<?x2048>) {
                %v1 = tq1.rms_norm %x, @blk.0.attn_norm {eps=1e-05}
                ...
              }
            }
        """
        lines = [
            f"tq1.module @{self.name} {{",
            f"  // fonte: {self.source_gguf}",
            f"  // {len(self.weights)} tensores | {len(self.functions)} função(ões)",
            "",
            "  // --- tabela de pesos ---",
        ]

        shown = list(self.weights.values())[:max_weights]
        for entry in shown:
            lines.append(f"  {entry.fmt()}")
        if len(self.weights) > max_weights:
            lines.append(f"  // ... e mais {len(self.weights) - max_weights} tensores")
        lines.append("")

        for fn in self.functions:
            ins = ", ".join(f"%{v.name}: {fmt_type(v.type)}" for v in fn.inputs)
            outs = ", ".join(fmt_type(v.type) for v in fn.outputs)
            lines.append(f"  tq1.func @{fn.name}({ins}) -> ({outs}) {{")
            for op in fn.body.ops:
                lines.append(f"    {op.fmt()}")
            lines.append("  }")
            lines.append("")

        lines.append("}")
        return "\n".join(lines)
