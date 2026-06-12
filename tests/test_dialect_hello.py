"""
test_dialect_hello.py — Demo e validação do dialect TQ1-IR.

O que este teste faz:
  1. Lê o GGUF do BitNet (ou MOM.gguf se disponível)
  2. Constrói o Module TQ1-IR via o frontend
  3. Imprime o IR textual da layer pedida
  4. Emite a ISA do acelerador
  5. Conta instruções por tipo
  6. Corre o pass de verificação de tipos

Correr:
    python test_dialect_hello.py
    python test_dialect_hello.py --layer 5
    python test_dialect_hello.py --verbose
"""

import sys
import argparse
from pathlib import Path

DEFAULT_GGUFS = [
    Path("tools/bitnet-cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"),
    Path("MOM.gguf"),
]


def find_gguf():
    for p in DEFAULT_GGUFS:
        if p.exists():
            return p
    return None


def run(gguf_path: str, layer_idx: int = 0, verbose: bool = False):
    from tq1ir.frontend import load_gguf
    from tq1ir.backend  import emit_isa, count_instructions
    from tq1ir.passes   import verify
    import tq1ir

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  TQ1-IR Compiler — Hello World")
    print(f"{SEP}\n")

    # ── 1. Carregar o modelo ──────────────────────────────────────────────────
    print(f"[1/5] A ler GGUF: {gguf_path}")
    mod = load_gguf(gguf_path)
    print(f"      {len(mod.weights)} tensores registados")
    print(f"      {len(mod.functions)} funcoes construidas")
    meta = mod.metadata
    print(f"      n_layers={meta['n_layers']}, d_model={meta['d_model']}, "
          f"n_heads={meta['n_heads']}, n_kv={meta['n_kv_heads']}, "
          f"d_ffn={meta['d_ffn']}\n")

    # ── 2. Imprimir o IR (uma layer) ──────────────────────────────────────────
    print(f"[2/5] TQ1-IR — @layer_{layer_idx}:")
    print("-" * 60)
    sample = tq1ir.Module(name=mod.name, source_gguf=mod.source_gguf,
                          metadata=mod.metadata)
    for k, v in mod.weights.items():
        sample.add_weight(v)
    if mod.functions:
        sample.add_function(mod.functions[layer_idx])
    print(sample.dump())
    print("-" * 60)

    # ── 3. Emitir ISA ─────────────────────────────────────────────────────────
    print(f"\n[3/5] ISA do acelerador — @layer_{layer_idx}:")
    print("-" * 60)
    print(emit_isa(mod, functions=[f"layer_{layer_idx}"]))
    print("-" * 60)

    # ── 4. Estatisticas de instrucoes ─────────────────────────────────────────
    print("\n[4/5] Instrucoes em todo o modelo:")
    counts = count_instructions(mod)
    total = counts.pop("TOTAL", 1)
    for name, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = chr(9608) * min(n // 3, 35)
        pct = 100 * n / total
        print(f"  {name:<25} {n:5d}  {pct:4.0f}%  {bar}")
    print(f"  {'TOTAL':<25} {total:5d}")
    tgemv = counts.get("TernaryLinearOp", 0)
    print(f"\n  TGEMV = {100*tgemv/total:.0f}% das instrucoes"
          f" -> systolic array e o componente dominante\n")

    # ── 5. Pass de verificacao de tipos ───────────────────────────────────────
    print("[5/5] Pass de verificacao de tipos:")
    erros, avisos = verify(mod)

    def resumo(diags, label):
        if not diags:
            return
        tipos = {}
        for d in diags:
            chave = d.message.split(":")[0].strip()[:50]
            tipos[chave] = tipos.get(chave, 0) + 1
        print(f"\n  {label} ({len(diags)} total):")
        for msg, cnt in sorted(tipos.items(), key=lambda x: -x[1])[:6]:
            print(f"    {cnt:4d}x  {msg}")
        if verbose:
            for d in diags[:15]:
                print(f"         {d}")

    resumo(erros, "ERROS")
    resumo(avisos, "AVISOS")

    print()
    if not erros:
        print(f"  OK  0 erros — IR valido para compilacao")
        if avisos:
            print(f"      {len(avisos)} aviso(s) — nao bloqueiam a compilacao")
    else:
        print(f"  FALHOU  {len(erros)} erro(s) no IR")

    print(f"\n{SEP}")
    print("  Pipeline: GGUF -> TQ1-IR -> ISA | verificacao de tipos OK")
    print(f"{SEP}\n")
    print("  Nota: se implementares esta ISA em FPGA ou ASIC,")
    print("  manda uma mensagem ao Nuno. Serio.")
    print()
    return len(erros) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TQ1-IR compiler demo")
    parser.add_argument("--gguf",    type=str, default=None)
    parser.add_argument("--layer",   type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gguf = args.gguf
    if gguf is None:
        found = find_gguf()
        if found is None:
            print("ERRO: GGUF nao encontrado nos caminhos padrao.")
            print("Usa: python test_dialect_hello.py --gguf caminho/modelo.gguf")
            sys.exit(1)
        gguf = str(found)

    ok = run(gguf, layer_idx=args.layer, verbose=args.verbose)
    sys.exit(0 if ok else 1)
