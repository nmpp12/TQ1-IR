"""
test_residual_v1.py - Valida a v1 dos residuals no TQ1-IR.

Tres gates:
  G1  ResidualAddOp isolado: lhs + rhs == h + tmm do mom/engine (bit-exact).
  G2  Topologia do layer_0: o IR insere residuais nos sitios certos
      (h_attn = x + o_proj e out = h_attn + ffn_out).
  G3  Layer_0 end-to-end contra referencia Python que reusa AS MESMAS ops
      do kernel backend (SiLUMul, sem sub_norms). Bit-exact prova que os
      residuals estao a juntar-se na ordem certa e que nenhum buffer
      intermedio foi esquecido. O mismatch contra mom/engine FULL fica
      a depender dos sub_norms BitNet + activacao relu^2, pre-existentes
      ao scope desta task.

Correr: python test_residual_v1.py
"""
import sys, copy
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

GGUF = "tools/bitnet-cpp/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"


def banner(t): print(f"\n=== {t} ===")
def ok(m):     print(f"  OK  {m}")
def fail(m):   print(f"  !!  {m}"); sys.exit(1)


def main():
    if not Path(GGUF).exists():
        fail(f"GGUF nao encontrado: {GGUF}")

    from tq1ir.frontend import load_gguf
    from tq1ir.ops import ResidualAddOp, TernaryLinearOp, RMSNormOp, RoPEOp, \
                          AttentionOp, SiLUMulOp, ReturnOp
    from tq1ir.backend.kernel_backend import WeightLoader, Executor

    banner("Build IR")
    mod = load_gguf(GGUF)
    fn0 = mod.functions[0]
    ok(f"{len(mod.weights)} tensores | layer_0 com {len(fn0.body.ops)} ops")

    # ---------- G2 ----------
    banner("G2  Topologia do layer_0")
    resid_ops = [op for op in fn0.body.ops if isinstance(op, ResidualAddOp)]
    if len(resid_ops) != 2:
        fail(f"esperava 2 ResidualAddOp, encontrou {len(resid_ops)}")

    # 1o residual: lhs deve ser x (fn input), rhs deve ser resultado de
    # TernaryLinear (o o_proj). 2o residual: lhs == result do 1o, rhs == ffn_down.
    in_x = fn0.inputs[0]
    r1, r2 = resid_ops
    if r1.lhs.name != in_x.name:
        fail(f"r1.lhs deveria ser %{in_x.name}, e %{r1.lhs.name}")

    # acha o produtor do rhs de r1
    name_of = {op.result.name: op for op in fn0.body.ops if op.result}
    prod_r1_rhs = name_of.get(r1.rhs.name)
    if not isinstance(prod_r1_rhs, TernaryLinearOp):
        fail(f"r1.rhs deveria vir de TernaryLinear, veio de {type(prod_r1_rhs).__name__}")
    if "attn_output" not in prod_r1_rhs.weight_ref:
        fail(f"r1.rhs nao vem de attn_output (vem de {prod_r1_rhs.weight_ref})")
    ok("r1: lhs=x, rhs=o_proj (attn_output)")

    if r2.lhs.name != r1.result.name:
        fail(f"r2.lhs deveria ser %{r1.result.name}, e %{r2.lhs.name}")
    prod_r2_rhs = name_of.get(r2.rhs.name)
    if not isinstance(prod_r2_rhs, TernaryLinearOp):
        fail(f"r2.rhs nao vem de TernaryLinear")
    if "ffn_down" not in prod_r2_rhs.weight_ref:
        fail(f"r2.rhs nao vem de ffn_down (vem de {prod_r2_rhs.weight_ref})")
    ok("r2: lhs=h_attn, rhs=ffn_out (ffn_down)")

    # ffn_norm deve consumir h_attn, NAO o_proj
    ffn_norm = next((op for op in fn0.body.ops
                     if isinstance(op, RMSNormOp) and "ffn_norm" in op.weight_ref), None)
    if ffn_norm is None:
        fail("nao encontrei RMSNorm com weight_ref ffn_norm")
    if ffn_norm.input.name != r1.result.name:
        fail(f"ffn_norm.input = %{ffn_norm.input.name}, esperado %{r1.result.name} "
             "(h_attn) - bug topologico do paper Sec 3.2 NAO foi corrigido")
    ok("ffn_norm consome h_attn (skip incluido)")

    # ReturnOp deve devolver o resultado do 2o residual
    ret = next(op for op in fn0.body.ops if isinstance(op, ReturnOp))
    if ret.values[0].name != r2.result.name:
        fail(f"return devolve %{ret.values[0].name}, esperava %{r2.result.name}")
    ok("return = out do segundo residual")

    # ---------- G1 ----------
    banner("G1  ResidualAddOp isolado vs mom/engine")
    rng = np.random.default_rng(0)
    h    = rng.standard_normal((4, 2560)).astype(np.float32) * 0.1
    proj = rng.standard_normal((4, 2560)).astype(np.float32) * 0.05

    # via mom/engine: simplesmente h + proj (essa e a semantica de h = h + tmm(...))
    expected = (h + proj).astype(np.float32)

    # via IR: cria um pequeno IR com so o ResidualAdd e executa
    from tq1ir.ir import Module as IRModule, Function as IRFunc, Block as IRBlock
    from tq1ir.ops import Value
    from tq1ir.types import F32TensorType

    ft = F32TensorType(shape=(-1, 2560))
    v_h = Value(type=ft, name="h")
    v_p = Value(type=ft, name="p")
    v_o = Value(type=ft, name="o")
    fn = IRFunc(name="resid", inputs=[v_h, v_p], outputs=[v_o], body=IRBlock())
    fn.body.append(ResidualAddOp(result=v_o, lhs=v_h, rhs=v_p))
    fn.body.append(ReturnOp(values=[v_o]))

    tiny_mod = IRModule(name="tiny", source_gguf=GGUF, functions=[fn])
    loader = WeightLoader(GGUF)
    ex = Executor(tiny_mod, loader)
    env = {v_h.name: h, v_p.name: proj}
    for op in fn.body.ops:
        r = ex._exec_op(op, env)
        if op.result and r is not None:
            env[op.result.name] = r
    got = env[v_o.name]
    loader.close()

    if got.shape != expected.shape:
        fail(f"shape: {got.shape} vs {expected.shape}")
    maxdiff = float(np.abs(got - expected).max())
    if maxdiff != 0.0:
        fail(f"NAO bit-exact: max|diff|={maxdiff}")
    ok(f"ResidualAddOp bit-exact (max|diff|=0, shape={got.shape})")

    # ---------- G3 ----------
    banner("G3  Layer_0 end-to-end vs referencia (mesmas ops)")
    # IR via kernel backend
    loader = WeightLoader(GGUF)
    ex = Executor(mod, loader)

    rng = np.random.default_rng(42)
    T, d = 4, 2560
    x_in  = rng.standard_normal((T, d)).astype(np.float32) * 0.1
    pos_in = np.arange(T, dtype=np.float32)

    env = {fn0.inputs[0].name: x_in, fn0.inputs[1].name: pos_in}
    for op in fn0.body.ops:
        r = ex._exec_op(op, env)
        if op.result and r is not None:
            env[op.result.name] = r
    y_ir = env[fn0.outputs[0].name]

    # Referencia em Python puro (numpy), espelhando as mesmas ops do
    # kernel backend - silu_mul (NAO relu^2) e SEM sub_norms.
    # Usa os mesmos pesos via WeightLoader (decodificacao i2_s ja gated).
    def W(name): return loader.load(mod.weights[name])

    def rmsnorm(x, w, eps=1e-5):
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        return (x / rms) * w

    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -80, 80)))

    def rope(x, pos, theta):
        T, d = x.shape[-2], x.shape[-1]
        half = d // 2
        freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / half))
        ang = np.outer(pos.astype(np.float64), freqs).astype(np.float32)
        c, s = np.cos(ang), np.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1*c - x2*s, x1*s + x2*c], axis=-1)

    def attn(q, k, v, n_h=20, n_kv=5, hd=128):
        T = q.shape[0]
        q = q.reshape(T, n_h, hd); k = k.reshape(T, n_kv, hd); v = v.reshape(T, n_kv, hd)
        rep = n_h // n_kv
        k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
        out = np.zeros((T, n_h, hd), dtype=np.float32)
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        for h_i in range(n_h):
            sc = q[:, h_i] @ k[:, h_i].T * (hd ** -0.5) + mask
            sc -= sc.max(-1, keepdims=True)
            p = np.exp(sc); p /= p.sum(-1, keepdims=True)
            out[:, h_i] = p @ v[:, h_i]
        return out.reshape(T, n_h * hd)

    x = x_in
    xn = rmsnorm(x, W("blk.0.attn_norm.weight"))
    q  = xn @ W("blk.0.attn_q.weight").T
    k  = xn @ W("blk.0.attn_k.weight").T
    v  = xn @ W("blk.0.attn_v.weight").T
    q_r = rope(q, pos_in, 500_000.0)
    k_r = rope(k, pos_in, 500_000.0)
    a   = attn(q_r, k_r, v)
    o   = a @ W("blk.0.attn_output.weight").T
    h_attn = x + o                                           # residual 1
    xn2 = rmsnorm(h_attn, W("blk.0.ffn_norm.weight"))
    gate = xn2 @ W("blk.0.ffn_gate.weight").T
    up   = xn2 @ W("blk.0.ffn_up.weight").T
    act  = silu(gate) * up
    ffn_out = act @ W("blk.0.ffn_down.weight").T
    y_ref = h_attn + ffn_out                                 # residual 2
    loader.close()

    if y_ir.shape != y_ref.shape:
        fail(f"shape: {y_ir.shape} vs {y_ref.shape}")
    maxdiff = float(np.abs(y_ir - y_ref).max())
    print(f"  layer_0 max|diff|={maxdiff:.6e} (referencia: mesmas ops)")
    if maxdiff != 0.0:
        fail("layer_0 NAO e bit-exact vs referencia - bug na ordem dos residuals")
    ok(f"layer_0 bit-exact (max|diff|=0) - residuals na ordem certa")

    # ---------- Nota de honestidade ----------
    print()
    print("Nota: G3 prova que os residuals estao no sitio certo.")
    print("Bit-exact contra mom/engine FULL fica bloqueado por gaps")
    print("pre-existentes ao scope desta task:")
    print("  (a) sub-norms intra-layer (attn_sub_norm, ffn_sub_norm).")
    print("  (b) activacao FFN: relu^2*up em vez de silu*up.")
    print("Sao independentes de residuals; ficam para v1.1.")

    print("\nTODOS OS GATES PASSARAM (G1, G2, G3).")


if __name__ == "__main__":
    main()
