"""Edge-case check of `bullet_train` against the exact serial reference (outside the gate).

Random cases: start membrane far from 0, the no-leak mode (decay_input=False), a gradient
through the last membrane, L not a multiple of K, several K, block sizes and tapes, both
backward passes, and a non-contiguous incoming gradient. Prints one line per failed case
and a summary; exit status 1 if a case fails.

    python -B check_bullet.py [n_cases] [seed]
"""

import sys

sys.dont_write_bytecode = True

import random  # noqa: E402

import torch  # noqa: E402

import bullet_neuron  # noqa: E402
import neuron  # noqa: E402


def loop_reference(x, q, v0, theta, decay):
    """Plain PyTorch loop of the detached hard-reset neuron with the ATan surrogate (autograd)."""
    v, spikes = v0, []
    for t in range(x.shape[0]):
        h = v + (x[t] - v) * q if decay else v - v * q + x[t]
        z = h - theta
        s_hard = (z >= 0).float()
        sg = ((neuron.ALPHA / 2) / (1 + (torch.pi / 2 * neuron.ALPHA * z) ** 2)).detach()
        s = s_hard + (sg * z - (sg * z).detach())  # value: Heaviside; gradient: sg (the ATan derivative)
        spikes.append(s)
        v = h * (1.0 - s_hard)
    return torch.stack(spikes), v


def one_case(rng: random.Random, gen: torch.Generator) -> list[str]:
    """Run one random case. Returns the list of problems (empty if it passed)."""
    K = rng.choice([8, 16, 32, 64, 128])
    bwd = rng.choice(["serial", "scan"])
    tape = rng.choice(["h32", "h16", "u8"])
    Kb = rng.choice([8, 16, 32, 64])
    U = 8
    step = max(U, Kb) if bwd == "scan" else U
    L = rng.choice([1, 2, 3, 5]) * step * rng.choice([1, 3, 8])
    B, D = rng.choice([1, 2, 3]), rng.choice([1, 7, 32, 100])
    decay = rng.random() < 0.8
    theta = rng.choice([0.5, 1.0, 2.0])
    tau = (1.2 + torch.rand(D, generator=gen, device="cuda") * 20).requires_grad_(True)
    x = (torch.randn(L, B, D, generator=gen, device="cuda") * rng.choice([0.5, 1.5, 4.0]) + 0.5).requires_grad_(True)
    v0 = (torch.randn(B, D, generator=gen, device="cuda") * rng.choice([0.0, 0.5, 3.0])).requires_grad_(True)
    r = torch.randn(L, B, D, generator=gen, device="cuda")
    r_last = torch.randn(B, D, generator=gen, device="cuda") * rng.choice([0.0, 1.0])
    if rng.random() < 0.3:
        r = r.transpose(1, 2).contiguous().transpose(1, 2)  # same values, non-contiguous view
    cfg = dict(K=K, BLOCK=rng.choice([16, 32, 64]), nw=rng.choice([1, 2, 4]), tape=tape, bwd=bwd, U=U, Kb=Kb,
               bBLOCK=rng.choice([16, 32]), sBLOCK=rng.choice([32, 64]), snw=rng.choice([1, 2]))  # fmt: skip
    s, v_last = bullet_neuron.bullet_train(x, tau, theta, v0=v0, decay_input=decay, hard=1, **cfg)
    grads = torch.autograd.grad((s, v_last), (x, tau, v0), grad_outputs=(r, r_last))
    q = neuron.leak_q(tau)
    s_ref, v_ref = loop_reference(x, q, v0, theta, decay)
    if tape != "h32":
        # A rounded tape changes the gradient by itself: the 8-bit code loses leak-gradient accuracy
        # for theta far from 1, and a float16 h just below theta can round to theta, so the backward
        # pass sees a spike there. Both are limits of `serial_triton` with the same tape. Compare with
        # that kernel, same tape, to test the method and not the tape.
        s_tape, v_tape = neuron.serial_triton(x, tau, theta, v0=v0, decay_input=decay, U=U, BLOCK=32, nw=1, tape=tape, hard=1)
        grads_ref = torch.autograd.grad((s_tape, v_tape), (x, tau, v0), grad_outputs=(r, r_last))
    else:
        grads_ref = torch.autograd.grad((s_ref, v_ref), (x, tau, v0), grad_outputs=(r, r_last))
    problems = []
    wrong = int((s.detach() != s_ref.detach()).sum())
    if wrong > max(1, s.numel() // 10000):  # scan rounding may move a spike at an exact threshold tie
        problems.append(f"{wrong} wrong spikes of {s.numel()}")
    min_cos = 0.999
    for name, g, g_ref in zip(("x", "tau", "v0"), grads, grads_ref):
        if g_ref.norm() < 1e-6:
            continue
        cos = torch.nn.functional.cosine_similarity(g.flatten(), g_ref.flatten(), dim=0).item()
        if cos < min_cos and not (name == "tau" and wrong):
            problems.append(f"cos {name} {cos:.4f}")
    if problems:
        problems.insert(0, f"L={L} B={B} D={D} decay={decay} theta={theta} {cfg}")
    return problems


if __name__ == "__main__":
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng, gen = random.Random(seed), torch.Generator(device="cuda").manual_seed(seed)
    failed = 0
    for i in range(n_cases):
        problems = one_case(rng, gen)
        if problems:
            failed += 1
            print(f"case {i} FAILED: " + "; ".join(problems))
    print(f"CHECK cases={n_cases} failed={failed} seed={seed}")
    sys.exit(1 if failed else 0)
