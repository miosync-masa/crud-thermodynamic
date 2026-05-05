"""
Landauer CRUD Thermodynamics Simulator - Final FIG Edition
==========================================================
Single-file CRUD information thermodynamics demo with:

  - Create / Read / Update / Delete in one codebase
  - Full accounting:
      DeltaF_total(full) = DeltaF_eq + kT * Delta D_KL(full)
  - Histogram-based macro / intra KL decomposition for C/U/D
  - Read as measurement with system-meter coupling
  - Write endpoint epsilon micro-calibration
  - One integrated final figure + CSV export
  - Relax scan figure for C/U/D
  - Optional GPU-side power logging via nvidia-smi
  - DKL bin sensitivity scan for KL-accounting robustness

Operation mapping used here:
  Create : blank0 -> 1
  Read   : non-destructive measurement
  Update : unknown old value -> 1  (overwrite-like update)
  Delete : unknown value -> blank0

Interpretation:
  - Delete / Update / Create are model-level thermodynamic demonstrations
  - Read is modeled as measurement thermodynamics
  - This is numerical verification under explicit model assumptions,
    not direct proof on production servers/hardware.

Author: Masamichi Iizumi & Tamaki Iizumi
Prepared for principle-presentation mode
"""

import csv
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random, jit, vmap


# ============================================================
# CONFIG
# ============================================================

VERSION = "CRUD-FIG-FINAL-v1.1"

KB = 1.0
GAMMA = 1.0
DT = 0.001

T0 = 1.0
N_CORE = 10000
N_SCAN = 8000
N_CALIB = 2500

N_LAMBDA_STEPS = 2000
RELAX_CORE = 5
RELAX_LIST = [1, 2, 5, 10, 20]

B_HI = 4.0
B_LO = 0.2
C_MAG_DEFAULT = 3.0
UNBIAS_FRAC = 0.05
END_MODE = "unbias"

WRITE_EPS_TARGET = 0.01
WRITE_ENDPOINT_TOL = 0.002
WRITE_ENDPOINT_REPS = 3
WRITE_ENDPOINT_MAX_ITERS = 8
WRITE_ENDPOINT_GROWTH = 1.20
WRITE_CALIB_GRID = 80000
WRITE_CALIB_TOL_PR = 2e-4

READ_GMAX = 2.0
READ_KMETER = 2.0
READ_STEPS = 2000
READ_NPART = 10000

DKL_BINS = 240
DKL_BINS_SENSITIVITY = [120, 240, 480]

# Optional hardware diagnostic.
# In Colab this usually captures GPU-side power only; it is NOT full server energy.
ENABLE_GPU_POWER_LOG = True
GPU_POWER_INTERVAL_SEC = 0.2

PFX = "landauer_crud_final_v11"


print("JAX devices:", jax.devices())
print("=" * 72)
print(f"Landauer CRUD Thermodynamics Simulator ({VERSION})")
print("DeltaF_eq + kT Delta D_KL(full) + Read peak/final MI + GPU logging + final FIG")
print("=" * 72)
print(f"Landauer kT ln2 = {KB * T0 * np.log(2.0):.6f} (T={T0})")
print(f"Particles(core): {N_CORE}")
print(f"Particles(scan): {N_SCAN}")
print(f"Lambda-steps:    {N_LAMBDA_STEPS}")
print(f"Relax(core):     {RELAX_CORE}")
print(f"b_hi={B_HI}, b_lo={B_LO}, c_mag(default)={C_MAG_DEFAULT}, unbias_frac={UNBIAS_FRAC}, end_mode={END_MODE}")
print(f"Write eps target={WRITE_EPS_TARGET}, Read g_max={READ_GMAX}, k_meter={READ_KMETER}")
print("=" * 72)



# ============================================================
# SECTION 0.5: Optional GPU-side power logging (Colab / nvidia-smi)
# ============================================================

class GPUPowerLogger:
    """
    Lightweight GPU-side power logger.

    Notes:
      - Uses nvidia-smi if available.
      - Estimates accelerator-side energy only.
      - This is a hardware diagnostic, not full server-level energy accounting.
    """
    def __init__(self, interval_sec=0.2):
        self.interval_sec = float(interval_sec)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.available = shutil.which("nvidia-smi") is not None

    def _query_once(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=2.0)
        # Use first visible GPU.
        line = out.strip().splitlines()[0]
        parts = [x.strip() for x in line.split(",")]
        name = parts[0]
        power_W = float(parts[1])
        util_pct = float(parts[2])
        mem_MiB = float(parts[3])
        return name, power_W, util_pct, mem_MiB

    def _loop(self):
        while not self._stop.is_set():
            t_mono = time.monotonic()
            t_wall = time.time()
            try:
                name, power_W, util_pct, mem_MiB = self._query_once()
                self.samples.append({
                    "t_monotonic": t_mono,
                    "t_wall": t_wall,
                    "gpu_name": name,
                    "power_W": power_W,
                    "util_pct": util_pct,
                    "memory_MiB": mem_MiB,
                })
            except Exception as exc:
                self.samples.append({
                    "t_monotonic": t_mono,
                    "t_wall": t_wall,
                    "gpu_name": "nvidia-smi-error",
                    "power_W": np.nan,
                    "util_pct": np.nan,
                    "memory_MiB": np.nan,
                    "error": str(exc),
                })
            self._stop.wait(self.interval_sec)

    def start(self):
        if not self.available:
            print("[GPU logger] nvidia-smi not found; GPU power logging disabled.")
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[GPU logger] started, interval={self.interval_sec:.3f}s")
        return self

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=3.0)
            print("[GPU logger] stopped.")
        return self.summary()

    def summary(self):
        valid = [s for s in self.samples if np.isfinite(s.get("power_W", np.nan))]
        if len(valid) < 2:
            return {
                "gpu_logger_available": bool(self.available),
                "gpu_samples": len(valid),
                "gpu_energy_J": None,
                "gpu_mean_power_W": None,
                "gpu_max_power_W": None,
                "gpu_elapsed_s": None,
                "gpu_name": valid[0]["gpu_name"] if valid else None,
            }

        t = np.array([s["t_monotonic"] for s in valid], dtype=np.float64)
        p = np.array([s["power_W"] for s in valid], dtype=np.float64)
        energy_J = float(np.trapz(p, t))
        elapsed_s = float(t[-1] - t[0])

        return {
            "gpu_logger_available": True,
            "gpu_samples": int(len(valid)),
            "gpu_energy_J": energy_J,
            "gpu_mean_power_W": float(np.mean(p)),
            "gpu_max_power_W": float(np.max(p)),
            "gpu_elapsed_s": elapsed_s,
            "gpu_name": valid[0]["gpu_name"],
        }

    def export_csv(self, prefix=PFX):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"{prefix}_gpu_power_log_{ts}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_monotonic", "t_wall", "gpu_name", "power_W", "util_pct", "memory_MiB", "error"])
            for s in self.samples:
                w.writerow([
                    s.get("t_monotonic"),
                    s.get("t_wall"),
                    s.get("gpu_name"),
                    s.get("power_W"),
                    s.get("util_pct"),
                    s.get("memory_MiB"),
                    s.get("error", ""),
                ])
        print(f"Exported: {fn}")
        return fn

def export_config_and_gpu_summary(config_dict, gpu_summary, prefix=PFX):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{prefix}_run_config_gpu_summary_{ts}.json"
    payload = {
        "config": config_dict,
        "gpu_summary": gpu_summary,
        "note": "GPU energy is accelerator-side nvidia-smi estimate only, not full server energy.",
    }
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Exported: {fn}")
    return fn


# ============================================================
# SECTION 1: Core physics
# ============================================================

@jit
def potential(x, b, c):
    return x**4 - b * x**2 - c * x

@jit
def force(x, b, c):
    return -4.0 * x**3 + 2.0 * b * x + c

@jit
def step_single(x, key, b, c, T):
    noise = random.normal(key)
    return x + (force(x, b, c) / GAMMA) * DT + jnp.sqrt(2.0 * KB * T * DT / GAMMA) * noise

@jit
def step_batch(x_batch, keys, b, c, T):
    return vmap(lambda x, k: step_single(x, k, b, c, T))(x_batch, keys)

@jit
def work_step(x, b_old, c_old, b_new, c_new):
    return potential(x, b_new, c_new) - potential(x, b_old, c_old)

work_step_batch = vmap(work_step, in_axes=(0, None, None, None, None))

@jit
def jarzynski_deltaF(work_samples, T):
    beta = 1.0 / (KB * T)
    y = -beta * work_samples
    m = jnp.max(y)
    return -KB * T * (m + jnp.log(jnp.mean(jnp.exp(y - m))))


# ============================================================
# SECTION 2: Numpy equilibrium helpers
# ============================================================

def potential_np(x, b, c):
    return x**4 - b * x**2 - c * x

def auto_L(b, c, T):
    return max(
        4.0,
        float(np.sqrt(abs(b))) + 3.0 * float(np.sqrt(T + 0.1)) + 1.8 * float(abs(c) ** (1.0 / 3.0)) + 1.0
    )

def logZ_region(b, c, T, region=None, n_grid=WRITE_CALIB_GRID):
    beta = 1.0 / (KB * T)
    L = auto_L(b, c, T)

    x = np.linspace(-L, L, int(n_grid), dtype=np.float64)
    dx = float(x[1] - x[0])
    V = potential_np(x, b, c)

    if region is None:
        mask = np.ones_like(x, dtype=bool)
    elif region == "left":
        mask = (x <= 0.0)
    elif region == "right":
        mask = (x > 0.0)
    else:
        raise ValueError("region must be None/'left'/'right'")

    V_reg = np.where(mask, V, np.inf)
    Vmin = np.min(V_reg)
    w = np.where(mask, np.exp(-beta * (V - Vmin)), 0.0)

    Z = np.sum(w) * dx
    Z = np.clip(Z, 1e-300, 1e300)

    return float(-beta * Vmin + np.log(Z))

def free_energy_eq(b, c, T, region=None, n_grid=WRITE_CALIB_GRID):
    return -KB * T * logZ_region(b, c, T, region=region, n_grid=n_grid)

def p_right_eq(b, c, T, n_grid=WRITE_CALIB_GRID):
    logZ_tot = logZ_region(b, c, T, region=None, n_grid=n_grid)
    logZ_R = logZ_region(b, c, T, region="right", n_grid=n_grid)
    return float(np.exp(logZ_R - logZ_tot))

def sample_equilibrium_positions(b, c, T, n_particles, key, region=None, n_grid=50000):
    L = auto_L(b, c, T)
    x = np.linspace(-L, L, int(n_grid), dtype=np.float64)
    dx = float(x[1] - x[0])

    beta = 1.0 / (KB * T)
    V = potential_np(x, b, c)

    if region is None:
        mask = np.ones_like(x, dtype=bool)
    elif region == "left":
        mask = (x <= 0.0)
    elif region == "right":
        mask = (x > 0.0)
    else:
        raise ValueError("region must be None/'left'/'right'")

    V_reg = np.where(mask, V, np.inf)
    Vmin = np.min(V_reg)
    w = np.where(mask, np.exp(-beta * (V - Vmin)), 0.0)

    cdf = np.cumsum(w) * dx
    Z = float(cdf[-1])
    if Z <= 0.0:
        raise RuntimeError("Sampling partition vanished.")
    cdf = cdf / Z

    key_u, key_j = random.split(key)
    u = np.array(random.uniform(key_u, shape=(n_particles,)), dtype=np.float64)
    idx = np.searchsorted(cdf, u, side="right")
    idx = np.clip(idx, 0, int(n_grid) - 1)
    samples = x[idx]

    jitter = (np.array(random.uniform(key_j, shape=(n_particles,)), dtype=np.float64) - 0.5) * dx
    samples = samples + jitter

    if region == "left":
        samples = np.minimum(samples, 0.0)
    elif region == "right":
        samples = np.maximum(samples, 0.0)

    samples = np.clip(samples, -L, L)
    return jnp.array(samples, dtype=jnp.float32)


# ============================================================
# SECTION 3: Safe KL + macro/intra decomposition
# ============================================================

def make_grid_edges_centers(b, c, T, n_bins):
    L = auto_L(b, c, T)
    edges = np.linspace(-L, L, int(n_bins) + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])
    return edges, centers, dx

def normalize_density(p, dx):
    Z = max(float(np.sum(p) * dx), 1e-300)
    return p / Z

def hist_density(x_samples, edges):
    counts, _ = np.histogram(np.asarray(x_samples, dtype=np.float64), bins=edges)
    dx = float(edges[1] - edges[0])
    N = max(int(np.sum(counts)), 1)
    return counts.astype(np.float64) / (N * dx)

def dkl_continuous_safe(p_emp, p_eq, dx):
    p_emp = np.asarray(p_emp, dtype=np.float64)
    p_eq = np.asarray(p_eq, dtype=np.float64)
    valid = (p_emp > 0.0) & (p_eq > 0.0)
    if not np.any(valid):
        return 0.0
    term = np.zeros_like(p_emp, dtype=np.float64)
    term[valid] = p_emp[valid] * (np.log(p_emp[valid]) - np.log(p_eq[valid]))
    val = float(np.sum(term) * dx)
    return max(0.0, val)

def compute_dkl_full_macro_intra(x_samples, b, c, T, n_bins=DKL_BINS):
    edges, centers, dx = make_grid_edges_centers(b, c, T, n_bins)

    p_emp = hist_density(x_samples, edges)
    p_emp = normalize_density(p_emp, dx)

    beta = 1.0 / (KB * T)
    V = potential_np(centers, b, c)
    Vmin = np.min(V)
    q_raw = np.exp(-beta * (V - Vmin))
    p_eq = normalize_density(q_raw, dx)

    D_full = dkl_continuous_safe(p_emp, p_eq, dx)

    mask_L = (centers <= 0.0)
    mask_R = (centers > 0.0)

    pL_emp = float(np.sum(p_emp[mask_L]) * dx)
    pR_emp = float(np.sum(p_emp[mask_R]) * dx)
    qL_eq = float(np.sum(p_eq[mask_L]) * dx)
    qR_eq = float(np.sum(p_eq[mask_R]) * dx)

    s_emp = max(pL_emp + pR_emp, 1e-300)
    pL_emp /= s_emp
    pR_emp /= s_emp

    s_eq = max(qL_eq + qR_eq, 1e-300)
    qL_eq /= s_eq
    qR_eq /= s_eq

    D_macro = 0.0
    if pL_emp > 1e-300 and qL_eq > 1e-300:
        D_macro += pL_emp * np.log(pL_emp / qL_eq)
    if pR_emp > 1e-300 and qR_eq > 1e-300:
        D_macro += pR_emp * np.log(pR_emp / qR_eq)

    D_intra_L = 0.0
    D_intra_R = 0.0

    if pL_emp > 1e-12 and qL_eq > 1e-12:
        p_emp_L = p_emp[mask_L] / pL_emp
        p_eq_L = p_eq[mask_L] / qL_eq
        D_intra_L = dkl_continuous_safe(p_emp_L, p_eq_L, dx)

    if pR_emp > 1e-12 and qR_eq > 1e-12:
        p_emp_R = p_emp[mask_R] / pR_emp
        p_eq_R = p_eq[mask_R] / qR_eq
        D_intra_R = dkl_continuous_safe(p_emp_R, p_eq_R, dx)

    D_intra_weighted = pL_emp * D_intra_L + pR_emp * D_intra_R
    recon_err = float(D_full - (D_macro + D_intra_weighted))

    return {
        "DKL_full": float(D_full),
        "DKL_macro": float(D_macro),
        "DKL_intra_weighted": float(D_intra_weighted),
        "recon_err": float(recon_err),
        "pL_emp": float(pL_emp),
        "pR_emp": float(pR_emp),
        "pL_eq": float(qL_eq),
        "pR_eq": float(qR_eq),
    }

def binary_entropy_nats(eps):
    eps = float(np.clip(eps, 1e-12, 1.0 - 1e-12))
    return -(eps * np.log(eps) + (1.0 - eps) * np.log(1.0 - eps))

def landauer_logical_deltaF(eps, T):
    return KB * T * (np.log(2.0) - binary_entropy_nats(eps))


# ============================================================
# SECTION 4: Protocol family for Create / Update / Delete
# ============================================================

def ease_cos(s):
    s = float(np.clip(s, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * s)

def protocol_core(t, b_hi, b_lo, c_bias, unbias_frac=0.05, end_mode="unbias"):
    """
    4-phase protocol:
      1) lower barrier
      2) bias at low barrier
      3) raise barrier under bias
      4) optional unbias at high barrier
    """
    u = float(np.clip(unbias_frac, 0.0, 0.4))
    t1 = 0.25
    t3 = 0.25
    t4 = u
    t2 = 1.0 - (t1 + t3 + t4)
    if t2 <= 0.05:
        t2 = 0.05
        t4 = 1.0 - (t1 + t2 + t3)

    if t < t1:
        s = ease_cos(t / t1)
        b = (1.0 - s) * b_hi + s * b_lo
        c = 0.0
    elif t < t1 + t2:
        s = ease_cos((t - t1) / t2)
        b = b_lo
        c = s * c_bias
    elif t < t1 + t2 + t3:
        s = ease_cos((t - (t1 + t2)) / t3)
        b = (1.0 - s) * b_lo + s * b_hi
        c = c_bias
    else:
        if end_mode == "unbias":
            if t4 <= 1e-9:
                b = b_hi
                c = 0.0
            else:
                s = ease_cos((t - (t1 + t2 + t3)) / t4)
                b = b_hi
                c = (1.0 - s) * c_bias
        else:
            b = b_hi
            c = c_bias

    return float(b), float(c)

def make_protocol(op_name, b_hi, b_lo, c_mag, unbias_frac, end_mode):
    """
    CRUD mapping:
      Create = blank0 -> 1      (init left, target right)
      Update = unknown -> 1     (init full, target right)
      Delete = unknown -> blank0(init full, target left)
    """
    if op_name == "Create":
        c_bias = +abs(c_mag)
        init_region = "left"
        target = "right"
    elif op_name == "Update":
        c_bias = +abs(c_mag)
        init_region = None
        target = "right"
    elif op_name == "Delete":
        c_bias = -abs(c_mag)
        init_region = None
        target = "left"
    else:
        raise ValueError("op_name must be Create/Update/Delete")

    def proto(t):
        return protocol_core(
            t,
            b_hi=b_hi,
            b_lo=b_lo,
            c_bias=c_bias,
            unbias_frac=unbias_frac,
            end_mode=end_mode,
        )
    return proto, init_region, target


# ============================================================
# SECTION 5: Write calibration (Create endpoint fixed)
# ============================================================

def solve_c_for_target_pR(b, T, target_pR,
                          c_min=0.0,
                          c_max=2.0,
                          n_grid=WRITE_CALIB_GRID,
                          tol_pR=WRITE_CALIB_TOL_PR,
                          max_iter=80,
                          verbose=False):
    target_pR = float(np.clip(target_pR, 1e-6, 1.0 - 1e-6))
    c_min = float(c_min)
    c_max = float(c_max)

    p_min = p_right_eq(b, c_min, T, n_grid=n_grid)
    p_max = p_right_eq(b, c_max, T, n_grid=n_grid)

    while p_max < target_pR and c_max < 1e4:
        c_max *= 2.0
        p_max = p_right_eq(b, c_max, T, n_grid=n_grid)

    if verbose:
        print(f"[analytic-calib] target_pR={target_pR:.6f}")
        print(f"  bracket: c_min={c_min:.6f} -> pR={p_min:.6f} | c_max={c_max:.6f} -> pR={p_max:.6f}")

    lo, hi = c_min, c_max
    for it in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        pmid = p_right_eq(b, mid, T, n_grid=n_grid)
        if verbose and it < 10:
            print(f"  it={it:02d} c={mid:10.6f} pR={pmid:.6f} err={pmid-target_pR:+.6f}")

        if abs(pmid - target_pR) < tol_pR:
            return float(mid)

        if pmid < target_pR:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))

def calibrate_write_cmag_analytic(b_lo, T, eps_target=WRITE_EPS_TARGET, verbose=True):
    target_pR = 1.0 - float(eps_target)
    return solve_c_for_target_pR(
        b=b_lo,
        T=T,
        target_pR=target_pR,
        c_min=0.0,
        c_max=2.0,
        n_grid=WRITE_CALIB_GRID,
        tol_pR=WRITE_CALIB_TOL_PR,
        verbose=verbose,
    )

def simulate_protocol(protocol_fn, x0, n_lambda_steps, relax_steps, T, key):
    bs = np.zeros(n_lambda_steps + 1, dtype=np.float64)
    cs = np.zeros(n_lambda_steps + 1, dtype=np.float64)
    for i in range(n_lambda_steps + 1):
        tt = i / n_lambda_steps
        bs[i], cs[i] = protocol_fn(tt)

    x = jnp.array(x0, dtype=jnp.float32)
    W = jnp.zeros(x.shape[0], dtype=jnp.float32)

    for i in range(n_lambda_steps):
        b_old, c_old = float(bs[i]), float(cs[i])
        b_new, c_new = float(bs[i + 1]), float(cs[i + 1])

        dW = work_step_batch(x, b_old, c_old, b_new, c_new)
        W = W + dW

        for _ in range(int(relax_steps)):
            key, sub = random.split(key)
            keys = random.split(sub, x.shape[0])
            x = step_batch(x, keys, b_new, c_new, T)

    return x, W

def estimate_eps(x_final, target):
    x_np = np.asarray(x_final)
    if target == "left":
        return float(np.mean(x_np > 0.0))
    elif target == "right":
        return float(np.mean(x_np <= 0.0))
    raise ValueError("target must be 'left'/'right'")

def run_create_endpoint_only(c_mag, relax_steps, n_lambda_steps, n_particles, T, key):
    proto, init_region, target = make_protocol(
        "Create",
        b_hi=B_HI,
        b_lo=B_LO,
        c_mag=float(c_mag),
        unbias_frac=UNBIAS_FRAC,
        end_mode=END_MODE,
    )
    b0, c0 = proto(0.0)
    key, k_init, k_run = random.split(key, 3)
    x0 = sample_equilibrium_positions(float(b0), float(c0), T, n_particles, k_init, region=init_region)
    x1, _ = simulate_protocol(proto, x0, n_lambda_steps, relax_steps, T, k_run)

    eps = estimate_eps(x1, target=target)
    pR_final = float(jnp.mean(x1 > 0.0))
    return float(eps), float(pR_final)

def evaluate_create_endpoint_avg(c_mag, relax_steps, n_lambda_steps, n_particles, T, base_key, n_reps=WRITE_ENDPOINT_REPS):
    rep_keys = random.split(base_key, int(n_reps))
    eps_list = []
    pR_list = []
    for rk in rep_keys:
        eps, pR = run_create_endpoint_only(
            c_mag=c_mag,
            relax_steps=relax_steps,
            n_lambda_steps=n_lambda_steps,
            n_particles=n_particles,
            T=T,
            key=rk,
        )
        eps_list.append(eps)
        pR_list.append(pR)

    eps_arr = np.array(eps_list, dtype=np.float64)
    pR_arr = np.array(pR_list, dtype=np.float64)

    return {
        "eps_mean": float(np.mean(eps_arr)),
        "eps_std": float(np.std(eps_arr)),
        "pR_mean": float(np.mean(pR_arr)),
        "pR_std": float(np.std(pR_arr)),
    }

def refine_write_cmag_endpoint(c_init, *,
                               relax_steps,
                               n_lambda_steps,
                               T,
                               base_key,
                               eps_target=WRITE_EPS_TARGET,
                               tol=WRITE_ENDPOINT_TOL,
                               n_particles=N_CALIB,
                               n_reps=WRITE_ENDPOINT_REPS,
                               max_iters=WRITE_ENDPOINT_MAX_ITERS,
                               growth=WRITE_ENDPOINT_GROWTH,
                               verbose=True):
    cache = {}

    def eval_c(c):
        c = float(max(c, 1e-6))
        key_cache = round(c, 8)
        if key_cache not in cache:
            cache[key_cache] = evaluate_create_endpoint_avg(
                c_mag=c,
                relax_steps=relax_steps,
                n_lambda_steps=n_lambda_steps,
                n_particles=n_particles,
                T=T,
                base_key=base_key,
                n_reps=n_reps,
            )
        return cache[key_cache]

    c0 = float(c_init)
    r0 = eval_c(c0)
    if verbose:
        print(f"[endpoint-refine] relax={relax_steps}  start c={c0:.6f}  eps={r0['eps_mean']:.6f}+-{r0['eps_std']:.6f}")

    if abs(r0["eps_mean"] - eps_target) <= tol:
        return {
            "c_analytic": c0,
            "c_refined": c0,
            "status": "init_ok",
            "eps_mean": r0["eps_mean"],
            "eps_std": r0["eps_std"],
            "pR_mean": r0["pR_mean"],
            "pR_std": r0["pR_std"],
        }

    if r0["eps_mean"] > eps_target:
        c_lo = c0
        c_hi = c0 * growth
        r_hi = eval_c(c_hi)
        guard = 0
        while r_hi["eps_mean"] > eps_target and guard < 12:
            c_hi *= growth
            r_hi = eval_c(c_hi)
            guard += 1
    else:
        c_hi = c0
        c_lo = c0 / growth
        r_lo = eval_c(c_lo)
        guard = 0
        while r_lo["eps_mean"] < eps_target and c_lo > 1e-6 and guard < 12:
            c_lo /= growth
            r_lo = eval_c(c_lo)
            guard += 1

    r_lo = eval_c(c_lo)
    r_hi = eval_c(c_hi)

    if verbose:
        print(f"  bracket: c_lo={c_lo:.6f} eps_lo={r_lo['eps_mean']:.6f} | c_hi={c_hi:.6f} eps_hi={r_hi['eps_mean']:.6f}")

    best = None
    for c in [c_lo, c_hi, c0]:
        rr = eval_c(c)
        cand = (abs(rr["eps_mean"] - eps_target), c, rr)
        if best is None or cand[0] < best[0]:
            best = cand

    if not (r_lo["eps_mean"] >= eps_target >= r_hi["eps_mean"]):
        _, c_best, rr_best = best
        return {
            "c_analytic": c0,
            "c_refined": float(c_best),
            "status": "best_effort_no_bracket",
            "eps_mean": rr_best["eps_mean"],
            "eps_std": rr_best["eps_std"],
            "pR_mean": rr_best["pR_mean"],
            "pR_std": rr_best["pR_std"],
        }

    for it in range(int(max_iters)):
        c_mid = 0.5 * (c_lo + c_hi)
        r_mid = eval_c(c_mid)

        if verbose:
            print(f"  it={it:02d} c={c_mid:.6f} eps={r_mid['eps_mean']:.6f}+-{r_mid['eps_std']:.6f}")

        cand = (abs(r_mid["eps_mean"] - eps_target), c_mid, r_mid)
        if cand[0] < best[0]:
            best = cand

        if abs(r_mid["eps_mean"] - eps_target) <= tol:
            return {
                "c_analytic": c0,
                "c_refined": float(c_mid),
                "status": "converged",
                "eps_mean": r_mid["eps_mean"],
                "eps_std": r_mid["eps_std"],
                "pR_mean": r_mid["pR_mean"],
                "pR_std": r_mid["pR_std"],
            }

        if r_mid["eps_mean"] > eps_target:
            c_lo = c_mid
        else:
            c_hi = c_mid

    _, c_best, rr_best = best
    return {
        "c_analytic": c0,
        "c_refined": float(c_best),
        "status": "max_iter_best",
        "eps_mean": rr_best["eps_mean"],
        "eps_std": rr_best["eps_std"],
        "pR_mean": rr_best["pR_mean"],
        "pR_std": rr_best["pR_std"],
    }


# ============================================================
# SECTION 6: CRUD operation run
# ============================================================

def run_crud_operation(op_name, *,
                       b_hi, b_lo, c_mag, unbias_frac, end_mode,
                       n_lambda_steps, relax_steps, n_particles, T, key,
                       dkl_bins=DKL_BINS):
    proto, init_region, target = make_protocol(op_name, b_hi, b_lo, c_mag, unbias_frac, end_mode)
    b0, c0 = proto(0.0)
    b1, c1 = proto(1.0)

    key, k_init, k_run = random.split(key, 3)
    x0 = sample_equilibrium_positions(float(b0), float(c0), T, n_particles, k_init, region=init_region)

    dkl_init = compute_dkl_full_macro_intra(x0, float(b0), float(c0), T, n_bins=dkl_bins)
    x1, W_samples = simulate_protocol(proto, x0, n_lambda_steps, relax_steps, T, k_run)
    dkl_final = compute_dkl_full_macro_intra(x1, float(b1), float(c1), T, n_bins=dkl_bins)

    meanW = float(jnp.mean(W_samples))
    stdW = float(jnp.std(W_samples))
    semW = float(stdW / np.sqrt(max(int(n_particles), 1)))

    deltaF_eq = free_energy_eq(float(b1), float(c1), T, region=None) - free_energy_eq(float(b0), float(c0), T, region=None)

    deltaF_info_full = KB * T * (dkl_final["DKL_full"] - dkl_init["DKL_full"])
    deltaF_macro = KB * T * (dkl_final["DKL_macro"] - dkl_init["DKL_macro"])
    deltaF_intra = KB * T * (dkl_final["DKL_intra_weighted"] - dkl_init["DKL_intra_weighted"])

    deltaF_total = deltaF_eq + deltaF_info_full
    W_diss = meanW - deltaF_total

    eps = estimate_eps(x1, target=target)
    eps_se = float(np.sqrt(max(eps * (1.0 - eps), 0.0) / max(int(n_particles), 1)))
    pR_init = float(dkl_init["pR_emp"])
    pR_final = float(dkl_final["pR_emp"])

    # Jarzynski diagnostic is only standard for full-canonical initial ensembles.
    # Create uses conditional initial equilibrium (blank0/left), so report N/A.
    jarz = None if op_name == "Create" else float(jarzynski_deltaF(W_samples, T))

    # Landauer logical error formula is meaningful for operations that compress an
    # unknown logical state into a target state (Delete, overwrite-like Update).
    # For Create (known blank0 -> known 1), report N/A to avoid false erasure framing.
    deltaF_logical_eps = None if op_name == "Create" else landauer_logical_deltaF(eps, T)

    note = {
        "Create": "write-like / known-to-known",
        "Update": "overwrite-like / unknown-to-target",
        "Delete": "landauer-like / unknown-to-blank0",
    }[op_name]

    return {
        "op": op_name,
        "note": note,
        "meanW": meanW,
        "stdW": stdW,
        "semW": semW,
        "deltaF_eq": float(deltaF_eq),
        "deltaF_info_full": float(deltaF_info_full),
        "deltaF_macro": float(deltaF_macro),
        "deltaF_intra": float(deltaF_intra),
        "deltaF_total": float(deltaF_total),
        "W_diss": float(W_diss),

        "DKL_init_full": float(dkl_init["DKL_full"]),
        "DKL_final_full": float(dkl_final["DKL_full"]),
        "DKL_init_macro": float(dkl_init["DKL_macro"]),
        "DKL_final_macro": float(dkl_final["DKL_macro"]),
        "DKL_init_intra": float(dkl_init["DKL_intra_weighted"]),
        "DKL_final_intra": float(dkl_final["DKL_intra_weighted"]),
        "recon_err_init": float(dkl_init["recon_err"]),
        "recon_err_final": float(dkl_final["recon_err"]),

        "eps": float(eps),
        "eps_se": eps_se,
        "pR_init": float(pR_init),
        "pR_final": float(pR_final),
        "deltaF_logical_eps": None if deltaF_logical_eps is None else float(deltaF_logical_eps),
        "jarzynski_diag_deltaF": jarz,

        "c_mag": float(c_mag),
        "relax_steps": int(relax_steps),
        "n_lambda_steps": int(n_lambda_steps),
        "n_particles": int(n_particles),
        "target": target,
        "dkl_bins": int(dkl_bins),
        "b0": float(b0),
        "c0": float(c0),
        "b1": float(b1),
        "c1": float(c1),
        "init_region": init_region,
        "init_x": x0,
        "final_x": x1,
    }

# ============================================================
# SECTION 7: Read operation
# ============================================================

@jit
def step_coupled_single(x, y, key, b, c, g, k_meter, T):
    key1, key2 = random.split(key)

    fx = force(x, b, c) + g * y
    fy = -k_meter * y + g * x

    nx = random.normal(key1)
    ny = random.normal(key2)

    x_new = x + (fx / GAMMA) * DT + jnp.sqrt(2.0 * KB * T * DT / GAMMA) * nx
    y_new = y + (fy / GAMMA) * DT + jnp.sqrt(2.0 * KB * T * DT / GAMMA) * ny

    return x_new, y_new

step_coupled_batch = vmap(
    lambda x, y, k, b, c, g, km, T: step_coupled_single(x, y, k, b, c, g, km, T),
    in_axes=(0, 0, 0, None, None, None, None, None)
)

def protocol_read(t, g_max=READ_GMAX):
    b = 2.0
    c = 0.0
    if t < 0.2:
        g = g_max * (t / 0.2)
    elif t < 0.8:
        g = g_max
    else:
        g = g_max * (1.0 - (t - 0.8) / 0.2)
    return float(b), float(c), float(g)

def estimate_mutual_information(x_samples, y_samples, n_bins=30):
    x = np.asarray(x_samples, dtype=np.float64)
    y = np.asarray(y_samples, dtype=np.float64)

    x_edges = np.linspace(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5, n_bins + 1)
    y_edges = np.linspace(float(np.min(y)) - 0.5, float(np.max(y)) + 0.5, n_bins + 1)

    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_xy = hist / max(float(np.sum(hist)), 1.0)
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    p_xp_y = p_x[:, None] * p_y[None, :]

    valid = (p_xy > 0.0) & (p_xp_y > 0.0)
    I = np.sum(p_xy[valid] * (np.log(p_xy[valid]) - np.log(p_xp_y[valid])))
    I_bits = I / np.log(2.0)
    return float(max(I, 0.0)), float(max(I_bits, 0.0))

def estimate_measurement_accuracy(x_samples, y_samples):
    x = np.asarray(x_samples)
    y = np.asarray(y_samples)
    true_state = np.sign(x)
    inferred_state = np.sign(y)
    return float(np.mean(true_state == inferred_state))

def run_read_simulation(n_steps, n_particles, T, key, g_max=READ_GMAX, k_meter=READ_KMETER):
    b0, c0, g0 = protocol_read(0.0, g_max=g_max)
    assert abs(g0) < 1e-12

    key, kx, ky = random.split(key, 3)
    x = sample_equilibrium_positions(float(b0), float(c0), T, n_particles, kx, region=None)

    sigma_y = np.sqrt(KB * T / k_meter)
    y = random.normal(ky, shape=(n_particles,)) * sigma_y

    total_work_sys = jnp.zeros(n_particles)
    total_work_cpl = jnp.zeros(n_particles)
    traj = []

    params = [protocol_read(i / n_steps, g_max=g_max) for i in range(n_steps + 1)]
    peak_index = int(round(0.8 * n_steps))

    x_peak = None
    y_peak = None

    for i in range(n_steps):
        b_old, c_old, g_old = params[i]
        b_new, c_new, g_new = params[i + 1]

        w_sys = work_step_batch(x, b_old, c_old, b_new, c_new)
        w_cpl = -(g_new - g_old) * x * y

        total_work_sys = total_work_sys + w_sys
        total_work_cpl = total_work_cpl + w_cpl

        traj.append(float(jnp.mean(total_work_sys + total_work_cpl)))

        key, sub = random.split(key)
        keys = random.split(sub, n_particles)
        x, y = step_coupled_batch(x, y, keys, b_new, c_new, g_new, k_meter, T)

        # End of measurement hold / just before substantial decoupling.
        if (i + 1) == peak_index:
            x_peak = x
            y_peak = y

    if x_peak is None:
        x_peak = x
        y_peak = y

    total_work = total_work_sys + total_work_cpl

    meanW = float(jnp.mean(total_work))
    stdW = float(jnp.std(total_work))
    semW = float(stdW / np.sqrt(max(int(n_particles), 1)))
    deltaF_theory = 0.0
    W_diss = meanW - deltaF_theory

    deltaF_jarz = float(jarzynski_deltaF(total_work, T))

    MI_nats_peak, MI_bits_peak = estimate_mutual_information(x_peak, y_peak)
    acc_peak = estimate_measurement_accuracy(x_peak, y_peak)

    MI_nats_final, MI_bits_final = estimate_mutual_information(x, y)
    acc_final = estimate_measurement_accuracy(x, y)

    # Main Read metric uses measurement-time correlation; final values are stored separately.
    MI_nats = MI_nats_peak
    MI_bits = MI_bits_peak
    acc = acc_peak

    sagawa_bound = -KB * T * MI_nats
    sagawa_ok = (W_diss >= sagawa_bound - 1e-2)

    return {
        "op": "Read",
        "note": "measurement / correlation generation",
        "meanW": meanW,
        "stdW": stdW,
        "semW": semW,
        "deltaF_eq": 0.0,
        "deltaF_info_full": 0.0,
        "deltaF_macro": None,
        "deltaF_intra": None,
        "deltaF_total": 0.0,
        "W_diss": W_diss,
        "deltaF_jarz": deltaF_jarz,

        "MI_nats": MI_nats,
        "MI_bits": MI_bits,
        "accuracy": acc,
        "measurement_error": 1.0 - acc,

        "MI_nats_peak": MI_nats_peak,
        "MI_bits_peak": MI_bits_peak,
        "accuracy_peak": acc_peak,
        "measurement_error_peak": 1.0 - acc_peak,

        "MI_nats_final": MI_nats_final,
        "MI_bits_final": MI_bits_final,
        "accuracy_final": acc_final,
        "measurement_error_final": 1.0 - acc_final,
        "MI_decay_bits": MI_bits_peak - MI_bits_final,

        "sagawa_bound": sagawa_bound,
        "sagawa_ok": bool(sagawa_ok),
        "final_x": x,
        "final_y": y,
        "peak_x": x_peak,
        "peak_y": y_peak,
        "trajectory": np.array(traj, dtype=np.float64),
    }

# ============================================================
# SECTION 8: Relax scan
# ============================================================

def run_relax_scan(relax_list, *, T, key):
    rows = []
    write_calib_map = {}

    print()
    print("=" * 72)
    print("RELAX SCAN: varying relax_steps per lambda-step")
    print(f"relax_steps: {relax_list}")
    print("Write calibration: analytic start + endpoint micro-correction")
    print("=" * 72)

    c_analytic = calibrate_write_cmag_analytic(B_LO, T, eps_target=WRITE_EPS_TARGET, verbose=False)

    for relax in relax_list:
        relax = int(relax)

        print()
        print(f"[relax={relax}] Write calibration ...")
        print(f"  analytic start c={c_analytic:.6f}")

        key, sub = random.split(key)
        info = refine_write_cmag_endpoint(
            c_init=c_analytic,
            relax_steps=relax,
            n_lambda_steps=N_LAMBDA_STEPS,
            T=T,
            base_key=sub,
            eps_target=WRITE_EPS_TARGET,
            tol=WRITE_ENDPOINT_TOL,
            n_particles=N_CALIB,
            n_reps=WRITE_ENDPOINT_REPS,
            max_iters=WRITE_ENDPOINT_MAX_ITERS,
            growth=WRITE_ENDPOINT_GROWTH,
            verbose=True,
        )
        write_calib_map[relax] = info
        c_write = info["c_refined"]

        print(f"  -> refined c_write={c_write:.6f}  status={info['status']}  eps~{info['eps_mean']:.6f}+-{info['eps_std']:.6f}")

        for op in ["Create", "Update", "Delete"]:
            key, sub = random.split(key)
            c_use = c_write if op == "Create" else (c_write if op == "Update" else C_MAG_DEFAULT)

            r = run_crud_operation(
                op,
                b_hi=B_HI,
                b_lo=B_LO,
                c_mag=c_use,
                unbias_frac=UNBIAS_FRAC,
                end_mode=END_MODE,
                n_lambda_steps=N_LAMBDA_STEPS,
                relax_steps=relax,
                n_particles=N_SCAN,
                T=T,
                key=sub,
            )
            rows.append(r)

            print(f"  {op:8} c_mag={c_use:8.4f}  <W>={r['meanW']:+.4f}  "
                  f"DeltaF_total={r['deltaF_total']:+.4f}  W_diss={r['W_diss']:+.4f}  eps={r['eps']:.4f}")

    return rows, write_calib_map



# ============================================================
# SECTION 8.5: DKL bin sensitivity scan
# ============================================================

def run_dkl_bin_sensitivity(core_results, bins_list=DKL_BINS_SENSITIVITY, T=T0):
    """
    Recompute KL accounting for stored initial/final samples at multiple bin counts.

    Purpose:
      Verify that the central state-function quantity kT Delta D_KL(full)
      is not an artifact of a single histogram resolution.
    """
    rows = []
    print()
    print("=" * 72)
    print("DKL BIN SENSITIVITY SCAN")
    print(f"bins_list: {bins_list}")
    print("=" * 72)

    for r in core_results:
        op = r["op"]
        print(f"\n[{op}]")
        for nb in bins_list:
            d0 = compute_dkl_full_macro_intra(r["init_x"], r["b0"], r["c0"], T, n_bins=int(nb))
            d1 = compute_dkl_full_macro_intra(r["final_x"], r["b1"], r["c1"], T, n_bins=int(nb))

            deltaF_info_full = KB * T * (d1["DKL_full"] - d0["DKL_full"])
            deltaF_macro = KB * T * (d1["DKL_macro"] - d0["DKL_macro"])
            deltaF_intra = KB * T * (d1["DKL_intra_weighted"] - d0["DKL_intra_weighted"])
            deltaF_total = r["deltaF_eq"] + deltaF_info_full
            W_diss = r["meanW"] - deltaF_total

            row = {
                "op": op,
                "n_bins": int(nb),
                "meanW": r["meanW"],
                "deltaF_eq": r["deltaF_eq"],
                "deltaF_info_full": float(deltaF_info_full),
                "deltaF_macro": float(deltaF_macro),
                "deltaF_intra": float(deltaF_intra),
                "deltaF_total": float(deltaF_total),
                "W_diss": float(W_diss),
                "DKL_init_full": d0["DKL_full"],
                "DKL_final_full": d1["DKL_full"],
                "recon_err_init": d0["recon_err"],
                "recon_err_final": d1["recon_err"],
            }
            rows.append(row)
            print(f"  bins={int(nb):4d}  DeltaF_total={deltaF_total:+.6f}  "
                  f"W_diss={W_diss:+.6f}  recon=({d0['recon_err']:+.2e},{d1['recon_err']:+.2e})")
    return rows


# ============================================================
# SECTION 9: CSV export
# ============================================================

def export_core_csv(core_results, read_result, write_calib, prefix=PFX):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{prefix}_core_summary_{ts}.csv"

    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "op", "note", "meanW", "stdW", "semW",
            "deltaF_eq", "deltaF_info_full", "deltaF_macro", "deltaF_intra",
            "deltaF_total", "W_diss",
            "eps_or_meas_error", "eps_se", "pR_init", "pR_final",
            "deltaF_logical_eps", "jarzynski_diag_deltaF",
            "DKL_init_full", "DKL_final_full",
            "DKL_init_macro", "DKL_final_macro",
            "DKL_init_intra", "DKL_final_intra",
            "recon_err_init", "recon_err_final",
            "MI_bits_peak", "MI_bits_final", "MI_decay_bits",
            "accuracy_peak", "accuracy_final",
            "sagawa_bound", "sagawa_ok",
            "c_mag", "write_c_analytic", "write_c_refined", "write_calib_status"
        ])

        for r in core_results:
            w.writerow([
                r["op"], r["note"], r["meanW"], r["stdW"], r["semW"],
                r["deltaF_eq"], r["deltaF_info_full"],
                r["deltaF_macro"], r["deltaF_intra"],
                r["deltaF_total"], r["W_diss"],
                r["eps"], r["eps_se"], r["pR_init"], r["pR_final"],
                r["deltaF_logical_eps"], r["jarzynski_diag_deltaF"],
                r["DKL_init_full"], r["DKL_final_full"],
                r["DKL_init_macro"], r["DKL_final_macro"],
                r["DKL_init_intra"], r["DKL_final_intra"],
                r["recon_err_init"], r["recon_err_final"],
                None, None, None, None, None, None, None,
                r["c_mag"], write_calib["c_analytic"], write_calib["c_refined"], write_calib["status"]
            ])

        rr = read_result
        w.writerow([
            rr["op"], rr["note"], rr["meanW"], rr["stdW"], rr["semW"],
            rr["deltaF_eq"], rr["deltaF_info_full"],
            rr["deltaF_macro"], rr["deltaF_intra"],
            rr["deltaF_total"], rr["W_diss"],
            rr["measurement_error"], None, None, None,
            None, rr["deltaF_jarz"],
            None, None, None, None, None, None, None, None,
            rr["MI_bits_peak"], rr["MI_bits_final"], rr["MI_decay_bits"],
            rr["accuracy_peak"], rr["accuracy_final"],
            rr["sagawa_bound"], rr["sagawa_ok"],
            None, write_calib["c_analytic"], write_calib["c_refined"], write_calib["status"]
        ])

    print(f"Exported: {fn}")
    return fn

def export_relax_csv(scan_rows, calib_map, prefix=PFX):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{prefix}_relax_scan_{ts}.csv"

    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "relax_steps", "op", "c_mag_used",
            "write_c_analytic", "write_c_refined", "write_status", "write_eps_mean", "write_eps_std",
            "meanW", "deltaF_total", "W_diss", "deltaF_info_full", "deltaF_macro", "deltaF_intra",
            "eps", "pR_init", "pR_final"
        ])

        for r in scan_rows:
            info = calib_map[int(r["relax_steps"])]
            w.writerow([
                r["relax_steps"], r["op"], r["c_mag"],
                info["c_analytic"], info["c_refined"], info["status"], info["eps_mean"], info["eps_std"],
                r["meanW"], r["deltaF_total"], r["W_diss"], r["deltaF_info_full"],
                r["deltaF_macro"], r["deltaF_intra"], r["eps"], r["pR_init"], r["pR_final"]
            ])

    print(f"Exported: {fn}")
    return fn



def export_dkl_sensitivity_csv(sensitivity_rows, prefix=PFX):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"{prefix}_dkl_bin_sensitivity_{ts}.csv"

    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "op", "n_bins", "meanW", "deltaF_eq",
            "deltaF_info_full", "deltaF_macro", "deltaF_intra",
            "deltaF_total", "W_diss",
            "DKL_init_full", "DKL_final_full",
            "recon_err_init", "recon_err_final",
        ])
        for r in sensitivity_rows:
            w.writerow([
                r["op"], r["n_bins"], r["meanW"], r["deltaF_eq"],
                r["deltaF_info_full"], r["deltaF_macro"], r["deltaF_intra"],
                r["deltaF_total"], r["W_diss"],
                r["DKL_init_full"], r["DKL_final_full"],
                r["recon_err_init"], r["recon_err_final"],
            ])

    print(f"Exported: {fn}")
    return fn


# ============================================================
# SECTION 10: Figures
# ============================================================

def plot_final_integrated_figure(core_results, read_result, scan_rows, write_calib, prefix=PFX):
    core_map = {r["op"]: r for r in core_results}

    ops = ["Create", "Read", "Update", "Delete"]
    meanW = [core_map["Create"]["meanW"], read_result["meanW"], core_map["Update"]["meanW"], core_map["Delete"]["meanW"]]
    dFtot = [core_map["Create"]["deltaF_total"], read_result["deltaF_total"], core_map["Update"]["deltaF_total"], core_map["Delete"]["deltaF_total"]]
    Wdiss = [core_map["Create"]["W_diss"], read_result["W_diss"], core_map["Update"]["W_diss"], core_map["Delete"]["W_diss"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # A: CRUD decomposition
    ax = axes[0, 0]
    x = np.arange(len(ops))
    width = 0.25
    ax.bar(x - width, meanW, width, label="<W>")
    ax.bar(x, dFtot, width, label="DeltaF_total")
    ax.bar(x + width, Wdiss, width, label="W_diss")
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel("Energy")
    ax.set_title("CRUD thermodynamic decomposition")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    # B: macro / intra decomposition for C/U/D
    ax = axes[0, 1]
    cud = ["Create", "Update", "Delete"]
    macro = [core_map[o]["deltaF_macro"] for o in cud]
    intra = [core_map[o]["deltaF_intra"] for o in cud]
    xx = np.arange(len(cud))
    ax.bar(xx, macro, label="DeltaF_macro")
    ax.bar(xx, intra, bottom=macro, label="DeltaF_intra")
    ax.axhline(KB * T0 * np.log(2.0), color="gray", linestyle="--", linewidth=1.0, label="kT ln2")
    ax.axhline(0.0, color="k", linewidth=0.7)
    ax.set_xticks(xx)
    ax.set_xticklabels(cud)
    ax.set_ylabel("Energy")
    ax.set_title("Information free-energy decomposition")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    # C: final task error
    ax = axes[0, 2]
    err_ops = ["Create", "Read", "Update", "Delete"]
    errs = [
        core_map["Create"]["eps"],
        read_result["measurement_error"],
        core_map["Update"]["eps"],
        core_map["Delete"]["eps"],
    ]
    ax.bar(np.arange(len(err_ops)), errs)
    ax.axhline(WRITE_EPS_TARGET, color="gray", linestyle="--", linewidth=1.0, label="write eps target")
    ax.set_xticks(np.arange(len(err_ops)))
    ax.set_xticklabels(err_ops)
    ax.set_ylabel("Error")
    ax.set_title("Final logical / measurement error")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8)

    # D: relax scan W_diss
    ax = axes[1, 0]
    for op in ["Create", "Update", "Delete"]:
        rows = [r for r in scan_rows if r["op"] == op]
        rows = sorted(rows, key=lambda z: int(z["relax_steps"]))
        rel = [int(r["relax_steps"]) for r in rows]
        y = [max(abs(float(r["W_diss"])), 1e-12) for r in rows]
        ax.plot(rel, y, "o-", label=op)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relax_steps (log)")
    ax.set_ylabel("|W_diss| (log)")
    ax.set_title("Quasi-static trend")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # E: Write calibration vs relax
    ax = axes[1, 1]
    relax_values = sorted(list(set(int(r["relax_steps"]) for r in scan_rows)))
    c_analytic = [write_calib["c_analytic"]] * len(relax_values)
    c_refined = []
    eps_refined = []
    # Need per-relax map from scan rows: recover from global per-relax calibration not available here
    # For this integrated fig, infer Create c_mag at each relax (same as Update)
    for rv in relax_values:
        rows = [r for r in scan_rows if r["op"] == "Create" and int(r["relax_steps"]) == rv]
        c_refined.append(float(rows[0]["c_mag"]))
        eps_refined.append(float(rows[0]["eps"]))
    ax.plot(relax_values, c_analytic, "s--", label="analytic c_write")
    ax.plot(relax_values, c_refined, "o-", label="refined c_write")
    ax.set_xscale("log")
    ax.set_xlabel("relax_steps (log)")
    ax.set_ylabel("c_write")
    ax.set_title("Write calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(relax_values, eps_refined, "x-", alpha=0.7)
    ax2.set_ylabel("Create eps")

    # F: Read panel
    ax = axes[1, 2]
    labels = ["W_diss", "-kT I"]
    vals = [read_result["W_diss"], read_result["sagawa_bound"]]
    ax.bar(labels, vals, alpha=0.8)
    ax.axhline(0.0, color="k", linewidth=0.7)
    title = f"Read: MI_peak={read_result['MI_bits_peak']:.3f} bits -> MI_final={read_result['MI_bits_final']:.3f} bits"
    ax.set_title(title)
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(f"{prefix}_final_integrated_fig.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{prefix}_final_integrated_fig.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {prefix}_final_integrated_fig.png/pdf")

def plot_read_scatter(read_result, prefix=PFX):
    x = np.array(read_result["peak_x"])
    y = np.array(read_result["peak_y"])

    n_plot = min(2500, len(x))
    plt.figure(figsize=(6, 5))
    plt.scatter(x[:n_plot], y[:n_plot], s=5, alpha=0.3)
    plt.axhline(0.0, color="k", linewidth=0.6)
    plt.axvline(0.0, color="k", linewidth=0.6)
    plt.xlabel("system x")
    plt.ylabel("meter y")
    plt.title(f"Read scatter at measurement peak: MI={read_result['MI_bits_peak']:.3f} bits, Acc={read_result['accuracy_peak']:.1%}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{prefix}_read_scatter.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{prefix}_read_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {prefix}_read_scatter.png/pdf")



def plot_dkl_bin_sensitivity(sensitivity_rows, prefix=PFX):
    if not sensitivity_rows:
        return

    ops = sorted(set(r["op"] for r in sensitivity_rows))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for op in ops:
        rows = sorted([r for r in sensitivity_rows if r["op"] == op], key=lambda z: z["n_bins"])
        ax.plot([r["n_bins"] for r in rows], [r["deltaF_total"] for r in rows], "o-", label=op)
    ax.set_xlabel("DKL histogram bins")
    ax.set_ylabel("DeltaF_total")
    ax.set_title("DKL bin sensitivity: DeltaF_total")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for op in ops:
        rows = sorted([r for r in sensitivity_rows if r["op"] == op], key=lambda z: z["n_bins"])
        ax.plot([r["n_bins"] for r in rows], [r["W_diss"] for r in rows], "o-", label=op)
    ax.set_xlabel("DKL histogram bins")
    ax.set_ylabel("W_diss")
    ax.set_title("DKL bin sensitivity: W_diss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{prefix}_dkl_bin_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{prefix}_dkl_bin_sensitivity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {prefix}_dkl_bin_sensitivity.png/pdf")


# ============================================================
# SECTION 11: MAIN
# ============================================================

def main():
    key = random.PRNGKey(42)
    landauer = KB * T0 * np.log(2.0)

    wall_start = time.time()
    gpu_logger = GPUPowerLogger(interval_sec=GPU_POWER_INTERVAL_SEC).start() if ENABLE_GPU_POWER_LOG else None

    print()
    print("=" * 72)
    print("CORE RUNS")
    print("=" * 72)

    # Write calibration from Create endpoint
    print()
    print("[Core] Analytic calibration for Write ...")
    c_write_analytic = calibrate_write_cmag_analytic(B_LO, T0, eps_target=WRITE_EPS_TARGET, verbose=True)
    pR_ana = p_right_eq(B_LO, c_write_analytic, T0, n_grid=WRITE_CALIB_GRID)
    print(f"[Core] Write analytic c_mag={c_write_analytic:.6f}  analytic eps={1.0 - pR_ana:.6f}")

    print()
    print("[Core] Endpoint micro-correction for Write ...")
    key, sub = random.split(key)
    write_calib = refine_write_cmag_endpoint(
        c_init=c_write_analytic,
        relax_steps=RELAX_CORE,
        n_lambda_steps=N_LAMBDA_STEPS,
        T=T0,
        base_key=sub,
        eps_target=WRITE_EPS_TARGET,
        tol=WRITE_ENDPOINT_TOL,
        n_particles=N_CALIB,
        n_reps=WRITE_ENDPOINT_REPS,
        max_iters=WRITE_ENDPOINT_MAX_ITERS,
        growth=WRITE_ENDPOINT_GROWTH,
        verbose=True,
    )
    c_write = write_calib["c_refined"]
    print(f"[Core] Refined Write c_mag={c_write:.6f}  status={write_calib['status']}  eps~{write_calib['eps_mean']:.6f}+-{write_calib['eps_std']:.6f}")

    core_results = []
    for op in ["Create", "Update", "Delete"]:
        key, sub = random.split(key)
        c_use = c_write if op in ("Create", "Update") else C_MAG_DEFAULT
        r = run_crud_operation(
            op,
            b_hi=B_HI,
            b_lo=B_LO,
            c_mag=c_use,
            unbias_frac=UNBIAS_FRAC,
            end_mode=END_MODE,
            n_lambda_steps=N_LAMBDA_STEPS,
            relax_steps=RELAX_CORE,
            n_particles=N_CORE,
            T=T0,
            key=sub,
        )
        core_results.append(r)

        print()
        print(f"[{op}]")
        print(f"  <W>={r['meanW']:+.6f}  DeltaF_eq={r['deltaF_eq']:+.6f}")
        print(f"  DeltaF_info(full)={r['deltaF_info_full']:+.6f}  (macro={r['deltaF_macro']:+.6f}, intra={r['deltaF_intra']:+.6f})")
        print(f"  DeltaF_total={r['deltaF_total']:+.6f}  W_diss={r['W_diss']:+.6f}")
        print(f"  eps={r['eps']:.4f}±{r['eps_se']:.4f}  pR_init={r['pR_init']:.3f} -> pR_final={r['pR_final']:.3f}")
        if r['deltaF_logical_eps'] is None:
            print("  Landauer-logical from eps = N/A (Create is known-to-known, not erasure)")
        else:
            print(f"  Landauer-logical from eps = {r['deltaF_logical_eps']:+.6f}")
        if r['jarzynski_diag_deltaF'] is None:
            print("  Jarzynski diagnostic = N/A (conditional initial ensemble)")
        else:
            print(f"  Jarzynski diagnostic = {r['jarzynski_diag_deltaF']:+.6f}")

    print()
    print("[Read]")
    key, sub = random.split(key)
    read_result = run_read_simulation(
        n_steps=READ_STEPS,
        n_particles=READ_NPART,
        T=T0,
        key=sub,
        g_max=READ_GMAX,
        k_meter=READ_KMETER,
    )
    print(f"  <W>={read_result['meanW']:+.6f}  DeltaF_theory=+0.000000  W_diss={read_result['W_diss']:+.6f}")
    print(f"  MI_peak={read_result['MI_bits_peak']:.4f} bits  Acc_peak={read_result['accuracy_peak']:.2%}  err_peak={read_result['measurement_error_peak']:.4f}")
    print(f"  MI_final={read_result['MI_bits_final']:.4f} bits  Acc_final={read_result['accuracy_final']:.2%}  MI_decay={read_result['MI_decay_bits']:+.4f} bits")
    print(f"  Sagawa bound={read_result['sagawa_bound']:+.6f}  satisfied={read_result['sagawa_ok']}")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Landauer kT ln2 = {landauer:.6f}")
    print("-" * 104)
    print(f"{'Op':10} {'<W>':>10} {'DeltaF_total':>14} {'W_diss':>12} {'Error':>10} {'Note':>20}")
    print("-" * 104)

    core_map = {r["op"]: r for r in core_results}
    for op in ["Create", "Read", "Update", "Delete"]:
        if op == "Read":
            print(f"{op:10} {read_result['meanW']:+10.4f} {read_result['deltaF_total']:+14.4f} {read_result['W_diss']:+12.4f} "
                  f"{read_result['measurement_error']:10.4f} {'measurement':>20}")
        else:
            rr = core_map[op]
            note = rr.get("note", "")
            print(f"{op:10} {rr['meanW']:+10.4f} {rr['deltaF_total']:+14.4f} {rr['W_diss']:+12.4f} {rr['eps']:10.4f} {note:>20}")

    # DKL bin sensitivity scan
    sensitivity_rows = run_dkl_bin_sensitivity(core_results, bins_list=DKL_BINS_SENSITIVITY, T=T0)

    # Relax scan
    scan_rows, write_calib_map = run_relax_scan(RELAX_LIST, T=T0, key=key)

    print()
    print("=" * 72)
    print("EXPORT")
    print("=" * 72)
    export_core_csv(core_results, read_result, write_calib, prefix=PFX)
    export_relax_csv(scan_rows, write_calib_map, prefix=PFX)
    export_dkl_sensitivity_csv(sensitivity_rows, prefix=PFX)

    print()
    print("=" * 72)
    print("PLOTS")
    print("=" * 72)
    plot_final_integrated_figure(core_results, read_result, scan_rows, write_calib, prefix=PFX)
    plot_read_scatter(read_result, prefix=PFX)
    plot_dkl_bin_sensitivity(sensitivity_rows, prefix=PFX)

    print()
    print("=" * 72)
    print("SIMULATION COMPLETE")
    gpu_summary = None
    if gpu_logger is not None:
        gpu_summary = gpu_logger.stop()
        gpu_summary["wall_elapsed_s"] = float(time.time() - wall_start)
        gpu_logger.export_csv(prefix=PFX)

        config = {
            "VERSION": VERSION,
            "KB": KB,
            "GAMMA": GAMMA,
            "DT": DT,
            "T0": T0,
            "N_CORE": N_CORE,
            "N_SCAN": N_SCAN,
            "N_CALIB": N_CALIB,
            "N_LAMBDA_STEPS": N_LAMBDA_STEPS,
            "RELAX_CORE": RELAX_CORE,
            "RELAX_LIST": RELAX_LIST,
            "B_HI": B_HI,
            "B_LO": B_LO,
            "C_MAG_DEFAULT": C_MAG_DEFAULT,
            "UNBIAS_FRAC": UNBIAS_FRAC,
            "END_MODE": END_MODE,
            "WRITE_EPS_TARGET": WRITE_EPS_TARGET,
            "READ_GMAX": READ_GMAX,
            "READ_KMETER": READ_KMETER,
            "DKL_BINS": DKL_BINS,
            "DKL_BINS_SENSITIVITY": DKL_BINS_SENSITIVITY,
            "ENABLE_GPU_POWER_LOG": ENABLE_GPU_POWER_LOG,
            "GPU_POWER_INTERVAL_SEC": GPU_POWER_INTERVAL_SEC,
        }
        export_config_and_gpu_summary(config, gpu_summary, prefix=PFX)

        print()
        print("=" * 72)
        print("GPU HARDWARE DIAGNOSTIC")
        print("=" * 72)
        print(gpu_summary)
        print("Note: GPU energy is accelerator-side only, not full server-level energy.")

    print("=" * 72)

    return core_results, read_result, scan_rows, write_calib_map, sensitivity_rows, gpu_summary


if __name__ == "__main__":
    results = main()
