"""Helpers shared by results.ipynb and demo.ipynb.

Nothing in sim.py imports this module. The estimators in sim.py never see a true
state; the true Bloch vector only appears here in (a) the state builders that make
test states, (b) HiddenDevice, which keeps it private, and (c) evaluation code that
compares an estimate with the truth AFTER the estimate has been made.
"""
import math
from functools import lru_cache
from statistics import NormalDist

import numpy as np

import sim

SQRT2 = math.sqrt(2.0)
N_VEC = np.array([1 / SQRT2, 0.0, 1 / SQRT2])  # Bloch vector of |T>
T_KET = np.array([math.cos(math.pi / 8), math.sin(math.pi / 8)], dtype=complex)


# ----------------------------------------------------------------------
# Test states (Bloch vectors). Used for simulation and evaluation only.
# ----------------------------------------------------------------------
def target_state():
    """|T> itself (F = 1)."""
    return sim.make_state(*N_VEC)


def on_axis_state(F):
    """State on the target axis with fidelity F (F=1 is |T>, F=0.5 is maximally mixed)."""
    c = 2 * F - 1
    return sim.make_state(c / SQRT2, 0.0, c / SQRT2)


def coherent_state(F):
    """Pure state obtained by rotating |T> about Y; Bloch rotation angle theta = arccos(2F-1)."""
    theta = math.acos(2 * F - 1)
    c, s = math.cos(theta), math.sin(theta)
    return sim.make_state((c + s) / SQRT2, 0.0, (c - s) / SQRT2)


def with_ry_state(ry=0.2):
    """Pure state with a nonzero y component (F = (1 + sqrt(1 - ry^2)) / 2)."""
    xz = math.sqrt(1 - ry ** 2) / SQRT2
    return sim.make_state(xz, ry, xz)


def random_bloch(rng, radius=None):
    """Random Bloch vector; uniform in the ball unless a radius is given."""
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    if radius is None:
        radius = rng.random() ** (1 / 3)
    return sim.make_state(*(radius * v))


def state_zoo():
    """Named test states used in the Level 2 experiments."""
    return {
        "target (F=1)": target_state(),
        "depolarized (F=0.95)": on_axis_state(0.95),
        "coherent rotation (F=0.95)": coherent_state(0.95),
        "with r_y=0.2 (F~0.99)": with_ry_state(0.2),
        "mixed (F=0.80)": on_axis_state(0.80),
    }


# ----------------------------------------------------------------------
# Density-matrix helpers (used to check the Bloch-vector shortcuts)
# ----------------------------------------------------------------------
def bloch_from_rho(rho):
    return np.real(np.array([np.trace(rho @ sim.PAULI[P]) for P in "XYZ"]))


def fidelity_from_rho(rho):
    """<T|rho|T> computed directly from the density matrix."""
    return float(np.real(T_KET.conj() @ rho @ T_KET))


def trace_distance_rho(rho, sigma):
    """0.5 * ||rho - sigma||_1 from eigenvalues of the (Hermitian) difference."""
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(rho - sigma))))


def check_physical(rho, tol=1e-9):
    """Hermitian, trace one, positive semidefinite."""
    herm = np.allclose(rho, rho.conj().T, atol=tol)
    trace_one = abs(np.trace(rho) - 1) < tol
    psd = np.linalg.eigvalsh((rho + rho.conj().T) / 2).min() >= -tol
    return bool(herm and trace_one and psd)


# ----------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def rmse_bias_se(err):
    """Bias, RMSE and their Monte Carlo standard errors from an array of (estimate - truth)."""
    err = np.asarray(err, dtype=float)
    n = len(err)
    bias = float(np.mean(err))
    bias_se = float(np.std(err, ddof=1) / math.sqrt(n))
    sq = err ** 2
    rmse = math.sqrt(float(np.mean(sq)))
    # delta method on the mean of squares (robust to non-Gaussian tails)
    rmse_se = float(np.std(sq, ddof=1) / (2 * rmse * math.sqrt(n))) if rmse > 0 else 0.0
    return bias, bias_se, rmse, rmse_se


# ----------------------------------------------------------------------
# Level 2 theory (standard deviation of the fidelity estimate, ignoring clipping)
# ----------------------------------------------------------------------
def sigma_equal(r, N):
    """Equal X/Z split: Var = [(1-rx^2) + (1-rz^2)] / (4N)."""
    return math.sqrt(((1 - r[0] ** 2) + (1 - r[2] ** 2)) / (4 * N))


def sigma_adaptive(r, N):
    """Neyman allocation N_i ~ sqrt(1-r_i^2) (large-N limit): Var = (sum_i sqrt(1-r_i^2))^2 / (8N)."""
    s = math.sqrt(1 - r[0] ** 2) + math.sqrt(1 - r[2] ** 2)
    return math.sqrt(s ** 2 / (8 * N))


def sigma_random_basis(r, N):
    """Random X/Z per shot: Var = (1 - m^2) / (2N) with m = (rx + rz)/2."""
    m = 0.5 * (r[0] + r[2])
    return math.sqrt((1 - m ** 2) / (2 * N))


# ----------------------------------------------------------------------
# Samplers and a device that hides the true state
# ----------------------------------------------------------------------
def fast_sampler(r, rng, a=0.0, b=1.0):
    """Distribution-equivalent to sim.make_sampler(r, rng, a, b).

    A random-basis shot is one +/-1 outcome with mean a + b * (rx + rz) / 2, so a single
    Bernoulli draw per shot is enough. results.ipynb checks the equivalence statistically.
    """
    r_eff = np.array([0.0, 0.0, 0.5 * (r[0] + r[2])])
    return lambda n: sim.measure(r_eff, "Z", n, rng, a, b)


class HiddenDevice:
    """A black box that holds a secret Bloch vector.

    Estimators may only call .measure(), .measure_reference() and .sample(); .reveal() is for
    evaluation after the estimate has been produced. This mirrors the challenge rule that the
    algorithm must treat the state as unknown.
    """

    def __init__(self, r, rng, a=0.0, b=1.0):
        self.__r = np.asarray(r, dtype=float)
        self._rng = rng
        self._a, self._b = a, b
        self.shots = 0
        self._sampler = sim.make_sampler(self.__r, rng, a, b)

    def measure(self, basis, n):
        self.shots += n
        return sim.measure(self.__r, basis, n, self._rng, self._a, self._b)

    def measure_reference(self, basis, eigenvalue, n):
        """Calibration shots: same (noisy) apparatus, but a known +/-1 eigenstate of `basis`."""
        self.shots += n
        return sim.measure(sim.eigenstate(basis, eigenvalue), basis, n, self._rng, self._a, self._b)

    def sample(self, n):
        """Random-basis (X or Z) outcomes, as needed by sim.certify."""
        self.shots += n
        return self._sampler(n)

    def reveal(self):
        return self.__r.copy()

    def __repr__(self):
        return "HiddenDevice(state hidden)"


def collect_data_device(device, N_total, f_cal):
    """Same shot accounting and data layout as sim.collect_data, but through a HiddenDevice."""
    n_cal = max(1, int(round(f_cal * N_total / 4)))
    n_state = max(1, int(round((1 - f_cal) * N_total / 2)))
    return {
        "state": {P: device.measure(P, n_state) for P in "XZ"},
        "cal": {(P, s): device.measure_reference(P, s, n_cal) for P in "XZ" for s in (+1, -1)},
    }


# ----------------------------------------------------------------------
# Level 4 helpers
# ----------------------------------------------------------------------
def certify_trace(sampler, thresh=0.95, delta=0.05, batch=50, max_shots=20000):
    """Same logic as sim.certify, but also returns the history [(N, F_hat, lo_run, hi_run), ...].

    Returns (decision, (lo_run, hi_run), N, history). results.ipynb asserts that the first three
    outputs match sim.certify under identical random seeds, so a drift between the two is caught.
    """
    s_sum, N, k = 0.0, 0, 0
    lo_run, hi_run = 0.0, 1.0
    history = []
    while N < max_shots:
        k += 1
        shots = min(batch * 2 ** (k - 1) - N, max_shots - N)
        s = np.asarray(sampler(shots), dtype=float)
        s_sum += float(np.sum(s))
        N += shots
        delta_k = delta * 6 / (math.pi ** 2 * k ** 2)
        t = math.sqrt(2 * math.log(1 / delta_k) / N)
        F_hat = 0.5 + (s_sum / N) / SQRT2
        lo, hi = F_hat - t / SQRT2, F_hat + t / SQRT2
        lo_run, hi_run = max(lo_run, lo), min(hi_run, hi)
        history.append((N, F_hat, lo_run, hi_run))
        if lo_run >= thresh:
            return "PASS", (lo_run, hi_run), N, np.array(history)
        if hi_run < thresh:
            return "FAIL", (lo_run, hi_run), N, np.array(history)
    return "INCONCLUSIVE", (lo_run, hi_run), N, np.array(history)


def predicted_hoeffding_shots(F, thresh=0.95, delta=0.05, batch=50, max_shots=20000):
    """Noise-free prediction of the stopping point of sim.certify.

    With F_hat = F, the interval clears the threshold at look k once N_k >= ln(1/delta_k) / gap^2,
    where gap = |F - thresh|, N_k = batch * 2^(k-1) and delta_k = 6 delta / (pi^2 k^2).
    """
    gap = abs(F - thresh)
    if gap == 0:
        return max_shots
    k = 1
    while True:
        N = batch * 2 ** (k - 1)
        if N >= max_shots:
            return max_shots
        delta_k = delta * 6 / (math.pi ** 2 * k ** 2)
        if N >= math.log(1 / delta_k) / gap ** 2:
            return N
        k += 1


@lru_cache(maxsize=None)
def _z_schedule(n_looks, delta, spend):
    k = np.arange(1, n_looks + 1)
    alpha = delta * 6 / (math.pi ** 2 * k ** 2) if spend else np.full(n_looks, delta)
    nd = NormalDist()
    return np.array([nd.inv_cdf(1 - a) for a in alpha])


def normal_rule(s, thresh=0.95, delta=0.05, batch=50, spend=False):
    """Sequential test using one-sided normal-approximation intervals, peeking every `batch` shots.

    s: array of +/-1 outcomes (length = max_shots; only the prefix up to the stopping look is "used").
    spend=False: naive rule (fixed delta at every look, current interval only).
    spend=True:  alpha spending (delta_k = 6 delta / (pi^2 k^2)) with the running intersection.
    Returns (decision, shots used).
    """
    s = np.asarray(s)
    n_looks = len(s) // batch
    N = batch * np.arange(1, n_looks + 1)
    mean = np.cumsum(s)[N - 1] / N
    var = np.maximum(1 - mean ** 2, 1e-12) * N / (N - 1)  # sample variance of +/-1 outcomes
    half = _z_schedule(n_looks, delta, spend) * np.sqrt(var / N) / SQRT2
    F_hat = 0.5 + mean / SQRT2
    lo, hi = F_hat - half, F_hat + half
    if spend:
        lo, hi = np.maximum.accumulate(lo), np.minimum.accumulate(hi)
    p = np.flatnonzero(lo >= thresh)
    f = np.flatnonzero(hi < thresh)
    ip = p[0] if p.size else n_looks
    jf = f[0] if f.size else n_looks
    if ip == n_looks and jf == n_looks:
        return "INCONCLUSIVE", int(N[-1])
    return ("PASS", int(N[ip])) if ip < jf else ("FAIL", int(N[jf]))


RULE_NAMES = ("naive normal", "alpha-spending normal", "alpha-spending Hoeffding")


def optional_stopping_experiment(F_values, trials, rng, thresh=0.95, delta=0.05,
                                 batch=50, max_shots=20000):
    """False-PASS rate and mean shots for three stopping rules on states with F < thresh.

    The two normal rules see the SAME simulated shots (paired comparison). The Hoeffding rule is
    sim.certify run on its own fresh shots. Returns {rule: {"passes": [...], "mean_shots": [...]}, "trials": n}.
    """
    out = {name: {"passes": [], "mean_shots": []} for name in RULE_NAMES}
    out["trials"] = trials
    for F in F_values:
        r = on_axis_state(F)
        sampler = fast_sampler(r, rng)
        passes = dict.fromkeys(RULE_NAMES, 0)
        shots = {name: [] for name in RULE_NAMES}
        for _ in range(trials):
            s = sampler(max_shots)
            for name, spend in (("naive normal", False), ("alpha-spending normal", True)):
                decision, n_used = normal_rule(s, thresh, delta, batch, spend)
                passes[name] += decision == "PASS"
                shots[name].append(n_used)
            decision, _, n_used = sim.certify(fast_sampler(r, rng), thresh=thresh, delta=delta,
                                              batch=batch, max_shots=max_shots)
            passes["alpha-spending Hoeffding"] += decision == "PASS"
            shots["alpha-spending Hoeffding"].append(n_used)
        for name in RULE_NAMES:
            out[name]["passes"].append(passes[name])
            out[name]["mean_shots"].append(float(np.mean(shots[name])))
    return out


# ----------------------------------------------------------------------
# Plot helper: the X-Z plane of the Bloch ball
# ----------------------------------------------------------------------
def plot_xz_disk(ax, points=None, thresh=0.95):
    """Draw the X-Z plane (r_x horizontal, r_z vertical) with the target direction and the F >= thresh region.

    F depends only on r.n with n = (1, 0, 1)/sqrt(2), so the F = thresh contour is a straight line
    perpendicular to n. r_y never matters, which is why the y axis is not drawn.
    points: {label: Bloch vector} to mark in the plane.
    """
    phi = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(phi), np.sin(phi), color="black", lw=1)
    c = 2 * thresh - 1  # r.n on the threshold line
    alpha = math.acos(c)
    arc = np.linspace(math.pi / 4 - alpha, math.pi / 4 + alpha, 100)
    ax.fill(np.cos(arc), np.sin(arc), color="tab:green", alpha=0.2, label=f"F >= {thresh}")
    n_perp = np.array([1 / SQRT2, -1 / SQRT2])
    half_len = math.sqrt(1 - c ** 2)
    p0 = c * np.array([1 / SQRT2, 1 / SQRT2])
    line = np.array([p0 - half_len * n_perp, p0 + half_len * n_perp])
    ax.plot(line[:, 0], line[:, 1], "--", color="tab:green", lw=1.5, label=f"F = {thresh}")
    ax.annotate("", xy=(N_VEC[0], N_VEC[2]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(N_VEC[0] + 0.02, N_VEC[2] + 0.04, "|T> direction n", color="gray", fontsize=9)
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (label, r) in enumerate((points or {}).items()):
        ax.plot([r[0]], [r[2]], markers[i % len(markers)], ms=8, label=label)
    ax.axhline(0, color="lightgray", lw=0.8)
    ax.axvline(0, color="lightgray", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("r_x")
    ax.set_ylabel("r_z")
    ax.legend(loc="lower left", fontsize=8)
