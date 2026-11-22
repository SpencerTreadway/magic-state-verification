import math
import numpy as np

# Basis index mapping for Bloch vector components
BASIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
# 2x2 identity matrix
I2 = np.eye(2, dtype=complex)
# Pauli matrices
PAULI = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def make_state(rx, ry, rz):
    """Return a Bloch vector, rejecting vectors outside the Bloch ball."""

    # read components as an array
    r = np.asarray([rx, ry, rz], dtype=float)

    # check that the components are finite and that the vector is inside the Bloch ball
    assert np.all(np.isfinite(r)), "Bloch-vector components must be finite"
    assert np.linalg.norm(r) <= 1.0 + 1e-12, "Bloch vector is outside the Bloch ball"

    return r

def fidelity(r):
    """Return the fidelity of a Bloch vector with the |T> state."""

    # read components as an array
    r = np.asarray(r, dtype=float)

    # return fidelity
    return 0.5 * (1 + (r[0] + r[2]) / math.sqrt(2))

def rho_from_bloch(r):
    """Return a density matrix from a Bloch vector."""

    # read components as an array
    r = np.asarray(r, dtype=float)

    # check that the components are finite and that the vector is inside the Bloch ball
    if np.linalg.norm(r) > 1.0 + 1e-12:
        raise ValueError("Bloch vector is outside the Bloch ball")

    # return the density matrix
    return 0.5 * (
        I2
        + r[0] * PAULI["X"]
        + r[1] * PAULI["Y"]
        + r[2] * PAULI["Z"]
    )

def eigenstate(P, eigenvalue):
    """Return a perfectly prepared +1 or -1 eigenstate as a Bloch vector."""

    # check that the basis and eigenvalue are valid
    if P not in BASIS_INDEX:
        raise ValueError("basis must be one of {'X', 'Y', 'Z'}")
    # check that the eigenvalue is valid
    if eigenvalue not in (+1, -1):
        raise ValueError("eigenvalue must be +1 or -1")

    # intialize a Bloch vector with all zeros
    r = np.zeros(3, dtype=float)
    # set the component corresponding to the basis to +1 or -1
    r[BASIS_INDEX[P]] = 1 if eigenvalue == +1 else -1

    return r

def measure(r, basis, n, rng, a=0.0, b=1.0):
    """Measure a Bloch vector in a given basis."""
    
    # check that the basis is valid
    if basis not in BASIS_INDEX:
        raise ValueError("basis must be one of {'X', 'Y', 'Z'}")
    # check that number of shots is non-negative
    if n < 0:
        raise ValueError("n must be non-negative")
    
    # read components as an array
    r = np.asarray(r, dtype=float)

    # mean of the true measurement outcome in the given basis
    mean_true = r[BASIS_INDEX[basis]]
    # symmetric flip: a=0, b=1-2*eps
    mean_obs = a + b * mean_true
    # probability of measuring +1 in the given basis
    p_plus = (1 + mean_obs) / 2

    # check that the probability is in the valid range [0, 1] with a small tolerance
    if not -1e-12 <= p_plus <= 1 + 1e-12:
        raise ValueError("a + b * <P> must lie in [-1, 1]; check readout parameters a and b")

    # generate n random outcomes according to the probability p_plus
    return np.where(rng.random(n) < p_plus, +1, -1)

def reconstruct(outX, outY, outZ):
    """Reconstruct a Bloch vector from measurement outcomes."""

    # create a Bloch vector from the mean of the measurement outcomes
    r_hat = np.asarray(
        [np.mean(outX), np.mean(outY), np.mean(outZ)],
        dtype=float,
    )

    # project onto Bloch ball
    if np.linalg.norm(r_hat) > 1:
        r_hat /= np.linalg.norm(r_hat)

    # return the density matrix corresponding to the reconstructed Bloch vector
    return rho_from_bloch(r_hat)

def trace_distance(r, r_hat):
    """Bloch-vector trace distance: |r - r_hat| / 2."""

    # read components as arrays
    r = np.asarray(r, dtype=float)
    r_hat = np.asarray(r_hat, dtype=float)

    # check that the Bloch vectors are three-component vectors
    if r.shape != (3,) or r_hat.shape != (3,):
        raise ValueError("r and r_hat must be three-component Bloch vectors")

    # return the trace distance between the two Bloch vectors
    return 0.5 * float(np.linalg.norm(r - r_hat))

def estimate_fidelity(outX, outZ):
    """Estimate fidelity from measurement outcomes in X and Z bases."""

    # create a Bloch vector from the mean of the measurement outcomes
    r_hat_x, r_hat_z = np.mean(outX), np.mean(outZ)
    # compute the fidelity with respect to the |T> state
    F = 0.5 * (1 + (r_hat_x + r_hat_z) / np.sqrt(2))

    # clip the fidelity to the range [0, 1] and return it
    return np.clip(F, 0, 1)

def calibrate(P, n_cal, rng, true_a, true_b):
    """Calibrate the readout parameters a and b for a given basis P using n_cal calibration shots."""

    # check that the n_cal is positive
    if n_cal <= 0:
        raise ValueError("n_cal must be positive")

    # measure the +1 and -1 eigenstates of the given basis P
    m_plus = np.mean(measure(eigenstate(P, +1), P, n_cal, rng, true_a, true_b))
    m_minus = np.mean(measure(eigenstate(P, -1), P, n_cal, rng, true_a, true_b))

    # compute the estimated readout parameters a_hat and b_hat
    a_hat = (m_plus + m_minus) / 2
    b_hat = (m_plus - m_minus) / 2

    # return the estimated readout parameters and the number of calibration shots
    return a_hat, b_hat, n_cal

def corrected_r(outP, a_hat, b_hat):
    """Correct the mean of measurement outcomes in basis P using estimated readout parameters a_hat and b_hat."""

    # check that b_hat is not zero to avoid division by zero
    if abs(b_hat) < 1e-12:
        raise ValueError("estimated readout contrast b_hat is zero")

    # return the corrected mean of the measurement outcomes
    return (np.mean(outP) - a_hat) / b_hat

def fidelity_with_calibration(
    N_total,
    f_cal,
    r=None,
    rng=None,
    true_a=0.0,
    true_b=1.0,
):
    """Estimate fidelity with calibration shots."""

    # check that N_total is positive
    if N_total < 1:
        raise ValueError("N_total must be positive")
    # check that f_cal is in the range [0, 1]
    if not 0 <= f_cal <= 1:
        raise ValueError("f_cal must be in [0, 1]")

    n_cal = int(round(f_cal * N_total / 4)) # per basis
    n_state = int(round((1 - f_cal) * N_total / 2)) # per basis

    # check that the calculated number of calibration and state shots are positive
    if n_cal < 1:
        n_cal = 1
    if n_state < 1:
        n_state = 1
    # check that r is a valid Bloch vector, if provided
    if r is None:
        # use the |T> state as the default Bloch vector
        r = make_state(1 / math.sqrt(2), 0.0, 1 / math.sqrt(2))
    # check that the rng is provided, if not, create a default random number generator
    if rng is None:
        rng = np.random.default_rng()
    
    # perform the calibration and state measurement

    # calibrate the readout parameters for the X and Z bases
    a_x, b_x, _ = calibrate("X", n_cal, rng, true_a, true_b)
    a_z, b_z, _ = calibrate("Z", n_cal, rng, true_a, true_b)

    # measure the state in the X and Z bases
    out_x = measure(r, "X", n_state, rng, true_a, true_b)
    out_z = measure(r, "Z", n_state, rng, true_a, true_b)

    # correct the measurement outcomes using the estimated readout parameters
    r_hat_x = corrected_r(out_x, a_x, b_x)
    r_hat_z = corrected_r(out_z, a_z, b_z)

    # compute the fidelity with respect to the |T> state using the corrected Bloch vector components
    F = 0.5 * (1 + (r_hat_x + r_hat_z) / math.sqrt(2))

    # clip the fidelity to the range [0, 1] and return it
    return float(np.clip(F, 0.0, 1.0))


def bootstrap_fidelity_with_calibration(
    N_total,
    f_cal,
    r=None,
    rng=None,
    true_a=0.0,
    true_b=1.0,
    n_bootstrap=500,
):
    """Monte Carlo spread of the full calibration-and-state pipeline (uses the true state; experiment-side only)."""

    # check that n_bootstrap is positive
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    # check that rng is provided, if not, create a default random number generator
    if rng is None:
        rng = np.random.default_rng()

    # perform the bootstrap resampling to estimate the fidelity and its uncertainty
    estimates = np.asarray(
        [
            fidelity_with_calibration(
                N_total,
                f_cal,
                r=r,
                rng=rng,
                true_a=true_a,
                true_b=true_b,
            )
            for _ in range(n_bootstrap)
        ],
        dtype=float,
    )

    # return the mean and standard deviation of the bootstrap estimates
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1)) if n_bootstrap > 1 else 0.0

def collect_data(r, N_total, f_cal, rng, true_a=0.0, true_b=1.0):
    """Simulator side: collect calibration and state outcomes. Only this function sees the true state."""

    # check that N_total is positive
    if N_total < 1:
        raise ValueError("N_total must be positive")
    # check that f_cal is in the range [0, 1]
    if not 0 <= f_cal <= 1:
        raise ValueError("f_cal must be in [0, 1]")

    # set the number of calibration and state shots based on the total number of shots and the fraction allocated to calibration
    n_cal = max(1, int(round(f_cal * N_total / 4))) # per calibration state
    n_state = max(1, int(round((1 - f_cal) * N_total / 2))) # per basis

    return {
        "state": {P: measure(r, P, n_state, rng, true_a, true_b) for P in "XZ"},
        "cal": {
            (P, s): measure(eigenstate(P, s), P, n_cal, rng, true_a, true_b)
            for P in "XZ" for s in (+1, -1)
        },
    }

def fidelity_from_data(data):
    """Estimator side: fidelity from collected outcomes only. Never sees the true state."""

    r_hat = {}
    for P in "XZ":
        # estimate the readout parameters from the calibration outcomes
        m_plus = np.mean(data["cal"][(P, +1)])
        m_minus = np.mean(data["cal"][(P, -1)])
        a_hat = (m_plus + m_minus) / 2
        b_hat = (m_plus - m_minus) / 2
        # correct the state outcomes
        if abs(b_hat) < 1e-12:
            r_hat[P] = 0.0
        else:
            r_hat[P] = corrected_r(data["state"][P], a_hat, b_hat)

    F = 0.5 * (1 + (r_hat["X"] + r_hat["Z"]) / math.sqrt(2))
    return float(np.clip(F, 0.0, 1.0))

def bootstrap_ci(data, rng, n_boot=1000, alpha=0.05):
    """Percentile bootstrap CI that resamples BOTH the state and the calibration outcomes,
    so it captures calibration uncertainty. Uses only observed data."""

    # resample one array of outcomes with replacement
    rs = lambda arr: rng.choice(arr, size=len(arr), replace=True)
    ests = [
        fidelity_from_data({
            "state": {P: rs(v) for P, v in data["state"].items()},
            "cal": {k: rs(v) for k, v in data["cal"].items()},
        })
        for _ in range(n_boot)
    ]
    # return the (lower, upper) percentile interval
    return np.percentile(ests, [100 * alpha / 2, 100 * (1 - alpha / 2)])

def estimate_fidelity_adaptive(r, N, rng, pilot=100, a=0.0, b=1.0):
    """Allocate state shots adaptively, with a pilot estimate for X and Z."""

    # check that N is at least 2 to allow for pilot measurements
    if N < 2:
        raise ValueError("N must be at least 2")

    # check that pilot is at least 1 and at most N // 2 to ensure valid allocation
    pilot = min(pilot, N // 2)
    pilot = max(1, pilot)

    # perform pilot measurements in the X and Z bases
    pilot_x = measure(r, "X", pilot, rng, a, b)
    pilot_z = measure(r, "Z", pilot, rng, a, b)
    # compute the mean of the pilot measurements to estimate the Bloch vector components
    r_x = float(np.mean(pilot_x))
    r_z = float(np.mean(pilot_z))

    # compute weights for adaptive allocation based on the estimated Bloch vector components
    weights = np.asarray(
        [math.sqrt(max(0.0, 1 - r_x * r_x)),
         math.sqrt(max(0.0, 1 - r_z * r_z))],
        dtype=float,
    )

    # ensure that the weights are not all zero to avoid division by zero in allocation
    if np.sum(weights) == 0:
        weights[:] = 1.0

    # calculate the number of remaining shots after the pilot measurements
    remaining = N - 2 * pilot
    # allocate the remaining shots to the X and Z bases based on the computed weights
    n_x_new = int(round(remaining * weights[0] / np.sum(weights)))
    n_z_new = remaining - n_x_new

    # perform additional measurements in the X and Z bases and concatenate with the pilot measurements
    out_x = np.concatenate([pilot_x, measure(r, "X", n_x_new, rng, a, b)])
    out_z = np.concatenate([pilot_z, measure(r, "Z", n_z_new, rng, a, b)])

    # compute and return the estimated fidelity based on the measurement outcomes
    return estimate_fidelity(out_x, out_z)

def estimate_fidelity_random_basis(r, N, rng, a=0.0, b=1.0):
    """Randomly choose X or Z for every shot and estimate fidelity."""

    # choose a random basis (X or Z) for each of the N shots
    bases = rng.choice(("X", "Z"), size=N)

    # initialize an array to hold the measurement outcomes
    outcomes = np.empty(N, dtype=int)
    # perform measurements in the chosen bases and store the outcomes
    for basis in ("X", "Z"):
        # create a boolean mask for the shots that were assigned to the current basis
        mask = bases == basis
        # perform measurements for the shots in the current basis and store the outcomes
        outcomes[mask] = measure(r, basis, int(np.sum(mask)), rng, a, b)

    # return the estimated fidelity
    return float(np.clip(0.5 + np.mean(outcomes) / math.sqrt(2), 0.0, 1.0))


# Descriptive aliases for callers that prefer explicit variant names.
adaptive_fidelity = estimate_fidelity_adaptive
random_basis_fidelity = estimate_fidelity_random_basis

def make_sampler(r, rng, a=0.0, b=1.0):
    """Return sampler(n) giving n outcomes, each measured in a randomly chosen X or Z basis."""

    # check that rng is provided, if not, create a default random number generator
    if rng is None:
        rng = np.random.default_rng()

    # define the sampler function that takes the number of shots n as input
    def sampler(n):
        # choose a random basis (X or Z) for each shot
        bases = rng.choice(("X", "Z"), size=n)
        # initialize an array to hold the measurement outcomes
        outcomes = np.empty(n, dtype=int)
        # perform measurements in the chosen bases and store the outcomes
        for basis in ("X", "Z"):
            # create a boolean mask for the shots that were assigned to the current basis
            mask = bases == basis
            # perform measurements for the shots in the current basis and store the outcomes
            outcomes[mask] = measure(r, basis, int(np.sum(mask)), rng, a, b)
        return outcomes

    return sampler

def certify(sampler, thresh=0.95, delta=0.05, batch=50, max_shots=20000):
    """Certify that the fidelity of a sampler is above a threshold with confidence 1 - delta."""

    # initialize variables for the cumulative sum of samples, number of samples, and iteration count
    s_sum, N, k = 0, 0, 0
    # initialize the lower and upper bounds of the running confidence interval
    lo_run, hi_run = 0.0, 1.0

    # loop until the maximum number of shots is reached
    while N < max_shots:
        # increment the iteration count
        k += 1

        # determine the number of shots to take in this batch, ensuring it does not exceed the maximum allowed
        shots = min(batch * 2 ** (k - 1) - N, max_shots - N)
        # sample the outcomes from the sampler for the current batch of shots and convert to a float array
        s = np.asarray(sampler(shots), dtype=float)

        # update the cumulative sum of samples and the total number of samples taken
        s_sum += float(np.sum(s))
        N += shots

        # compute the adjusted delta for the current iteration based on the number of looks
        delta_k = delta * 6 / (math.pi**2 * k**2)
        # compute the half-width of the confidence interval using the Hoeffding bound
        t = math.sqrt(2 * math.log(1 / delta_k) / N)

        # compute the estimated fidelity based on the cumulative sum of samples
        F_hat = 0.5 + (s_sum / N) / math.sqrt(2)

        # compute the lower and upper bounds of the confidence interval for the estimated fidelity
        lo = F_hat - t / math.sqrt(2)
        hi = F_hat + t / math.sqrt(2)

        # update the running lower and upper bounds of the confidence interval to ensure they hold jointly across all iterations
        lo_run, hi_run = max(lo_run, lo), min(hi_run, hi) 
        # check if the running lower bound exceeds the threshold or if the running upper bound is below the threshold to determine if we can conclude PASS or FAIL
        if lo_run >= thresh: return "PASS", (lo_run, hi_run), N
        if hi_run <  thresh: return "FAIL", (lo_run, hi_run), N

    # if the loop completes without reaching a conclusion, return INCONCLUSIVE along with the running confidence interval and total number of samples
    return "INCONCLUSIVE", (lo_run, hi_run), N