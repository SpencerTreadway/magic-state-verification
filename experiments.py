import argparse
import csv
import math
import statistics
import textwrap
from pathlib import Path

import numpy as np

import sim

# Constants
TRIALS_DEFAULT = 2000 # Number of trials for fidelity and calibration experiments
CERTIFY_TRIALS_DEFAULT = 1000 # Number of trials for certification experiments
SHOT_COUNTS = [50, 100, 200, 500, 1000, 5000] # Number of shots to use in fidelity and calibration experiments
TRUE_FIDELITIES = np.linspace(0.90, 1.0, 21) # True fidelities to test in certification experiments
MIN_CAL = 20 # Minimum number of calibration shots to use in calibration experiments

def target_state():
    """Return the target magic state |T> = (1, 0, 1)/sqrt(2)."""

    return sim.make_state(1 / math.sqrt(2), 0.0, 1 / math.sqrt(2))

def depolarized(visibility):
    """Mix the target state with the maximally mixed state using visibility."""

    # Compute the Bloch vector of the depolarized state
    r = target_state() * visibility
    # Add the contribution from the maximally mixed state
    r += (1 - visibility) * np.array([0.0, 0.0, 0.0])

    return sim.make_state(*r)

def coherent_rotation(theta):
    """Rotate the target Bloch vector coherently about the Y axis."""

    # set the x and z components of the target state
    x = 1 / math.sqrt(2)
    z = 1 / math.sqrt(2)
    # Rotate the Bloch vector (x, 0, z) about the Y axis by angle theta
    rx = z * math.sin(theta) + x * math.cos(theta)
    rz = z * math.cos(theta) - x * math.sin(theta)

    return sim.make_state(rx, 0.0, rz)

def near_threshold_state(fid=0.95):
    """Place a state on the target axis with the requested fidelity."""

    return state_for_fidelity(fid)

def state_with_ry(ry=0.20):
    """Place a state with a nonzero y component of the Bloch vector."""

    # xz component of the Bloch vector is sqrt(1 - ry^2) / sqrt(2)
    xz = math.sqrt(1 - ry * ry) / math.sqrt(2)

    return sim.make_state(xz, ry, xz)

def test_states():
    """Return a dictionary of test states for fidelity estimation experiments."""

    return {
        "target": target_state(),
        "depolarized(visibility=0.90)": depolarized(0.90),
        "coherent_rotation(theta=acos(0.90))": coherent_rotation(math.acos(0.90)),
        "near_threshold": near_threshold_state(),
        "with_ry": state_with_ry(),
    }

def fidelity_estimator(name, r, n, rng, a=0.0, b=1.0):
    """Estimate the fidelity of a state r using n shots and the specified estimator."""

    # if the estimator is "equal", split the shots evenly between X and Z measurements
    if name == "equal":
        nx = n // 2
        nz = n - nx
        return float(
            sim.estimate_fidelity(
                sim.measure(r, "X", nx, rng, a, b),
                sim.measure(r, "Z", nz, rng, a, b),
            )
        )
    # if the estimator is "adaptive", use the adaptive estimator
    if name == "adaptive":
        return float(sim.estimate_fidelity_adaptive(r, n, rng, pilot=max(1, n // 20), a=a, b=b))
    # if the estimator is "random_basis", use the random basis estimator
    if name == "random_basis":
        return float(sim.estimate_fidelity_random_basis(r, n, rng, a=a, b=b))
    raise ValueError("unknown estimator")


def run_fidelity_experiment(trials=TRIALS_DEFAULT, seed=12345, a=0.0, b=1.0):
    """Run fidelity estimation experiments for various test states and estimators."""

    # Set up the random number generator and prepare to record results
    rng = np.random.default_rng(seed)

    # initialize an empty list to hold the results of the experiments
    records = []
    # Loop over each test state and compute the true fidelity
    for state_name, r in test_states().items():
        # Compute the true fidelity of the state r with respect to the target state
        true_f = float(sim.fidelity(r))
        # Loop over each shot count and perform trials for each estimator
        for n in SHOT_COUNTS:
            # Initialize a dictionary to hold the estimates for each estimator
            estimates = {name: [] for name in ("equal", "adaptive", "random_basis")}
            # Perform the specified number of trials for each estimator
            for _ in range(trials):
                # Loop over each estimator
                for name in estimates:
                    # Compute the fidelity estimate using the specified estimator and append it to the list of estimates for that estimator
                    estimates[name].append(
                        fidelity_estimator(name, r, n, rng, a=a, b=b)
                    )
            # After all trials are complete, compute the bias and RMSE for each estimator and record the results
            for name, values in estimates.items():
                # Convert the list of estimates to a NumPy array for easier computation
                values = np.asarray(values)
                # Append a dictionary containing the results of the experiment for this state, shot count, and estimator to the records list
                records.append({
                    "level": "fidelity",
                    "state": state_name,
                    "N": n,
                    "estimator": name,
                    "true_F": true_f,
                    "bias": float(np.mean(values - true_f)),
                    "bias_SE": float(np.std(values - true_f, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0,
                    "RMSE": float(np.sqrt(np.mean((values - true_f) ** 2))),
                    "RMSE_SE": float(np.sqrt(np.mean((values - true_f) ** 2)) / math.sqrt(max(2 * trials, 1))),
                })

    return records

def run_calibration_experiment(trials=TRIALS_DEFAULT, seed=23456,
                               r=None, true_a=0.0, true_b=0.90):
    """Run readout calibration experiments for various shot counts and calibration fractions."""

    # Set up the random number generator and prepare to record results
    rng = np.random.default_rng(seed)

    # If no state is provided, use the target state for the calibration experiments
    if r is None:
        r = target_state()

    # Compute the true fidelity of the state r with respect to the target state
    true_f = float(sim.fidelity(r))

    # initialize an empty list to hold the results of the calibration experiments
    records = []
    # Loop over each shot count and perform trials for each calibration fraction
    for n in SHOT_COUNTS:
        # Loop over each calibration fraction and perform trials for each estimator
        for name, f_cal in (
            ("uncorrected", None),
            ("calibrated_f0.05", 0.05),
            ("calibrated_f0.10", 0.10),
            ("calibrated_f0.20", 0.20),
            ("calibrated_f0.30", 0.30),
            ("calibrated_f0.40", 0.40),
            ("asymmetric_uncorrected", None),
            ("asymmetric_calibrated_f0.20", 0.20),
        ):
            # If a calibration fraction is provided, compute the number of calibration shots to use, and skip this combination if the number of calibration shots is below the minimum threshold
            n_cal = max(1, int(round(f_cal * n / 4))) if f_cal is not None else None
            if f_cal is not None and n_cal < MIN_CAL:
                continue
            # Initialize a list to hold the estimates for this estimator
            estimates = []
            # Perform the specified number of trials for this estimator
            for _ in range(trials):
                # If no calibration fraction is provided, use the uncorrected estimator
                if f_cal is None:
                    # Use the uncorrected estimator by measuring half the shots in the X basis and half in the Z basis, and then estimating the fidelity from these measurements
                    estimates.append(float(sim.estimate_fidelity(
                        sim.measure(r, "X", n // 2, rng, true_a, true_b),
                        sim.measure(r, "Z", n - n // 2, rng, true_a, true_b),
                    )))
                # If a calibration fraction is provided, use the calibrated estimator by collecting calibration data and then estimating the fidelity from this data
                else:
                    data = sim.collect_data(r, n, f_cal, rng, true_a, true_b)
                    estimates.append(sim.fidelity_from_data(data))

            # Convert the list of estimates to a NumPy array for easier computation
            estimates = np.asarray(estimates)
            # After all trials are complete, compute the bias and RMSE for this estimator and record the results
            records.append({
                "level": "calibration",
                "state": "target",
                "scenario": "symmetric",
                "N": n,
                "estimator": name,
                "true_F": true_f,
                "bias": float(np.mean(estimates - true_f)),
                "RMSE": float(np.sqrt(np.mean((estimates - true_f) ** 2))),
                "RMSE_SE": float(np.sqrt(np.mean((estimates - true_f) ** 2)) / math.sqrt(max(2 * trials, 1))),
            })

    # Perform a sweep of calibration experiments for various shot counts and calibration fractions
    calibration_fractions = np.linspace(0.01, 0.60, 12)
    # Loop over each shot count and calibration fraction and perform trials for each combination
    for n in [100, 500, 1000]:
        for f_cal in calibration_fractions:
            n_cal = max(1, int(round(f_cal * n / 4)))
            if n_cal < MIN_CAL:
                continue
            # estimate the fidelity using the specified calibration fraction and number of trials, and record the results
            estimates = np.asarray([
                sim.fidelity_from_data(sim.collect_data(r, n, f_cal, rng, true_a, true_b))
                for _ in range(trials)
            ])
            # After all trials are complete, compute the bias and RMSE for this calibration fraction and record the results
            rmse_boot = []
            if trials > 1:
                for _ in range(200):
                    sampled = rng.choice(estimates, size=trials, replace=True)
                    rmse_boot.append(
                        np.sqrt(np.mean((sampled - true_f) ** 2))
                    )
            rmse_boot_se = (
                float(np.std(rmse_boot, ddof=1)) if len(rmse_boot) > 1 else 0.0
            )
            records.append({
                "level": "calibration_sweep",
                "state": "target",
                "scenario": "symmetric",
                "N": n,
                "estimator": f"f_cal={f_cal:.2f}",
                "f_cal": float(f_cal),
                "true_F": true_f,
                "bias": float(np.mean(estimates - true_f)),
                "RMSE": float(np.sqrt(np.mean((estimates - true_f) ** 2))),
                "RMSE_SE": float(np.sqrt(np.mean((estimates - true_f) ** 2)) / math.sqrt(max(2 * trials, 1))),
                "RMSE_BOOT_SE": rmse_boot_se,
            })

    # Perform a sweep of asymmetric readout experiments for various shot counts and calibration fractions
    asymmetric_a = 0.10 # Set the readout error parameter a for asymmetric readout experiments
    asymmetric_b = 0.85 # Set the readout error parameter b for asymmetric readout experiments

    # Loop over each shot count and perform trials for each asymmetric readout estimator
    for n in SHOT_COUNTS:
        # Loop over each asymmetric readout estimator and perform trials for each one
        for name, f_cal in (
            ("asymmetric_uncorrected", None),
            ("asymmetric_calibrated_f0.20", 0.20),
        ):
            # Initialize a list to hold the estimates for this estimator
            estimates = []
            # Perform the specified number of trials for this estimator
            for _ in range(trials):
                # If no calibration fraction is provided, use the uncorrected estimator with asymmetric readout errors
                if f_cal is None:
                    estimates.append(float(sim.estimate_fidelity(
                        sim.measure(r, "X", n // 2, rng, asymmetric_a, asymmetric_b),
                        sim.measure(r, "Z", n - n // 2, rng, asymmetric_a, asymmetric_b),
                    )))
                # If a calibration fraction is provided, use the calibrated estimator with asymmetric readout errors by collecting calibration data and then estimating the fidelity from this data
                else:
                    # If a calibration fraction is provided, compute the number of calibration shots to use, and skip this combination if the number of calibration shots is below the minimum threshold
                    n_cal = max(1, int(round(f_cal * n / 4)))
                    if n_cal < MIN_CAL:
                        break

                    # Collect calibration data with asymmetric readout errors and estimate the fidelity from this data
                    data = sim.collect_data(
                        r, n, f_cal, rng, asymmetric_a, asymmetric_b
                    )
                    # Append the estimated fidelity from the calibration data to the list of estimates for this estimator
                    estimates.append(sim.fidelity_from_data(data))

            # After all trials are complete, compute the bias and RMSE for this estimator and record the results
            if estimates:
                estimates = np.asarray(estimates)
                records.append({
                    "level": "calibration",
                    "state": "target",
                    "scenario": "asymmetric",
                    "N": n,
                    "estimator": name,
                    "true_F": true_f,
                    "bias": float(np.mean(estimates - true_f)),
                    "bias_SE": float(np.std(estimates - true_f, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0,
                    "variance": float(np.var(estimates, ddof=1)) if trials > 1 else 0.0,
                    "SE": float(np.std(estimates, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0,
                    "RMSE": float(np.sqrt(np.mean((estimates - true_f) ** 2))),
                    "RMSE_SE": float(np.sqrt(np.mean((estimates - true_f) ** 2)) / math.sqrt(max(2 * trials, 1))),
                })

    low_fidelity_r = state_for_fidelity(0.90)
    low_fidelity_true = float(sim.fidelity(low_fidelity_r))
    for n in SHOT_COUNTS:
        for name, f_cal in (
            ("asymmetric_uncorrected_lowF", None),
            ("asymmetric_calibrated_f0.20_lowF", 0.20),
        ):
            estimates = []
            for _ in range(trials):
                if f_cal is None:
                    estimates.append(float(sim.estimate_fidelity(
                        sim.measure(low_fidelity_r, "X", n // 2, rng, asymmetric_a, asymmetric_b),
                        sim.measure(low_fidelity_r, "Z", n - n // 2, rng, asymmetric_a, asymmetric_b),
                    )))
                else:
                    n_cal = max(1, int(round(f_cal * n / 4)))
                    if n_cal < MIN_CAL:
                        break
                    data = sim.collect_data(
                        low_fidelity_r, n, f_cal, rng, asymmetric_a, asymmetric_b
                    )
                    estimates.append(sim.fidelity_from_data(data))
            if estimates:
                estimates = np.asarray(estimates)
                records.append({
                    "level": "calibration",
                    "state": "low_fidelity_0.90",
                    "scenario": "asymmetric",
                    "N": n,
                    "estimator": name,
                    "true_F": low_fidelity_true,
                    "bias": float(np.mean(estimates - low_fidelity_true)),
                    "bias_SE": float(np.std(estimates - low_fidelity_true, ddof=1) / math.sqrt(trials)) if trials > 1 else 0.0,
                    "RMSE": float(np.sqrt(np.mean((estimates - low_fidelity_true) ** 2))),
                    "RMSE_SE": float(np.sqrt(np.mean((estimates - low_fidelity_true) ** 2)) / math.sqrt(max(2 * trials, 1))),
                })

    return records

def state_for_fidelity(true_f):
    """Return a state with the specified fidelity along the target axis."""

    component = 2 * true_f - 1

    return sim.make_state(
        component / math.sqrt(2),
        0.0,
        component / math.sqrt(2),
    )

def run_bootstrap_coverage(trials=1000, seed=45678, n_boot=300):
    """Measure percentile-bootstrap coverage for target and near-threshold states."""

    # Set up the random number generator and prepare to record results
    rng = np.random.default_rng(seed)

    # initialize an empty list to hold the results of the bootstrap coverage experiments
    records = []
    # Loop over the target state and a near-threshold state and perform trials for each one
    for state_name, r in (
        ("target", target_state()),
        ("near_threshold", near_threshold_state(0.95)),
    ):
        # Compute the true fidelity of the state r with respect to the target state
        true_f = float(sim.fidelity(r))

        # Loop over each shot count and perform trials for each one
        for n in (100, 250, 500, 1000, 2000):
            # Initialize a counter for the number of times the true fidelity falls within the bootstrap confidence interval
            hits = 0
            # Perform the specified number of trials for this shot count and state
            for _ in range(trials):
                # Collect data for the state r using the specified number of shots, calibration fraction, and readout error parameters
                data = sim.collect_data(r, n, 0.20, rng, 0.0, 0.90)
                # Compute the bootstrap confidence interval for the collected data and check if the true fidelity falls within this interval
                lo, hi = sim.bootstrap_ci(data, rng, n_boot=n_boot)

                # Increment the hits counter if the true fidelity falls within the bootstrap confidence interval
                hits += lo <= true_f <= hi

            # After all trials are complete, compute the coverage rate and record the results
            records.append({
                "level": "bootstrap_coverage",
                "state": state_name,
                "N": n,
                "true_F": true_f,
                "nominal_coverage": 0.95,
                "coverage": hits / trials,
                "coverage_SE": math.sqrt((hits / trials) * (1 - hits / trials) / trials),
            })
    return records

def naive_normal_certify(sampler, thresh=0.95, delta=0.05,
                         batch=50, max_shots=20000):
    """Non-alpha-spending normal-approximation baseline for optional stopping."""

    # Set up the initial values for the sequential certification process
    s_sum = 0.0 # Cumulative sum of the sampled values
    N = 0 # Total number of shots taken
    z = 1.96 # Z-score for a 95% confidence interval
    lo_run = 0.0 # Lower bound of the running confidence interval
    hi_run = 1.0 # Upper bound of the running confidence interval
    # Perform the sequential certification process until the maximum number of shots is reached
    while N < max_shots:
        # Determine the number of shots to take in this batch, ensuring that it does not exceed the maximum allowed shots
        shots = min(batch, max_shots - N)

        # Sample the state using the provided sampler and convert the results to a NumPy array of floats
        s = np.asarray(sampler(shots), dtype=float)
        # Update the cumulative sum of the sampled values and the total number of shots taken
        s_sum += float(np.sum(s))

        # Update the total number of shots taken and compute the mean of the sampled values
        N += shots
        mean_s = s_sum / N

        # Compute the estimated fidelity and the standard error of the mean, and update the running confidence interval
        F_hat = 0.5 + mean_s / math.sqrt(2)
        se_s = math.sqrt(max(0.0, 1 - mean_s * mean_s) / N)
        lo = F_hat - z * se_s / math.sqrt(2)
        hi = F_hat + z * se_s / math.sqrt(2)
        lo_run = max(lo_run, lo)
        hi_run = min(hi_run, hi)

        # Check if the running confidence interval is entirely above or below the threshold, and return the appropriate decision along with the interval and number of shots taken
        if lo_run >= thresh:
            return "PASS", (lo_run, hi_run), N
        if hi_run < thresh:
            return "FAIL", (lo_run, hi_run), N  

    # If the maximum number of shots is reached without a conclusive decision, return "INCONCLUSIVE" along with the running confidence interval and number of shots taken
    return "INCONCLUSIVE", (lo_run, hi_run), N

def normal_alpha_certify(sampler, thresh=0.95, delta=0.05,
                         batch=50, max_shots=20000):
    """Normal-approximation intervals with the same alpha spending as Hoeffding."""
    s_sum = 0.0
    N = 0
    k = 0
    lo_run = 0.0
    hi_run = 1.0
    while N < max_shots:
        k += 1
        shots = min(batch, max_shots - N)
        s = np.asarray(sampler(shots), dtype=float)
        s_sum += float(np.sum(s))
        N += shots
        mean_s = s_sum / N
        delta_k = delta * 6 / (math.pi ** 2 * k ** 2)
        z = statistics.NormalDist().inv_cdf(1 - delta_k / 2)
        F_hat = 0.5 + mean_s / math.sqrt(2)
        se_s = math.sqrt(max(0.0, 1 - mean_s * mean_s) / N)
        lo = F_hat - z * se_s / math.sqrt(2)
        hi = F_hat + z * se_s / math.sqrt(2)
        lo_run = max(lo_run, lo)
        hi_run = min(hi_run, hi)
        if lo_run >= thresh:
            return "PASS", (lo_run, hi_run), N
        if hi_run < thresh:
            return "FAIL", (lo_run, hi_run), N
    return "INCONCLUSIVE", (lo_run, hi_run), N

def run_certification_experiment(trials=CERTIFY_TRIALS_DEFAULT, seed=34567,
                                 thresh=0.95, delta=0.05, batch=50,
                                 max_shots=20000, a=0.0, b=1.0):
    """Run sequential certification experiments for various true fidelities."""

    # Set up the random number generator and prepare to record results
    rng = np.random.default_rng(seed)

    # initialize an empty list to hold the results of the certification experiments
    records = []
    # Loop over each true fidelity and perform trials for each one
    for true_f in TRUE_FIDELITIES:
        # Create a state with the specified true fidelity along the target axis
        r = state_for_fidelity(float(true_f))

        # Initialize a dictionary to hold the counts of each decision type (PASS, FAIL, INCONCLUSIVE) and a list to hold the number of shots used in each trial
        counts = {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0}
        # Initialize an empty list to hold the number of shots used in each trial
        shots = []
        # Perform the specified number of trials for this true fidelity
        for _ in range(trials):
            # Create a sampler for the state r using the specified random number generator and readout error parameters
            sampler = sim.make_sampler(r, rng, a=a, b=b)
            # Use the sampler to perform sequential certification with the specified threshold, delta, batch size, and maximum number of shots, and record the decision, interval, and number of shots used
            decision, interval, n = sim.certify(
                sampler,
                thresh=thresh,
                delta=delta,
                batch=batch,
                max_shots=max_shots,
            )

            # Increment the count for the decision type (PASS, FAIL, INCONCLUSIVE) and append the number of shots used to the list of shots
            counts[decision] += 1
            shots.append(n)

        # Compute the false-PASS rate for this true fidelity and record the results
        false_pass = counts["PASS"] / trials if true_f < thresh - 1e-9 else 0.0
        false_fail = counts["FAIL"] / trials if true_f > thresh + 1e-9 else 0.0
        # Append a dictionary containing the results of the certification experiment for this true fidelity to the records list
        records.append({
            "level": "certification",
            "true_F": float(true_f),
            "threshold": thresh,
            "delta": delta,
            "PASS": counts["PASS"] / trials,
            "FAIL": counts["FAIL"] / trials,
            "INCONCLUSIVE": counts["INCONCLUSIVE"] / trials,
            "false_pass": false_pass,
            "false_fail": false_fail,
            "mean_shots": float(np.mean(shots)),
        })

    return records

def run_optional_stopping_experiment(trials=1000, seed=56789,
                                     thresh=0.95, delta=0.05,
                                     batch=50, max_shots=20000):
    """Compare alpha-spending Hoeffding with a naive normal-approximation baseline."""

    # Set up the random number generator
    rng = np.random.default_rng(seed)

    # initialize an empty list to hold the results of the optional stopping experiments
    records = []
    # Loop over a range of true fidelities and perform trials for each one
    fidelities = np.linspace(0.90, 0.949, 10)
    # Loop over each true fidelity and perform trials for each one
    for true_f in fidelities:
        # Create a state with the specified true fidelity along the target axis
        r = state_for_fidelity(float(true_f))

        # Initialize a dictionary to hold the counts of each decision type (PASS, FAIL, INCONCLUSIVE) for both the Hoeffding and normal-approximation methods
        counts = {
            "hoeffding": {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0},
            "normal_alpha": {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0},
            "normal": {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0},
        }
        shot_totals = {method: 0 for method in counts}

        # Perform trials for each method
        for method in ("hoeffding", "normal_alpha", "normal"):
            # Perform the specified number of trials for this method
            for _ in range(trials):
                # Create a sampler for the state r using the specified random number generator
                sampler = sim.make_sampler(r, rng)

                # if the method is "hoeffding", use the Hoeffding-based sequential certification, otherwise use the naive normal-approximation baseline for optional stopping
                if method == "hoeffding":
                    decision, _, n_shots = sim.certify(
                        sampler, thresh=thresh, delta=delta,
                        batch=batch, max_shots=max_shots,
                    )
                elif method == "normal_alpha":
                    decision, _, n_shots = normal_alpha_certify(
                        sampler, thresh=thresh, delta=delta,
                        batch=batch, max_shots=max_shots,
                    )
                else:
                    decision, _, n_shots = naive_normal_certify(
                        sampler, thresh=thresh, delta=delta,
                        batch=batch, max_shots=max_shots,
                    )

                # Increment the count for the decision type (PASS, FAIL, INCONCLUSIVE) for this method
                counts[method][decision] += 1
                shot_totals[method] += n_shots

        # After all trials are complete, compute the false-PASS rate for this true fidelity and record the results
        records.append({
            "level": "optional_stopping",
            "true_F": float(true_f),
            "threshold": thresh,
            "hoeffding_false_pass": counts["hoeffding"]["PASS"] / trials,
            "normal_alpha_false_pass": counts["normal_alpha"]["PASS"] / trials,
            "normal_false_pass": counts["normal"]["PASS"] / trials,
            "hoeffding_mean_shots": shot_totals["hoeffding"] / trials,
            "normal_alpha_mean_shots": shot_totals["normal_alpha"] / trials,
            "normal_mean_shots": shot_totals["normal"] / trials,
            "delta": delta,
        })

    return records

def write_csv(path, records):
    """Write a list of dictionaries to a CSV file at the specified path."""

    # Create the parent directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # If there are no records to write, return early
    if not records:
        return

    # Determine the fieldnames for the CSV file by extracting the keys from the first record
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    
    # Write the records to the CSV file using a DictWriter
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def plot_results(fidelity_records, calibration_records, certification_records, output_dir,
                 coverage_records=None, optional_records=None,
                 fidelity_trials=None, certify_trials=None, coverage_trials=None,
                 optional_trials=None, fidelity_seed=None, certify_seed=None):
    """Plot the results of the experiments and save the figures to the specified output directory."""

    try:
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        print("matplotlib is unavailable; CSV summaries were still generated.")
        return

    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    states = test_states()
    fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    estimator_styles = {
        "equal": ("tab:blue", "o"),
        "adaptive": ("tab:orange", "s"),
        "random_basis": ("tab:green", "^"),
    }
    for ax, (state_name, r) in zip(axes, states.items()):
        rx, rz = float(r[0]), float(r[2])
        for estimator, (color, marker) in estimator_styles.items():
            rows = [
                row for row in fidelity_records
                if row["state"] == state_name and row["estimator"] == estimator
            ]
            if rows:
                ns = np.asarray([row["N"] for row in rows], dtype=float)
                values = np.asarray([row["RMSE"] for row in rows])
                errors = np.asarray([row.get("RMSE_SE", 0.0) for row in rows])
                ax.errorbar(ns, values, yerr=errors, marker=marker,
                            color=color, capsize=3, label=estimator)
        theory = np.asarray(SHOT_COUNTS, dtype=float)
        theory_rmse = np.sqrt(
            ((1 - rx * rx) / (theory / 2)
             + (1 - rz * rz) / (theory - theory / 2)) / 8
        )
        ax.plot(theory, theory_rmse, "--", color="black", label="Theory")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(state_name)
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    handles = [
        mlines.Line2D([], [], color=color, marker=marker, label=name)
        for name, (color, marker) in estimator_styles.items()
    ]
    handles.append(
        mlines.Line2D(
            [], [], color="black", linestyle="--", label="Theory"
        )
    )
    axes[-1].legend(handles=handles, loc="center", title="Estimator", fontsize=9)
    fig.suptitle(
        f"Fidelity RMSE by state"
        + (f" | {fidelity_trials} trials, seed {fidelity_seed}" if fidelity_trials else ""),
        fontsize=14,
    )
    fig.supxlabel("Shots N", y=0.05)
    fig.supylabel("RMSE")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.text(0.50, 0.01,
             "Takeaway: States near F=1 fall below theory due to clipping; F=0.95 states track theory. "
             "Coherent-state adaptive allocation matches Neyman theory at large N.",
             ha="center", fontsize=9)
    fig.savefig(output_dir / "rmse_vs_n.png", dpi=150)
    plt.close(fig)

    # create side-by-side calibration RMSE plots for symmetric and asymmetric readout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    # Filter the calibration records to only include rows for the "calibration" level
    calibration_rows = [
        row for row in calibration_records
        if row["level"] == "calibration"
    ]
    # Loop over each estimator and plot the RMSE vs. number of shots N for the calibration experiments
    estimator_colors = {
        "uncorrected": "tab:blue",
        "calibrated_f0.05": "tab:orange",
        "calibrated_f0.10": "tab:green",
        "calibrated_f0.20": "tab:red",
        "calibrated_f0.30": "tab:purple",
        "calibrated_f0.40": "tab:brown",
        "asymmetric_uncorrected": "tab:pink",
        "asymmetric_calibrated_f0.20": "tab:gray",
        "asymmetric_uncorrected_lowF": "tab:cyan",
        "asymmetric_calibrated_f0.20_lowF": "tab:olive",
    }
    for estimator in estimator_colors:
        # Filter the calibration rows to only include rows for the current estimator
        if estimator.startswith("asymmetric_"):
            scenario = "asymmetric"
        else:
            scenario = "symmetric"
        rows = [
            row for row in calibration_rows
            if row["estimator"] == estimator
            and row.get("scenario") == scenario
        ]
        rows = sorted(rows, key=lambda row: row["N"])
        # Plot the RMSE vs. number of shots N for the current estimator
        if rows:
            ax = axes[0] if scenario == "symmetric" else axes[1]
            ax.errorbar(
                [row["N"] for row in rows],
                [row["RMSE"] for row in rows],
                yerr=[row.get("RMSE_SE", 0.0) for row in rows],
                marker="o", capsize=3,
                color=estimator_colors[estimator],
                label=(
                    estimator.removeprefix("asymmetric_").replace("_lowF", " (F=0.90)")
                    if scenario == "asymmetric"
                    else estimator
                ),
            )

    # Set the x and y labels, title, legend, grid, and save the figure to the output directory
    for ax, title in zip(
        axes,
        ("Symmetric readout: (a,b)=(0,0.90)",
         "Asymmetric readout: (a,b)=(0.10,0.85); fixed point ≈0.67"),
    ):
        ax.set(xscale="log", xlabel="Shots N", ylabel="RMSE",
               title=f"{title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.text(0.50, 0.01,
             "Takeaway: Asymmetric uncorrected beats every calibrated run from N=500 (state-specific cancellation); "
             "symmetric calibration reduces bias.",
             ha="center", fontsize=9)
    fig.text(0.50, 0.055,
             "Calibration guard: f0.05/f0.10 appear only when ≥20 shots/state.",
             ha="center", fontsize=8)
    fig.savefig(output_dir / "calibration_vs_n.png", dpi=150)
    plt.close(fig)

    # create a figure and axis for the calibration fraction plot
    fig, ax = plt.subplots(figsize=(9, 6))
    # Filter the calibration records to only include rows for the "calibration_sweep" level
    sweep = [row for row in calibration_records if row["level"] == "calibration_sweep"]
    # Loop over each shot count N and plot the RMSE vs. calibration fraction for the calibration sweep experiments
    for n in [100, 500, 1000]:
        # Filter the sweep rows to only include rows for the current shot count N
        rows = [row for row in sweep if row["N"] == n]
        # Plot the RMSE vs. calibration fraction for the current shot count N
        if rows:
            uncorrected = [
                row["RMSE"] for row in calibration_rows
                if row["estimator"] == "uncorrected" and row["N"] == n
            ]
            baseline = uncorrected[0] if uncorrected else 1.0
            ax.errorbar(
                [row["f_cal"] for row in rows],
                [row["RMSE"] / baseline for row in rows],
                yerr=[
                    row.get("RMSE_BOOT_SE", row.get("RMSE_SE", 0.0)) / baseline
                    for row in rows
                ],
                marker="o", capsize=3, label=f"N={n}",
            )
    ax.axhline(1.0, color="tab:gray", linestyle=":", label="uncorrected")

    # Set the x and y labels, title, legend, grid, and save the figure to the output directory
    ax.set(xlabel="Calibration fraction", ylabel="RMSE / uncorrected RMSE",
           title="Calibration-fraction sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.text(0.50, 0.01,
             "Takeaway: Calibration reaches ~45–60% of uncorrected RMSE; shallow minimum, no sharp optimum.",
             ha="center", fontsize=9)
    fig.savefig(output_dir / "calibration_fraction.png", dpi=150)
    plt.close(fig)

    # if there are certification records, create a figure with two subplots: one for the decision rates (PASS, FAIL, INCONCLUSIVE) vs. true fidelity, and another for the mean number of shots vs. true fidelity
    if certification_records:
        # Extract the true fidelities from the certification records
        fs = [row["true_F"] for row in certification_records]

        # Create a figure with two subplots, one for the decision rates and one for the mean number of shots
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Plot the decision rates (PASS, FAIL, INCONCLUSIVE) vs. true fidelity on the first subplot
        axes[0].plot(fs, [row["PASS"] for row in certification_records],
                     label="PASS")
        axes[0].plot(fs, [row["FAIL"] for row in certification_records],
                     label="FAIL")
        axes[0].plot(fs, [row["INCONCLUSIVE"] for row in certification_records],
                     label="INCONCLUSIVE")
        # Add a vertical dashed line at the threshold fidelity of 0.95 to indicate the certification threshold
        axes[0].axvline(0.95, color="black", linestyle="--", label="threshold")
        # Set the x and y labels, title, legend, grid, and save the figure to the output directory
        axes[0].set(xlabel="True fidelity", ylabel="Decision rate",
                    title=f"Sequential certification | {certify_trials} trials, δ={certification_records[0]['delta']}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot the mean number of shots vs. true fidelity on the second subplot
        axes[1].plot(fs, [row["mean_shots"] for row in certification_records],
                     color="tab:purple")
        # Add a vertical dashed line at the threshold fidelity of 0.95 to indicate the certification threshold
        axes[1].axvline(0.95, color="black", linestyle="--", label="threshold")
        # Set the x and y labels, title, legend, grid, and save the figure to the output directory
        axes[1].set(xlabel="True fidelity", ylabel="Mean shots",
                    title=f"Decision cost | {certify_trials} trials, δ={certification_records[0]['delta']}")
        axes[1].set_yscale("log")
        axes[1].axhline(20000, color="tab:red", linestyle=":",
                        label="20k-shot cap")
        axes[1].set_yticks([50, 100, 500, 1000, 5000, 10000, 20000])
        axes[1].set_ylim(3000, 25000)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.text(
            0.50, 0.01,
            "Takeaway: No false PASS below 0.95; F≈0.93–0.97 is almost always INCONCLUSIVE at the 20k cap.",
            ha="center", fontsize=9,
        )
        fig.savefig(output_dir / "certification.png", dpi=150)
        plt.close(fig)

    # If there are coverage records, create a figure for the bootstrap confidence-interval coverage vs. number of shots N for each state
    if coverage_records:
        # Create a figure and axis for the bootstrap confidence-interval coverage plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Loop over each unique state name in the coverage records and plot the bootstrap confidence-interval coverage vs. number of shots N for each state
        for state_name in sorted(set(row["state"] for row in coverage_records)):
            # Filter the coverage records to only include rows for the current state name
            rows = [row for row in coverage_records if row["state"] == state_name]

            # Plot the bootstrap confidence-interval coverage vs. number of shots N for the current state
            ax.errorbar(
                [row["N"] for row in rows],
                [row["coverage"] for row in rows],
                yerr=[row["coverage_SE"] for row in rows],
                marker="o", capsize=3, label=state_name,
            )
            
        # Add a horizontal dashed line at the nominal 95% coverage level to indicate the target coverage
        ax.axhline(0.95, color="black", linestyle="--", label="nominal 95%")

        # Set the x and y labels, title, legend, grid, and save the figure to the output directory
        ax.set(xlabel="Shots N", ylabel="Bootstrap coverage",
               title="Bootstrap confidence-interval coverage")
        ax.set_xscale("log")
        ax.set_ylim(0.75, 1.00)
        ax.set_title(
            f"Bootstrap coverage | {coverage_trials or 'N'} trials"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        coverage_takeaway = (
            "Percentile bootstrap under-covers at small N (≈0.89 at N≤250); "
            "near threshold it plateaus at ≈0.93–0.94, while the target reaches nominal by N≈1000."
        )
        coverage_takeaway = textwrap.fill(
            "Takeaway: " + coverage_takeaway,
            width=90,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.text(
            0.50, 0.01,
            coverage_takeaway,
            ha="center", va="bottom", fontsize=8,
        )
        fig.savefig(output_dir / "bootstrap_coverage.png", dpi=150)
        plt.close(fig)

    # If there are optional records, create a figure for the optional stopping comparison
    if optional_records:
        # Create a figure and axis for the optional stopping comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]

        # Plot the false-PASS rate vs. true fidelity for the alpha-spending Hoeffding method
        series = (
            ("hoeffding_false_pass", "alpha-spending Hoeffding"),
            ("normal_alpha_false_pass", "alpha-spending normal"),
            ("normal_false_pass", "naive normal"),
        )
        for key, label in series:
            fs = np.asarray([row["true_F"] for row in optional_records])
            rates = np.asarray([row[key] for row in optional_records])
            se = np.sqrt(rates * (1 - rates) / max(optional_trials or 1, 1))
            ax.errorbar(fs, rates, yerr=se, marker="o", capsize=3, label=label)
        
        # Add a horizontal dashed line at the delta value to indicate the false-PASS rate threshold
        ax.axhline(optional_records[0]["delta"], color="black",
                   linestyle="--", label="delta")

        # Set the x and y labels, title, legend, grid, and save the figure to the output directory
        ax.set(xlabel="True fidelity", ylabel="False-PASS rate",
               title="Optional stopping comparison")
        ax.set_title(
            f"Optional stopping comparison | {optional_trials or 'N'} trials"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        for key, label in (
            ("hoeffding_mean_shots", "alpha-spending Hoeffding"),
            ("normal_alpha_mean_shots", "alpha-spending normal"),
            ("normal_mean_shots", "naive normal"),
        ):
            axes[1].plot(
                [row["true_F"] for row in optional_records],
                [row[key] for row in optional_records],
                marker="o", label=label,
            )
        axes[1].set_xlabel("True fidelity")
        axes[1].set_ylabel("Mean shots")
        axes[1].set_title("Decision cost")
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.text(0.50, 0.01,
                 "Takeaway: Naive normal intervals inflate false-PASS risk; alpha spending keeps normal risk near δ, "
                 "but normal intervals have no finite-sample guarantee.",
                 ha="center", fontsize=9)
        fig.savefig(output_dir / "optional_stopping.png", dpi=150)
        plt.close(fig)

def main():
    """Run the fidelity, calibration, and certification experiments and save the results."""

    # Set up the argument parser to allow the user to specify the number of trials, seeds, output directory, and whether to skip certification experiments
    parser = argparse.ArgumentParser(description="Run magic-state verification experiments")
    parser.add_argument("--trials", type=int, default=TRIALS_DEFAULT)
    parser.add_argument("--certify-trials", type=int, default=CERTIFY_TRIALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--certify-seed", type=int, default=34567)
    parser.add_argument("--output", type=Path, default=Path("experiment_results"))
    parser.add_argument("--skip-certification", action="store_true")
    parser.add_argument("--coverage-trials", type=int, default=500)
    parser.add_argument("--optional-trials", type=int, default=200)
    args = parser.parse_args()

    # Run the fidelity and calibration experiments, and optionally the certification experiments, using the specified number of trials and seeds
    fidelity_records = run_fidelity_experiment(
        trials=args.trials,
        seed=args.seed,
    )
    # Run the calibration experiments using the specified number of trials and a seed that is offset from the fidelity experiment seed to ensure different random number sequences
    calibration_records = run_calibration_experiment(
        trials=args.trials,
        seed=args.seed + 1,
    )

    # Initialize an empty list to hold the certification records
    certification_records = []
    # Run the certification experiments, if they are not skipped
    if not args.skip_certification:
        certification_records = run_certification_experiment(
            trials=args.certify_trials,
            seed=args.certify_seed,
        )

    # Run the bootstrap coverage experiments using the specified number of trials and a seed that is offset from the fidelity and calibration experiment seeds to ensure different random number sequences
    coverage_records = run_bootstrap_coverage(
        trials=args.coverage_trials,
        seed=args.seed + 2,
    )
    
    # Run the optional stopping experiments using the specified number of trials and a seed that is offset from the fidelity, calibration, and bootstrap coverage experiment seeds to ensure different random number sequences, if certification experiments are not skipped
    optional_records = run_optional_stopping_experiment(
        trials=args.optional_trials,
        seed=args.seed + 3,
    ) if not args.skip_certification else []

    # Create the output directory if it does not exist, and write the results of the experiments to CSV files in that directory
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fidelity.csv", fidelity_records)
    write_csv(args.output / "calibration.csv", calibration_records)
    write_csv(args.output / "certification.csv", certification_records)
    write_csv(args.output / "bootstrap_coverage.csv", coverage_records)
    write_csv(args.output / "optional_stopping.csv", optional_records)
    # Plot the results of the experiments and save the figures to the output directory
    plot_results(
        fidelity_records,
        calibration_records,
        certification_records,
        args.output,
        coverage_records,
        optional_records,
        args.trials,
        args.certify_trials,
        args.coverage_trials,
        args.optional_trials,
        args.seed,
        args.certify_seed,
    )

    # Print a message indicating that the experiment outputs have been written to the specified output directory, and if certification records are available, print the certification results and the maximum false-PASS rate below the threshold
    print(f"Wrote experiment outputs to {args.output}")
    # If certification records are available, print the certification results and the maximum false-PASS rate below the threshold
    if certification_records:
        print("Certification results:")
        for record in certification_records:
            # Print the true fidelity, threshold, and false-PASS rate for each certification record
            print(f"  {record['true_F']:.4f} (threshold: {record['threshold']:.4f}) - False PASS: {record['false_pass']:.4f}")
        # Find the maximum false-PASS rate for true fidelities below the threshold and print it
        below_threshold = [
            row["false_pass"] for row in certification_records
            if row["true_F"] < row["threshold"]
        ]
        # If there are any false-PASS rates below the threshold, print the maximum value
        if below_threshold:
            print(f"Maximum false-PASS rate below threshold: {max(below_threshold):.4f}")


if __name__ == "__main__":
    main()
