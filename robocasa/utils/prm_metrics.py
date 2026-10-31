import numpy as np


def preprocess_progress_curve(
    progress_curve,
    *,
    despike_max_run=64,
    gaussian_sigma=0.0,
):
    """
    Preprocess a state-grounded semantic progress curve before metric computation.

    Default behavior:
      1) clip values to [0, 1]
      2) remove short A -> B -> A flickers whose middle run lasts at most
         `despike_max_run` timesteps
      3) optionally apply Gaussian smoothing if gaussian_sigma > 0

    For semantic milestone curves, despiking is preferred over Gaussian
    smoothing because true milestone transitions should remain sharp.
    """
    p = np.asarray(progress_curve, dtype=np.float32).reshape(-1)

    if p.size == 0:
        return p

    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    if despike_max_run > 0 and p.size >= 3:
        p = p.copy()
        changed = True

        while changed:
            changed = False

            starts = [0]
            for i in range(1, len(p)):
                if not np.isclose(p[i], p[i - 1], atol=1e-8):
                    starts.append(i)
            starts.append(len(p))

            if len(starts) <= 3:
                break

            for run_idx in range(1, len(starts) - 2):
                start = starts[run_idx]
                end = starts[run_idx + 1]
                run_len = end - start

                left_value = p[start - 1]
                right_value = p[end]

                if (
                    run_len <= despike_max_run
                    and np.isclose(left_value, right_value, atol=1e-8)
                ):
                    p[start:end] = left_value
                    changed = True
                    break

    if gaussian_sigma and gaussian_sigma > 0:
        sigma = float(gaussian_sigma)
        radius = max(1, int(np.ceil(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= np.sum(kernel)

        padded = np.pad(p, (radius, radius), mode="edge")
        p = np.convolve(padded, kernel, mode="valid").astype(np.float32)
        p = np.clip(p, 0.0, 1.0)

    return p


def compute_prm_metrics(
    progress_curve,
    success=False,
    *,
    despike_max_run=64,
    gaussian_sigma=0.0,
):
    """
    Compute PRM-as-a-Judge metrics from a state-grounded progress curve.

    DRR / FNS / SQS are NaN when the episode is not eligible:
      - FNS: failed rollouts only
      - DRR: rollouts with a drawdown only
      - SQS: successful rollouts only
    """
    p = preprocess_progress_curve(
        progress_curve,
        despike_max_run=despike_max_run,
        gaussian_sigma=gaussian_sigma,
    )

    if p.size == 0:
        return {
            "MP": 0.0,
            "MC25": 0.0,
            "MC50": 0.0,
            "MC75": 0.0,
            "PPL": 0.0,
            "CRA": 0.0,
            "STR": 0.0,
            "DRR": np.nan,
            "FNS": np.nan if success else 0.0,
            "SQS": 0.0 if success else np.nan,
        }

    mp = float(np.max(p))

    metrics = {
        "MP": mp,
        "MC25": float(mp >= 0.25),
        "MC50": float(mp >= 0.50),
        "MC75": float(mp >= 0.75),
    }

    # PPL = MP^2 / total variation; paper sets PPL=0 when variation is 0.
    path_length = float(np.sum(np.abs(np.diff(p))))
    metrics["PPL"] = (
        float(mp * mp / path_length)
        if path_length > 1e-8
        else 0.0
    )

    # CRA
    if mp > 1e-8:
        running_max = np.maximum.accumulate(p)
        regret = running_max - p
        metrics["CRA"] = float(np.mean(regret) / mp)
    else:
        metrics["CRA"] = 0.0

    # STR, epsilon = 0.005
    if p.size > 1:
        metrics["STR"] = float(np.mean(np.abs(np.diff(p)) < 0.005))
    else:
        metrics["STR"] = 1.0

    # DRR at maximum drawdown
    running_max = np.maximum.accumulate(p)
    drawdown = running_max - p
    t_star = int(np.argmax(drawdown))
    max_drawdown = float(drawdown[t_star])

    if max_drawdown > 1e-8:
        recovery = float(np.max(p[t_star:]) - p[t_star])
        metrics["DRR"] = float(min(1.0, recovery / max_drawdown))
    else:
        metrics["DRR"] = np.nan

    # FNS for failed rollouts only
    if success:
        metrics["FNS"] = np.nan
    else:
        metrics["FNS"] = float(
            0.5 * metrics["MP"]
            + 0.3 * metrics["MC75"]
            + 0.2 * metrics["MC50"]
        )

    # SQS for successful rollouts only
    if success:
        metrics["SQS"] = float(
            0.5 * metrics["PPL"]
            + 0.3 * (1.0 - metrics["CRA"])
            + 0.2 * (1.0 - metrics["STR"])
        )
    else:
        metrics["SQS"] = np.nan

    return metrics
