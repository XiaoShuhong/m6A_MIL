import numpy as np
from scipy import stats
 

def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()
 
    valid = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[valid]
    targets = targets[valid]

    if len(predictions) < 2:
        return {k: 0.0 for k in [
            "pearson_r", "spearman_r", "mse", "mae", "r_squared", "direction_acc",
        ]}

    pearson_r, pearson_p = stats.pearsonr(predictions, targets)

    spearman_r, spearman_p = stats.spearmanr(predictions, targets)

    mse = float(np.mean((predictions - targets) ** 2))

    mae = float(np.mean(np.abs(predictions - targets)))
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r_squared = 1 - ss_res / max(ss_tot, 1e-8)

    significant = np.abs(targets) > 0.1
    if significant.sum() > 0:
        direction_acc = float(np.mean(
            np.sign(predictions[significant]) == np.sign(targets[significant])
        ))
    else:
        direction_acc = 0.0
 
    return {
        "pearson_r": float(pearson_r),
        "spearman_r": float(spearman_r),
        "mse": mse,
        "mae": mae,
        "r_squared": float(r_squared),
        "direction_acc": direction_acc,
    }