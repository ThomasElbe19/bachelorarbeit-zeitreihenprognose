import numpy as np

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rmse_by_vol_quantiles(
    y_true,
    y_pred,
    vol,
    min_bin_size: int = 5,
):
    """
    RMSE getrennt nach drei Volatilitätsregimen (niedrig / mittel / hoch).

    - nutzt 33%- und 66%-Quantile der Volatilität
    - wenn Quantile kollabieren oder Bins zu klein sind, wird
      auf den globalen RMSE zurückgefallen
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    vol    = np.asarray(vol,    dtype=float)

    # gemeinsame Maske: nur Einträge mit vollen Infos
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & ~np.isnan(vol)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    vol    = vol[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan

    # globaler RMSE (Fallback für leere / instabile Bins)
    global_rmse = rmse(y_true, y_pred)

    # wenn Volatilität praktisch konstant ist → alle Regime identisch
    if len(np.unique(vol)) == 1:
        return global_rmse, global_rmse, global_rmse

    q1, q2 = np.quantile(vol, [0.33, 0.66])

    # Falls Quantile kollabieren (z.B. q1 == q2), kein sauberes Split möglich
    if q1 == q2:
        return global_rmse, global_rmse, global_rmse

    low_mask  = vol <= q1
    mid_mask  = (vol > q1) & (vol <= q2)
    high_mask = vol > q2

    def rmse_for_mask(m):
        n = int(np.sum(m))
        if n < min_bin_size:
            # zu wenig Punkte im Bin → globaler RMSE als robustes Fallback
            return global_rmse
        return rmse(y_true[m], y_pred[m])

    rmse_low  = rmse_for_mask(low_mask)
    rmse_mid  = rmse_for_mask(mid_mask)
    rmse_high = rmse_for_mask(high_mask)

    return rmse_low, rmse_mid, rmse_high
