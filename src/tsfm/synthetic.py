import numpy as np 

def _reject_sample_roots(rng, order, max_attempts=200):

    for _ in range(max_attempts):
        c = rng.normal(0,0.4, size= order)
        roots = np.roots(np.concatenate([[1.0], -c]))
        if np.all(np.abs(roots)<0.99):
            return c 
    return np.array([0.5]+[0.5]* (order-1))

def generate_wave(length, rng, fn):
    period = float(rng.integers(4, max(5, length // 2 + 1)))
    phase = float(rng.uniform(0, 2 * np.pi))
    amp = float(np.clip(rng.normal(1.0, 0.5), 0.1, 3.0))
    t = np.arange(length, dtype=np.float32)
    return (amp * fn(2 * np.pi * t / period + phase)).astype(np.float32)


def generate_piecewise_trend(length, rng):
    k = int(rng.integers(2,9))
    bp = np.sort(rng.choice(np.arange(1,length), size = k-1 , replace=False)) if k >1  else np.array([], int)
    breaks = np.concatenate([[0], bp , [length]]).astype(int)
    slopes = rng.normal(0,0.5, size=k)
    trend = np.zeros(length , np.float32)
    intercept = 0.0 
    for i in range(k):
        t0,t1 = breaks[i] , breaks[i+1]
        t = np.arange(t0, t1 , dtype=np.float32)
        trend[t0:t1] = intercept + slopes[i] * (t - t0)
        if t1 > t0:
            intercept = float(trend[t1 - 1])           # keep the line continuous at each break
    return trend

def generate_arma(length, rng, max_attempts=100):
    p = int(rng.integers(1, 9))
    q = int(rng.integers(1, 9))
    ar = _reject_sample_roots(rng, p, max_attempts)        # stationary AR coefficients
    ma = _reject_sample_roots(rng, q, max_attempts)        # invertible MA coefficients
    burn = max(p, q) * 10                                  # let the process forget its zero start
    n = burn + length
    eps = rng.normal(0, 1.0, size=n)
    y = np.zeros(n, np.float32)
    for t in range(n):
        pp = min(p, t); qq = min(q, t)
        ar_part = float(np.dot(ar[:pp], y[t - pp:t][::-1])) if pp else 0.0
        ma_part = float(np.dot(ma[:qq], eps[t - qq:t][::-1])) if qq else 0.0
        y[t] = ar_part + eps[t] + ma_part
    return y[burn:].astype(np.float32)

def generate_series(length, rng, rw_prob=0.2):
    if rng.random() < rw_prob:                                          # ~20% structureless RW/white-noise
        y = generate_random_walk(length, rng)
    else:
        on = {k: bool(rng.random() < 0.5) for k in ("trend", "arma", "sine", "cosine")}
        if not any(on.values()):
            on[("trend", "arma", "sine", "cosine")[int(rng.integers(0, 4))]] = True
        comp = {}
        if on["trend"]:
            comp["trend"] = generate_piecewise_trend(length, rng)
        if on["arma"]:
            comp["arma"] = generate_arma(length, rng)
        if on["sine"]:
            comp["sine"] = generate_wave(length, rng, np.sin)
        if on["cosine"]:
            comp["cosine"] = generate_wave(length, rng, np.cos)
        names = list(comp)
        w = rng.uniform(0, 1, size=len(names))
        w = w / w.sum()
        y_add = np.zeros(length, np.float32)
        for name, wi in zip(names, w):
            if name != "trend":
                y_add += wi * comp[name]
        if "trend" in comp:                                            # trend can be additive or multiplicative
            tr = comp["trend"]
            wt = float(w[names.index("trend")])
            denom = 1.0 + float(np.std(np.abs(tr)))
            if rng.random() < 0.5:
                y = (1.0 + wt * (tr / denom)) * y_add
            else:
                y = wt * tr + y_add
        else:
            y = y_add
        if rng.random() < 0.1:
            y = y + rng.normal(0, 0.1, size=length)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)              # never emit nan/inf
    y = np.clip(y, -1e4, 1e4)
    s = float(y.std())
    if s > 1e-6:
        y = (y - float(y.mean())) / s                                  # standardize to ~unit scale
    return y.astype(np.float32)