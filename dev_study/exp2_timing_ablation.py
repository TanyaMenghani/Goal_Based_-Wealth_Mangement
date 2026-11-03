import time, json, numpy as np, torch
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from gen_states import Case, BASE
from run_rl_simulation import load_actor
torch.set_num_threads(1)
out = {}

# ------------------------------------------------------------------ 1. TIMING
adv = load_actor(BASE / "models" / "ppo_actor_goal_0.pth", 27)
inv = load_actor(BASE / "models" / "ppo_actor_port_0.pth", 27)

# surrogate MLP (same 64-32 shape as exp1), trained on all data just for a speed test
d = np.load("dataset.npz")
GS = d["goal_state"]; FEAT = [1, 3, 4] + list(range(6, 27))
def feats(S):
    return np.hstack([S[:, FEAT], np.log(np.maximum(S[:, [3, 4]], 1e-6))]).astype(np.float32)
Xs = feats(GS[:20000]); ys = GS[:20000, 0]
net = torch.nn.Sequential(torch.nn.Linear(Xs.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 32),
                          torch.nn.Tanh(), torch.nn.Linear(32, 1), torch.nn.Sigmoid()).eval()
hgb = HistGradientBoostingRegressor(max_iter=300).fit(Xs, ys)
# numpy-only MLP forward (what you'd deploy; no torch overhead)
W1, b1 = net[0].weight.detach().numpy().T.copy(), net[0].bias.detach().numpy()
W2, b2 = net[2].weight.detach().numpy().T.copy(), net[2].bias.detach().numpy()
W3, b3 = net[4].weight.detach().numpy().T.copy(), net[4].bias.detach().numpy()
def np_mlp(X): return 1 / (1 + np.exp(-(np.tanh(np.tanh(X @ W1 + b1) @ W2 + b2) @ W3 + b3)))

def bench(f, reps):
    f(); t0 = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t0) / reps

rows = {}
for k, t in [(24, 20), (22, 30), (33, 8), (66, 5)]:
    c = Case(k)
    for n in [1, 1000, 10000]:
        W = np.exp(np.linspace(np.log(20), np.log(2000), n))
        c.states(t, W[:3])
        tm_all = []
        for _ in range(5):
            tm = {}; c.states(t, W, tm); tm_all.append((tm["goal_s"], tm["port_s"]))
        g_s, p_s = np.median(np.array(tm_all), axis=0)
        s = c.states(t, W)
        X = feats(s["goal"])
        rows[f"case{k}_t{t}_n{n}"] = dict(
            true_gsim_ms=g_s * 1e3, true_psim_ms_each=p_s / 2 * 1e3,
            actor_torch_ms=bench(lambda: adv(torch.tensor(s["goal"], dtype=torch.float32)), 20) * 1e3,
            surrogate_np_mlp_ms=bench(lambda: np_mlp(X), 50) * 1e3,
            surrogate_hgb_ms=bench(lambda: hgb.predict(X), 5) * 1e3)
        print(f"case{k:2d} t={t:2d} n={n:5d}", {a: round(b, 3) for a, b in rows[f'case{k}_t{t}_n{n}'].items()}, flush=True)
out["timing"] = rows

# ------------------------------------------------------------------ 2. FEATURE ABLATION (how simple can the "formula" be?)
case = d["case"]; grp = (case - 1) % 33
has = GS[:, 6] > 0
sets = {"A: W/pess, W/optm": [3, 4], "B: A + t/T": [1, 3, 4], "C: B + current goal (cost/W,util,max)": [1, 3, 4, 6, 13, 20],
        "D: all 24 features": FEAT}
abl = {}
for target, S, mask in [("g_sim", GS, has), ("p_sim", d["port_t"], np.ones(len(GS), bool))]:
    y = S[mask, 0 if target == "g_sim" else 2]; g = grp[mask]
    for nm, F in sets.items():
        X = np.hstack([S[mask][:, F], np.log(np.maximum(S[mask][:, [3, 4]], 1e-6))]).astype(np.float32)
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, g):
            pred[te] = HistGradientBoostingRegressor(max_iter=150).fit(X[tr], y[tr]).predict(X[te])
        pred = np.clip(pred, 0, 1)
        r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        abl[f"{target}|{nm}"] = dict(MAE=float(np.abs(pred - y).mean()), R2=float(r2))
        print(target, nm, abl[f"{target}|{nm}"], flush=True)
out["ablation"] = abl
json.dump(out, open("exp2_results.json", "w"), indent=1)
