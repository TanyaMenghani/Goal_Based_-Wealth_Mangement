"""
Exp 1: Can a cheap model predict g_sim and p_sim from the 23 non-indicator state features?
Evaluated with GroupKFold over *goal structures* (case k and k+33 are the same structure, with/without
infusions, so they share a group) -> measures generalisation to unseen investor problems.
Reports (a) regression error and (b) decision agreement when the *actor networks* receive the predicted
indicator instead of the true one.
"""
import json, time, numpy as np, torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from run_rl_simulation import load_actor
from gen_states import BASE

torch.manual_seed(0); np.random.seed(0)
d = np.load("dataset.npz")
case = d["case"]; grp = (case - 1) % 33
FEAT = [1, 3, 4] + list(range(6, 27))            # 24 informative inputs (drop g, p slots and the constant inflation slot)
adv = load_actor(BASE / "models" / "ppo_actor_goal_0.pth", 27)
inv = load_actor(BASE / "models" / "ppo_actor_port_0.pth", 27)

def actor(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

def feats(S):  # add log-ratio features (the indicators depend on W through ratios)
    X = S[:, FEAT]
    eps = 1e-6
    extra = np.log(np.maximum(S[:, [3, 4]], eps))
    return np.hstack([X, extra]).astype(np.float32)

class MLP(nn.Module):
    def __init__(s, i, h=(64, 32)):
        super().__init__()
        L = []; p = i
        for k in h: L += [nn.Linear(p, k), nn.Tanh()]; p = k
        L += [nn.Linear(p, 1), nn.Sigmoid()]
        s.net = nn.Sequential(*L)
    def forward(s, x): return s.net(x).squeeze(-1)

def fit_mlp(X, y, h=(64, 32), epochs=40):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = MLP(X.shape[1], h); opt = torch.optim.Adam(m.parameters(), 3e-3)
    Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y, dtype=torch.float32)
    n = len(Xt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 2048):
            idx = perm[i:i + 2048]
            opt.zero_grad(); loss = ((m(Xt[idx]) - yt[idx]) ** 2).mean(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    return lambda Z: m(torch.tensor((Z - mu) / sd)).detach().numpy()

def fit_hgb(X, y):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=31).fit(X, y)
    return lambda Z: np.clip(m.predict(Z), 0, 1)

def fit_ridge(X, y):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = Ridge(1e-3).fit((X - mu) / sd, y)
    return lambda Z: np.clip(m.predict((Z - mu) / sd), 0, 1)

MODELS = {"ridge(linear)": fit_ridge, "MLP 64-32": fit_mlp, "HistGB": fit_hgb}

# ---------- build the two supervised problems ----------
GS = d["goal_state"]; has_goal = GS[:, 6] > 0           # bucket-0 cost/W > 0  <=> a goal exists at current step
Xg = feats(GS)[has_goal]; yg = GS[has_goal, 0]; gg = grp[has_goal]; ag_true = d["a_goal"][has_goal]
GS_h = GS[has_goal]
PS = np.vstack([d["port_t"], d["port_n"]]); yp = PS[:, 2]; gp = np.concatenate([grp, grp])
Xp = feats(PS); ap_true = np.concatenate([d["a_pt"], d["a_pn"]])
print(f"goal problem rows={len(yg)}   port problem rows={len(yp)}")

res = {}
def run(name, X, y, g, S, actfn, kind):
    for mname, fit in MODELS.items():
        pred = np.zeros_like(y); t_fit = 0
        for tr, te in GroupKFold(5).split(X, y, g):
            t0 = time.time(); f = fit(X[tr], y[tr]); t_fit += time.time() - t0
            pred[te] = f(X[te])
        mae = np.abs(pred - y).mean(); r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        S2 = S.copy(); S2[:, 0 if kind == "goal" else 2] = pred
        a_hat = actfn(S2)
        S1 = S.copy()  # actor with TRUE indicator
        a_true_ind = actfn(S1)
        if kind == "goal":
            agree_actor = (np.round(a_hat) == np.round(a_true_ind)).mean()
            agree_direct = (np.round(pred) == np.round(y)).mean()
            r = dict(MAE=mae, R2=r2, actor_action_agree=agree_actor, direct_round_agree=agree_direct)
        else:
            ih, it = np.round(a_hat * 14), np.round(a_true_ind * 14)
            r = dict(MAE=mae, R2=r2, actor_idx_exact=(ih == it).mean(), actor_idx_pm1=(np.abs(ih - it) <= 1).mean(),
                     direct_idx_pm1=(np.abs(np.round(pred * 14) - np.round(y * 14)) <= 1).mean())
        r["fit_s_total_5fold"] = t_fit
        res[f"{name}|{mname}"] = {k: float(v) for k, v in r.items()}
        print(f"{name:7s} {mname:14s}", {k: round(float(v), 4) for k, v in r.items()}, flush=True)

run("g_sim", Xg, yg, gg, GS_h, lambda S: actor(adv, S), "goal")
run("p_sim", Xp, yp, gp, PS, lambda S: actor(inv, S), "port")
json.dump(res, open("exp1_results.json", "w"), indent=1)
