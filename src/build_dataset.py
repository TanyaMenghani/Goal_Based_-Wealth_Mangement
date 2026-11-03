import sys, time, numpy as np, torch
from gen_states import Case, BASE
from run_rl_simulation import load_actor

rng = np.random.default_rng(0)
NW = int(sys.argv[1]) if len(sys.argv) > 1 else 300
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
out = sys.argv[3] if len(sys.argv) > 3 else "dataset.npz"

adv = inv = None
cols = {k: [] for k in ["case", "t", "NN", "W", "Wt", "Wn", "goal_state", "port_t", "port_n",
                          "a_goal", "a_pt", "a_pn"]}
t_start = time.time()
tot_goal = tot_port = 0.0
nrows = 0
for k in range(1, 67):
    c = Case(k)
    if adv is None:
        adv = load_actor(BASE / "models" / "ppo_actor_goal_0.pth", 27)
        inv = load_actor(BASE / "models" / "ppo_actor_port_0.pth", 27)
    ts = np.unique(np.linspace(0, c.NN - 1, min(MAXT, c.NN)).round().astype(int))
    gt = [t for t in range(c.NN) if c.gutil[t] > 0]
    ts = np.unique(np.concatenate([ts, gt[:MAXT]])) if len(gt) else ts
    for t in ts:
        slide = np.array(list(c.gcost[t:]) + [0.0] * t)
        pess = max(0.001, sum(slide / c.env.minGR))
        W = pess * np.exp(rng.uniform(np.log(0.05), np.log(3.0), NW))
        tm = {}
        s = c.states(int(t), W, tm)
        tot_goal += tm["goal_s"]; tot_port += tm["port_s"]
        with torch.no_grad():
            ag = adv(torch.tensor(s["goal"], dtype=torch.float32)).numpy().ravel()
            apt = inv(torch.tensor(s["port_t"], dtype=torch.float32)).numpy().ravel()
            apn = inv(torch.tensor(s["port_n"], dtype=torch.float32)).numpy().ravel()
        n = len(W)
        cols["case"].append(np.full(n, k)); cols["t"].append(np.full(n, t)); cols["NN"].append(np.full(n, c.NN))
        for key in ["W", "Wt", "Wn"]: cols[key].append(s[key])
        cols["goal_state"].append(s["goal"]); cols["port_t"].append(s["port_t"]); cols["port_n"].append(s["port_n"])
        cols["a_goal"].append(ag); cols["a_pt"].append(apt); cols["a_pn"].append(apn)
        nrows += n
    print(f"case {k:2d} done  rows={nrows}  elapsed={time.time()-t_start:.0f}s", flush=True)
np.savez_compressed(out, **{k: np.concatenate(v) for k, v in cols.items()})
print("TOTAL rows", nrows, "goal-indicator s", round(tot_goal, 2), "port-indicator s", round(tot_port, 2))
