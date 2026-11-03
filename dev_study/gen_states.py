"""
Vectorised generator of (state, indicator, actor-output) rows using the project's own
create_state_matrix(), so the labels are *exactly* what the paper's GSIM/PSIM produce.

State layout (27 = 6 lead + 3*7 aggregated goal features):
  [0] g_sim   (only filled when call=="goal")
  [1] t/T
  [2] p_sim   (only filled when call=="port")
  [3] W / pessimistic PV of all goals
  [4] W / optimistic  PV of all goals
  [5] inflation multiplier (=1 here, inflation off)
  [6:13]  GCmin / W   (7 aggregated goal buckets)
  [13:20] Gu / totut
  [20:27] GCmax / W
"""
import os, time, contextlib, io
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
BASE = Path(__file__).resolve().parent
os.chdir(BASE)
from infl_test_rl_functions import create_state_matrix
from run_rl_simulation import build_case


class Case:
    def __init__(self, k, ef_index=15, wealth_ratio=0.58):
        (self.env, obs, self.goal_list, self.W0, self.NN, self.H, self.INFLCOST,
         self.UTIL, self.EF_mu, self.EF_sig) = build_case(k, ef_index, wealth_ratio)
        self.k = k
        env = self.env
        self.T = env.TT
        self.infusions = np.array(env.infusn)
        amg = np.zeros((self.T + 2, 2))
        for t in range(self.T + 1):
            for goal in self.goal_list[t]:
                amg[t, 0 if goal[0][3] == "Real" else 1] += goal[-1][1]
        self.all_max_goals = amg
        self.gcost = np.array(env.max_c)
        self.gutil = np.array(env.max_u)

    def _args(self, t, W, call, slide_gc, slide_gu):
        env = self.env
        n = len(W)
        if len(self.INFLCOST[t]) > 0:
            mc = np.array(self.INFLCOST[t], float).reshape(1, -1)
            mu = np.array(self.UTIL[t], float).reshape(1, -1)
        else:
            mc = np.zeros((1, 1)); mu = np.zeros((1, 1))
        mc = np.tile(mc, (n, 1)); mu = np.tile(mu, (n, 1))
        return (np.asarray(W, float), env.gc, env.gu, env.gt, self.EF_mu, self.EF_sig, env.Z,
                env.infl_grow, env.minGR, env.maxGR, env.goal_arr_size, slide_gc, slide_gu, t,
                self.T, env.totut, self.infusions[t + 1:], call, env.clip, "unsorted",
                np.ones(n), env.lead_state_size, mc, mu, self.all_max_goals[t + 1:, :],
                np.ones(n), 1.0, env.real)

    def states(self, t, W, timing=None):
        """Return dict with the goal state and the two port states for a vector of wealths."""
        W = np.asarray(W, float)
        sgc = np.array(list(self.gcost[t:]) + [0.0] * t, float)
        sgu = np.array(list(self.gutil[t:]) + [0.0] * t, float)
        t0 = time.perf_counter()
        gs, oc, nc, ou, nu = create_state_matrix(*self._args(t, W, "goal", sgc, sgu))
        t1 = time.perf_counter()
        psgc, psgu = sgc.copy(), sgu.copy(); psgc[0] = 0.0; psgu[0] = 0.0
        Wt = np.maximum(0.0, W - oc); Wn = np.maximum(0.0, W - nc)
        ps_t, *_ = create_state_matrix(*self._args(t, Wt, "port", psgc, psgu))
        ps_n, *_ = create_state_matrix(*self._args(t, Wn, "port", psgc, psgu))
        t2 = time.perf_counter()
        if timing is not None:
            timing["goal_s"] = t1 - t0; timing["port_s"] = t2 - t1
        return dict(goal=gs, port_t=ps_t, port_n=ps_n, W=W, Wt=Wt, Wn=Wn)


if __name__ == "__main__":
    c = Case(1)
    W = np.exp(np.linspace(np.log(5), np.log(300), 500))
    c.states(3, W[:5])  # warm-up / compile
    tm = {}
    s = c.states(3, W, tm)
    print(tm, s["goal"].shape)
    print("gsim range", s["goal"][:, 0].min(), s["goal"][:, 0].max())
    print("psim values", np.unique(np.round(s["port_t"][:, 2] * 14)))
