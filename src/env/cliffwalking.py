import numpy as np
import gymnasium as gym
from dataclasses import dataclass

from src.utils.utils import (
    finite_horizon_optimal_Qh,
    stochastic_policy_from_Qh,
    calculation_Q_policy_evaluation,
)


def get_base_P(env_id="CliffWalking-v0"):
    env = gym.make(env_id)
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n
    env.close()
    return P, nS, nA


def slip_kernel(P, nS, nA, eps):
    P_eps = {s: {a: [] for a in range(nA)} for s in range(nS)}

    def aggregate(transitions):
        agg = {}
        for p, s2, r, done in transitions:
            key = (int(s2), float(r), bool(done))
            agg[key] = agg.get(key, 0.0) + float(p)
        return agg

    avg_agg = {}
    for s in range(nS):
        agg_sum = {}
        for a in range(nA):
            agg_a = aggregate(P[s][a])
            for key, prob in agg_a.items():
                agg_sum[key] = agg_sum.get(key, 0.0) + prob / nA
        avg_agg[s] = agg_sum

    for s in range(nS):
        for a in range(nA):
            agg_intended = aggregate(P[s][a])
            agg_rand = avg_agg[s]
            agg_mix = {}

            for key, prob in agg_intended.items():
                agg_mix[key] = agg_mix.get(key, 0.0) + (1.0 - eps) * prob
            for key, prob in agg_rand.items():
                agg_mix[key] = agg_mix.get(key, 0.0) + eps * prob

            P_eps[s][a] = [(prob, key[0], key[1], key[2]) for key, prob in agg_mix.items() if prob > 0]

    return P_eps


class TabularKernelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, P, nS, nA, d0=None, seed=0):
        super().__init__()
        self.P = P
        self.nS = nS
        self.nA = nA
        self._seed = int(seed) if seed is not None else None
        self.rng = np.random.default_rng(self._seed)

        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)

        if d0 is None:
            d0 = np.ones(nS, dtype=np.float64) / nS
        d0 = np.asarray(d0, dtype=np.float64)
        self.d0 = d0 / max(d0.sum(), 1e-12)

        self.s = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(self._seed)
        self.s = int(self.rng.choice(self.nS, p=self.d0))
        return self.s, {}

    def step(self, action):
        a = int(action)
        trans = self.P[self.s][a]
        ps = np.asarray([t[0] for t in trans], dtype=np.float64)
        ps = ps / max(ps.sum(), 1e-12)
        idx = int(self.rng.choice(len(trans), p=ps))
        _, s2, r, done = trans[idx]
        self.s = int(s2)
        terminated = bool(done)
        truncated = False
        return self.s, float(r), terminated, truncated, {}


class KernelRandomizedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, kernels, nS, nA, d0, seed=0, weights=None):
        super().__init__()
        self.kernels = list(kernels)
        self.nS = nS
        self.nA = nA

        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)

        self.rng = np.random.default_rng(seed)

        d0 = np.asarray(d0, dtype=np.float64)
        self.d0 = d0 / max(d0.sum(), 1e-12)

        if weights is None:
            weights = np.ones(len(self.kernels), dtype=np.float64) / max(len(self.kernels), 1)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / max(self.weights.sum(), 1e-12)

        self.P = None
        self.s = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))

        k = int(self.rng.choice(len(self.kernels), p=self.weights))
        self.P = self.kernels[k]
        self.s = int(self.rng.choice(self.nS, p=self.d0))
        return self.s, {}

    def step(self, action):
        a = int(action)
        trans = self.P[self.s][a]
        ps = np.asarray([t[0] for t in trans], dtype=np.float64)
        ps = ps / max(ps.sum(), 1e-12)
        idx = int(self.rng.choice(len(trans), p=ps))
        _, s2, r, done = trans[idx]
        self.s = int(s2)
        terminated = bool(done)
        truncated = False
        return self.s, float(r), terminated, truncated, {}


def average_kernels(P_list, nS, nA):
    def aggregate(transitions):
        agg = {}
        for p, s2, r, done in transitions:
            key = (int(s2), float(r), bool(done))
            agg[key] = agg.get(key, 0.0) + float(p)
        return agg

    P_avg = {s: {a: [] for a in range(nA)} for s in range(nS)}
    K = len(P_list)

    for s in range(nS):
        for a in range(nA):
            agg_sum = {}
            for k in range(K):
                agg_k = aggregate(P_list[k][s][a])
                for key, prob in agg_k.items():
                    agg_sum[key] = agg_sum.get(key, 0.0) + prob / K
            P_avg[s][a] = [(prob, key[0], key[1], key[2]) for key, prob in agg_sum.items() if prob > 0]

    return P_avg


def _try_get_cliff_shape(env):
    unwrapped = env.unwrapped
    nrow = getattr(unwrapped, "nrow", None)
    ncol = getattr(unwrapped, "ncol", None)
    if isinstance(nrow, int) and isinstance(ncol, int) and nrow > 0 and ncol > 0:
        return nrow, ncol
    return 4, 12


def _try_get_start_state(env, nrow, ncol):
    unwrapped = env.unwrapped
    s = getattr(unwrapped, "start_state", None)
    if isinstance(s, (int, np.integer)):
        return int(s)
    return (nrow - 1) * ncol + 0


def make_d0_cliff_start(env, nS):
    """Gym-like: start state one-hot."""
    nrow, ncol = _try_get_cliff_shape(env)
    start = _try_get_start_state(env, nrow, ncol)
    d0 = np.zeros(nS, dtype=np.float64)
    d0[start] = 1.0
    return d0


def make_d0_cliff_hard_band(env, nS, mode="near_cliff"):
    nrow, ncol = _try_get_cliff_shape(env)
    start = _try_get_start_state(env, nrow, ncol)
    goal = (nrow - 1) * ncol + (ncol - 1)

    d0 = np.zeros(nS, dtype=np.float64)

    if mode == "near_cliff":
        r = nrow - 2
        idxs = [r * ncol + c for c in range(ncol)]
        idxs = [s for s in idxs if 0 <= s < nS]
    elif mode == "cliff_row":
        r = nrow - 1
        idxs = [r * ncol + c for c in range(ncol)]
        idxs = [s for s in idxs if s not in (start, goal) and 0 <= s < nS]
    elif mode == "uniform_nonterminal":
        P = env.unwrapped.P
        idxs = []
        for s in range(nS):
            always_done = True
            for a in range(env.action_space.n):
                for p, s2, r, done in P[s][a]:
                    if not bool(done):
                        always_done = False
                        break
                if not always_done:
                    break
            if not always_done:
                idxs.append(s)
    else:
        raise ValueError(f"Unknown hard mode: {mode}")

    if len(idxs) == 0:
        d0[:] = 1.0 / nS
        return d0

    d0[idxs] = 1.0 / len(idxs)
    return d0


def mix_d0(d0, d0_hard, alpha):
    alpha = float(alpha)
    alpha = min(max(alpha, 0.0), 1.0)
    d = (1.0 - alpha) * np.asarray(d0, dtype=np.float64) + alpha * np.asarray(d0_hard, dtype=np.float64)
    s = d.sum()
    return d / max(s, 1e-12)


@dataclass
class CliffKernelExperimentFH:
    env_id: str
    nS: int
    nA: int
    H: int
    P_train: dict
    P_infer: dict
    d0_train: np.ndarray
    d0_inf: np.ndarray
    Q_star_infer_h: np.ndarray  
    Q_star_train_h: np.ndarray  
    pi_star_h: np.ndarray       
    env_train: gym.Env
    rng_eval: np.random.Generator


def build_experiment_finite_horizon(
    env_id="CliffWalking-v0",
    eps_infer=0.0,
    eps_train=0.2,
    alpha_d0=0.0,
    max_steps_per_ep=200,
    seed=0,
    tau_star=1e-6,
    env_randomization=False,
    mix_kernels=5,
    mix_eps_low=0.0,
    mix_eps_high=0.3,
    hard_d0_mode="near_cliff",
):
    rng_eval = np.random.default_rng(seed)

    P_base, nS, nA = get_base_P(env_id)
    P_infer = slip_kernel(P_base, nS, nA, eps_infer)

    env0 = gym.make(env_id)
    d0_train = make_d0_cliff_start(env0, nS)
    d0_hard = make_d0_cliff_hard_band(env0, nS, mode=hard_d0_mode)
    d0_inf = mix_d0(d0_train, d0_hard, alpha=alpha_d0)
    env0.close()

    H = int(max_steps_per_ep)

    Q_star_infer_h = finite_horizon_optimal_Qh(P_infer, nS, nA, H=H)            
    pi_star_h = stochastic_policy_from_Qh(Q_star_infer_h, tau_star=tau_star)    

    if not env_randomization:
        P_train = slip_kernel(P_base, nS, nA, eps_train)
        env_train = TabularKernelEnv(P_train, nS, nA, d0=d0_train, seed=seed)
        _, Q_star_train_h = calculation_Q_policy_evaluation(P_train, nS, nA, pi_star_h, H=H)
    else:
        rng = np.random.default_rng(seed)
        eps_list = rng.uniform(mix_eps_low, mix_eps_high, size=int(mix_kernels)).tolist()
        P_train_list = [slip_kernel(P_base, nS, nA, float(eps)) for eps in eps_list]
        env_train = KernelRandomizedEnv(P_train_list, nS, nA, d0=d0_train, seed=seed)

        P_train = average_kernels(P_train_list, nS, nA)
        _, Q_star_train_h = calculation_Q_policy_evaluation(P_train, nS, nA, pi_star_h, H=H)

    return CliffKernelExperimentFH(
        env_id=env_id,
        nS=nS, nA=nA, H=H,
        P_train=P_train, P_infer=P_infer,
        d0_train=d0_train, d0_inf=d0_inf,
        Q_star_infer_h=Q_star_infer_h,
        Q_star_train_h=Q_star_train_h,
        pi_star_h=pi_star_h,
        env_train=env_train,
        rng_eval=rng_eval,
    )
