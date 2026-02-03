import numpy as np
import gymnasium as gym
from dataclasses import dataclass
from src.utils.utils import finite_horizon_optimal_Qh, greedy_policy_from_Qh, stochastic_policy_from_Qh, calculation_Q_policy_evaluation

def get_base_P(env_id="Taxi-v3"):
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
            key = (s2, r, done)
            agg[key] = agg.get(key, 0.0) + p
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
                agg_mix[key] = agg_mix.get(key, 0.0) + (1 - eps) * prob
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
        self.rng = np.random.default_rng(seed)

        if d0 is None:
            d0 = np.ones(nS, dtype=np.float64) / nS
        d0 = np.asarray(d0, dtype=np.float64)
        self.d0 = d0 / d0.sum()

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
        ps = np.array([t[0] for t in trans], dtype=np.float64)
        idx = int(self.rng.choice(len(trans), p=ps))
        _, s2, r, done = trans[idx]
        self.s = int(s2)
        terminated = bool(done)
        truncated = False
        return self.s, float(r), terminated, truncated, {}


def make_d0_gym_valid_uniform(env, nS):
    d0 = np.zeros(nS, dtype=np.float64)
    valid = []
    for s in range(nS):
        _, _, pass_loc, dest = env.unwrapped.decode(s)
        if pass_loc != 4 and pass_loc != dest:
            valid.append(s)
    d0[valid] = 1.0 / len(valid)
    return d0


def make_d0_hard_passloc(env, nS, pass_locs=(0, 3)):
    d0_hard = np.zeros(nS, dtype=np.float64)
    hard = []
    for s in range(nS):
        _, _, pass_loc, dest = env.unwrapped.decode(s)
        if pass_loc in pass_locs and pass_loc != 4 and pass_loc != dest:
            hard.append(s)
    d0_hard[hard] = 1.0 / len(hard)
    return d0_hard


def mix_d0(d0, d0_hard, alpha):
    d = (1 - alpha) * d0 + alpha * d0_hard
    return d / d.sum()


@dataclass
class TaxiKernelExperimentFH:
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


class KernelRandomizedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, kernels, nS, nA, d0, seed=0, weights=None):
        super().__init__()
        self.kernels = kernels
        self.nS = nS
        self.nA = nA
        self.observation_space = gym.spaces.Discrete(nS)
        self.action_space = gym.spaces.Discrete(nA)
        self.rng = np.random.default_rng(seed)

        d0 = np.asarray(d0, dtype=np.float64)
        self.d0 = d0 / d0.sum()

        if weights is None:
            weights = np.ones(len(kernels), dtype=np.float64) / len(kernels)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()

        self.P = None
        self.s = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # sample kernel
        k = int(self.rng.choice(len(self.kernels), p=self.weights))
        self.P = self.kernels[k]
        self.s = int(self.rng.choice(self.nS, p=self.d0))
        return self.s, {}

    def step(self, action):
        a = int(action)
        trans = self.P[self.s][a]
        ps = np.array([t[0] for t in trans], dtype=np.float64)
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
            key = (s2, r, done)
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


def build_experiment_finite_horizon(
    eps_infer=0.0,
    eps_train=0.0,
    alpha_d0=0.5,
    max_steps_per_ep=500, 
    seed=0,
    tau_star=1e-6,
    env_randomization=False,
    mix_kernels=5,
    mix_eps_low=0.0,
    mix_eps_high=0.3,
):
    
    rng_eval = np.random.default_rng(seed)
    P_base, nS, nA = get_base_P("Taxi-v3")
    P_infer = slip_kernel(P_base, nS, nA, eps_infer)
    P_train = slip_kernel(P_base, nS, nA, eps_train)

    env0 = gym.make("Taxi-v3")
    d0_train = make_d0_gym_valid_uniform(env0, nS)
    d0_hard = make_d0_hard_passloc(env0, nS, pass_locs=(0, 3))
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
        P_train_list = [slip_kernel(P_base, nS, nA, eps) for eps in eps_list]
        env_train = KernelRandomizedEnv(P_train_list, nS, nA, d0=d0_train, seed=seed)
        P_train = average_kernels(P_train_list, nS, nA)
        _, Q_star_train_h = calculation_Q_policy_evaluation(P_train, nS, nA, pi_star_h, H=H)


    return TaxiKernelExperimentFH(
        nS=nS, nA=nA, H=H,
        P_train=P_train, P_infer=P_infer,
        d0_train=d0_train, d0_inf=d0_inf,
        Q_star_infer_h=Q_star_infer_h,
        Q_star_train_h=Q_star_train_h,
        pi_star_h=pi_star_h,
        env_train=env_train,
        rng_eval=rng_eval
    )
