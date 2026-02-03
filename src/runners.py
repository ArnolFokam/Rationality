# src/runners.py
import random
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from src.model.DQN import DQNNet
from src.regularisers import l2_param_loss

from src.utils.utils import (
    sample_pi_star_actions_h,
    sample_dqn_policy_actions_h,
    build_Ppi_h,
    calculate_expected_rational_gap,
    calculate_empirical_rational_gap,
    EmpiricalGapSum,
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_dqn(
    exp,
    num_episodes=5000,
    eval_every=20,
    lr=1e-3,
    batch_size=64,
    replay_size=50000,
    warmup=1000,
    target_sync=500,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=3000,
    device="cuda",
    seed=0,
    dqn_tau=1e-7,
    layernorm=False,
    weightnorm=False,
    l2_coef=0.0,
    logger=None,
):
    set_global_seed(seed)

    device = device.lower()
    nS, nA, H = exp.nS, exp.nA, exp.H
    env = exp.env_train

    Qnet = DQNNet(
        nS, nA,
        layernorm=layernorm,
        weightnorm=weightnorm,
    ).to(device)

    Tnet = DQNNet(
        nS, nA,
        layernorm=layernorm,
        weightnorm=weightnorm,
    ).to(device)
    Tnet.load_state_dict(Qnet.state_dict())
    opt = optim.Adam(Qnet.parameters(), lr=lr)

    eye = torch.eye(nS, device=device, dtype=torch.float32)

    buf_s = np.zeros(replay_size, dtype=np.int32)
    buf_a = np.zeros(replay_size, dtype=np.int32)
    buf_r = np.zeros(replay_size, dtype=np.float32)
    buf_s2 = np.zeros(replay_size, dtype=np.int32)
    buf_d = np.zeros(replay_size, dtype=np.float32)
    ptr, size = 0, 0

    def add_transition(s, a, r, s2, done):
        nonlocal ptr, size
        buf_s[ptr] = s
        buf_a[ptr] = a
        buf_r[ptr] = r
        buf_s2[ptr] = s2
        buf_d[ptr] = 1.0 if done else 0.0
        ptr = (ptr + 1) % replay_size
        size = min(size + 1, replay_size)

    def sample_batch():
        idx = np.random.randint(0, size, size=batch_size)
        return buf_s[idx], buf_a[idx], buf_r[idx], buf_s2[idx], buf_d[idx]

    def epsilon_by_episode(ep):
        t = min(1.0, ep / eps_decay_episodes)
        return eps_start + t * (eps_end - eps_start)

    emp_gap = EmpiricalGapSum()
    episodes_axis, exp_gap_axis, emp_gap_axis = [], [], []
    total_steps = 0
    gamma_train = 1.0  

    for ep in range(1, num_episodes + 1):
        ep_states, ep_dones = [], []

        s, _ = env.reset()
        done = False
        steps = 0
        eps = epsilon_by_episode(ep)
        ep_reward = 0.0

        while not done and steps < H:
            steps += 1
            total_steps += 1

            if np.random.rand() < eps:
                a = np.random.randint(nA)
            else:
                with torch.no_grad():
                    q = Qnet(eye[s].unsqueeze(0))
                    a = int(torch.argmax(q, dim=1).item())

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_reward += float(r)

            add_transition(s, a, r, s2, done)
            ep_states.append(int(s))
            ep_dones.append(bool(done))

            s = s2

            if size >= max(warmup, batch_size):
                bs, ba, br, bs2, bd = sample_batch()
                idx_s = torch.tensor(bs, device=device, dtype=torch.long)
                idx_s2 = torch.tensor(bs2, device=device, dtype=torch.long)

                x = eye[idx_s]
                x2 = eye[idx_s2]

                a_t = torch.tensor(ba, device=device, dtype=torch.long)
                r_t = torch.tensor(br, device=device, dtype=torch.float32)
                d_t = torch.tensor(bd, device=device, dtype=torch.float32)

                q_all = Qnet(x)  
                q_sa = q_all.gather(1, a_t.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    q2 = Tnet(x2).max(dim=1).values
                    target = r_t + (1.0 - d_t) * gamma_train * q2

                td_loss = ((q_sa - target) ** 2).mean()

                reg_loss = 0.0
                if l2_coef > 0.0:
                    reg_loss = reg_loss + l2_coef * l2_param_loss(Qnet)

                loss = td_loss + reg_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                if total_steps % target_sync == 0:
                    Tnet.load_state_dict(Qnet.state_dict())

        if ep % eval_every == 0:
            a_circ_h = sample_pi_star_actions_h(
                pi_probs_h=exp.pi_star_h,
                rng=exp.rng_eval
            )  

            a_pi_h = sample_dqn_policy_actions_h(
                Qnet=Qnet,
                nS=nS, nA=nA,
                H=H,
                eye=eye,
                device=device,
                rng=exp.rng_eval,
                tau_pi=dqn_tau,
            )  

            g_h = np.zeros((H, nS), dtype=np.float64)
            for h in range(H):
                g_h[h] = (
                    exp.Q_star_infer_h[h, np.arange(nS), a_circ_h[h]]
                    - exp.Q_star_infer_h[h, np.arange(nS), a_pi_h[h]]
                )

            Ppi_h = build_Ppi_h(exp.P_infer, nS, a_circ_h, H)
            exp_risk_gap = calculate_expected_rational_gap(Ppi_h, exp.d0_inf, g_h)

            ep_emp_risk = calculate_empirical_rational_gap(
                states=ep_states,
                dones=ep_dones,
                Q_star_train_h=exp.Q_star_train_h,
                a_circ_h=a_circ_h,
                a_pi_h=a_pi_h,
            )
            emp_gap.update(ep_emp_risk)
            emp_risk_gap = emp_gap.mean()

            rational_risk_gap = abs(exp_risk_gap - emp_risk_gap)

            episodes_axis.append(ep)
            exp_gap_axis.append(exp_risk_gap)
            emp_gap_axis.append(emp_risk_gap)

            print(
                f"[DQN ep {ep}] reward={ep_reward:.1f} | "
                f"exp_risk_gap={exp_risk_gap:.4f} | emp_risk_gap={emp_risk_gap:.4f} | "
                f"rational_risk_gap={rational_risk_gap:.4f}"
            )

            if logger is not None:
                logger.log([ep, ep_reward, exp_risk_gap, emp_risk_gap, rational_risk_gap])

    return episodes_axis, exp_gap_axis, emp_gap_axis

