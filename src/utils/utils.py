import numpy as np
import torch
import numpy as np
from scipy.sparse import csr_matrix


def finite_horizon_optimal_Qh(P, nS, nA, H):
    Qh = np.zeros((H, nS, nA), dtype=np.float64)
    V_next = np.zeros(nS, dtype=np.float64)

    for h in reversed(range(H)):
        for s in range(nS):
            for a in range(nA):
                q = 0.0
                for p, s2, r, done in P[s][a]:
                    q += p * (r + (0.0 if done else V_next[s2]))
                Qh[h, s, a] = q
        V_next = Qh[h].max(axis=1)

    return Qh

def softmax_probs(q_row, tau=1.0):
    q = np.asarray(q_row, dtype=np.float64)
    z = q / max(float(tau), 1e-8)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def stochastic_policy_from_Qh(Qh, tau_star=1.0):
    H, nS, nA = Qh.shape
    pi_probs_h = np.zeros((H, nS, nA), dtype=np.float64)
    for h in range(H):
        pi_probs_h[h] = softmax_probs(Qh[h], tau=tau_star)  
    return pi_probs_h

def greedy_policy_from_Qh(Qh):
    return Qh.argmax(axis=2).astype(int)


def sample_actions_from_probs(probs, rng):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probs, axis=1)                 
    u = rng.random(probs.shape[0])[:, None]       
    return (u <= cdf).argmax(axis=1).astype(int)   


def sample_pi_star_actions_h(pi_probs_h, rng):
    H, nS, nA = pi_probs_h.shape
    a_circ_h = np.zeros((H, nS), dtype=int)
    for h in range(H):
        a_circ_h[h] = sample_actions_from_probs(pi_probs_h[h], rng) 
    return a_circ_h

def sample_dqn_policy_actions_h(Qnet, nS, nA, H, eye, device, rng, tau_pi=1.0):
    with torch.no_grad():
        q = Qnet(eye).detach().cpu().numpy()      
    probs = softmax_probs(q, tau=tau_pi)          
    a_pi = sample_actions_from_probs(probs, rng)  
    return np.repeat(a_pi[None, :], H, axis=0)    

def sample_a_pi_abs_from_policy(model, nS, device, rng):
    a_pi_abs = np.zeros(nS, dtype=int)
    model.eval()
    with torch.no_grad():
        for s in range(nS):
            x = torch.zeros((1, nS), device=device, dtype=torch.float32)
            x[0, s] = 1.0
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            a_pi_abs[s] = int(rng.choice(len(probs), p=probs))
    return a_pi_abs



def build_Ppi_h(P_infer, nS, a_circ_h, H):
    terminal = nS
    nS_ext = nS + 1

    mats = []
    for h in range(H):
        rows = []
        cols = []
        data = []

        for s in range(nS):
            a = int(a_circ_h[h, s])
            for p, s2, r, done in P_infer[s][a]:
                rows.append(s)
                cols.append(terminal if done else int(s2))
                data.append(float(p))

        rows.append(terminal)
        cols.append(terminal)
        data.append(1.0)

        Ppi = csr_matrix((data, (rows, cols)), shape=(nS_ext, nS_ext))
        mats.append(Ppi)

    return mats


def calculate_expected_rational_gap(Ppi_h, d0, g_h, tol=1e-12):
    H = len(Ppi_h)
    nS = g_h.shape[1]
    terminal = nS
    nS_ext = nS + 1

    D = np.zeros(nS_ext, dtype=np.float64)
    D[:nS] = d0 / d0.sum()

    total = 0.0
    for h in range(H):
        total += float(D[:nS] @ g_h[h])

        alive = D[:nS].sum()
        if alive < tol:
            break


        D = D @ Ppi_h[h] 

    return total

def calculate_empirical_rational_gap(
    states, dones,
    Q_star_train_h,   
    a_circ_h,         
    a_pi_h,          
):
    H = Q_star_train_h.shape[0]
    T_steps = min(len(states), H)

    total = 0.0
    for h in range(T_steps):
        s = int(states[h])
        a_c = int(a_circ_h[h, s])
        a_p = int(a_pi_h[h, s])
        total += float(Q_star_train_h[h, s, a_c] - Q_star_train_h[h, s, a_p])
        if bool(dones[h]):
            break
    return total

class EmpiricalGapSum:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, x):
        self.sum += float(x)
        self.n += 1

    def mean(self):
        return self.sum / max(self.n, 1)


def calculation_Q_policy_evaluation(P, nS, nA, pi_probs_h, H):
    pi = np.asarray(pi_probs_h, dtype=np.float64)
    assert pi.shape == (H, nS, nA), f"pi_probs_h must be (H,nS,nA), got {pi.shape}"

    V = np.zeros((H + 1, nS), dtype=np.float64)   
    Q = np.zeros((H, nS, nA), dtype=np.float64)

    for h in range(H - 1, -1, -1):
        V_next = V[h + 1]

        for s in range(nS):
            for a in range(nA):
                q = 0.0
                for p, s2, r, done in P[s][a]:
                    q += p * (r + (0.0 if done else V_next[s2]))
                Q[h, s, a] = q

        V[h] = (Q[h] * pi[h]).sum(axis=1)

    return V, Q

