import argparse
from datetime import datetime

from src.runners import run_dqn
from src.utils.logger import ExperimentLogger
from pathlib import Path


def make_run_name(args):
    parts = [
        f"algo={args.algo}",
        f"epsTr={args.eps_train}",
        f"epsInf={args.eps_infer}",
        f"a0={args.alpha_d0}",
        f"H={args.horizon}",
        f"mix={args.env_randomization}",
        f"mixK={args.mix_kernels}",
        f"mixLo={args.mix_eps_low}",
        f"mixHi={args.mix_eps_high}",
        f"ln={args.layernorm}",
        f"l2={args.l2_coef}",
        f"tauPi={args.dqn_tau}",
        f"tauStar={args.tau_star}",
        f"seed={args.seed}",
        f"gap={'eval' if args.expected_rational_gap == 'evaluated policy' else 'opt'}",
    ]
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eps_train", type=float, default=0.0)
    parser.add_argument("--eps_infer", type=float, default=0.0)
    parser.add_argument("--alpha_d0", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="taxi")

    parser.add_argument("--env_randomization", action="store_true")
    parser.add_argument("--mix_kernels", type=int, default=5)        
    parser.add_argument("--mix_eps_low", type=float, default=0.1)    
    parser.add_argument("--mix_eps_high", type=float, default=0.3)   

    parser.add_argument("--tau_star", type=float, default=1e-6)

    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_size", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--target_sync", type=int, default=500)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_episodes", type=int, default=3000)
    parser.add_argument("--dqn_tau", type=float, default=1e-7)  

    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)

    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--weightnorm", action="store_true")
    parser.add_argument("--l2_coef", type=float, default=0.0)   
    
    parser.add_argument("--experiment", type=str, default="exp_test", help="experiment name (e.g. exp1_main, exp2_ablation)")
    parser.add_argument("--variable", type=str, default="default", help="variable name (e.g. reg)")
    parser.add_argument(
        "--expected_rational_gap",
        type=str,
        choices=["evaluated policy", "optimal policy"],
        default="evaluated policy",
        help="state distribution to use for the expected rational gap: 'optimal policy' (d_h^{pi_circ}) or 'evaluated policy' (d_h^{pi})",
    )

    args = parser.parse_args()

    if args.env == "taxi":
        from src.env.taxi import build_experiment_finite_horizon
        exp = build_experiment_finite_horizon(
            eps_infer=args.eps_infer,
            eps_train=args.eps_train,
            alpha_d0=args.alpha_d0,
            max_steps_per_ep=args.horizon,
            seed=args.seed,
            tau_star=args.tau_star,
            env_randomization=args.env_randomization,
            mix_kernels=args.mix_kernels,
            mix_eps_low=args.mix_eps_low,
            mix_eps_high=args.mix_eps_high,
        )
    elif args.env == "cliffwalking":
        from src.env.cliffwalking import build_experiment_finite_horizon
        exp = build_experiment_finite_horizon(
            eps_infer=args.eps_infer,
            eps_train=args.eps_train,
            alpha_d0=args.alpha_d0,
            max_steps_per_ep=args.horizon,
            seed=args.seed,
            tau_star=args.tau_star,
            env_randomization=args.env_randomization,
            mix_kernels=args.mix_kernels,
            mix_eps_low=args.mix_eps_low,
            mix_eps_high=args.mix_eps_high,
        )



    header=["episode", "episode_reward", "exp_risk_gap", "emp_risk_gap", "rational_risk_gap"]

    LOG_DIR = Path("~/rational_exp/logs").expanduser()
    log_dir = LOG_DIR / args.env / args.experiment / args.variable
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"result_{args.seed}.csv"

    logger = ExperimentLogger(log_dir=log_dir, 
                              filename=log_filename, 
                              header=header)

    try:
        if args.algo == "dqn":
            run_dqn(
                exp=exp,
                num_episodes=args.num_episodes,
                eval_every=args.eval_every,
                lr=args.lr,
                batch_size=args.batch_size,
                replay_size=args.replay_size,
                warmup=args.warmup,
                target_sync=args.target_sync,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_episodes=args.eps_decay_episodes,
                device=args.device,
                seed=args.seed,
                dqn_tau=args.dqn_tau,
                layernorm=args.layernorm,
                weightnorm=args.weightnorm,
                l2_coef=args.l2_coef,
                logger=logger,
                gap_method=args.expected_rational_gap,
            )
        else:
            print("Another training not yet implemented.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
