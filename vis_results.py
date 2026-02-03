import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from cycler import cycler


def list_subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def should_keep_var(var_name, include_vars, exclude_vars):
    if include_vars is not None and len(include_vars) > 0:
        return var_name in include_vars
    if exclude_vars is not None and len(exclude_vars) > 0:
        return var_name not in exclude_vars
    return True


def set_color_cycle(ax, use_orange: bool):
    default_colors = rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        return 

    if use_orange:
        colors = default_colors
    else:
        colors = [c for i, c in enumerate(default_colors) if i != 1]

    ax.set_prop_cycle(cycler(color=colors))


def plot_one(
    base_dir,
    env,
    experiment,
    x_col,
    y_col,
    min_episode,
    max_episode,
    include_vars,
    exclude_vars,
    out_dir,
    use_orange=False,
    figsize=(3.0, 2.2),
    ymin=None,
    ymax=None,
):
    exp_dir = os.path.join(base_dir, env, experiment)
    if not os.path.isdir(exp_dir):
        print(f"[Skip] Not found: {exp_dir}")
        return

    all_vars = list_subdirs(exp_dir)
    all_vars = [v for v in all_vars if should_keep_var(v, include_vars, exclude_vars)]

    if len(all_vars) == 0:
        print(f"[Skip] No variables to plot under: {exp_dir}")
        return

    var_stats = {}

    for var in all_vars:
        var_dir = os.path.join(exp_dir, var)
        csv_files = sorted(glob.glob(os.path.join(var_dir, "result_*.csv")))
        if len(csv_files) == 0:
            continue

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError(
                    f"Missing columns in {f}. Need {x_col},{y_col}. Got {list(df.columns)}"
                )
            dfs.append(df[[x_col, y_col]])

        df_all = pd.concat(dfs, axis=0, ignore_index=True)

        if min_episode is not None:
            df_all = df_all[df_all[x_col] >= min_episode]
        if max_episode is not None:
            df_all = df_all[df_all[x_col] <= max_episode]

        if df_all.empty:
            continue

        stat = (
            df_all
            .groupby(x_col)[y_col]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(x_col)
        )

        var_stats[var] = stat

    if len(var_stats) == 0:
        print(f"[Skip] No valid CSV stats for: {env}/{experiment}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)

    set_color_cycle(ax, use_orange=use_orange)

    for var, stat in var_stats.items():
        x = stat[x_col].values
        y = stat["mean"].values
        yerr = stat["std"].values


        ax.plot(x, y, label=var, linewidth=1.2, alpha=0.9)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.25)

    ax.grid(True, alpha=0.25)


    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)

    ax.legend(fontsize=6, frameon=False)

    final_out_dir = os.path.join(out_dir, y_col)
    os.makedirs(final_out_dir, exist_ok=True)
    out_fig = os.path.join(final_out_dir, f"{env}_{experiment}.png")

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"[OK] Saved: {out_fig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="rational_exp/logs")
    parser.add_argument("--env", type=str, default="taxi", help="env name or 'all'")
    parser.add_argument("--exp", type=str, default="exp2_eps_train", help="experiment name or 'all'")

    parser.add_argument("--x_col", type=str, default="episode")
    parser.add_argument("--y_col", type=str, default="rational_risk_gap")

    parser.add_argument("--min_episode", type=float, default=None)
    parser.add_argument("--max_episode", type=float, default=None)

    parser.add_argument(
        "--include_vars",
        type=str,
        default=None,
        help="comma-separated variable folder names to include, e.g. eps_train_01,eps_train_03",
    )
    parser.add_argument(
        "--exclude_vars",
        type=str,
        default=None,
        help="comma-separated variable folder names to exclude",
    )

    parser.add_argument("--out_dir", type=str, default="figs")


    parser.add_argument(
        "--orange",
        action="store_true",
        help="Include the orange color (2nd color) in matplotlib's default color cycle",
    )


    parser.add_argument("--ymin", type=float, default=None, help="Minimum value for y-axis (e.g. 100)")
    parser.add_argument("--ymax", type=float, default=None, help="Maximum value for y-axis (e.g. 750)")

    args = parser.parse_args()

    include_vars = args.include_vars.split(",") if args.include_vars else None
    exclude_vars = args.exclude_vars.split(",") if args.exclude_vars else None

    envs = list_subdirs(args.base_dir) if args.env == "all" else [args.env]

    for env in envs:
        env_dir = os.path.join(args.base_dir, env)
        exps = list_subdirs(env_dir) if args.exp == "all" else [args.exp]

        for exp in exps:
            plot_one(
                base_dir=args.base_dir,
                env=env,
                experiment=exp,
                x_col=args.x_col,
                y_col=args.y_col,
                min_episode=args.min_episode,
                max_episode=args.max_episode,
                include_vars=include_vars,
                exclude_vars=exclude_vars,
                out_dir=args.out_dir,
                use_orange=args.orange,
                figsize=(3.0, 2.2),
                ymin=args.ymin,
                ymax=args.ymax,
            )


if __name__ == "__main__":
    main()
