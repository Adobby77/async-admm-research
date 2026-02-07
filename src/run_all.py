"""
run_all.py — 4개 알고리즘 비교 실행 및 시각화
=============================================================================
사용법:
    python run_all.py

결과:
    ./results/ 폴더에 4장의 PNG 그래프 저장
      - fig_objective.png      : F^k vs iteration (논문 Fig. 2 스타일)
      - fig_convergence.png    : ε^k vs iteration (논문 Fig. 6 스타일)
      - fig_walltime.png       : F^k vs wall time (논문의 주 비교 지표)
      - fig_eps_walltime.png   : ε^k vs wall time
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from common import (generate_data, split_random, split_dirichlet,
                    compute_rho, obj_val, optimal_solution, rel_diff)
from cc_admm import cc_admm
from sr_admm import sr_admm
from srad_admm import srad_admm
from srad_admm_ii import srad_admm_ii

# 결과 저장 디렉토리
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_experiment(N, K, distribution, seed=42):
    """하나의 (N, distribution) 설정에 대해 4개 알고리즘 실행."""
    np.random.seed(seed)
    A, b = generate_data()
    data = split_random(A, b, N) if distribution == 'random' \
        else split_dirichlet(A, b, N)

    rho = compute_rho(data)
    z_star = optimal_solution(data)
    f_star = obj_val(data, z_star)
    n = A.shape[1]

    x_init = [np.random.uniform(-0.5, 0.5, n) for _ in range(N)]

    label = f'N={N}, {distribution}'
    print(f'\n  {label}: ρ={rho:.1f}, F*={f_star:.2f}')

    algos = OrderedDict()

    np.random.seed(100)
    algos['CC-ADMM'] = cc_admm(data, rho, K, x_init)

    np.random.seed(200)
    algos['SR-ADMM'] = sr_admm(data, rho, K, x_init, Nmin=2)

    np.random.seed(300)
    algos['SRAD-ADMM'] = srad_admm(data, rho, K, x_init, Nmin=2)

    np.random.seed(400)
    algos['SRAD-ADMM-II'] = srad_admm_ii(data, rho, K, x_init,
                                          Nmin=N, Nmin_star=2)

    metrics = OrderedDict()
    for name, (zh, th) in algos.items():
        eps = [rel_diff(z, z_star) for z in zh]
        obj = [obj_val(data, z) for z in zh]
        metrics[name] = {'eps': eps, 'obj': obj, 'time': th}

        # 수렴 iteration 찾기 (ε ≤ 0.01)
        k_conv = '>K'
        for ki in range(len(eps)):
            if eps[ki] <= 0.01:
                k_conv = ki
                break
        t_conv = f'{th[k_conv]:.1f}s' if isinstance(k_conv, int) else '-'
        print(f'    {name:<18} ε_final={eps[-1]:.4e}  '
              f'k*(0.01)={k_conv}  t*={t_conv}')

    return metrics, f_star


def plot_all(all_results):
    """4개 설정 × 4개 그래프 생성."""
    colors = {'CC-ADMM': '#2196F3', 'SR-ADMM': '#FF9800',
              'SRAD-ADMM': '#4CAF50', 'SRAD-ADMM-II': '#F44336'}
    styles = {'CC-ADMM': '-', 'SR-ADMM': '--',
              'SRAD-ADMM': '-.', 'SRAD-ADMM-II': ':'}
    lw = {'CC-ADMM': 2, 'SR-ADMM': 2,
          'SRAD-ADMM': 2.5, 'SRAD-ADMM-II': 2.5}

    n_configs = len(all_results)
    nrows = (n_configs + 1) // 2
    ncols = 2

    # --- Figure 1: Objective vs Iteration ---
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes1 = axes1.flatten() if n_configs > 1 else [axes1]
    for idx, (label, (metrics, f_star)) in enumerate(all_results.items()):
        ax = axes1[idx]
        for name, d in metrics.items():
            ax.plot(d['obj'], color=colors[name], ls=styles[name],
                    lw=lw[name], label=name, alpha=0.9)
        ax.axhline(f_star, color='gray', ls='--', alpha=0.4, lw=1)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration k')
        ax.set_ylabel('$F^k$')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    for idx in range(len(all_results), len(axes1)):
        axes1[idx].set_visible(False)
    fig1.suptitle('Average Objective Value $F^k$ over Iterations\n'
                  '(He et al., IEEE TSP 2025 — Fig. 2 style)',
                  fontsize=14, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    fig1.savefig(os.path.join(OUTPUT_DIR, 'fig_objective.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # --- Figure 2: ε vs Iteration ---
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes2 = axes2.flatten() if n_configs > 1 else [axes2]
    for idx, (label, (metrics, f_star)) in enumerate(all_results.items()):
        ax = axes2[idx]
        for name, d in metrics.items():
            ax.plot(d['eps'], color=colors[name], ls=styles[name],
                    lw=lw[name], label=name, alpha=0.9)
        ax.axhline(0.001, color='gray', ls=':', alpha=0.5, lw=1,
                   label='α=0.001')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration k')
        ax.set_ylabel('$\\epsilon^k = \\|z^k - z^*\\| / \\|z^*\\|$')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    for idx in range(len(all_results), len(axes2)):
        axes2[idx].set_visible(False)
    fig2.suptitle('Relative Difference $\\epsilon^k$ over Iterations\n'
                  '(He et al., IEEE TSP 2025 — Fig. 6 style)',
                  fontsize=14, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(os.path.join(OUTPUT_DIR, 'fig_convergence.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # --- Figure 3: Objective vs Wall Time ---
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes3 = axes3.flatten() if n_configs > 1 else [axes3]
    for idx, (label, (metrics, f_star)) in enumerate(all_results.items()):
        ax = axes3[idx]
        for name, d in metrics.items():
            ax.plot(d['time'], d['obj'], color=colors[name], ls=styles[name],
                    lw=lw[name], label=name, alpha=0.9)
        ax.axhline(f_star, color='gray', ls='--', alpha=0.4, lw=1)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Simulated Wall Time (seconds)')
        ax.set_ylabel('$F^k$')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    for idx in range(len(all_results), len(axes3)):
        axes3[idx].set_visible(False)
    fig3.suptitle('Average Objective Value $F^k$ over Simulated Wall Time\n'
                  '(He et al., IEEE TSP 2025 — Primary comparison metric)',
                  fontsize=14, fontweight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    fig3.savefig(os.path.join(OUTPUT_DIR, 'fig_walltime.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig3)

    # --- Figure 4: ε vs Wall Time ---
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes4 = axes4.flatten() if n_configs > 1 else [axes4]
    for idx, (label, (metrics, f_star)) in enumerate(all_results.items()):
        ax = axes4[idx]
        for name, d in metrics.items():
            ax.plot(d['time'], d['eps'], color=colors[name], ls=styles[name],
                    lw=lw[name], label=name, alpha=0.9)
        ax.axhline(0.001, color='gray', ls=':', alpha=0.5, lw=1,
                   label='α=0.001')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Simulated Wall Time (seconds)')
        ax.set_ylabel('$\\epsilon^k$')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    for idx in range(len(all_results), len(axes4)):
        axes4[idx].set_visible(False)
    fig4.suptitle('Relative Difference $\\epsilon^k$ over Simulated Wall Time\n'
                  '(He et al., IEEE TSP 2025)',
                  fontsize=14, fontweight='bold')
    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    fig4.savefig(os.path.join(OUTPUT_DIR, 'fig_eps_walltime.png'),
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)


# =============================================================================
if __name__ == '__main__':
    K = 2500

    print('=' * 72)
    print('  SRAD-ADMM for Distributed Least Squares')
    print('  He et al., IEEE Trans. Signal Processing, 2025')
    print('=' * 72)

    configs = [
        (10, 'random'),
        (10, 'dirichlet'),
        (20, 'random'),
        (20, 'dirichlet'),
    ]

    all_results = OrderedDict()
    for N, dist in configs:
        metrics, f_star = run_experiment(N, K, dist, seed=42)
        all_results[f'N={N}, {dist}'] = (metrics, f_star)

    plot_all(all_results)

    print(f'\n{"=" * 72}')
    print(f'  ✓ All figures saved to: {OUTPUT_DIR}')
    print(f'{"=" * 72}')
