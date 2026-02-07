"""
sr_admm.py — Straggler-Resilient ADMM
=============================================================================
논문 Section IV-A, Eq. (15), Algorithms 1-2.

CC-ADMM과의 핵심 차이:
  - 매 iteration에 일부 노드(N^k ⊂ N)만 참여해도 z^k 계산 가능
  - Straggler를 기다리지 않고 Nmin개 이상 도착하면 진행
  - 중앙 노드가 z를 계산 (centralized)

Key formulas:
  z^k = (1/N_{1:k}) Σ_{i∈N^{1:k}} s_i^{k_i}          — Eq. (13)

  Dual update (Eq. 15):
    Returning (i ∈ N^{1:k-1}): y_i^k = y_i^{k-1} + ρ(x_i^k - z^{k-1})
    New       (i ∉ N^{1:k-1}): y_i^k = ρ(x_i^k - x_i^0)   — Eq. (10)

  s_i^k = x_i^k + y_i^k / ρ                            — Eq. (14)
=============================================================================
"""

import numpy as np
from common import sim_times


def sr_admm(data, rho, K, x_init, Nmin=2):
    """
    Straggler-Resilient ADMM (centralized).

    Args:
        data:   [(A_i, b_i), ...] — 노드별 데이터 (길이 N)
        rho:    ρ — penalty parameter
        K:      총 iteration 수
        x_init: [x_0^0, ...] — 초기값 (길이 N)
        Nmin:   z 계산을 위한 최소 참여 노드 수

    Returns:
        z_list: z 히스토리
        t_list: 누적 wall time 히스토리
    """
    N = len(data)
    n = data[0][0].shape[1]

    # Pre-compute
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    # Initialize
    x  = [xi.copy() for xi in x_init]
    x0 = [xi.copy() for xi in x_init]   # 초기값 보관 (New member용)
    y  = [-rho * xi.copy() for xi in x_init]
    z  = np.zeros(n)
    s  = [np.zeros(n) for _ in range(N)]  # 각 노드의 최신 s_i

    contribs = [0] * N   # 노드별 누적 기여 횟수
    ever = set()          # N^{1:k} — 한 번이라도 기여한 노드 집합

    z_list = [z.copy()]
    t_list = [0.0]
    cum_time = 0.0

    for k in range(1, K + 1):
        # --- Straggler 시뮬레이션 ---
        # 각 노드의 계산 시간을 시뮬레이션, 빠른 순으로 정렬
        times = sim_times(N, straggler_prob=0.3)
        order = np.argsort(times)

        # Nmin번째 노드가 끝나는 시간 + 약간의 여유(1.1배) 내에 도착한 노드들
        Nk = list(order[:max(Nmin,
                             np.sum(times <= np.sort(times)[Nmin - 1] * 1.1).astype(int))])

        # Wall time: Nmin번째 노드가 끝나는 시간
        cum_time += times[order[min(len(Nk) - 1, Nmin - 1)]]

        # --- 참여 노드들의 업데이트 ---
        for i in Nk:
            # x-update: x_i^k = Q_i @ (A_i^T b_i - y_i + ρ z)
            x[i] = Q[i] @ (g[i] - y[i] + rho * z)

            # y-update (Eq. 15, line 2)
            if contribs[i] > 0:
                # Returning member: 표준 dual update
                y[i] = y[i] + rho * (x[i] - z)
            else:
                # New member: y_i = ρ(x_i - x_i^0)  — Eq. (10)
                y[i] = rho * (x[i] - x0[i])

            # s-update: s_i = x_i + y_i / ρ  — Eq. (14)
            s[i] = x[i] + y[i] / rho

            ever.add(i)
            contribs[i] += 1

        # --- 중앙 노드: z 계산 ---
        # z^k = (1/N_{1:k}) Σ_{i∈N^{1:k}} s_i  — Eq. (13)
        if ever:
            z = np.mean([s[i] for i in ever], axis=0)

        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list
