"""
cc_admm.py — Classical Centralized ADMM
=============================================================================
논문 Eq. (2), Section I.

완전 동기식 ADMM. 매 iteration마다:
  1) 모든 N개 노드가 x_i, y_i를 계산
  2) 중앙 노드가 z를 계산
  3) 가장 느린 노드가 끝날 때까지 대기

Update steps:
  x_i^k = argmin_x  L_{ρ,i}(x, y_i^{k-1}, z^{k-1})
  y_i^k = y_i^{k-1} + ρ(x_i^k - z^{k-1})
  z^k   = (1/N) Σ_i (x_i^k + y_i^k / ρ)

Least Squares closed-form:
  x_i^k = (A_i^T A_i + ρI)^{-1} (A_i^T b_i - y_i^{k-1} + ρ z^{k-1})
=============================================================================
"""

import numpy as np
from common import sim_times


def cc_admm(data, rho, K, x_init):
    """
    Classical Centralized ADMM.

    Args:
        data:   [(A_i, b_i), ...] — 노드별 데이터 리스트 (길이 N)
        rho:    ρ — penalty parameter
        K:      총 iteration 수
        x_init: [x_0^0, x_1^0, ...] — 초기값 리스트 (길이 N)

    Returns:
        z_list: [z^0, z^1, ..., z^K] — z 히스토리
        t_list: [t^0, t^1, ..., t^K] — 누적 wall time 히스토리
    """
    N = len(data)
    n = data[0][0].shape[1]

    # Pre-compute: Q_i = (A_i^T A_i + ρI)^{-1}, g_i = A_i^T b_i
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    # Initialize (논문 Section VI)
    #   x_i^0 ~ U[-0.5, 0.5]   (x_init으로 전달받음)
    #   y_i^0 = -ρ x_i^0        (초기 평형 가정)
    #   z^0   = 0
    x = [xi.copy() for xi in x_init]
    y = [-rho * xi.copy() for xi in x_init]
    z = np.zeros(n)

    z_list = [z.copy()]
    t_list = [0.0]
    cum_time = 0.0

    for k in range(1, K + 1):
        # --- Simulated wall time ---
        # CC-ADMM must wait for ALL nodes → time = max(node times)
        times = sim_times(N, straggler_prob=0.3)
        cum_time += np.max(times)

        # --- Step 1: All nodes compute x_i^k (parallel) ---
        # x_i^k = Q_i @ (A_i^T b_i - y_i^{k-1} + ρ z^{k-1})
        for i in range(N):
            x[i] = Q[i] @ (g[i] - y[i] + rho * z)

        # --- Step 2: All nodes compute y_i^k ---
        # y_i^k = y_i^{k-1} + ρ(x_i^k - z^{k-1})
        for i in range(N):
            y[i] = y[i] + rho * (x[i] - z)

        # --- Step 3: Central node computes z^k ---
        # z^k = (1/N) Σ_i (x_i^k + y_i^k / ρ)
        z = np.mean([x[i] + y[i] / rho for i in range(N)], axis=0)

        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list
