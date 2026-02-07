"""
srad_admm.py — Straggler-Resilient Asynchronous Decentralized ADMM
=============================================================================
논문 Section IV-B ~ IV-C, Eq. (32), Algorithms 3-4.

SR-ADMM과의 핵심 차이:
  - 탈중앙화: 중앙 노드 없이 각 노드가 z_i를 직접 계산
  - 비동기: 노드들이 독립적으로 병렬 실행
  - 증분적 z 업데이트: 각 노드가 순차적으로 z를 갱신

Key formula — Eq. (32):
  Returning member (i ∈ N^{1:k-1}):
    z_i^k = z_{prev} + (1/N_j^k) · (s_i^k − s_i^{prev})    [Replacement]
    N_j^k 변화 없음

  New member (i ∉ N^{1:k-1}):
    z_i^k = (1/N_j^k) · ((N_j^k − 1) · z_{prev} + s_i^k)   [Additive]
    N_j^k = N_{j-1}^k + 1

Conflict Resolution (Rule 1, Eq. 34):
  같은 rank j에 여러 노드가 경쟁 → ID가 작은 노드가 승리
  → 구현: 참여 노드를 ID 오름차순으로 정렬하여 처리
=============================================================================
"""

import numpy as np
from common import sim_times


def srad_admm(data, rho, K, x_init, Nmin=2):
    """
    Straggler-Resilient Asynchronous Decentralized ADMM.

    Args:
        data:   [(A_i, b_i), ...] — 노드별 데이터 (길이 N)
        rho:    ρ — penalty parameter
        K:      총 iteration 수
        x_init: [x_0^0, ...] — 초기값 (길이 N)
        Nmin:   다음 iteration 진행을 위한 최소 참여 노드 수

    Returns:
        z_list: z 히스토리
        t_list: 누적 wall time 히스토리
    """
    N = len(data)
    n = data[0][0].shape[1]

    # Pre-compute: Q_i = (A_i^T A_i + ρI)^{-1}, g_i = A_i^T b_i
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    # --- Per-node state (Algorithm 3 초기화) ---
    x      = [xi.copy() for xi in x_init]       # x_curr
    x0     = [xi.copy() for xi in x_init]       # x_initial (New member용)
    y      = [-rho * xi.copy() for xi in x_init] # y_curr = -ρ x^0
    z      = np.zeros(n)                          # z_curr = 0
    s_prev = [np.zeros(n) for _ in range(N)]     # s_{prev}: 이전 기여값

    contribs = [0] * N   # 노드별 contributions
    ever     = set()      # N^{1:k} — 한 번이라도 기여한 노드
    Ncount   = 0          # |N^{1:k}|

    z_list = [z.copy()]
    t_list = [0.0]
    cum_time = 0.0

    for k in range(1, K + 1):
        # =================================================================
        # Straggler 시뮬레이션: 어떤 노드가 이번 iteration에 참여하는가?
        # =================================================================
        times = sim_times(N, straggler_prob=0.3)
        order = np.argsort(times)

        # SRAD-ADMM에서는 aggregation이 computation과 겹침 (overlap)
        # Nmin번째 노드 완료 시간 기준, 그 시간의 1.3배 이내 완료 노드 참여
        threshold_time = np.sort(times)[min(Nmin - 1, N - 1)]
        Nk = [int(i) for i in order if times[i] <= threshold_time * 1.3]
        if len(Nk) < Nmin:
            Nk = list(order[:Nmin])

        # Conflict Resolution: lower ID first — Rule 1, Eq. (34)
        Nk.sort()

        # Wall time: overlap 효과로 Nmin번째 노드 시간의 70%
        cum_time += threshold_time * 0.7

        # =================================================================
        # 증분적 z 업데이트 — Eq. (32)
        # =================================================================
        z_base = z.copy()    # z_{i_0^k} = z^{k-1}  — Eq. (17)
        Nj = Ncount          # N_j^k 시작값 = |N^{1:k-1}|

        for nid in Nk:
            # --- Step 2: x-update ---
            x_new = Q[nid] @ (g[nid] - y[nid] + rho * z)

            # --- Step 3: y-update (Rule 4) ---
            if contribs[nid] > 0:
                # Returning: y = y + ρ(x_new - z)
                y_new = y[nid] + rho * (x_new - z)
            else:
                # New: y = ρ(x_new - x_0)  — Eq. (10)
                y_new = rho * (x_new - x0[nid])

            # s-update: s = x + y/ρ
            s_new = x_new + y_new / rho

            # --- Step 5: z 증분 업데이트 — Eq. (32) ---
            is_returning = (nid in ever)

            if Nj == 0:
                # Case A: 최초 기여자
                Nj = 1
                z_base = s_new.copy()
            elif is_returning:
                # Case C: Returning member → Replacement update
                # z = z_prev + (1/Nj)(s_new - s_old)
                # Nj 변화 없음
                z_base = z_base + (1.0 / Nj) * (s_new - s_prev[nid])
            else:
                # Case B: New member → Additive update
                # Nj += 1; z = (1/Nj)((Nj-1)*z_prev + s_new)
                Nj += 1
                z_base = (1.0 / Nj) * ((Nj - 1) * z_base + s_new)

            # --- Step 8: Commit ---
            s_prev[nid] = s_new.copy()
            x[nid] = x_new.copy()
            y[nid] = y_new.copy()
            contribs[nid] += 1
            ever.add(nid)

        # 글로벌 상태 갱신
        Ncount = len(ever)
        z = z_base.copy()

        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list
