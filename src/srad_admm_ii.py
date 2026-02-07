"""
srad_admm_ii.py — SRAD-ADMM with Time Tracking
=============================================================================
논문 Section IV-D, Algorithm 5.

SRAD-ADMM과의 핵심 차이:
  동적 종료 조건 — 다음 중 하나가 만족되면 다음 iteration으로 진행:
    조건 1: N^k ≥ Nmin                           (충분한 노드 참여)
    조건 2: N^k ≥ Nmin* AND |t - t_last| ≥ δ     (시간 초과)

  δ = 마지막 두 유효 shared variable 사이의 시간 간격

이를 통해:
  - Nmin을 N으로 높게 설정해도 straggler에 의한 무한 대기 방지
  - 동적 환경에서 노드 가용성/성능 변동에 자동 적응
=============================================================================
"""

import numpy as np
from common import sim_times


def srad_admm_ii(data, rho, K, x_init, Nmin=None, Nmin_star=2):
    """
    SRAD-ADMM-II: Time-tracking extension.

    Args:
        data:      [(A_i, b_i), ...] — 노드별 데이터 (길이 N)
        rho:       ρ — penalty parameter
        K:         총 iteration 수
        x_init:    [x_0^0, ...] — 초기값 (길이 N)
        Nmin:      주 종료 조건 임계값 (기본값: N)
        Nmin_star: 보조 종료 조건 임계값 (시간 초과 시 적용)

    Returns:
        z_list: z 히스토리
        t_list: 누적 wall time 히스토리
    """
    N = len(data)
    n = data[0][0].shape[1]
    if Nmin is None:
        Nmin = N

    # Pre-compute
    Q = [np.linalg.inv(Ai.T @ Ai + rho * np.eye(n)) for Ai, _ in data]
    g = [Ai.T @ bi for Ai, bi in data]

    # Initialize
    x      = [xi.copy() for xi in x_init]
    x0     = [xi.copy() for xi in x_init]
    y      = [-rho * xi.copy() for xi in x_init]
    z      = np.zeros(n)
    s_prev = [np.zeros(n) for _ in range(N)]

    contribs = [0] * N
    ever     = set()
    Ncount   = 0

    z_list = [z.copy()]
    t_list = [0.0]
    cum_time = 0.0

    # 시간 추적 변수 (Algorithm 5, line 3)
    delta  = float('inf')    # δ: 마지막 두 유효 z 사이의 시간 간격
    t_last = 0.0             # t_pre: 마지막 유효 z의 발행 시각

    for k in range(1, K + 1):
        # =================================================================
        # 노드 도착 시뮬레이션 (도착 시간순 정렬)
        # =================================================================
        times = sim_times(N, straggler_prob=0.3)
        order = np.argsort(times)   # 빠른 순서

        z_base = z.copy()
        Nj = Ncount
        count = 0

        for idx in order:
            nid = int(idx)

            # --- x, y, s 업데이트 (SRAD-ADMM과 동일) ---
            x_new = Q[nid] @ (g[nid] - y[nid] + rho * z)

            if contribs[nid] > 0:
                y_new = y[nid] + rho * (x_new - z)
            else:
                y_new = rho * (x_new - x0[nid])

            s_new = x_new + y_new / rho

            # --- z 증분 업데이트 (Eq. 32) ---
            is_returning = (nid in ever)

            if Nj == 0:
                Nj = 1
                z_base = s_new.copy()
            elif is_returning:
                z_base = z_base + (1.0 / Nj) * (s_new - s_prev[nid])
            else:
                Nj += 1
                z_base = (1.0 / Nj) * ((Nj - 1) * z_base + s_new)

            # Commit
            s_prev[nid] = s_new.copy()
            x[nid] = x_new.copy()
            y[nid] = y_new.copy()
            contribs[nid] += 1
            ever.add(nid)
            count += 1

            # =============================================================
            # 종료 조건 확인 (Section IV-D)
            # =============================================================
            t_node = times[nid]

            # 조건 1: N^k ≥ Nmin
            if count >= Nmin:
                delta_new = abs(t_node - t_last) if t_last > 0 else delta
                delta = delta_new
                t_last = t_node
                cum_time += t_node * 0.7
                break

            # 조건 2: N^k ≥ Nmin* AND |t - t_last| ≥ δ
            if count >= Nmin_star and abs(t_node - t_last) >= delta:
                delta = abs(t_node - t_last)
                t_last = t_node
                cum_time += t_node * 0.7
                break

        # 글로벌 상태 갱신
        Ncount = len(ever)
        z = z_base.copy()

        z_list.append(z.copy())
        t_list.append(cum_time)

    return z_list, t_list
