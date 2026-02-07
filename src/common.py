"""
common.py — 공통 유틸리티
=============================================================================
Based on: He et al., "Straggler-Resilient Asynchronous ADMM for
          Distributed Consensus Optimization", IEEE TSP, 2025.

데이터 생성, 분배, ρ 계산, 목적함수, 최적해 계산, 시간 시뮬레이션 등
모든 알고리즘이 공유하는 함수들을 모아둔 모듈.
=============================================================================
"""

import numpy as np


# =============================================================================
# 데이터 생성
# =============================================================================

def generate_data(n_samples=4177, n_features=7):
    """
    Abalone 유사 합성 데이터 생성.
    논문 Section VI: abalone dataset (4177 samples, 7 numeric features).
    네트워크가 없으므로 동일 구조의 합성 데이터를 사용.

    Returns:
        A: (n_samples, n_features) — 특성 행렬
        b: (n_samples,) — 타깃 벡터
    """
    rng = np.random.RandomState(2025)
    A = rng.randn(n_samples, n_features)
    scales = np.array([0.1, 0.08, 0.1, 0.5, 0.2, 0.1, 0.15])
    A = A * scales + np.array([0.5, 0.4, 0.13, 0.83, 0.37, 0.18, 0.27])
    x_true = rng.randn(n_features) * 2
    b = A @ x_true + rng.randn(n_samples) * 0.5
    return A, b


# =============================================================================
# 데이터 분배
# =============================================================================

def split_random(A, b, N):
    """N개 노드에 무작위 균등 분배."""
    idx = np.random.permutation(A.shape[0])
    parts = np.array_split(idx, N)
    return [(A[p], b[p]) for p in parts]


def split_dirichlet(A, b, N, beta=0.5):
    """
    Dirichlet 분포를 이용한 비균일(Non-IID) 분배.
    논문 Section VI, [31] Li et al., ICDE 2022 참조.
    """
    order = np.argsort(b)
    chunks = np.array_split(order, N * 5)
    props = np.random.dirichlet(np.ones(N) * beta, len(chunks))
    bins = [[] for _ in range(N)]
    for ci, ch in enumerate(chunks):
        alloc = (props[ci] * len(ch)).astype(int)
        alloc[np.argmax(alloc)] += len(ch) - alloc.sum()
        s = 0
        for ni in range(N):
            bins[ni].extend(ch[s:s + alloc[ni]].tolist())
            s += alloc[ni]
    out = []
    for idx_list in bins:
        idx = np.array(idx_list) if len(idx_list) > 0 else np.array([0])
        out.append((A[idx], b[idx]))
    return out


# =============================================================================
# 수치 유틸리티
# =============================================================================

def compute_rho(data):
    """
    ρ = 2 · max_i{L_i} + 2
    논문 Section VI, Proposition V.5 (Eq. 49).
    L_i = λ_max(A_i^T A_i)  (Lipschitz 상수)
    """
    Lmax = max(float(np.max(np.linalg.eigvalsh(Ai.T @ Ai))) for Ai, _ in data)
    return 2.0 * Lmax + 2.0


def obj_val(data, z):
    """
    전체 목적함수 값 계산.
    F(z) = Σ_i f_i(z) = Σ_i (1/2)||A_i z - b_i||²
    논문 Eq. (77).
    """
    return sum(0.5 * np.dot(Ai @ z - bi, Ai @ z - bi) for Ai, bi in data)


def optimal_solution(data):
    """
    전역 최적해 계산 (closed-form).
    x* = (Σ A_i^T A_i)^{-1} (Σ A_i^T b_i)
    """
    n = data[0][0].shape[1]
    H, g = np.zeros((n, n)), np.zeros(n)
    for Ai, bi in data:
        H += Ai.T @ Ai
        g += Ai.T @ bi
    return np.linalg.solve(H, g)


def rel_diff(z, z_star):
    """
    상대 차이 계산.
    ε^k = ||z^k - z*|| / ||z*||
    논문 Eq. (79).
    """
    return np.linalg.norm(z - z_star) / max(np.linalg.norm(z_star), 1e-12)


# =============================================================================
# Straggler 시간 시뮬레이션
# =============================================================================

def sim_times(N, straggler_prob=0.3):
    """
    노드별 계산 시간 시뮬레이션.
      - 정상 노드: t ~ U(0.8, 1.2)
      - Straggler:  t ~ U(3.0, 8.0)  (3~10배 느림)
    
    CC-ADMM은 가장 느린 노드를 매 iteration 기다려야 하고,
    SR/SRAD-ADMM은 Nmin개만 기다리면 됨 → wall time 이득.
    """
    times = np.random.uniform(0.8, 1.2, N)
    for i in range(N):
        if np.random.random() < straggler_prob:
            times[i] = np.random.uniform(3.0, 8.0)
    return times
