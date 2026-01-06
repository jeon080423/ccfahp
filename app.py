import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import io
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fuzzy AHP 분석 시스템", layout="wide", page_icon="📊")

# -----------------------------
# 1. 기본 상수
# -----------------------------
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Saaty 척도 → 삼각퍼지수 (Chang 1996 근사)
FUZZY_SCALE = {
    1: (1, 1, 1),
    2: (1, 2, 3),
    3: (2, 3, 4),
    4: (3, 4, 5),
    5: (4, 5, 6),
    6: (5, 6, 7),
    7: (6, 7, 8),
    8: (7, 8, 9),
    9: (9, 9, 9),
}

# -----------------------------
# 2. AHP 관련 함수
# -----------------------------
def convert_punch_to_matrix(punch_data, n_factors):
    """펀칭 데이터를 쌍대비교 행렬로 변환 (음수=좌측 중요, 양수=우측 중요)."""
    mat = np.ones((n_factors, n_factors))
    idx = 0
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            v = punch_data[idx]
            if v < 0:
                a = abs(v)
                if a > 1:
                    mat[i, j] = 1 / a
                    mat[j, i] = a
            elif v > 1:
                mat[i, j] = v
                mat[j, i] = 1 / v
            # v == 1이면 이미 1
            idx += 1
    return mat


def ahp_weights(matrix):
    """고유벡터 기반 AHP 가중치 및 CR."""
    n = matrix.shape[0]
    eigvals, eigvecs = linalg.eig(matrix)
    max_idx = np.argmax(eigvals.real)
    w = np.abs(eigvecs[:, max_idx].real)
    w = w / w.sum()
    lam_max = eigvals[max_idx].real
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    CR = CI / RI.get(n, 1.49) if n > 2 else 0
    return w, lam_max, CI, CR


def correct_matrix(matrix, threshold=0.1, max_iter=10):
    """CR이 threshold 이하가 되도록 간단 보정."""
    mat = matrix.copy()
    _, _, _, CR = ahp_weights(mat)
    orig_CR = CR
    it = 0
    while CR > threshold and it < max_iter:
        n = mat.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                g = np.sqrt(mat[i, j] * mat[j, i])
                if g <= 0:
                    g = 1
                mat[i, j] = g
                mat[j, i] = 1 / g
        _, _, _, CR = ahp_weights(mat)
        it += 1
    return mat, orig_CR, CR, it


def geometric_mean_matrix(mats):
    """여러 행렬의 기하평균."""
    if len(mats) == 0:
        return None
    mats = np.array(mats)
    logm = np.log(mats)
    gm = np.exp(logm.mean(axis=0))
    return gm


# -----------------------------
# 3. Fuzzy 연산 함수
# -----------------------------
def saaty_to_fuzzy_scalar(v):
    """양수 Saaty 값 v (>=1)를 TFN으로 변환."""
    v = max(1, min(9, int(round(v))))
    return FUZZY_SCALE[v]


def reciprocal_fuzzy(tfn):
    """TFN의 역수."""
    l, m, u = tfn
    return (1 / u, 1 / m, 1 / l)


def fuzzy_add(f1, f2):
    l1, m1, u1 = f1
    l2, m2, u2 = f2
    return (l1 + l2, m1 + m2, u1 + u2)


def defuzzify_tfn_array(Si, method="weighted"):
    """Si: shape (n,3) TFN 배열 → 정규화된 crisp 가중치."""
    L = Si[:, 0]
    M = Si[:, 1]
    U = Si[:, 2]
    if method == "weighted":
        c = (L + 2 * M + U) / 4
    elif method == "arithmetic":
        c = (L + M + U) / 3
    elif method == "geometric":
        L2 = np.where(L <= 0, 1e-9, L)
        M2 = np.where(M <= 0, 1e-9, M)
        U2 = np.where(U <= 0, 1e-9, U)
        c = (L2 * M2 * U2) ** (1 / 3)
    else:
        c = M.copy()
    s = c.sum()
    return c / s if s > 0 else c


# -----------------------------
# 4. Chang Extent Fuzzy AHP
# -----------------------------
def fuzzy_ahp_chang(matrix, defuzzy_method="weighted"):
    """
    Chang(1996)의 Extent Analysis 기반 Fuzzy AHP.
    입력: AHP 쌍대비교 행렬
    출력: Si (n,3), priority(정규화 가중치), crisp(참고용)
    """
    n = matrix.shape[0]

    # 1) Fuzzy pairwise matrix
    F = np.empty((n, n, 3), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                F[i, j] = (1, 1, 1)
            else:
                v = matrix[i, j]
                if v >= 1:
                    F[i, j] = saaty_to_fuzzy_scalar(v)
                else:
                    inv = 1 / v
                    F[i, j] = reciprocal_fuzzy(saaty_to_fuzzy_scalar(inv))

    # 2) 행별 fuzzy 합
    row_sum = np.zeros((n, 3))
    for i in range(n):
        s = (0.0, 0.0, 0.0)
        for j in range(n):
            s = fuzzy_add(s, tuple(F[i, j]))
        row_sum[i] = s

    # 3) 전체 합
    total = row_sum.sum(axis=0)
    total_l, total_m, total_u = total

    # 4) Si 계산
    Si = np.zeros((n, 3))
    for i in range(n):
        l_i, m_i, u_i = row_sum[i]
        Si[i, 0] = l_i / total_u
        Si[i, 1] = m_i / total_m
        Si[i, 2] = u_i / total_l

    # 5) Degree of possibility V(Si >= Sj)
    def V_geq(si, sj):
        l1, m1, u1 = si
        l2, m2, u2 = sj
        if m1 >= m2:
            return 1.0
        elif l2 >= u1:
            return 0.0
        else:
            return (u1 - l2) / ((u1 - m1) + (m2 - l2))

    # 6) d_i = min_j V(Si >= Sj)
    d = np.zeros(n)
    for i in range(n):
        vals = []
        for j in range(n):
            if i == j:
                continue
            vals.append(V_geq(Si[i], Si[j]))
        d[i] = min(vals) if vals else 1.0

    priority = d / d.sum() if d.sum() > 0 else np.ones(n) / n
    crisp = defuzzify_tfn_array(Si, method=defuzzy_method)

    return Si, priority, crisp


# -----------------------------
# 5. Streamlit UI
# -----------------------------
st.title("📊 Fuzzy AHP 분석 시스템")
st.markdown("AHP와 Fuzzy AHP를 동시에 분석하는 웹 기반 도구 (Chang Extent, 0.25 오류 수정 버전).")

with st.sidebar:
    st.header("⚙️ 분석 옵션")
    cr_th = st.slider("CR 허용 임계값", 0.0, 0.2, 0.1, 0.01)
    defuzz_disp = st.selectbox(
        "비퍼지화 방법",
        ["가중평균 (l+2m+u)/4", "산술평균 (l+m+u)/3", "기하평균 (l×m×u)^(1/3)"],
    )
    defuzz_map = {
        "가중평균 (l+2m+u)/4": "weighted",
        "산술평균 (l+m+u)/3": "arithmetic",
        "기하평균 (l×m×u)^(1/3)": "geometric",
    }
    defuzz_method = defuzz_map[defuzz_disp]

# 샘플 데이터 (원하시면 제거 가능)
st.markdown("### 📥 샘플 데이터")
sample_df = pd.DataFram
