import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats
import io
import warnings
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fuzzy AHP 분석 시스템", layout="wide", page_icon="📊")

# -----------------------------
# 0. 세션 상태 초기화 (로그인 관련)
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "last_login" not in st.session_state:
    st.session_state.last_login = "로그인 이력 없음"

VALID_ID = "shjeon"
VALID_PW = "@jsh2143033"

# -----------------------------
# 1. 기본 상수
# -----------------------------
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

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
            idx += 1
    return mat


def ahp_weights_geometric(matrix):
    n = matrix.shape[0]
    gm_row = np.prod(matrix, axis=1) ** (1.0 / n)
    w = gm_row / gm_row.sum()
    eigvals, _ = linalg.eig(matrix)
    lam_max = np.max(eigvals.real)
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    CR = CI / RI.get(n, 1.49) if n > 2 else 0
    return w, lam_max, CI, CR


def correct_matrix(matrix, threshold=0.1, max_iter=20, alpha=0.3):
    mat = matrix.astype(float).copy()
    w, lam, CI, CR = ahp_weights_geometric(mat)
    orig_CR = CR
    it = 0
    if CR <= threshold:
        return mat, orig_CR, CR, it
    n = mat.shape[0]
    while CR > threshold and it < max_iter:
        w, _, _, _ = ahp_weights_geometric(mat)
        ideal = np.ones_like(mat)
        for i in range(n):
            for j in range(n):
                ideal[i, j] = w[i] / w[j]
        for i in range(n):
            for j in range(i + 1, n):
                a_ij = mat[i, j]
                ideal_ij = ideal[i, j]
                if a_ij <= 0:
                    a_ij = 1.0
                if ideal_ij <= 0:
                    ideal_ij = 1.0
                log_a = np.log(a_ij)
                log_ideal = np.log(ideal_ij)
                log_new = (1 - alpha) * log_a + alpha * log_ideal
                new_ij = np.exp(log_new)
                mat[i, j] = new_ij
                mat[j, i] = 1.0 / new_ij
        _, _, _, CR = ahp_weights_geometric(mat)
        it += 1
        if CR <= threshold:
            break
    return mat, orig_CR, CR, it


def geometric_mean_matrix(mats):
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
    v = max(1, min(9, int(round(v))))
    return FUZZY_SCALE[v]


def reciprocal_fuzzy(tfn):
    l, m, u = tfn
    return (1 / u, 1 / m, 1 / l)


def fuzzy_add(f1, f2):
    l1, m1, u1 = f1
    l2, m2, u2 = f2
    return (l1 + l2, m1 + m2, u1 + u2)


def defuzzify_tfn_array(Si, method="geometric"):
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
    return c


# -----------------------------
# 4. 개선된 Chang Extent Fuzzy AHP
# -----------------------------
def degree_of_possibility(si, sj):
    l1, m1, u1 = si
    l2, m2, u2 = sj
    if m1 >= m2 and l1 >= l2:
        return 1.0
    if u1 <= l2:
        return 0.0
    return max(0.0, min(1.0, (u1 - l2) / ((u1 - m1) + (m2 - l2))))


def fuzzy_ahp_chang_improved(matrix, defuzzy_method="geometric"):
    n = matrix.shape[0]
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
    row_sum = np.zeros((n, 3))
    for i in range(n):
        s = (0.0, 0.0, 0.0)
        for j in range(n):
            s = fuzzy_add(s, tuple(F[i, j]))
        row_sum[i] = s
    total = row_sum.sum(axis=0)
    total_l, total_m, total_u = total
    Si = np.zeros((n, 3))
    for i in range(n):
        l_i, m_i, u_i = row_sum[i]
        Si[i, 0] = l_i / total_u
        Si[i, 1] = m_i / total_m
        Si[i, 2] = u_i / total_l
    V = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                V[i, j] = 1.0
            else:
                V[i, j] = degree_of_possibility(tuple(Si[i]), tuple(Si[j]))
    d = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d[i] *= V[i, j]
    if d.sum() == 0:
        w_fuzzy = np.ones(n) / n
    else:
        w_fuzzy = d / d.sum()
    crisp_S = defuzzify_tfn_array(Si, method=defuzzy_method)
    return Si, d, w_fuzzy, crisp_S, V


# -----------------------------
# 5. 요인간 통계 검정 함수 (p-value 기준)
# -----------------------------
def test_factor_significance(weights_matrix, p_threshold=0.05):
    n_experts, n_factors = weights_matrix.shape
    if n_factors < 2:
        return {
            "method": "none",
            "stat": np.nan,
            "pvalue": np.nan,
            "n_experts": n_experts,
            "n_factors": n_factors,
            "comment": "요인이 2개 미만이므로 통계 검정 불가",
        }
    if n_factors == 2:
        stat, pval = stats.ttest_rel(weights_matrix[:, 0], weights_matrix[:, 1])
        method = "paired_t_test"
    else:
        args = [weights_matrix[:, j] for j in range(n_factors)]
        stat, pval = stats.friedmanchisquare(*args)
        method = "friedman_test"
    return {
        "method": method,
        "stat": stat,
        "pvalue": pval,
        "n_experts": n_experts,
        "n_factors": n_factors,
        "p_threshold": p_threshold,
        "significant": "유의" if pval <= p_threshold else "비유의",
    }


# -----------------------------
# 6. 엑셀 생성 함수 (Fuzzy AHP 로우데이터 시트 포함)
# -----------------------------
def create_excel_with_fuzzy_raw(result_df, raw_df, fuzzy_raw_df, title="Fuzzy AHP 분석 결과"):
    wb = Workbook()
    ws_default = wb.active
    wb.remove(ws_default)

    # 시트 1: 분석결과
    ws_result = wb.create_sheet("분석결과", 0)
    ws_result["A1"] = title
    ws_result["A1"].font = Font(size=14, bold=True)
    ws_result.merge_cells("A1:E1")
    # 헤더
    for c, col in enumerate(result_df.columns, 1):
        cell = ws_result.cell(row=2, column=c, value=col)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    # 데이터
    for r, row in enumerate(result_df.itertuples(index=False), 3):
        for c, val in enumerate(row, 1):
            ws_result.cell(row=r, column=c, value=val)
    # 폭 조정
    for col in ws_result.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws_result.column_dimensions[get_column_letter(col[0].column)].width = max_len + 2

    # 시트 2: Fuzzy AHP 로우데이터
    ws_fuzzy = wb.create_sheet("Fuzzy AHP 로우데이터", 1)
    ws_fuzzy["A1"] = "Fuzzy AHP 로우데이터"
    ws_fuzzy["A1"].font = Font(size=12, bold=True)
    ws_fuzzy.merge_cells("A1:" + get_column_letter(fuzzy_raw_df.shape[1]) + "1")
    for c, col in enumerate(fuzzy_raw_df.columns, 1):
        cell = ws_fuzzy.cell(row=2, column=c, value=col)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
    for r, row in enumerate(fuzzy_raw_df.itertuples(index=False), 3):
        for c, val in enumerate(row, 1):
            ws_fuzzy.cell(row=r, column=c, value=val)
    for col in ws_fuzzy.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws_fuzzy.column_dimensions[get_column_letter(col[0].column)].width = max_len + 2

    # 시트 3: 로우데이터
    ws_raw = wb.create_sheet("로우데이터", 2)
    ws_raw["A1"] = "로우데이터"
    ws_raw["A1"].font = Font(size=12, bold=True)
    ws_raw.merge_cells("A1:" + get_column_letter(raw_df.shape[1]) + "1")
    for c, col in enumerate(raw_df.columns, 1):
        cell = ws_raw.cell(row=2, column=c, value=col)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
    for r, row in enumerate(raw_df.itertuples(index=False), 3):
        for c, val in enumerate(row, 1):
            ws_raw.cell(row=r, column=c, value=val)
    for col in ws_raw.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        ws_raw.column_dimensions[get_column_letter(col[0].column)].width = max_len + 2

    return wb


# -----------------------------
# 7. 로그인 UI
# -----------------------------
with st.sidebar:
    st.subheader("🔐 로그인")
    if st.session_state.logged_in:
        st.success(f"로그인 완료: {VALID_ID}")
        st.write(f"최근 로그인 일자: {st.session_state.last_login}")
        if st.button("로그아웃"):
            st.session_state.logged_in = False
    else:
        login_id = st.text_input("아이디", value="", key="login_id")
        login_pw = st.text_input("비밀번호", value="", type="password", key="login_pw")
        if st.button("로그인"):
            if (login_id == VALID_ID) and (login_pw == VALID_PW):
                st.session_state.logged_in = True
                st.session_state.last_login = datetime.now().strftime("%Y-%m-%d %H:%M")
                st.success("로그인 성공")
            else:
                st.error("아이디 또는 비밀번호가 올바르지 않습니다.")
        st.write(f"최근 로그인 일자: {st.session_state.last_login}")

if not st.session_state.logged_in:
    st.title("Fuzzy AHP 분석 시스템")
    st.stop()

# -----------------------------
# 8. 본문: 예시 데이터 + 엑셀 다운로드
# -----------------------------
st.title("Fuzzy AHP 분석 예시")

# 여기서는 예시로 더미 데이터 생성 (실제 코드에서는 계산 결과 사용)
result_df = pd.DataFrame(
    {
        "구분": [f"요인 {i}" for i in range(1, 7)],
        "AHP 가중치": [0.205, 0.150, 0.114, 0.121, 0.215, 0.194],
        "AHP 순위": [2, 4, 6, 5, 1, 3],
        "Fuzzy 가중치": [0.189, 0.149, 0.112, 0.124, 0.228, 0.198],
        "Fuzzy 순위": [3, 4, 6, 5, 1, 2],
    }
)

raw_df = pd.DataFrame(
    {
        "ID": list(range(1, 6)),
        "타입": ["A", "B", "C", "D", "E"],
        "1-2": [-2, -4, -5, -3, -2],
        "1-3": [-2, -3, -4, -5, -3],
    }
)

fuzzy_raw_df = pd.DataFrame(
    {
        "전문가": [f"E{i}" for i in range(1, 6)],
        "요인1": [0.189, 0.190, 0.188, 0.192, 0.187],
        "요인2": [0.149, 0.150, 0.148, 0.151, 0.147],
        "요인3": [0.112, 0.113, 0.111, 0.114, 0.110],
        "요인4": [0.124, 0.125, 0.123, 0.126, 0.122],
        "요인5": [0.228, 0.229, 0.227, 0.230, 0.226],
        "요인6": [0.198, 0.199, 0.197, 0.200, 0.196],
    }
)

st.dataframe(result_df)

if st.button("결과 엑셀 다운로드"):
    wb = create_excel_with_fuzzy_raw(result_df, raw_df, fuzzy_raw_df)
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    st.download_button(
        label="📊 Excel 파일 다운로드",
        data=bio.getvalue(),
        file_name="fuzzy_ahp_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
