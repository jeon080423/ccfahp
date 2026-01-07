import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats
import io
import warnings
from datetime import datetime

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import PatternFill, Alignment, Font, Border, Side
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
# 5. 통계 검정 함수 (요인/그룹)
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
        # F 검정이나 분산 분석으로 대체 가능[web:445]
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


def test_group_significance(all_results, groups, labels_kr, p_threshold=0.05):
    rows = []
    if len(groups) < 2:
        return pd.DataFrame(
            [{"요인": "전체", "method": "none", "stat": np.nan, "pvalue": np.nan,
              "p_threshold": p_threshold, "significant": "그룹 2개 미만"}]
        )

    for fi, lab in enumerate(labels_kr):
        samples = []
        for g in groups:
            w = all_results[g]["w_fuzzy"][fi]
            samples.append(w)
        try:
            stat, pval = stats.f_oneway(*[[w] for w in samples])
        except Exception:
            stat, pval = np.nan, np.nan
        rows.append(
            {
                "요인": lab,
                "method": "oneway_anova",
                "stat": stat,
                "pvalue": pval,
                "p_threshold": p_threshold,
                "significant": "유의" if (pd.notna(pval) and pval <= p_threshold) else "비유의",
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# 6. 삼각 퍼지 멤버십 함수 (그래프용)
# -----------------------------
def triangular_membership(x, a, b, c):
    a, b, c = sorted([a, b, c])
    y = np.zeros_like(x, dtype=float)

    if c == a:
        return y

    if b > a:
        idx = (x > a) & (x < b)
        y[idx] = (x[idx] - a) / (b - a)

    y[(x == b)] = 1.0

    if c > b:
        idx = (x > b) & (x < c)
        y[idx] = (c - x[idx]) / (c - b)

    y[(x <= a) | (x >= c)] = 0.0
    return y


# -----------------------------
# 7. 행렬_All 시트 출력 함수 (일반AHP + 퍼지AHP)
# -----------------------------
def export_to_excel_with_formatting(all_results, labels_kr):
    """
    all_results['All']['ahp_matrix'], all_results['All']['fuzzy_matrix']를
    '행렬_All' 통합 워크북으로 생성.
    - 가로/세로 레이블: 요인1, 요인2 ...
    - 대각선: 1 (정수), 회색 음영
    - 나머지: 소수점 3자리
    """
    wb = Workbook()
    if 'Sheet' in wb.sheetnames:
        wb.remove(wb['Sheet'])

    n_factors = len(labels_kr)

    diagonal_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # ---------- 7-1. 일반 AHP ----------
    ws_ahp = wb.create_sheet("일반AHP_행렬")

    # 헤더
    ws_ahp.append([''] + labels_kr)

    for i in range(n_factors):
        row_data = [labels_kr[i]]
        for j in range(n_factors):
            if i == j:
                row_data.append(1)
            else:
                val = all_results["All"]["ahp_matrix"][i, j]
                row_data.append(round(val, 3))
        ws_ahp.append(row_data)

    # 서식 적용
    for col in range(1, n_factors + 2):
        cell = ws_ahp.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border

    for row_idx in range(2, n_factors + 2):
        for col_idx in range(1, n_factors + 2):
            cell = ws_ahp.cell(row=row_idx, column=col_idx)
            cell.border = border
            cell.alignment = Alignment(horizontal='center', vertical='center')

            if col_idx == 1:
                cell.font = Font(bold=True)

            if col_idx - 1 == row_idx - 2:
                cell.fill = diagonal_fill
                cell.number_format = '0'
            else:
                if col_idx != 1:
                    cell.number_format = '0.000'

    ws_ahp.column_dimensions['A'].width = 12
    for c in range(2, n_factors + 2):
        ws_ahp.column_dimensions[get_column_letter(c)].width = 12

    # ---------- 7-2. 퍼지 AHP ----------
    ws_fuzzy = wb.create_sheet("퍼지AHP_행렬")

    ws_fuzzy.append([''] + labels_kr)

    for i in range(n_factors):
        row_data = [labels_kr[i]]
        for j in range(n_factors):
            if i == j:
                row_data.append(1)
            else:
                val = all_results["All"]["fuzzy_matrix"][i, j]
                row_data.append(round(val, 3))
        ws_fuzzy.append(row_data)

    for col in range(1, n_factors + 2):
        cell = ws_fuzzy.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = border

    for row_idx in range(2, n_factors + 2):
        for col_idx in range(1, n_factors + 2):
            cell = ws_fuzzy.cell(row=row_idx, column=col_idx)
            cell.border = border
            cell.alignment = Alignment(horizontal='center', vertical='center')

            if col_idx == 1:
                cell.font = Font(bold=True)

            if col_idx - 1 == row_idx - 2:
                cell.fill = diagonal_fill
                cell.number_format = '0'
            else:
                if col_idx != 1:
                    cell.number_format = '0.000'

    ws_fuzzy.column_dimensions['A'].width = 12
    for c in range(2, n_factors + 2):
        ws_fuzzy.column_dimensions[get_column_letter(c)].width = 12

    # 메모리 버퍼에 저장해서 Streamlit에서 바로 다운로드
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio


# -----------------------------
# 8. 로그인 UI
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
    st.title("📊 Fuzzy AHP 분석 시스템")
    st.markdown("제작: 전상현 / jeon080423@gmail.com")
    st.warning("좌측 로그인 후에만 분석 기능을 사용할 수 있습니다.")
    st.stop()

# -----------------------------
# 9. 메인 UI
# -----------------------------
st.title("📊 Fuzzy AHP 분석 시스템")
st.markdown("제작: 전상현 / jeon080423@gmail.com")
st.markdown("AHP와 Fuzzy AHP를 동시에 분석하는 웹 기반 도구.")

with st.sidebar:
    st.header("⚙️ 분석 옵션")

    options = [
        "기하평균 ((l×m×u)^(1/3))",
        "산술평균 ((l+m+u)/3)",
        "가중평균 ((l+2m+u)/4)",
    ]
    defuzz_disp = st.selectbox("비퍼지화 방법 (Si 비퍼지화)", options)

# 여기부터는 기존에 작성하신 업로드/분석/결과표 생성 코드를 그대로 이어 붙이면 된다.
# 마지막 Excel 다운로드 버튼만 아래처럼 수정해서 사용.

# ====== 예시: 분석 완료 후 all_results, labels_kr가 준비된 상황이라고 가정 ======
# all_results = {...}
# labels_kr = ["요인1", "요인2", ...]  # 실제 코드에서 설정

if 'all_results' in st.session_state and 'labels_kr' in st.session_state:
    all_results = st.session_state['all_results']
    labels_kr = st.session_state['labels_kr']

    st.subheader("📥 Excel 다운로드")
    if st.button("행렬_All 엑셀 다운로드"):
        bio = export_to_excel_with_formatting(all_results, labels_kr)
        st.download_button(
            label="✅ 행렬_All 다운로드",
            data=bio,
            file_name="행렬_All_일반AHP_퍼지AHP.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
