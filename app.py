import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats
import io
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fuzzy AHP 분석 시스템", layout="wide", page_icon="📊")

# -----------------------------
# 0. 세션 상태 초기화 (로그인 관련)
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "last_login" not in st.session_state:
    st.session_state.last_login = "로그인 이력 없음"

# 고정 계정 정보
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
# 2. AHP 관련 함수 (기하평균법 사용)
# -----------------------------
def convert_punch_to_matrix(punch_data, n_factors):
    """펀칭 데이터를 쌍대비교 행렬로 변환 (음수=좌측 중요, 양수=우측 중요)."""
    mat = np.ones((n_factors, n_factors))
    idx = 0
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            v = punch_data[idx]
            if v < 0:       # 좌측이 더 중요
                a = abs(v)
                if a > 1:
                    mat[i, j] = 1 / a
                    mat[j, i] = a
            elif v > 1:     # 우측이 더 중요
                mat[i, j] = v
                mat[j, i] = 1 / v
            idx += 1
    return mat


def ahp_weights_geometric(matrix):
    """기하평균법 기반 AHP 가중치 및 일관성 지표 계산."""
    n = matrix.shape[0]
    gm_row = np.prod(matrix, axis=1) ** (1.0 / n)
    w = gm_row / gm_row.sum()

    eigvals, _ = linalg.eig(matrix)
    lam_max = np.max(eigvals.real)
    CI = (lam_max - n) / (n - 1) if n > 1 else 0
    CR = CI / RI.get(n, 1.49) if n > 2 else 0
    return w, lam_max, CI, CR


def correct_matrix(matrix, threshold=0.1, max_iter=20, alpha=0.3):
    """CR이 threshold 이하가 되도록 최소한으로 보정."""
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
    """여러 행렬의 기하평균 (집단 통합 단계)."""
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


def defuzzify_tfn_array(Si, method="geometric"):
    """Si: shape (n,3) TFN 배열 → 비퍼지화 값 (정규화 전)."""
    L = Si[:, 0]; M = Si[:, 1]; U = Si[:, 2]
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
    """V(Si >= Sj) 계산."""
    l1, m1, u1 = si
    l2, m2, u2 = sj
    if m1 >= m2 and l1 >= l2:
        return 1.0
    if u1 <= l2:
        return 0.0
    return max(0.0, min(1.0, (u1 - l2) / ((u1 - m1) + (m2 - l2))))


def fuzzy_ahp_chang_improved(matrix, defuzzy_method="geometric"):
    """개선된 Fuzzy AHP (Chang + d_i 곱 방식)."""
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

    # 5) V 행렬 계산
    V = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                V[i, j] = 1.0
            else:
                V[i, j] = degree_of_possibility(tuple(Si[i]), tuple(Si[j]))

    # 6) d_i: V 값 곱
    d = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d[i] *= V[i, j]

    # 7) 정규화하여 최종 Fuzzy 가중치
    if d.sum() == 0:
        w_fuzzy = np.ones(n) / n
    else:
        w_fuzzy = d / d.sum()

    # 8) Si 비퍼지화 (참고용)
    crisp_S = defuzzify_tfn_array(Si, method=defuzzy_method)

    return Si, d, w_fuzzy, crisp_S, V


# -----------------------------
# 5. 요인간 통계 검정 함수 (p-value 기준)
# -----------------------------
def test_factor_significance(weights_matrix, p_threshold=0.05):
    """
    요인별 가중치(전문가 x 요인)를 입력받아
    - 요인 수가 2개면 대응 t-검정
    - 3개 이상이면 Friedman 검정
    을 수행하고 p-value 기준으로 유의성 판정.
    """
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
        stat, pval = stats.ttest_rel(weights_matrix[:, 0], weights_matrix[:, 1])  # paired t-test[web:304]
        method = "paired_t_test"
    else:
        args = [weights_matrix[:, j] for j in range(n_factors)]
        stat, pval = stats.friedmanchisquare(*args)  # Friedman test[web:148]
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
# 6. 로그인 UI (사이드바 맨 위)
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
    st.warning("좌측 로그인 후에만 분석 기능을 사용할 수 있습니다.")
    st.stop()

# -----------------------------
# 7. (로그인 후 메인 분석 UI)
# -----------------------------
st.title("📊 Fuzzy AHP 분석 시스템")
st.markdown("AHP와 Fuzzy AHP를 동시에 분석하는 웹 기반 도구 (Geometric Mean Method + 개선된 Chang Extent + 통계 검정).")

with st.sidebar:
    st.header("⚙️ 분석 옵션")

    options = [
        "기하평균 ((l×m×u)^(1/3))",
        "산술평균 ((l+m+u)/3)",
        "가중평균 ((l+2m+u)/4)",
    ]
    defuzz_disp = st.selectbox("비퍼지화 방법 (Si 비퍼지화)", options)
    defuzz_map = {
        "기하평균 ((l×m×u)^(1/3))": "geometric",
        "산술평균 ((l+m+u)/3)": "arithmetic",
        "가중평균 ((l+2m+u)/4)": "weighted",
    }
    defuzz_method = defuzz_map[defuzz_disp]

    cr_th = st.slider("CR 허용 임계값", 0.0, 0.2, 0.1, 0.01)
    alpha = st.slider("CR 보정 강도 (alpha)", 0.1, 0.5, 0.3, 0.05)
    max_iter = st.slider("CR 최대 보정 횟수", 1, 30, 20, 1)

    p_ttest = st.number_input(
        "모형간 t-검정 p-value 기준", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f"
    )
    p_factor = st.number_input(
        "요인간 유의성 p-value 기준", min_value=0.0, max_value=1.0, value=0.05, step=0.01, format="%.2f"
    )

# --- 샘플 데이터 (1_2 형식 예시) ---
st.markdown("### 📥 샘플 데이터 (1_2 형식 예시)")
sample_df = pd.DataFrame(
    {
        "ID": [1, 2, 3, 4, 5, 6],
        "type": [1, 1, 1, 1, 1, 1],
        "1_2": [3, 5, 2, -2, -3, -1],
        "1_3": [5, 7, 4, 3, 5, 2],
        "1_4": [7, 9, 5, 5, 7, 4],
        "2_3": [3, 5, 3, 5, 7, 4],
        "2_4": [5, 7, 4, 7, 9, 6],
        "3_4": [3, 5, 2, 5, 7, 3],
    }
)
buf_sample = io.BytesIO()
with pd.ExcelWriter(buf_sample) as w:
    sample_df.to_excel(w, index=False, sheet_name="Sample")
st.download_button(
    "📄 샘플 다운로드",
    buf_sample.getvalue(),
    "fuzzy_ahp_sample_1_2.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# --- 데이터 업로드 ---
st.markdown("### 📤 데이터 업로드")
file = st.file_uploader("Excel 파일을 업로드하세요", type=["xlsx", "xls"])

if not file:
    st.info("👆 Excel 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.stop()

excel_file_obj = pd.ExcelFile(file)
uploaded_sheet_names = excel_file_obj.sheet_names
first_sheet_name = uploaded_sheet_names[0] if uploaded_sheet_names else "Data"

df = pd.read_excel(file)
st.success(f"파일 업로드 완료: {len(df)}행")

with st.expander("📋 데이터 미리보기"):
    st.dataframe(df.head())

id_col = df.columns[0]
type_col = df.columns[1]
comp_cols = df.columns[2:]

n_comp = len(comp_cols)
n_factor = int((1 + np.sqrt(1 + 8 * n_comp)) / 2)

# --- 요인 라벨 ---
index_set = set()
for c in comp_cols:
    name = str(c)
    if "_" in name:
        a, b = name.split("_")
        index_set.add(int(a))
        index_set.add(int(b))
if len(index_set) == n_factor:
    labels_kr = [f"요인{i}" for i in sorted(index_set)]
else:
    labels_kr = [f"요인{i+1}" for i in range(n_factor)]
labels_en = [f"Factor {i+1}" for i in range(n_factor)]

st.info(f"자동 인식: 요인 {n_factor}개, 쌍대비교 {n_comp}개  (라벨: {', '.join(labels_kr)})")

has_group = df[type_col].notna().any()
groups = df[type_col].dropna().unique() if has_group else ["All"]

if st.button("🚀 분석 시작", type="primary"):
    all_results = {}
    cons_list = []
    prog = st.progress(0.0)
    step = 1.0 / len(groups)
    factor_tests = []
    fuzzy_raw_rows = []  # 응답자별 Fuzzy AHP 로우데이터
    comp_all = {}

    for gi, g in enumerate(groups):
        gdf = df[df[type_col] == g] if has_group else df

        matrices = []
        for _, row in gdf.iterrows():
            punch = pd.to_numeric(row[comp_cols], errors="coerce").fillna(1).values
            mat = convert_punch_to_matrix(punch, n_factor)
            cmat, cr0, cr1, it = correct_matrix(
                mat, threshold=cr_th, max_iter=max_iter, alpha=alpha
            )
            cons_list.append(
                {
                    "ID": row[id_col],
                    "Group": g if has_group else "All",
                    "보정 전 CR": round(cr0, 4),
                    "보정 후 CR": round(cr1, 4),
                    "보정 횟수": it,
                    "일관성": "○" if cr1 <= cr_th else "×",
                }
            )
            matrices.append(cmat)

            # ---- 응답자별 Fuzzy AHP (보정 행렬 기준) ----
            Si_i, d_i, w_fuzzy_i, crisp_S_i, V_i = fuzzy_ahp_chang_improved(cmat, defuzz_method)
            row_dict = {
                "ID": row[id_col],
                "Group": g if has_group else "All",
            }
            for fi, lab in enumerate(labels_kr):
                row_dict[f"{lab}_Lower"] = Si_i[fi, 0]
                row_dict[f"{lab}_Medium"] = Si_i[fi, 1]
                row_dict[f"{lab}_Upper"] = Si_i[fi, 2]
                row_dict[f"{lab}_Norm"] = w_fuzzy_i[fi]
            fuzzy_raw_rows.append(row_dict)
            # --------------------------------------------

        # 집단 기하평균 행렬로 최종 AHP/Fuzzy 가중치
        gm = geometric_mean_matrix(matrices)
        w_ahp, lam, CI, CR = ahp_weights_geometric(gm)
        Si, d_raw, w_fuzzy, crisp_S, V = fuzzy_ahp_chang_improved(gm, defuzz_method)

        fuzzy_matrix = np.ones_like(gm)
        for i in range(n_factor):
            for j in range(n_factor):
                fuzzy_matrix[i, j] = w_fuzzy[i] / w_fuzzy[j]

        all_results[g] = {
            "matrix": gm,
            "fuzzy_matrix": fuzzy_matrix,
            "ahp_w": w_ahp,
            "lam": lam,
            "CI": CI,
            "CR": CR,
            "Si": Si,
            "d_raw": d_raw,
            "w_fuzzy": w_fuzzy,
            "crisp_S": crisp_S,
            "V": V,
        }

        # 요인간 유의성 검정 (집단 Fuzzy 가중치, p-value 기준)
        weights_mat = np.tile(w_fuzzy, (len(gdf), 1))
        test_res = test_factor_significance(weights_mat, p_threshold=p_factor)
        test_res["Group"] = g
        factor_tests.append(test_res)

        prog.progress((gi + 1) * step)

    st.success("분석 완료")

    cons_df = pd.DataFrame(cons_list)
    factor_test_df = pd.DataFrame(factor_tests)
    fuzzy_raw_df = pd.DataFrame(fuzzy_raw_rows)

    tabs = st.tabs(
        [
            "일관성 검증",
            "AHP 행렬",
            "비교 분석",
            "Fuzzy 상세",
            "Visualization",
            "모형간 t-검정",
            "요인간 유의성",
            "엑셀 저장",
        ]
    )

    # 1) 일관성
    with tabs[0]:
        st.dataframe(cons_df, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("총 응답자", len(cons_df))
        with c2:
            ok = (cons_df["일관성"] == "○").sum()
            st.metric("일관성 통과", f"{ok}/{len(cons_df)}")
        with c3:
            st.metric("평균 CR", f"{cons_df['보정 후 CR'].mean():.4f}")

    # 2) AHP 행렬
    with tabs[1]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            mat_df = pd.DataFrame(r["matrix"], index=labels_kr, columns=labels_kr)
            st.dataframe(mat_df.style.format("{:.4f}"), use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("λmax", f"{r['lam']:.4f}")
            with c2:
                st.metric("CI", f"{r['CI']:.4f}")
            with c3:
                st.metric("CR", f"{r['CR']:.4f}")
            with c4:
                st.metric("일관성", "✅" if r["CR"] <= cr_th else "⚠️")

    # 3) 비교 분석 (AHP vs Fuzzy)
    with tabs[2]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            ahp_rank = pd.Series(r["ahp_w"]).rank(ascending=False, method="min").astype(int)
            fuzzy_rank = pd.Series(r["w_fuzzy"]).rank(ascending=False, method="min").astype(int)
            diff = fuzzy_rank - ahp_rank
            comp = pd.DataFrame(
                {
                    "항목": labels_kr,
                    "AHP 가중치": r["ahp_w"],
                    "AHP 순위": ahp_rank,
                    "Fuzzy 가중치": r["w_fuzzy"],
                    "Fuzzy 순위": fuzzy_rank,
                    "순위 변동": diff.apply(
                        lambda x: f"▼ {abs(x)}" if x > 0 else (f"▲ {abs(x)}" if x < 0 else "—")
                    ),
                }
            )
            comp_all[g] = comp
            st.dataframe(
                comp.style.format({"AHP 가중치": "{:.4f}", "Fuzzy 가중치": "{:.4f}"}),
                use_container_width=True,
            )

    # 4) Fuzzy 상세 (집단 기준)
    with tabs[3]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            st.info(f"비퍼지화 방법(Si용): {defuzz_disp}")
            Si = r["Si"]
            detail = pd.DataFrame(
                {
                    "구분": labels_kr,
                    "Fuzzy (Lower)": Si[:, 0],
                    "Fuzzy (Medium)": Si[:, 1],
                    "Fuzzy (Upper)": Si[:, 2],
                    "Crisp(Si)": r["crisp_S"],
                    "d_i (raw)": r["d_raw"],
                    "Norm": r["w_fuzzy"],
                    "순위": pd.Series(r["w_fuzzy"]).rank(ascending=False, method="min").astype(int),
                }
            )
            st.dataframe(
                detail.style.format(
                    {
                        "Fuzzy (Lower)": "{:.4f}",
                        "Fuzzy (Medium)": "{:.4f}",
                        "Fuzzy (Upper)": "{:.4f}",
                        "Crisp(Si)": "{:.4f}",
                        "d_i (raw)": "{:.6f}",
                        "Norm": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

    # 5) Visualization (간단 예시 – 필요 시 추가 커스터마이징)
    with tabs[4]:
        for g, r in all_results.items():
            st.markdown(f"#### Group: {g}")
            Si = r["Si"]

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(labels_kr))
            ax.bar(x - 0.2, Si[:, 0], width=0.2, label="Lower")
            ax.bar(x, Si[:, 1], width=0.2, label="Medium")
            ax.bar(x + 0.2, Si[:, 2], width=0.2, label="Upper")
            ax.set_xticks(x)
            ax.set_xticklabels(labels_kr)
            ax.set_title("Fuzzy Si (Lower/Medium/Upper)")
            ax.legend()
            st.pyplot(fig)

    # 6) 모형간 t-검정 (요약 테이블 위주)
    with tabs[5]:
        st.markdown("#### 모형간 차이 (AHP vs Fuzzy)")
        t_rows = []
        for g, r in all_results.items():
            for fi, lab in enumerate(labels_kr):
                ahp_val = r["ahp_w"][fi]
                fuzzy_val = r["w_fuzzy"][fi]
                diff = fuzzy_val - ahp_val
                pct_diff = (diff / ahp_val * 100) if ahp_val != 0 else 0
                t_rows.append(
                    {
                        "Group": g,
                        "항목": lab,
                        "AHP_가중치": ahp_val,
                        "Fuzzy_가중치": fuzzy_val,
                        "차이(Fuzzy-AHP)": diff,
                        "변화율(%)": pct_diff,
                    }
                )
        t_df = pd.DataFrame(t_rows)
        st.dataframe(
            t_df.style.format(
                {"AHP_가중치": "{:.4f}", "Fuzzy_가중치": "{:.4f}", "차이(Fuzzy-AHP)": "{:.4f}", "변화율(%)": "{:.2f}"}
            ),
            use_container_width=True,
        )

    # 7) 요인간 유의성
    with tabs[6]:
        st.markdown("#### 요인간 통계적 유의성 검정 결과")
        st.dataframe(factor_test_df, use_container_width=True)

    # ============================================
    # 8) 엑셀 저장 (모든 시트 포함)
    # ============================================
    with tabs[7]:
        st.markdown("### 📊 분석 결과 엑셀 저장")

        def create_excel_report(
            all_results,
            cons_df,
            factor_test_df,
            fuzzy_raw_df,
            comp_all,
            labels_kr,
            defuzz_method,
            cr_th,
        ):
            """
            다중 시트 엑셀 파일 생성:
            1. 요약
            2. 일관성_검증
            3. AHP_행렬 (그룹별)
            4. Fuzzy_행렬 (그룹별)
            5. 비교분석 (그룹별)
            6. Fuzzy_상세
            7. FuzzyAHP_로우데이터 (응답자별)
            8. 모형간_비교
            9. 요인간_유의성
            """
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                # -------- Sheet 2: 일관성_검증 --------
                cons_df.to_excel(writer, sheet_name="일관성_검증", index=False)

                # -------- Sheet: AHP_행렬 (그룹별) --------
                for gi, g in enumerate(all_results.keys()):
                    r = all_results[g]
                    mat_df = pd.DataFrame(r["matrix"], index=labels_kr, columns=labels_kr)

                    sheet_name_ahp = f"AHP_행렬_{gi+1}" if len(all_results) > 1 else "AHP_행렬"
                    mat_df.to_excel(writer, sheet_name=sheet_name_ahp)

                    ws = writer.sheets[sheet_name_ahp]
                    base_row = len(labels_kr) + 3
                    ws[f"A{base_row}"] = "λmax"
                    ws[f"B{base_row}"] = float(r["lam"])
                    ws[f"A{base_row+1}"] = "CI"
                    ws[f"B{base_row+1}"] = float(r["CI"])
                    ws[f"A{base_row+2}"] = "CR"
                    ws[f"B{base_row+2}"] = float(r["CR"])
                    ws[f"A{base_row+3}"] = "Group"
                    ws[f"B{base_row+3}"] = str(g)

                # -------- Sheet: Fuzzy_행렬 (그룹별) --------
                for gi, g in enumerate(all_results.keys()):
                    r = all_results[g]
                    fuzzy_mat_df = pd.DataFrame(r["fuzzy_matrix"], index=labels_kr, columns=labels_kr)
                    sheet_name_fuzzy = f"Fuzzy_행렬_{gi+1}" if len(all_results) > 1 else "Fuzzy_행렬"
                    fuzzy_mat_df.to_excel(writer, sheet_name=sheet_name_fuzzy)

                # -------- Sheet: 비교분석 --------
                for gi, g in enumerate(comp_all.keys()):
                    comp = comp_all[g]
                    sheet_name_comp = f"비교분석_{gi+1}" if len(comp_all) > 1 else "비교분석"
                    comp.to_excel(writer, sheet_name=sheet_name_comp, index=False)

                # -------- Sheet: Fuzzy_상세 --------
                fuzzy_detail_rows = []
                for g, r in all_results.items():
                    Si = r["Si"]
                    ranks = pd.Series(r["w_fuzzy"]).rank(ascending=False, method="min").astype(int)
                    for fi, lab in enumerate(labels_kr):
                        fuzzy_detail_rows.append(
                            {
                                "Group": g,
                                "항목": lab,
                                "Fuzzy_Lower": Si[fi, 0],
                                "Fuzzy_Medium": Si[fi, 1],
                                "Fuzzy_Upper": Si[fi, 2],
                                "Crisp(Si)": r["crisp_S"][fi],
                                "d_i(raw)": r["d_raw"][fi],
                                "Norm": r["w_fuzzy"][fi],
                                "순위": int(ranks[fi]),
                            }
                        )
                fuzzy_detail_df = pd.DataFrame(fuzzy_detail_rows)
                fuzzy_detail_df.to_excel(writer, sheet_name="Fuzzy_상세", index=False)

                # -------- Sheet: FuzzyAHP_로우데이터 (응답자별) --------
                fuzzy_raw_df.to_excel(writer, sheet_name="FuzzyAHP_로우데이터", index=False)

                # -------- Sheet: 모형간_비교 --------
                ttest_rows = []
                for g, r in all_results.items():
                    for fi, lab in enumerate(labels_kr):
                        ahp_val = r["ahp_w"][fi]
                        fuzzy_val = r["w_fuzzy"][fi]
                        diff = fuzzy_val - ahp_val
                        pct_diff = (diff / ahp_val * 100) if ahp_val != 0 else 0
                        ttest_rows.append(
                            {
                                "Group": g,
                                "항목": lab,
                                "AHP_가중치": ahp_val,
                                "Fuzzy_가중치": fuzzy_val,
                                "차이(Fuzzy-AHP)": diff,
                                "변화율(%)": pct_diff,
                            }
                        )
                ttest_df = pd.DataFrame(ttest_rows)
                ttest_df.to_excel(writer, sheet_name="모형간_비교", index=False)

                # -------- Sheet: 요인간_유의성 --------
                factor_test_df.to_excel(writer, sheet_name="요인간_유의성", index=False)

                # -------- Sheet: 요약(Summary) --------
                summary_data = []
                for g, r in all_results.items():
                    ahp_rank = pd.Series(r["ahp_w"]).rank(ascending=False, method="min").astype(int)
                    fuzzy_rank = pd.Series(r["w_fuzzy"]).rank(ascending=False, method="min").astype(int)

                    if "Group" in cons_df.columns:
                        cons_sub = cons_df[cons_df["Group"] == g]
                    else:
                        cons_sub = cons_df

                    summary_data.append(
                        {
                            "Group": g,
                            "응답자_수": len(cons_sub),
                            "요인_수": len(labels_kr),
                            "평균CR(보정후)": cons_sub["보정 후 CR"].mean(),
                            "일관성통과율": (cons_sub["일관성"] == "○").sum() / len(cons_sub) if len(cons_sub) > 0 else np.nan,
                            "최상위_요인(AHP)": labels_kr[ahp_rank.idxmin()],
                            "최상위_가중치(AHP)": r["ahp_w"][ahp_rank.idxmin()],
                            "최상위_요인(Fuzzy)": labels_kr[fuzzy_rank.idxmin()],
                            "최상위_가중치(Fuzzy)": r["w_fuzzy"][fuzzy_rank.idxmin()],
                            "비퍼지화_방법": defuzz_method,
                            "CR_임계값": cr_th,
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="요약", index=False)

            output.seek(0)
            return output.getvalue()

        excel_bytes = create_excel_report(
            all_results,
            cons_df,
            factor_test_df,
            fuzzy_raw_df,
            comp_all,
            labels_kr,
            defuzz_method,
            cr_th,
        )

        st.download_button(
            label="📥 분석 결과 다운로드 (Excel)",
            data=excel_bytes,
            file_name=f"Fuzzy_AHP_분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

        st.info(
            """
            포함 시트:
            1. 요약
            2. 일관성_검증
            3. AHP_행렬 (그룹별)
            4. Fuzzy_행렬 (그룹별)
            5. 비교분석 (그룹별)
            6. Fuzzy_상세
            7. FuzzyAHP_로우데이터 (응답자별)
            8. 모형간_비교
            9. 요인간_유의성
            """
        )
