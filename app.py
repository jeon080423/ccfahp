import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from scipy import linalg
import io
import warnings
import os

warnings.filterwarnings("ignore")

# =============================
# 0. 한글 폰트: 프로젝트 내 TTF 사용
# =============================
def set_korean_font_from_file():
    """
    프로젝트 폴더 안에 포함된 TTF 한글 폰트를 Matplotlib 기본 폰트로 설정.
    우선 fonts/NanumGothic.ttf 를 찾고, 없으면 현재 폴더의 NanumGothic.ttf 를 찾음.
    """
    candidate_paths = [
        os.path.join("fonts", "NanumGothic.ttf"),
        os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.ttf")
        if "__file__" in globals() else os.path.join("fonts", "NanumGothic.ttf"),
        "NanumGothic.ttf",
    ]

    font_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            font_path = p
            break

    if font_path is not None:
        font_prop = font_manager.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        rcParams["font.family"] = font_name
    # 폰트 파일을 못 찾더라도 앱이 죽지 않도록 함
    rcParams["axes.unicode_minus"] = False

set_korean_font_from_file()

st.set_page_config(page_title="Fuzzy AHP 분석 시스템", layout="wide", page_icon="📊")

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
            # v == 1 이거나 그 외는 동등
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


def correct_matrix(matrix, threshold=0.1, max_iter=20, alpha=0.3):
    """
    CR 임계값(threshold)을 만족하는 수준까지만 '최소한으로' 보정.
    """
    mat = matrix.astype(float).copy()
    w, lam, CI, CR = ahp_weights(mat)
    orig_CR = CR
    it = 0

    if CR <= threshold:
        return mat, orig_CR, CR, it

    n = mat.shape[0]

    while CR > threshold and it < max_iter:
        w, _, _, _ = ahp_weights(mat)
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

        _, _, _, CR = ahp_weights(mat)
        it += 1
        if CR <= threshold:
            break

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
# 5. Streamlit UI
# -----------------------------
st.title("📊 Fuzzy AHP 분석 시스템")
st.markdown("AHP와 Fuzzy AHP를 동시에 분석하는 웹 기반 도구 (개선된 Chang Extent + 최소 CR 보정).")

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
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as w:
    sample_df.to_excel(w, index=False, sheet_name="Sample")
st.download_button(
    "📄 샘플 다운로드",
    buf.getvalue(),
    "fuzzy_ahp_sample_1_2.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# --- 데이터 업로드 ---
st.markdown("### 📤 데이터 업로드")
file = st.file_uploader("Excel 파일을 업로드하세요", type=["xlsx", "xls"])

if not file:
    st.info("👆 Excel 파일을 업로드하면 분석을 시작할 수 있습니다.")
    st.stop()

df = pd.read_excel(file)
st.success(f"파일 업로드 완료: {len(df)}행")

with st.expander("📋 데이터 미리보기"):
    st.dataframe(df.head())

id_col = df.columns[0]
type_col = df.columns[1]
comp_cols = df.columns[2:]

n_comp = len(comp_cols)
n_factor = int((1 + np.sqrt(1 + 8 * n_comp)) / 2)

# --- 1_2 / 1_3 형식으로 요인 라벨 자동 생성 ---
index_set = set()
for c in comp_cols:
    name = str(c)
    if "_" in name:
        a, b = name.split("_")
        index_set.add(int(a))
        index_set.add(int(b))
if len(index_set) == n_factor:
    labels = [f"요인{i}" for i in sorted(index_set)]
else:
    labels = [f"요인{i+1}" for i in range(n_factor)]

st.info(f"자동 인식: 요인 {n_factor}개, 쌍대비교 {n_comp}개  (라벨: {', '.join(labels)})")

has_group = df[type_col].notna().any()
groups = df[type_col].dropna().unique() if has_group else ["All"]

if st.button("🚀 분석 시작", type="primary"):
    all_results = {}
    cons_list = []
    prog = st.progress(0.0)
    step = 1.0 / len(groups)

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

        gm = geometric_mean_matrix(matrices)
        w_ahp, lam, CI, CR = ahp_weights(gm)
        Si, d_raw, w_fuzzy, crisp_S, V = fuzzy_ahp_chang_improved(gm, defuzz_method)

        all_results[g] = {
            "matrix": gm,
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

        prog.progress((gi + 1) * step)

    st.success("분석 완료")

    tabs = st.tabs(["일관성 검증", "AHP 행렬", "비교 분석", "Fuzzy 상세", "시각화"])

    # 1) 일관성
    with tabs[0]:
        cons_df = pd.DataFrame(cons_list)
        st.dataframe(cons_df, use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 응답자", len(cons_df))
        with col2:
            ok = (cons_df["일관성"] == "○").sum()
            st.metric("일관성 통과", f"{ok}/{len(cons_df)}")
        with col3:
            st.metric("평균 CR", f"{cons_df['보정 후 CR'].mean():.4f}")

    # 2) AHP 행렬
    with tabs[1]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            mat_df = pd.DataFrame(r["matrix"], index=labels, columns=labels)
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

    # 3) 비교 분석
    with tabs[2]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            ahp_rank = pd.Series(r["ahp_w"]).rank(ascending=False, method="min").astype(int)
            fuzzy_rank = pd.Series(r["w_fuzzy"]).rank(ascending=False, method="min").astype(int)
            diff = fuzzy_rank - ahp_rank
            comp = pd.DataFrame(
                {
                    "항목": labels,
                    "AHP 가중치": r["ahp_w"],
                    "AHP 순위": ahp_rank,
                    "Fuzzy 가중치": r["w_fuzzy"],
                    "Fuzzy 순위": fuzzy_rank,
                    "순위 변동": diff.apply(
                        lambda x: f"▼ {abs(x)}" if x > 0 else (f"▲ {abs(x)}" if x < 0 else "—"
                        )
                    ),
                }
            )
            st.dataframe(
                comp.style.format({"AHP 가중치": "{:.4f}", "Fuzzy 가중치": "{:.4f}"}),
                use_container_width=True,
            )

    # 4) Fuzzy 상세
    with tabs[3]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            st.info(f"비퍼지화 방법(Si용): {defuzz_disp}")
            Si = r["Si"]
            detail = pd.DataFrame(
                {
                    "구분": labels,
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

    # 5) 시각화
    with tabs[4]:
        for g, r in all_results.items():
            st.markdown(f"#### 그룹: {g}")
            Si = r["Si"]

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            for i, lab in enumerate(labels):
                L, M, U = Si[i]
                ax.plot([L, M, U], [0, 1, 0], marker="o", label=lab, color=colors[i])
            ax.set_xlabel("Weight")
            ax.set_ylabel("Membership degree")
            ax.set_title("Fuzzy Membership Functions")
            ax.grid(True, alpha=0.3)
            ax.legend()   # 여기 범례에 요인1~4 한글 표시
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            x = np.arange(len(labels))
            w1 = r["ahp_w"]
            w2 = r["w_fuzzy"]
            ax2.bar(x - 0.2, w1, width=0.4, label="AHP")
            ax2.bar(x + 0.2, w2, width=0.4, label="Fuzzy")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel("Weight")
            ax2.set_title("AHP vs Fuzzy AHP Weights")
            ax2.grid(True, axis="y", alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)
