import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg
import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(page_title="Fuzzy AHP 분석 시스템", layout="wide", page_icon="📊")

# Random Index (RI) 값
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# Saaty 척도 → 삼각퍼지수 매핑
FUZZY_SCALE = {
    1: (1, 1, 1),
    2: (1, 2, 3),
    3: (2, 3, 4),
    4: (3, 4, 5),
    5: (4, 5, 6),
    6: (5, 6, 7),
    7: (6, 7, 8),
    8: (7, 8, 9),
    9: (8, 9, 9)
}

def convert_punch_to_matrix(punch_data, n_factors):
    """펀칭 데이터를 AHP 쌍대비교 행렬로 변환"""
    matrix = np.ones((n_factors, n_factors))
    idx = 0
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            value = punch_data[idx]
            if value < 0:
                # 음수: 좌측이 더 중요
                abs_val = abs(value)
                if abs_val > 1:
                    matrix[i][j] = 1 / abs_val
                    matrix[j][i] = abs_val
                else:
                    matrix[i][j] = 1
                    matrix[j][i] = 1
            elif value > 1:
                # 양수: 우측이 더 중요
                matrix[i][j] = value
                matrix[j][i] = 1 / value
            else:
                # 1: 동등
                matrix[i][j] = 1
                matrix[j][i] = 1
            idx += 1
    return matrix

def calculate_ahp_weights(matrix):
    """고유벡터 방법으로 AHP 가중치 계산"""
    n = len(matrix)
    eigenvalues, eigenvectors = linalg.eig(matrix)
    max_eigenvalue = max(eigenvalues.real)
    max_index = list(eigenvalues.real).index(max_eigenvalue)
    weights = eigenvectors[:, max_index].real
    weights = np.abs(weights)  # 음수 방지
    weights = weights / weights.sum()

    # CR 계산
    CI = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
    CR = CI / RI.get(n, 1.49) if n > 2 else 0

    return weights, max_eigenvalue.real, CI.real, CR.real

def correct_matrix(matrix, max_iterations=10, threshold=0.1):
    """CR 보정 프로세스"""
    n = len(matrix)
    corrected = matrix.copy()
    iterations = 0

    _, _, _, CR = calculate_ahp_weights(corrected)
    original_cr = CR

    while CR > threshold and iterations < max_iterations:
        for i in range(n):
            for j in range(i + 1, n):
                # 대칭성 강제
                geometric_mean = np.sqrt(corrected[i][j] * corrected[j][i])
                corrected[i][j] = geometric_mean
                corrected[j][i] = 1 / geometric_mean if geometric_mean > 0 else 1

        _, _, _, CR = calculate_ahp_weights(corrected)
        iterations += 1

    return corrected, original_cr, CR, iterations

def saaty_to_fuzzy(value):
    """Saaty 척도를 삼각퍼지수로 변환"""
    if value <= 0:
        value = 1
    rounded = int(round(value))
    if rounded < 1:
        rounded = 1
    elif rounded > 9:
        rounded = 9
    return FUZZY_SCALE[rounded]

def fuzzy_inverse(fuzzy_num):
    """삼각퍼지수의 역수 계산"""
    l, m, u = fuzzy_num
    if l > 0 and m > 0 and u > 0:
        return (1/u, 1/m, 1/l)
    else:
        return (1, 1, 1)

def fuzzy_multiply(f1, f2):
    """두 삼각퍼지수의 곱셈"""
    l1, m1, u1 = f1
    l2, m2, u2 = f2
    return (l1*l2, m1*m2, u1*u2)

def fuzzy_add(f1, f2):
    """두 삼각퍼지수의 덧셈"""
    l1, m1, u1 = f1
    l2, m2, u2 = f2
    return (l1+l2, m1+m2, u1+u2)

def defuzzify(fuzzy_values, method='weighted'):
    """비퍼지화 - 삼각퍼지수를 crisp 값으로 변환"""
    crisp_values = []
    for tfn in fuzzy_values:
        l, m, u = tfn
        if method == 'weighted':
            crisp = (l + 2*m + u) / 4
        elif method == 'arithmetic':
            crisp = (l + m + u) / 3
        elif method == 'geometric':
            if l > 0 and m > 0 and u > 0:
                crisp = (l * m * u) ** (1/3)
            else:
                crisp = 0
        else:
            crisp = m
        crisp_values.append(crisp)

    crisp_values = np.array(crisp_values)
    total = crisp_values.sum()
    if total > 0:
        return crisp_values / total
    else:
        return crisp_values

def fuzzy_ahp_changs_method(matrix, defuzzy_method='weighted'):
    """
    Chang's Extent Analysis Method로 Fuzzy AHP 분석
    완전히 재작성된 정확한 구현
    """
    n = len(matrix)

    # Step 1: 삼각퍼지수 행렬 생성
    fuzzy_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append((1.0, 1.0, 1.0))
            else:
                # 원본 행렬 값을 삼각퍼지수로 변환
                value = matrix[i][j]
                if value >= 1:
                    tfn = saaty_to_fuzzy(value)
                else:
                    # 1보다 작으면 역수의 퍼지수를 역변환
                    inv_value = 1 / value if value > 0 else 1
                    inv_tfn = saaty_to_fuzzy(inv_value)
                    tfn = fuzzy_inverse(inv_tfn)
                row.append(tfn)
        fuzzy_matrix.append(row)

    # Step 2: 각 행의 퍼지 합 계산
    fuzzy_row_sums = []
    for i in range(n):
        row_sum = (0.0, 0.0, 0.0)
        for j in range(n):
            row_sum = fuzzy_add(row_sum, fuzzy_matrix[i][j])
        fuzzy_row_sums.append(row_sum)

    # Step 3: 전체 행렬의 퍼지 합 계산
    total_fuzzy_sum = (0.0, 0.0, 0.0)
    for row_sum in fuzzy_row_sums:
        total_fuzzy_sum = fuzzy_add(total_fuzzy_sum, row_sum)

    # Step 4: Si 계산 = 각 행 합 / 전체 합의 역수
    Si = []
    total_l, total_m, total_u = total_fuzzy_sum

    for i in range(n):
        row_l, row_m, row_u = fuzzy_row_sums[i]

        # Si = 행합 × (1/전체합)
        if total_l > 0 and total_m > 0 and total_u > 0:
            si_l = row_l / total_u  # 주의: 역수 관계
            si_m = row_m / total_m
            si_u = row_u / total_l
        else:
            si_l, si_m, si_u = 0, 0, 0

        Si.append((si_l, si_m, si_u))

    Si = np.array(Si)

    # Step 5: V(Si >= Sj) 계산
    def degree_of_possibility(si, sj):
        """V(Si >= Sj) - 퍼지수 Si가 Sj보다 큰 정도"""
        l1, m1, u1 = si
        l2, m2, u2 = sj

        if m1 >= m2:
            return 1.0
        elif l1 >= u2:
            return 0.0
        else:
            numerator = u2 - l1
            denominator = (m1 - u1) + (u2 - m2)
            if denominator != 0:
                return max(0.0, min(1.0, numerator / denominator))
            else:
                return 0.0

    # Step 6: 각 요인의 우선순위 벡터 계산
    priority_vector = []
    for i in range(n):
        # V(Si >= S1, S2, ..., Sn) = min(V(Si >= Sj)) for all j != i
        min_degree = 1.0
        for j in range(n):
            if i != j:
                degree = degree_of_possibility(Si[i], Si[j])
                min_degree = min(min_degree, degree)
        priority_vector.append(min_degree)

    priority_vector = np.array(priority_vector)

    # Step 7: 정규화
    total_priority = priority_vector.sum()
    if total_priority > 0:
        weights_norm = priority_vector / total_priority
    else:
        weights_norm = np.ones(n) / n

    # Step 8: Crisp 값 계산 (비퍼지화)
    crisp_norm = defuzzify(Si, method=defuzzy_method)

    return Si, weights_norm, crisp_norm

def geometric_mean_matrix(matrices):
    """행렬들의 기하평균 계산"""
    if len(matrices) == 0:
        return None
    n = len(matrices[0])
    result = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            values = [m[i][j] for m in matrices if m[i][j] > 0]
            if len(values) > 0:
                result[i][j] = np.prod(values) ** (1 / len(values))
            else:
                result[i][j] = 1
    return result

# 메인 UI
st.title("📊 Fuzzy AHP 분석 시스템")
st.markdown("### AHP와 Fuzzy AHP를 동시에 분석하는 웹 기반 도구")

# 사이드바 - 옵션 설정
with st.sidebar:
    st.header("⚙️ 분석 옵션")
    cr_threshold = st.slider("CR 허용 임계값", 0.0, 0.2, 0.1, 0.01)
    defuzzy_method_display = st.selectbox(
        "비퍼지화 방법", 
        ["가중평균 (l+2m+u)/4", "산술평균 (l+m+u)/3", "기하평균 (l×m×u)^(1/3)"]
    )

    defuzzy_method_map = {
        "가중평균 (l+2m+u)/4": "weighted",
        "산술평균 (l+m+u)/3": "arithmetic",
        "기하평균 (l×m×u)^(1/3)": "geometric"
    }
    defuzzy_method = defuzzy_method_map[defuzzy_method_display]

    st.markdown("---")
    st.markdown("### 📖 사용 안내")
    st.markdown("""
    1. Excel 파일 업로드
    2. 데이터 형식 확인
    3. 분석 옵션 선택
    4. 분석 실행
    5. 결과 확인 및 다운로드
    """)

# 샘플 데이터 다운로드
st.markdown("### 📥 샘플 데이터")
col1, col2 = st.columns([1, 4])
with col1:
    sample_data = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'Type': ['A', 'A', 'A', 'B', 'B', 'B'],
        '요인1 vs 요인2': [3, 5, 2, -2, -3, -1],
        '요인1 vs 요인3': [5, 7, 4, 3, 5, 2],
        '요인1 vs 요인4': [7, 9, 5, 5, 7, 4],
        '요인2 vs 요인3': [3, 5, 3, 5, 7, 4],
        '요인2 vs 요인4': [5, 7, 4, 7, 9, 6],
        '요인3 vs 요인4': [3, 5, 2, 5, 7, 3]
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        sample_data.to_excel(writer, index=False, sheet_name='Sample')

    st.download_button(
        label="📄 샘플 다운로드",
        data=buffer.getvalue(),
        file_name="fuzzy_ahp_sample.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 파일 업로드
st.markdown("### 📤 데이터 업로드")
uploaded_file = st.file_uploader("Excel 파일을 업로드하세요", type=['xlsx', 'xls'])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ 파일 업로드 성공: {len(df)}개 응답")

        with st.expander("📋 업로드된 데이터 미리보기"):
            st.dataframe(df.head(10))

        id_col = df.columns[0]
        type_col = df.columns[1]
        comparison_cols = df.columns[2:]

        n_comparisons = len(comparison_cols)
        n_factors = int((1 + np.sqrt(1 + 8 * n_comparisons)) / 2)

        factor_labels = []
        for col in comparison_cols:
            parts = col.split(' vs ')
            if len(parts) == 2:
                if parts[0] not in factor_labels:
                    factor_labels.append(parts[0])
                if parts[1] not in factor_labels:
                    factor_labels.append(parts[1])

        if len(factor_labels) != n_factors:
            factor_labels = [f'요인{i+1}' for i in range(n_factors)]

        st.info(f"🔍 자동 인식: {n_factors}개 요인, {n_comparisons}개 쌍대비교")

        has_groups = df[type_col].notna().any()
        if has_groups:
            groups = df[type_col].dropna().unique()
            st.info(f"👥 그룹 분석 모드: {len(groups)}개 그룹 ({', '.join(map(str, groups))})")
        else:
            st.info("👥 전체 그룹 분석 모드")

        if st.button("🚀 분석 시작", type="primary"):
            with st.spinner("분석 진행 중..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                all_results = {}
                consistency_data = []

                if has_groups:
                    for group_idx, group in enumerate(groups):
                        group_df = df[df[type_col] == group]
                        group_matrices = []

                        for idx, row in group_df.iterrows():
                            punch_data = row[comparison_cols].values
                            matrix = convert_punch_to_matrix(punch_data, n_factors)
                            corrected, orig_cr, final_cr, iters = correct_matrix(matrix, threshold=cr_threshold)

                            consistency_data.append({
                                'ID': row[id_col],
                                'Group': group,
                                '보정 전 CR': round(orig_cr, 4),
                                '보정 후 CR': round(final_cr, 4),
                                '보정 횟수': iters,
                                '일관성': '○' if final_cr <= cr_threshold else '×'
                            })

                            group_matrices.append(corrected)

                        group_matrix = geometric_mean_matrix(group_matrices)
                        ahp_weights, lambda_max, CI, CR = calculate_ahp_weights(group_matrix)
                        fuzzy_si, fuzzy_weights, fuzzy_crisp = fuzzy_ahp_changs_method(group_matrix, defuzzy_method)

                        all_results[group] = {
                            'matrix': group_matrix,
                            'ahp_weights': ahp_weights,
                            'fuzzy_weights': fuzzy_weights,
                            'fuzzy_si': fuzzy_si,
                            'fuzzy_crisp': fuzzy_crisp,
                            'lambda_max': lambda_max,
                            'CI': CI,
                            'CR': CR
                        }

                        progress = (group_idx + 1) / len(groups)
                        progress_bar.progress(progress)
                        status_text.text(f"처리 중: 그룹 {group_idx + 1}/{len(groups)}")

                else:
                    all_matrices = []

                    for idx, row in df.iterrows():
                        punch_data = row[comparison_cols].values
                        matrix = convert_punch_to_matrix(punch_data, n_factors)
                        corrected, orig_cr, final_cr, iters = correct_matrix(matrix, threshold=cr_threshold)

                        consistency_data.append({
                            'ID': row[id_col],
                            '보정 전 CR': round(orig_cr, 4),
                            '보정 후 CR': round(final_cr, 4),
                            '보정 횟수': iters,
                            '일관성': '○' if final_cr <= cr_threshold else '×'
                        })

                        all_matrices.append(corrected)

                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"처리 중: {idx + 1}/{len(df)} 응답자")

                    combined_matrix = geometric_mean_matrix(all_matrices)
                    ahp_weights, lambda_max, CI, CR = calculate_ahp_weights(combined_matrix)
                    fuzzy_si, fuzzy_weights, fuzzy_crisp = fuzzy_ahp_changs_method(combined_matrix, defuzzy_method)

                    all_results['All'] = {
                        'matrix': combined_matrix,
                        'ahp_weights': ahp_weights,
                        'fuzzy_weights': fuzzy_weights,
                        'fuzzy_si': fuzzy_si,
                        'fuzzy_crisp': fuzzy_crisp,
                        'lambda_max': lambda_max,
                        'CI': CI,
                        'CR': CR
                    }

                progress_bar.progress(1.0)
                status_text.text("✅ 분석 완료!")

                st.success("🎉 분석이 완료되었습니다!")

                # 결과 표시
                tabs = st.tabs(["📊 일관성 검증", "🔢 AHP 행렬", "⚖️ 비교 분석", "🔺 Fuzzy 상세", "📈 시각화"])

                with tabs[0]:
                    st.markdown("### 응답자별 일관성 정보")
                    consistency_df = pd.DataFrame(consistency_data)
                    st.dataframe(consistency_df, use_container_width=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 응답자 수", len(consistency_df))
                    with col2:
                        consistent = (consistency_df['일관성'] == '○').sum()
                        st.metric("일관성 통과", f"{consistent}/{len(consistency_df)}")
                    with col3:
                        avg_cr = consistency_df['보정 후 CR'].mean()
                        st.metric("평균 CR", f"{avg_cr:.4f}")

                with tabs[1]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

                        matrix_df = pd.DataFrame(result['matrix'], 
                                                columns=factor_labels, 
                                                index=factor_labels)
                        st.write("**쌍대비교 행렬**")
                        st.dataframe(matrix_df.style.format("{:.4f}"), use_container_width=True)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("λmax", f"{result['lambda_max']:.4f}")
                        with col2:
                            st.metric("CI", f"{result['CI']:.4f}")
                        with col3:
                            st.metric("CR", f"{result['CR']:.4f}")
                        with col4:
                            consistency_status = "✅ 일관성 통과" if result['CR'] <= cr_threshold else "⚠️ 일관성 미달"
                            st.metric("일관성", consistency_status)

                        st.markdown("---")

                with tabs[2]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

                        ahp_ranks = pd.Series(result['ahp_weights']).rank(ascending=False, method='min').astype(int)
                        fuzzy_ranks = pd.Series(result['fuzzy_weights']).rank(ascending=False, method='min').astype(int)
                        rank_change = fuzzy_ranks - ahp_ranks

                        comparison_df = pd.DataFrame({
                            '항목': factor_labels,
                            'AHP 가중치': result['ahp_weights'],
                            'AHP 순위': ahp_ranks,
                            'Fuzzy 가중치': result['fuzzy_weights'],
                            'Fuzzy 순위': fuzzy_ranks,
                            '순위 변동': rank_change.apply(lambda x: f'▼ {abs(x)}' if x > 0 else (f'▲ {abs(x)}' if x < 0 else '—'))
                        })

                        st.dataframe(comparison_df.style.format({
                            'AHP 가중치': '{:.4f}',
                            'Fuzzy 가중치': '{:.4f}'
                        }), use_container_width=True)

                        st.markdown("---")

                with tabs[3]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")
                        st.info(f"📌 비퍼지화 방법: {defuzzy_method_display}")

                        fuzzy_detail_df = pd.DataFrame({
                            '구분': factor_labels,
                            'Fuzzy (Lower)': result['fuzzy_si'][:, 0],
                            'Fuzzy (Medium)': result['fuzzy_si'][:, 1],
                            'Fuzzy (Upper)': result['fuzzy_si'][:, 2],
                            'Crisp': result['fuzzy_crisp'],
                            'Norm': result['fuzzy_weights'],
                            '순위': pd.Series(result['fuzzy_weights']).rank(ascending=False, method='min').astype(int)
                        })

                        st.dataframe(fuzzy_detail_df.style.format({
                            'Fuzzy (Lower)': '{:.4f}',
                            'Fuzzy (Medium)': '{:.4f}',
                            'Fuzzy (Upper)': '{:.4f}',
                            'Crisp': '{:.4f}',
                            'Norm': '{:.4f}'
                        }), use_container_width=True)

                        st.markdown("---")

                with tabs[4]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

                        fig, ax = plt.subplots(figsize=(12, 6))
                        colors = plt.cm.Set3(np.linspace(0, 1, len(factor_labels)))

                        for i, label in enumerate(factor_labels):
                            lower, medium, upper = result['fuzzy_si'][i]
                            ax.plot([lower, medium, upper], [0, 1, 0], 
                                   marker='o', label=label, linewidth=2.5, 
                                   color=colors[i], markersize=8)

                        ax.set_xlabel('Weight (가중치)', fontsize=13, fontweight='bold')
                        ax.set_ylabel('Membership Degree (소속도)', fontsize=13, fontweight='bold')
                        ax.set_title('Fuzzy Membership Functions', fontsize=15, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=10)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.set_ylim(-0.1, 1.1)

                        st.pyplot(fig)
                        plt.close()

                        fig, ax = plt.subplots(figsize=(10, 6))
                        x = np.arange(len(factor_labels))
                        width = 0.35

                        bars1 = ax.bar(x - width/2, result['ahp_weights'], width, 
                                      label='AHP', alpha=0.8, color='#3498db')
                        bars2 = ax.bar(x + width/2, result['fuzzy_weights'], width, 
                                      label='Fuzzy AHP', alpha=0.8, color='#e74c3c')

                        ax.set_xlabel('요인', fontsize=13, fontweight='bold')
                        ax.set_ylabel('가중치', fontsize=13, fontweight='bold')
                        ax.set_title('AHP vs Fuzzy AHP 가중치 비교', fontsize=15, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(factor_labels)
                        ax.legend(fontsize=11)
                        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

                        for bars in [bars1, bars2]:
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

                        st.pyplot(fig)
                        plt.close()

                        st.markdown("---")

                # Excel 다운로드
                st.markdown("### 📥 결과 다운로드")

                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Raw Data', index=False)
                    consistency_df.to_excel(writer, sheet_name='Consistency', index=False)

                    for group_name, result in all_results.items():
                        sheet_name = 'All' if group_name == 'All' else f'Group_{group_name}'
                        sheet_name = sheet_name[:31]

                        matrix_df = pd.DataFrame(result['matrix'], 
                                                columns=factor_labels, 
                                                index=factor_labels)
                        matrix_df.to_excel(writer, sheet_name=f'{sheet_name}_Matrix'[:31])

                        ahp_ranks = pd.Series(result['ahp_weights']).rank(ascending=False, method='min').astype(int)
                        fuzzy_ranks = pd.Series(result['fuzzy_weights']).rank(ascending=False, method='min').astype(int)

                        comparison_df = pd.DataFrame({
                            '항목': factor_labels,
                            'AHP 가중치': result['ahp_weights'],
                            'AHP 순위': ahp_ranks,
                            'Fuzzy 가중치': result['fuzzy_weights'],
                            'Fuzzy 순위': fuzzy_ranks
                        })
                        comparison_df.to_excel(writer, sheet_name=f'{sheet_name}_Compare'[:31], index=False)

                        fuzzy_detail_df = pd.DataFrame({
                            '구분': factor_labels,
                            'Fuzzy (Lower)': result['fuzzy_si'][:, 0],
                            'Fuzzy (Medium)': result['fuzzy_si'][:, 1],
                            'Fuzzy (Upper)': result['fuzzy_si'][:, 2],
                            'Crisp': result['fuzzy_crisp'],
                            'Norm': result['fuzzy_weights']
                        })
                        fuzzy_detail_df.to_excel(writer, sheet_name=f'{sheet_name}_Fuzzy'[:31], index=False)

                st.download_button(
                    label="📊 전체 결과 Excel 다운로드",
                    data=output.getvalue(),
                    file_name="fuzzy_ahp_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Excel 파일을 업로드하여 분석을 시작하세요.")

    with st.expander("📚 상세 사용 가이드"):
        st.markdown("""
        ### 데이터 형식 요구사항

        #### Excel 파일 구조
        - **1열**: 응답자 ID
        - **2열**: 그룹 타입 (선택사항)
        - **3열 이후**: 쌍대비교 펀칭 데이터

        #### 펀칭 규칙
        - **1**: 동등
        - **음수 (-1~-9)**: 좌측이 더 중요
        - **양수 (1~9)**: 우측이 더 중요

        ### Fuzzy AHP (Chang's Method)
        - 삼각퍼지수(TFN) 변환
        - Extent Analysis 계산
        - 3가지 비퍼지화 방법 지원
        """)
