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
                matrix[i][j] = 1 / abs(value) if abs(value) > 1 else 1
                matrix[j][i] = abs(value) if abs(value) > 1 else 1
            elif value > 1:
                matrix[i][j] = value
                matrix[j][i] = 1 / value
            else:
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
    weights = weights / weights.sum()

    # CR 계산
    CI = (max_eigenvalue - n) / (n - 1) if n > 1 else 0
    CR = CI / RI.get(n, 1.49) if n > 2 else 0

    return weights, max_eigenvalue, CI, CR

def correct_matrix(matrix, max_iterations=10, threshold=0.1):
    """CR 보정 프로세스"""
    n = len(matrix)
    corrected = matrix.copy()
    iterations = 0

    _, _, _, CR = calculate_ahp_weights(corrected)
    original_cr = CR

    while CR > threshold and iterations < max_iterations:
        # 기하평균 기반 보정
        for i in range(n):
            for j in range(i + 1, n):
                geometric_mean = np.sqrt(corrected[i][j] * corrected[j][i])
                corrected[i][j] = geometric_mean
                corrected[j][i] = 1 / geometric_mean

        _, _, _, CR = calculate_ahp_weights(corrected)
        iterations += 1

    return corrected, original_cr, CR, iterations

def saaty_to_fuzzy(value):
    """Saaty 척도를 삼각퍼지수로 변환"""
    rounded = round(value)
    if rounded < 1:
        rounded = 1
    elif rounded > 9:
        rounded = 9
    return FUZZY_SCALE[rounded]

def fuzzy_ahp_changs_method(matrix):
    """Chang's Extent Analysis Method로 Fuzzy AHP 분석"""
    n = len(matrix)

    # 삼각퍼지수 행렬 생성
    fuzzy_matrix = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                fuzzy_matrix[i][j] = (1, 1, 1)
            else:
                fuzzy_matrix[i][j] = saaty_to_fuzzy(matrix[i][j])

    # Si 계산 (퍼지 종합값)
    Si = np.zeros((n, 3))
    for i in range(n):
        row_sum = fuzzy_matrix[i].sum(axis=0)
        total_sum = fuzzy_matrix.sum(axis=(0, 1))
        Si[i] = [row_sum[0] / total_sum[2], row_sum[1] / total_sum[1], row_sum[2] / total_sum[0]]

    # V 값 계산 (퍼지수 비교)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if Si[i][1] >= Si[j][1]:
                    V[i][j] = 1
                elif Si[i][0] >= Si[j][2]:
                    V[i][j] = 0
                else:
                    V[i][j] = (Si[j][2] - Si[i][0]) / ((Si[i][1] - Si[i][0]) + (Si[j][2] - Si[j][1]))

    # 가중치 계산
    weights = np.zeros(n)
    for i in range(n):
        weights[i] = min([V[i][j] for j in range(n) if i != j] + [1])

    # 정규화
    weights_norm = weights / weights.sum() if weights.sum() > 0 else weights

    # Crisp 값 계산
    crisp = (Si[:, 0] + 2 * Si[:, 1] + Si[:, 2]) / 4
    crisp_norm = crisp / crisp.sum()

    return Si, weights_norm, crisp_norm

def geometric_mean_matrix(matrices):
    """행렬들의 기하평균 계산"""
    if len(matrices) == 0:
        return None
    n = len(matrices[0])
    result = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            values = [m[i][j] for m in matrices]
            result[i][j] = np.prod(values) ** (1 / len(values))
    return result

# 메인 UI
st.title("📊 Fuzzy AHP 분석 시스템")
st.markdown("### AHP와 Fuzzy AHP를 동시에 분석하는 웹 기반 도구")

# 사이드바 - 옵션 설정
with st.sidebar:
    st.header("⚙️ 분석 옵션")
    cr_threshold = st.slider("CR 허용 임계값", 0.0, 0.2, 0.1, 0.01)
    defuzzy_method = st.selectbox("비퍼지화 방법", ["가중평균 (l+2m+u)/4", "산술평균 (l+m+u)/3"])

    st.markdown("---")
    st.markdown("### 📖 사용 안내")
    st.markdown("""
    1. Excel 파일 업로드
    2. 데이터 형식 확인
    3. 분석 실행
    4. 결과 확인 및 다운로드
    """)

# 샘플 데이터 다운로드
st.markdown("### 📥 샘플 데이터")
col1, col2 = st.columns([1, 4])
with col1:
    sample_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'Type': ['A', 'A', 'B'],
        '요인1 vs 요인2': [3, 5, -2],
        '요인1 vs 요인3': [5, 7, 3],
        '요인1 vs 요인4': [7, 9, 5],
        '요인2 vs 요인3': [3, 5, 2],
        '요인2 vs 요인4': [5, 7, 4],
        '요인3 vs 요인4': [3, 5, 3]
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
        # 데이터 읽기
        df = pd.read_excel(uploaded_file)

        st.success(f"✅ 파일 업로드 성공: {len(df)}개 응답")

        # 데이터 미리보기
        with st.expander("📋 업로드된 데이터 미리보기"):
            st.dataframe(df.head(10))

        # 데이터 파싱
        id_col = df.columns[0]
        type_col = df.columns[1]
        comparison_cols = df.columns[2:]

        # 요인 수 계산
        n_comparisons = len(comparison_cols)
        n_factors = int((1 + np.sqrt(1 + 8 * n_comparisons)) / 2)

        # 요인 레이블 추출
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

        # 그룹 분석 여부 확인
        has_groups = df[type_col].notna().any()
        if has_groups:
            groups = df[type_col].dropna().unique()
            st.info(f"👥 그룹 분석 모드: {len(groups)}개 그룹 감지 ({', '.join(map(str, groups))})")
        else:
            st.info("👥 전체 그룹 분석 모드")

        # 분석 시작 버튼
        if st.button("🚀 분석 시작", type="primary"):
            with st.spinner("분석 진행 중..."):

                # 진행률 표시
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 결과 저장 변수
                all_results = []
                consistency_data = []

                # 그룹별 분석
                if has_groups:
                    group_results = {}
                    for group in groups:
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

                        # 그룹 통합 행렬 (기하평균)
                        group_matrix = geometric_mean_matrix(group_matrices)
                        ahp_weights, lambda_max, CI, CR = calculate_ahp_weights(group_matrix)
                        fuzzy_si, fuzzy_weights, fuzzy_crisp = fuzzy_ahp_changs_method(group_matrix)

                        group_results[group] = {
                            'matrix': group_matrix,
                            'ahp_weights': ahp_weights,
                            'fuzzy_weights': fuzzy_weights,
                            'fuzzy_si': fuzzy_si,
                            'fuzzy_crisp': fuzzy_crisp,
                            'lambda_max': lambda_max,
                            'CI': CI,
                            'CR': CR
                        }

                    all_results = group_results
                else:
                    # 전체 그룹 분석
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

                    # 전체 통합 행렬
                    combined_matrix = geometric_mean_matrix(all_matrices)
                    ahp_weights, lambda_max, CI, CR = calculate_ahp_weights(combined_matrix)
                    fuzzy_si, fuzzy_weights, fuzzy_crisp = fuzzy_ahp_changs_method(combined_matrix)

                    all_results = {
                        'All': {
                            'matrix': combined_matrix,
                            'ahp_weights': ahp_weights,
                            'fuzzy_weights': fuzzy_weights,
                            'fuzzy_si': fuzzy_si,
                            'fuzzy_crisp': fuzzy_crisp,
                            'lambda_max': lambda_max,
                            'CI': CI,
                            'CR': CR
                        }
                    }

                progress_bar.progress(1.0)
                status_text.text("✅ 분석 완료!")

                st.success("🎉 분석이 완료되었습니다!")

                # 결과 표시
                tabs = st.tabs(["📊 일관성 검증", "🔢 AHP 행렬", "⚖️ 비교 분석", "🔺 Fuzzy 상세", "📈 시각화"])

                # 탭 1: 일관성 검증
                with tabs[0]:
                    st.markdown("### 응답자별 일관성 정보")
                    consistency_df = pd.DataFrame(consistency_data)
                    st.dataframe(consistency_df, use_container_width=True)

                    # 요약 통계
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 응답자 수", len(consistency_df))
                    with col2:
                        consistent = (consistency_df['일관성'] == '○').sum()
                        st.metric("일관성 통과", f"{consistent}/{len(consistency_df)}")
                    with col3:
                        avg_cr = consistency_df['보정 후 CR'].mean()
                        st.metric("평균 CR", f"{avg_cr:.4f}")

                # 탭 2: AHP 행렬
                with tabs[1]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

                        # 쌍대비교 행렬
                        matrix_df = pd.DataFrame(result['matrix'], 
                                                columns=factor_labels, 
                                                index=factor_labels)
                        st.write("**쌍대비교 행렬**")
                        st.dataframe(matrix_df.style.format("{:.4f}"), use_container_width=True)

                        # 가중치
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

                # 탭 3: 비교 분석
                with tabs[2]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

                        # AHP vs Fuzzy AHP 비교표
                        ahp_ranks = pd.Series(result['ahp_weights']).rank(ascending=False, method='min').astype(int)
                        fuzzy_ranks = pd.Series(result['fuzzy_crisp']).rank(ascending=False, method='min').astype(int)
                        rank_change = fuzzy_ranks - ahp_ranks

                        comparison_df = pd.DataFrame({
                            '항목': factor_labels,
                            'AHP 가중치': result['ahp_weights'],
                            'AHP 순위': ahp_ranks,
                            'Fuzzy 가중치': result['fuzzy_crisp'],
                            'Fuzzy 순위': fuzzy_ranks,
                            '순위 변동': rank_change.apply(lambda x: f'▼ {abs(x)}' if x > 0 else (f'▲ {abs(x)}' if x < 0 else '—'))
                        })

                        st.dataframe(comparison_df.style.format({
                            'AHP 가중치': '{:.4f}',
                            'Fuzzy 가중치': '{:.4f}'
                        }), use_container_width=True)

                        st.markdown("---")

                # 탭 4: Fuzzy 상세
                with tabs[3]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

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

                # 탭 5: 시각화
                with tabs[4]:
                    for group_name, result in all_results.items():
                        st.markdown(f"### {'전체 그룹' if group_name == 'All' else f'그룹: {group_name}'}")

                        # Fuzzy Membership Functions 그래프
                        fig, ax = plt.subplots(figsize=(12, 6))

                        for i, label in enumerate(factor_labels):
                            lower, medium, upper = result['fuzzy_si'][i]
                            ax.plot([lower, medium, upper], [0, 1, 0], marker='o', label=label, linewidth=2)

                        ax.set_xlabel('Weight (가중치)', fontsize=12)
                        ax.set_ylabel('Membership Degree (소속도)', fontsize=12)
                        ax.set_title('Fuzzy Membership Functions', fontsize=14, fontweight='bold')
                        ax.legend(loc='upper right')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(-0.1, 1.1)

                        st.pyplot(fig)

                        # 가중치 비교 바 차트
                        fig, ax = plt.subplots(figsize=(10, 6))
                        x = np.arange(len(factor_labels))
                        width = 0.35

                        ax.bar(x - width/2, result['ahp_weights'], width, label='AHP', alpha=0.8)
                        ax.bar(x + width/2, result['fuzzy_weights'], width, label='Fuzzy AHP', alpha=0.8)

                        ax.set_xlabel('요인', fontsize=12)
                        ax.set_ylabel('가중치', fontsize=12)
                        ax.set_title('AHP vs Fuzzy AHP 가중치 비교', fontsize=14, fontweight='bold')
                        ax.set_xticks(x)
                        ax.set_xticklabels(factor_labels)
                        ax.legend()
                        ax.grid(True, axis='y', alpha=0.3)

                        st.pyplot(fig)

                        st.markdown("---")

                # Excel 다운로드
                st.markdown("### 📥 결과 다운로드")

                # Excel 파일 생성
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 시트 1: 원본 데이터
                    df.to_excel(writer, sheet_name='Raw Data', index=False)

                    # 시트 2: 일관성 정보
                    consistency_df.to_excel(writer, sheet_name='Consistency', index=False)

                    # 시트 3-6: 그룹별 결과
                    for group_name, result in all_results.items():
                        sheet_name = 'All' if group_name == 'All' else f'Group_{group_name}'

                        # AHP 행렬
                        matrix_df = pd.DataFrame(result['matrix'], 
                                                columns=factor_labels, 
                                                index=factor_labels)
                        matrix_df.to_excel(writer, sheet_name=f'{sheet_name}_Matrix')

                        # 비교표
                        ahp_ranks = pd.Series(result['ahp_weights']).rank(ascending=False, method='min').astype(int)
                        fuzzy_ranks = pd.Series(result['fuzzy_crisp']).rank(ascending=False, method='min').astype(int)

                        comparison_df = pd.DataFrame({
                            '항목': factor_labels,
                            'AHP 가중치': result['ahp_weights'],
                            'AHP 순위': ahp_ranks,
                            'Fuzzy 가중치': result['fuzzy_crisp'],
                            'Fuzzy 순위': fuzzy_ranks
                        })
                        comparison_df.to_excel(writer, sheet_name=f'{sheet_name}_Compare', index=False)

                        # Fuzzy 상세
                        fuzzy_detail_df = pd.DataFrame({
                            '구분': factor_labels,
                            'Fuzzy (Lower)': result['fuzzy_si'][:, 0],
                            'Fuzzy (Medium)': result['fuzzy_si'][:, 1],
                            'Fuzzy (Upper)': result['fuzzy_si'][:, 2],
                            'Crisp': result['fuzzy_crisp'],
                            'Norm': result['fuzzy_weights']
                        })
                        fuzzy_detail_df.to_excel(writer, sheet_name=f'{sheet_name}_Fuzzy', index=False)

                st.download_button(
                    label="📊 전체 결과 Excel 다운로드",
                    data=output.getvalue(),
                    file_name="fuzzy_ahp_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"❌ 오류 발생: {str(e)}")
        st.info("데이터 형식을 확인해주세요. 샘플 데이터를 참고하세요.")

else:
    st.info("👆 Excel 파일을 업로드하여 분석을 시작하세요.")

    # 사용 가이드
    with st.expander("📚 상세 사용 가이드"):
        st.markdown("""
        ### 데이터 형식 요구사항

        #### Excel 파일 구조
        - **1열**: 응답자 ID (예: 1, 2, 3, ...)
        - **2열**: 그룹 타입 (선택사항, 비어있으면 전체 분석)
        - **3열 이후**: 쌍대비교 펀칭 데이터

        #### 쌍대비교 펀칭 규칙
        - **1**: 동등한 중요도
        - **음수 (-1~-9)**: 좌측 요인이 더 중요
        - **양수 (1~9)**: 우측 요인이 더 중요
        - 예시: "요인A vs 요인B" 컬럼에 -5 입력 → A가 B보다 강하게 중요

        #### 요인 수와 쌍대비교 수
        - 4개 요인 → 6개 쌍대비교 (4×3/2)
        - 5개 요인 → 10개 쌍대비교 (5×4/2)
        - 6개 요인 → 15개 쌍대비교 (6×5/2)

        ### 분석 방법론

        #### 일반 AHP
        - 고유벡터 방법으로 가중치 계산
        - CR(Consistency Ratio) ≤ 0.1 기준
        - 자동 CR 보정 (최대 10회)

        #### Fuzzy AHP (Chang's Method)
        - 삼각퍼지수(TFN) 변환
        - Extent Analysis로 퍼지 종합값 계산
        - 비퍼지화로 최종 가중치 도출

        ### 출력 결과

        1. **일관성 검증**: 응답자별 CR 값 및 보정 정보
        2. **AHP 행렬**: 통합 쌍대비교 행렬 및 가중치
        3. **비교 분석**: AHP와 Fuzzy AHP 순위 비교
        4. **Fuzzy 상세**: 삼각퍼지수 상세 값
        5. **시각화**: Membership Functions 및 가중치 비교 차트
        """)
