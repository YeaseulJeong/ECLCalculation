import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="ECL 시나리오 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 설정
st.sidebar.title("🔧 분석 설정")
st.sidebar.markdown("---")

class ECLScenarioAnalyzer:
    def __init__(self):
        self.comparison_df = None
        self.scn1_df = None
        self.scn2_df = None
    
    def generate_sample_data(self, n_samples, scenario1_params, scenario2_params):
        """샘플 데이터 생성"""
        np.random.seed(42)
        
        # 실제 손실 데이터 (로그정규분포)
        actual_loss = np.abs(np.random.lognormal(3.5, 0.8, n_samples))
        
        # 기본 ECL
        basic_ecl = actual_loss * (1 + np.random.normal(0.1, 0.3, n_samples))
        
        # Scenario 1: 구성요소별 조정
        scn1_bias = scenario1_params['bias']
        scn1_volatility = scenario1_params['volatility']
        scenario1_ecl = actual_loss * (1 + np.random.normal(scn1_bias, scn1_volatility, n_samples))
        
        # Scenario 2: 전체 승수
        scn2_bias = scenario2_params['bias']
        scn2_volatility = scenario2_params['volatility']
        scenario2_ecl = actual_loss * (1 + np.random.normal(scn2_bias, scn2_volatility, n_samples))
        
        # 경제지표 데이터
        unemployment = np.random.uniform(0.03, 0.08, n_samples)
        gdp_growth = np.random.uniform(-0.05, 0.1, n_samples)
        interest_rate = np.random.uniform(0.5, 5.0, n_samples)
        csi = np.random.uniform(80, 120, n_samples)
        
        self.comparison_df = pd.DataFrame({
            'Period': range(1, n_samples + 1),
            'Actual_Loss': actual_loss,
            'Basic_ECL': np.maximum(basic_ecl, 0),
            'Scenario1_ECL': np.maximum(scenario1_ecl, 0),
            'Scenario2_ECL': np.maximum(scenario2_ecl, 0)
        })
        
        self.scn1_df = pd.DataFrame({
            'YEAR': np.random.choice(range(2020, 2025), n_samples),
            '실업률': unemployment,
            '경제성장률': gdp_growth,
            '기준금리': interest_rate,
            '현재경기판단CSI': csi,
            '경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)': scenario1_ecl
        })
        
        self.scn2_df = self.scn1_df.copy()
        self.scn2_df['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'] = scenario2_ecl
        
        return self.comparison_df, self.scn1_df, self.scn2_df
    
    def load_uploaded_data(self, uploaded_files):
        """업로드된 파일 처리"""
        try:
            if len(uploaded_files) == 3:
                scn1_file, scn2_file, actual_file = uploaded_files
                
                scn1_df = pd.read_csv(scn1_file)
                scn2_df = pd.read_csv(scn2_file)
                actual_df = pd.read_csv(actual_file)
                
                # 데이터 정리 및 매칭
                min_length = min(len(scn1_df), len(scn2_df), len(actual_df))
                
                actual_df['발생신용손실(단위:십억원)'] = actual_df['발생신용손실(단위:백만원)'] / 1000
                
                self.comparison_df = pd.DataFrame({
                    'Period': range(1, min_length + 1),
                    'Actual_Loss': actual_df['발생신용손실(단위:십억원)'].head(min_length),
                    'Basic_ECL': scn1_df['기대신용손실(ECL)(단위:십억원)'].head(min_length),
                    'Scenario1_ECL': scn1_df['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'].head(min_length),
                    'Scenario2_ECL': scn2_df['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'].head(min_length)
                })
                
                self.scn1_df = scn1_df.head(min_length)
                self.scn2_df = scn2_df.head(min_length)
                
                return True
            return False
        except Exception as e:
            st.error(f"파일 로드 중 오류: {e}")
            return False
    
    def create_error_distribution_analysis(self):
        """오차 분포 분석"""
        actual = self.comparison_df['Actual_Loss'].values
        scn1 = self.comparison_df['Scenario1_ECL'].values
        scn2 = self.comparison_df['Scenario2_ECL'].values
        
        # 상대 오차 계산
        scn1_rel_errors = (scn1 - actual) / np.maximum(actual, 0.1) * 100
        scn2_rel_errors = (scn2 - actual) / np.maximum(actual, 0.1) * 100
        
        # 절대 오차 계산
        scn1_abs_errors = np.abs(scn1 - actual)
        scn2_abs_errors = np.abs(scn2 - actual)
        
        # Plotly 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['상대 오차 분포', '절대 오차 박스플롯', '실제값 대비 오차 패턴', '누적 분포 함수'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 상대 오차 분포 히스토그램
        fig.add_trace(
            go.Histogram(x=scn1_rel_errors, name='Scenario 1', opacity=0.7, 
                        marker_color='#E74C3C', nbinsx=30),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=scn2_rel_errors, name='Scenario 2', opacity=0.7, 
                        marker_color='#3498DB', nbinsx=30),
            row=1, col=1
        )
        
        # 2. 절대 오차 박스플롯
        fig.add_trace(
            go.Box(y=scn1_abs_errors, name='Scenario 1', marker_color='#E74C3C'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=scn2_abs_errors, name='Scenario 2', marker_color='#3498DB'),
            row=1, col=2
        )
        
        # 3. 산점도
        fig.add_trace(
            go.Scatter(x=actual, y=scn1_abs_errors, mode='markers', 
                      name='Scenario 1', marker=dict(color='#E74C3C', opacity=0.6)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=actual, y=scn2_abs_errors, mode='markers', 
                      name='Scenario 2', marker=dict(color='#3498DB', opacity=0.6)),
            row=2, col=1
        )
        
        # 4. CDF
        scn1_sorted = np.sort(scn1_abs_errors)
        scn2_sorted = np.sort(scn2_abs_errors)
        scn1_cdf = np.arange(1, len(scn1_sorted) + 1) / len(scn1_sorted)
        scn2_cdf = np.arange(1, len(scn2_sorted) + 1) / len(scn2_sorted)
        
        fig.add_trace(
            go.Scatter(x=scn1_sorted, y=scn1_cdf, mode='lines', 
                      name='Scenario 1 CDF', line=dict(color='#E74C3C', width=3)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=scn2_sorted, y=scn2_cdf, mode='lines', 
                      name='Scenario 2 CDF', line=dict(color='#3498DB', width=3)),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="📊 오차 분포 특성 분석", showlegend=True)
        
        return fig, {
            'scn1_mean_rel_error': np.mean(scn1_rel_errors),
            'scn2_mean_rel_error': np.mean(scn2_rel_errors),
            'scn1_std_rel_error': np.std(scn1_rel_errors),
            'scn2_std_rel_error': np.std(scn2_rel_errors),
            'scn1_median_abs_error': np.median(scn1_abs_errors),
            'scn2_median_abs_error': np.median(scn2_abs_errors)
        }
    
    def create_improvement_analysis(self):
        """정확도 개선도 분석"""
        actual = self.comparison_df['Actual_Loss'].values
        basic = self.comparison_df['Basic_ECL'].values
        scn1 = self.comparison_df['Scenario1_ECL'].values
        scn2 = self.comparison_df['Scenario2_ECL'].values
        
        # 개선도 계산
        basic_errors = np.abs(basic - actual)
        scn1_errors = np.abs(scn1 - actual)
        scn2_errors = np.abs(scn2 - actual)
        
        scn1_improvement = (basic_errors - scn1_errors) / basic_errors * 100
        scn2_improvement = (basic_errors - scn2_errors) / basic_errors * 100
        
        # 무한대값과 NaN 제거
        scn1_improvement = scn1_improvement[~(np.isnan(scn1_improvement) | np.isinf(scn1_improvement))]
        scn2_improvement = scn2_improvement[~(np.isnan(scn2_improvement) | np.isinf(scn2_improvement))]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['개선도 분포', '누적 개선도', '손실 규모별 개선도', '개선도 일관성'],
        )
        
        # 1. 개선도 분포
        fig.add_trace(
            go.Histogram(x=scn1_improvement, name='Scenario 1', opacity=0.7, 
                        marker_color='#E74C3C', nbinsx=25),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=scn2_improvement, name='Scenario 2', opacity=0.7, 
                        marker_color='#3498DB', nbinsx=25),
            row=1, col=1
        )
        
        # 2. 누적 개선도
        periods = range(1, len(basic_errors) + 1)
        scn1_cum_improvement = np.cumsum(basic_errors - scn1_errors) / np.cumsum(basic_errors) * 100
        scn2_cum_improvement = np.cumsum(basic_errors - scn2_errors) / np.cumsum(basic_errors) * 100
        
        fig.add_trace(
            go.Scatter(x=list(periods), y=scn1_cum_improvement, mode='lines', 
                      name='Scenario 1 누적', line=dict(color='#E74C3C', width=3)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=list(periods), y=scn2_cum_improvement, mode='lines', 
                      name='Scenario 2 누적', line=dict(color='#3498DB', width=3)),
            row=1, col=2
        )
        
        # 3. 손실 규모별 개선도
        loss_bins = np.percentile(actual, [0, 25, 50, 75, 100])
        bin_labels = ['Low', 'Med-Low', 'Med-High', 'High']
        
        scn1_improvements_by_bin = []
        scn2_improvements_by_bin = []
        
        for i in range(len(loss_bins)-1):
            mask = (actual >= loss_bins[i]) & (actual < loss_bins[i+1])
            if i == len(loss_bins)-2:
                mask = (actual >= loss_bins[i]) & (actual <= loss_bins[i+1])
            
            if mask.sum() > 0:
                scn1_imp = np.mean((basic_errors[mask] - scn1_errors[mask]) / basic_errors[mask] * 100)
                scn2_imp = np.mean((basic_errors[mask] - scn2_errors[mask]) / basic_errors[mask] * 100)
                scn1_improvements_by_bin.append(scn1_imp)
                scn2_improvements_by_bin.append(scn2_imp)
            else:
                scn1_improvements_by_bin.append(0)
                scn2_improvements_by_bin.append(0)
        
        fig.add_trace(
            go.Bar(x=bin_labels, y=scn1_improvements_by_bin, name='Scenario 1', 
                   marker_color='#E74C3C', opacity=0.8),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=bin_labels, y=scn2_improvements_by_bin, name='Scenario 2', 
                   marker_color='#3498DB', opacity=0.8),
            row=2, col=1
        )
        
        # 4. 이동 평균 개선도
        window = min(10, len(periods)//4)
        if len(scn1_improvement) >= window and len(scn2_improvement) >= window:
            scn1_rolling = pd.Series(scn1_improvement[:len(periods)]).rolling(window=window).mean()
            scn2_rolling = pd.Series(scn2_improvement[:len(periods)]).rolling(window=window).mean()
            
            fig.add_trace(
                go.Scatter(x=list(periods), y=scn1_rolling, mode='lines', 
                          name='Scenario 1 이동평균', line=dict(color='#E74C3C', width=2)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(periods), y=scn2_rolling, mode='lines', 
                          name='Scenario 2 이동평균', line=dict(color='#3498DB', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="📈 정확도 개선도 분석", showlegend=True)
        
        return fig, {
            'scn1_avg_improvement': np.mean(scn1_improvement),
            'scn2_avg_improvement': np.mean(scn2_improvement),
            'scn1_success_rate': (scn1_improvement > 0).sum() / len(scn1_improvement) * 100,
            'scn2_success_rate': (scn2_improvement > 0).sum() / len(scn2_improvement) * 100,
            'scn1_std_improvement': np.std(scn1_improvement),
            'scn2_std_improvement': np.std(scn2_improvement)
        }
    
    def create_stability_analysis(self):
        """예측 안정성 분석"""
        actual = self.comparison_df['Actual_Loss'].values
        scn1 = self.comparison_df['Scenario1_ECL'].values
        scn2 = self.comparison_df['Scenario2_ECL'].values
        
        scn1_errors = scn1 - actual
        scn2_errors = scn2 - actual
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['이동 표준편차 (변동성)', '오차 자기상관', '연속 우위 기간', '변동성 비교'],
        )
        
        # 1. 이동 표준편차
        windows = [5, 10, 20]
        colors = ['lightcoral', 'red', 'darkred']
        
        for i, window in enumerate(windows):
            if len(scn1_errors) >= window:
                scn1_rolling_std = pd.Series(scn1_errors).rolling(window=window).std()
                fig.add_trace(
                    go.Scatter(x=list(range(len(scn1_rolling_std))), y=scn1_rolling_std, 
                              mode='lines', name=f'Scn1 (w={window})', 
                              line=dict(color=colors[i], width=2)),
                    row=1, col=1
                )
        
        colors = ['lightblue', 'blue', 'darkblue']
        for i, window in enumerate(windows):
            if len(scn2_errors) >= window:
                scn2_rolling_std = pd.Series(scn2_errors).rolling(window=window).std()
                fig.add_trace(
                    go.Scatter(x=list(range(len(scn2_rolling_std))), y=scn2_rolling_std, 
                              mode='lines', name=f'Scn2 (w={window})', 
                              line=dict(color=colors[i], width=2, dash='dash')),
                    row=1, col=1
                )
        
        # 2. 자기상관 분석
        max_lags = min(20, len(scn1_errors)//4)
        if max_lags > 1:
            scn1_autocorr = []
            scn2_autocorr = []
            lags = range(1, max_lags + 1)
            
            for lag in lags:
                if len(scn1_errors) > lag:
                    corr1 = np.corrcoef(scn1_errors[:-lag], scn1_errors[lag:])[0, 1]
                    corr2 = np.corrcoef(scn2_errors[:-lag], scn2_errors[lag:])[0, 1]
                    scn1_autocorr.append(corr1 if not np.isnan(corr1) else 0)
                    scn2_autocorr.append(corr2 if not np.isnan(corr2) else 0)
            
            fig.add_trace(
                go.Bar(x=[l - 0.2 for l in lags], y=scn1_autocorr, 
                       name='Scenario 1', marker_color='#E74C3C', opacity=0.8, width=0.4),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=[l + 0.2 for l in lags], y=scn2_autocorr, 
                       name='Scenario 2', marker_color='#3498DB', opacity=0.8, width=0.4),
                row=1, col=2
            )
        
        # 3. 연속 우위 기간 분석
        scn1_better = (np.abs(scn1_errors) < np.abs(scn2_errors))
        scn2_better = ~scn1_better
        
        def get_consecutive_periods(boolean_series):
            consecutive = []
            current_length = 0
            for val in boolean_series:
                if val:
                    current_length += 1
                else:
                    if current_length > 0:
                        consecutive.append(current_length)
                    current_length = 0
            if current_length > 0:
                consecutive.append(current_length)
            return consecutive
        
        scn1_consecutive = get_consecutive_periods(scn1_better)
        scn2_consecutive = get_consecutive_periods(scn2_better)
        
        if scn1_consecutive and scn2_consecutive:
            max_length = max(max(scn1_consecutive), max(scn2_consecutive))
            bins = list(range(1, max_length + 2))
            
            fig.add_trace(
                go.Histogram(x=scn1_consecutive, name='Scenario 1', opacity=0.7, 
                            marker_color='#E74C3C', nbinsx=len(bins)),
                row=2, col=1
            )
            fig.add_trace(
                go.Histogram(x=scn2_consecutive, name='Scenario 2', opacity=0.7, 
                            marker_color='#3498DB', nbinsx=len(bins)),
                row=2, col=1
            )
        
        # 4. 변동성 비교
        actual_volatility = np.std(actual)
        scn1_volatility = np.std(scn1)
        scn2_volatility = np.std(scn2)
        
        window = min(10, len(actual)//4)
        if len(actual) >= window:
            actual_rolling_vol = pd.Series(actual).rolling(window=window).std()
            scn1_rolling_vol = pd.Series(scn1).rolling(window=window).std()
            scn2_rolling_vol = pd.Series(scn2).rolling(window=window).std()
            
            periods = range(len(actual))
            
            fig.add_trace(
                go.Scatter(x=list(periods), y=actual_rolling_vol, mode='lines', 
                          name='Actual', line=dict(color='#27AE60', width=3)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(periods), y=scn1_rolling_vol, mode='lines', 
                          name='Scenario 1', line=dict(color='#E74C3C', width=2)),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=list(periods), y=scn2_rolling_vol, mode='lines', 
                          name='Scenario 2', line=dict(color='#3498DB', width=2)),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="📉 예측 안정성 분석", showlegend=True)
        
        stability_stats = {
            'actual_volatility': actual_volatility,
            'scn1_volatility': scn1_volatility,
            'scn2_volatility': scn2_volatility,
            'scn1_error_volatility': np.std(scn1_errors),
            'scn2_error_volatility': np.std(scn2_errors),
            'scn1_win_rate': scn1_better.sum() / len(scn1_better) * 100,
            'scn2_win_rate': scn2_better.sum() / len(scn2_better) * 100
        }
        
        if scn1_consecutive and scn2_consecutive:
            stability_stats['scn1_avg_consecutive'] = np.mean(scn1_consecutive)
            stability_stats['scn2_avg_consecutive'] = np.mean(scn2_consecutive)
        
        return fig, stability_stats

def main():
    # 메인 타이틀
    st.title("📊 ECL 시나리오 분석 대시보드")
    st.markdown("### 🎯 Expected Credit Loss 시나리오별 성능 비교 및 인사이트 분석")
    st.markdown("---")
    
    # 분석기 인스턴스 생성
    analyzer = ECLScenarioAnalyzer()
    
    # 사이드바: 데이터 입력 방식 선택
    data_source = st.sidebar.radio(
        "📁 데이터 소스 선택",
        ["샘플 데이터 생성", "CSV 파일 업로드"]
    )
    
    if data_source == "샘플 데이터 생성":
        st.sidebar.subheader("🔧 샘플 데이터 설정")
        
        # 데이터 크기 설정
        n_samples = st.sidebar.slider("데이터 포인트 수", 50, 500, 100)
        
        # Scenario 1 파라미터
        st.sidebar.subheader("📈 Scenario 1 설정")
        scn1_bias = st.sidebar.slider("Scenario 1 편향", -0.2, 0.2, 0.05, 0.01, 
                                     help="양수: 과대평가, 음수: 과소평가")
        scn1_volatility = st.sidebar.slider("Scenario 1 변동성", 0.05, 0.5, 0.15, 0.01)
        
        # Scenario 2 파라미터
        st.sidebar.subheader("📉 Scenario 2 설정")
        scn2_bias = st.sidebar.slider("Scenario 2 편향", -0.2, 0.2, 0.0, 0.01)
        scn2_volatility = st.sidebar.slider("Scenario 2 변동성", 0.05, 0.5, 0.25, 0.01)
        
        # 데이터 생성
        scenario1_params = {'bias': scn1_bias, 'volatility': scn1_volatility}
        scenario2_params = {'bias': scn2_bias, 'volatility': scn2_volatility}
        
        comparison_df, scn1_df, scn2_df = analyzer.generate_sample_data(
            n_samples, scenario1_params, scenario2_params
        )
        
        st.success(f"✅ {n_samples}개 샘플 데이터가 생성되었습니다!")
        
    else:
        st.sidebar.subheader("📂 CSV 파일 업로드")
        st.sidebar.markdown("다음 3개 파일을 업로드해주세요:")
        st.sidebar.markdown("1. 시나리오 1 데이터")
        st.sidebar.markdown("2. 시나리오 2 데이터") 
        st.sidebar.markdown("3. 실제 발생신용손실 데이터")
        
        uploaded_files = st.sidebar.file_uploader(
            "CSV 파일들을 선택하세요",
            type=['csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files and len(uploaded_files) == 3:
            if analyzer.load_uploaded_data(uploaded_files):
                st.success("✅ 파일이 성공적으로 로드되었습니다!")
                comparison_df = analyzer.comparison_df
            else:
                st.error("❌ 파일 로드에 실패했습니다.")
                return
        else:
            if uploaded_files:
                st.warning("⚠️ 정확히 3개의 CSV 파일을 업로드해주세요.")
            return
    
    # 분석 옵션
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 분석 옵션")
    
    analysis_options = st.sidebar.multiselect(
        "분석할 항목을 선택하세요",
        ["오차 분포 분석", "정확도 개선도 분석", "예측 안정성 분석", "종합 인사이트"],
        default=["오차 분포 분석", "정확도 개선도 분석", "예측 안정성 분석", "종합 인사이트"]
    )
    
    # 메인 대시보드
    if analyzer.comparison_df is not None:
        
        # 기본 통계 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 데이터 포인트", 
                f"{len(analyzer.comparison_df)}개"
            )
        
        with col2:
            actual_mean = analyzer.comparison_df['Actual_Loss'].mean()
            st.metric(
                "💰 평균 실제 손실", 
                f"{actual_mean:.1f}억원"
            )
        
        with col3:
            scn1_mean = analyzer.comparison_df['Scenario1_ECL'].mean()
            st.metric(
                "🔴 Scenario 1 평균", 
                f"{scn1_mean:.1f}억원",
                f"{((scn1_mean - actual_mean) / actual_mean * 100):+.1f}%"
            )
        
        with col4:
            scn2_mean = analyzer.comparison_df['Scenario2_ECL'].mean()
            st.metric(
                "🔵 Scenario 2 평균", 
                f"{scn2_mean:.1f}억원",
                f"{((scn2_mean - actual_mean) / actual_mean * 100):+.1f}%"
            )
        
        st.markdown("---")
        
        # 시계열 비교 차트 (항상 표시)
        st.subheader("📈 시계열 비교")
        
        fig_timeseries = go.Figure()
        
        fig_timeseries.add_trace(go.Scatter(
            x=analyzer.comparison_df['Period'],
            y=analyzer.comparison_df['Actual_Loss'],
            mode='lines+markers',
            name='실제 발생신용손실',
            line=dict(color='#27AE60', width=3),
            marker=dict(size=6)
        ))
        
        fig_timeseries.add_trace(go.Scatter(
            x=analyzer.comparison_df['Period'],
            y=analyzer.comparison_df['Scenario1_ECL'],
            mode='lines+markers',
            name='Scenario 1 (구성요소별 조정)',
            line=dict(color='#E74C3C', width=2),
            marker=dict(size=4)
        ))
        
        fig_timeseries.add_trace(go.Scatter(
            x=analyzer.comparison_df['Period'],
            y=analyzer.comparison_df['Scenario2_ECL'],
            mode='lines+markers',
            name='Scenario 2 (전체 승수)',
            line=dict(color='#3498DB', width=2),
            marker=dict(size=4)
        ))
        
        fig_timeseries.update_layout(
            title="ECL 시나리오별 시계열 비교",
            xaxis_title="기간",
            yaxis_title="ECL (십억원)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_timeseries, use_container_width=True)
        
        # 선택된 분석 실행
        if "오차 분포 분석" in analysis_options:
            st.markdown("---")
            st.subheader("📊 1. 오차 분포 특성 분석")
            
            error_fig, error_stats = analyzer.create_error_distribution_analysis()
            st.plotly_chart(error_fig, use_container_width=True)
            
            # 오차 분석 결과 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Scenario 1 (구성요소별 조정)**")
                st.write(f"• 평균 상대 오차: {error_stats['scn1_mean_rel_error']:.2f}%")
                st.write(f"• 상대 오차 표준편차: {error_stats['scn1_std_rel_error']:.2f}%")
                st.write(f"• 중위 절대 오차: {error_stats['scn1_median_abs_error']:.2f}억원")
                
                if error_stats['scn1_mean_rel_error'] > 0:
                    st.info("💡 과대평가 경향 (보수적 접근)")
                else:
                    st.warning("⚠️ 과소평가 경향 (공격적 접근)")
            
            with col2:
                st.markdown("**🔵 Scenario 2 (전체 승수)**")
                st.write(f"• 평균 상대 오차: {error_stats['scn2_mean_rel_error']:.2f}%")
                st.write(f"• 상대 오차 표준편차: {error_stats['scn2_std_rel_error']:.2f}%")
                st.write(f"• 중위 절대 오차: {error_stats['scn2_median_abs_error']:.2f}억원")
                
                if error_stats['scn2_mean_rel_error'] > 0:
                    st.info("💡 과대평가 경향 (보수적 접근)")
                else:
                    st.warning("⚠️ 과소평가 경향 (공격적 접근)")
        
        if "정확도 개선도 분석" in analysis_options:
            st.markdown("---")
            st.subheader("📈 2. 정확도 개선도 분석")
            
            improvement_fig, improvement_stats = analyzer.create_improvement_analysis()
            st.plotly_chart(improvement_fig, use_container_width=True)
            
            # 개선도 분석 결과
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Scenario 1 개선 성과**")
                st.write(f"• 평균 개선도: {improvement_stats['scn1_avg_improvement']:.2f}%")
                st.write(f"• 개선 성공률: {improvement_stats['scn1_success_rate']:.1f}%")
                st.write(f"• 개선도 안정성: {improvement_stats['scn1_std_improvement']:.2f}%")
                
                if improvement_stats['scn1_avg_improvement'] > improvement_stats['scn2_avg_improvement']:
                    st.success("🏆 평균 개선도 우위")
                
                if improvement_stats['scn1_success_rate'] > improvement_stats['scn2_success_rate']:
                    st.success("🎯 성공률 우위")
            
            with col2:
                st.markdown("**🔵 Scenario 2 개선 성과**")
                st.write(f"• 평균 개선도: {improvement_stats['scn2_avg_improvement']:.2f}%")
                st.write(f"• 개선 성공률: {improvement_stats['scn2_success_rate']:.1f}%")
                st.write(f"• 개선도 안정성: {improvement_stats['scn2_std_improvement']:.2f}%")
                
                if improvement_stats['scn2_avg_improvement'] > improvement_stats['scn1_avg_improvement']:
                    st.success("🏆 평균 개선도 우위")
                
                if improvement_stats['scn2_success_rate'] > improvement_stats['scn1_success_rate']:
                    st.success("🎯 성공률 우위")
        
        if "예측 안정성 분석" in analysis_options:
            st.markdown("---")
            st.subheader("📉 3. 예측 안정성 분석")
            
            stability_fig, stability_stats = analyzer.create_stability_analysis()
            st.plotly_chart(stability_fig, use_container_width=True)
            
            # 안정성 분석 결과
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔴 Scenario 1 안정성**")
                st.write(f"• 예측 변동성: {stability_stats['scn1_volatility']:.2f}")
                st.write(f"• 오차 변동성: {stability_stats['scn1_error_volatility']:.2f}")
                st.write(f"• 예측 우위율: {stability_stats['scn1_win_rate']:.1f}%")
                
                if 'scn1_avg_consecutive' in stability_stats:
                    st.write(f"• 평균 연속 우위: {stability_stats['scn1_avg_consecutive']:.1f}기간")
                
                # 실제 변동성 대비 비율
                vol_ratio1 = stability_stats['scn1_volatility'] / stability_stats['actual_volatility']
                if 0.8 <= vol_ratio1 <= 1.2:
                    st.success(f"✅ 적정 변동성 (실제 대비 {vol_ratio1:.2f}배)")
                else:
                    st.warning(f"⚠️ 변동성 주의 (실제 대비 {vol_ratio1:.2f}배)")
            
            with col2:
                st.markdown("**🔵 Scenario 2 안정성**")
                st.write(f"• 예측 변동성: {stability_stats['scn2_volatility']:.2f}")
                st.write(f"• 오차 변동성: {stability_stats['scn2_error_volatility']:.2f}")
                st.write(f"• 예측 우위율: {stability_stats['scn2_win_rate']:.1f}%")
                
                if 'scn2_avg_consecutive' in stability_stats:
                    st.write(f"• 평균 연속 우위: {stability_stats['scn2_avg_consecutive']:.1f}기간")
                
                # 실제 변동성 대비 비율
                vol_ratio2 = stability_stats['scn2_volatility'] / stability_stats['actual_volatility']
                if 0.8 <= vol_ratio2 <= 1.2:
                    st.success(f"✅ 적정 변동성 (실제 대비 {vol_ratio2:.2f}배)")
                else:
                    st.warning(f"⚠️ 변동성 주의 (실제 대비 {vol_ratio2:.2f}배)")
        
        if "종합 인사이트" in analysis_options:
            st.markdown("---")
            st.subheader("🎯 4. 종합 인사이트 및 권고사항")
            
            # 성능 매트릭스 생성
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 성능 비교 매트릭스")
                
                # 각 지표별 점수 계산 (0-100점)
                metrics_df = pd.DataFrame({
                    '지표': ['정확도', '안정성', '개선도', '일관성', '종합'],
                    'Scenario 1': [0, 0, 0, 0, 0],
                    'Scenario 2': [0, 0, 0, 0, 0]
                })
                
                # 정확도 점수 (낮은 오차가 높은 점수)
                if 'error_stats' in locals():
                    if error_stats['scn1_median_abs_error'] < error_stats['scn2_median_abs_error']:
                        metrics_df.loc[0, 'Scenario 1'] = 85
                        metrics_df.loc[0, 'Scenario 2'] = 70
                    else:
                        metrics_df.loc[0, 'Scenario 1'] = 70
                        metrics_df.loc[0, 'Scenario 2'] = 85
                
                # 안정성 점수 (낮은 변동성이 높은 점수)
                if 'stability_stats' in locals():
                    if stability_stats['scn1_error_volatility'] < stability_stats['scn2_error_volatility']:
                        metrics_df.loc[1, 'Scenario 1'] = 90
                        metrics_df.loc[1, 'Scenario 2'] = 65
                    else:
                        metrics_df.loc[1, 'Scenario 1'] = 65
                        metrics_df.loc[1, 'Scenario 2'] = 90
                
                # 개선도 점수
                if 'improvement_stats' in locals():
                    if improvement_stats['scn1_avg_improvement'] > improvement_stats['scn2_avg_improvement']:
                        metrics_df.loc[2, 'Scenario 1'] = 80
                        metrics_df.loc[2, 'Scenario 2'] = 75
                    else:
                        metrics_df.loc[2, 'Scenario 1'] = 75
                        metrics_df.loc[2, 'Scenario 2'] = 80
                
                # 일관성 점수 (높은 성공률이 높은 점수)
                if 'improvement_stats' in locals():
                    if improvement_stats['scn1_success_rate'] > improvement_stats['scn2_success_rate']:
                        metrics_df.loc[3, 'Scenario 1'] = 85
                        metrics_df.loc[3, 'Scenario 2'] = 70
                    else:
                        metrics_df.loc[3, 'Scenario 1'] = 70
                        metrics_df.loc[3, 'Scenario 2'] = 85
                
                # 종합 점수 (가중 평균)
                metrics_df.loc[4, 'Scenario 1'] = (
                    metrics_df.loc[0, 'Scenario 1'] * 0.3 +  # 정확도 30%
                    metrics_df.loc[1, 'Scenario 1'] * 0.25 + # 안정성 25%
                    metrics_df.loc[2, 'Scenario 1'] * 0.25 + # 개선도 25%
                    metrics_df.loc[3, 'Scenario 1'] * 0.2    # 일관성 20%
                )
                
                metrics_df.loc[4, 'Scenario 2'] = (
                    metrics_df.loc[0, 'Scenario 2'] * 0.3 +
                    metrics_df.loc[1, 'Scenario 2'] * 0.25 +
                    metrics_df.loc[2, 'Scenario 2'] * 0.25 +
                    metrics_df.loc[3, 'Scenario 2'] * 0.2
                )
                
                # 점수를 정수로 변환
                metrics_df['Scenario 1'] = metrics_df['Scenario 1'].astype(int)
                metrics_df['Scenario 2'] = metrics_df['Scenario 2'].astype(int)
                
                st.dataframe(
                    metrics_df.style.background_gradient(subset=['Scenario 1', 'Scenario 2'], cmap='RdYlGn'),
                    use_container_width=True
                )
            
            with col2:
                st.markdown("#### 🏆 최종 권고사항")
                
                # 승자 결정
                scn1_total = metrics_df.loc[4, 'Scenario 1']
                scn2_total = metrics_df.loc[4, 'Scenario 2']
                
                if scn1_total > scn2_total:
                    st.success(f"🏆 **Scenario 1 우위** ({scn1_total}점 vs {scn2_total}점)")
                    recommended = "Scenario 1 (구성요소별 조정)"
                    alternative = "Scenario 2 (전체 승수)"
                else:
                    st.success(f"🏆 **Scenario 2 우위** ({scn2_total}점 vs {scn1_total}점)")
                    recommended = "Scenario 2 (전체 승수)"
                    alternative = "Scenario 1 (구성요소별 조정)"
                
                st.markdown(f"""
                **📋 권고 전략:**
                
                **1단계 (즉시 적용)**
                - 주 모델: {recommended}
                - 높은 종합 점수로 우선 적용 권장
                
                **2단계 (보완 활용)**  
                - 보조 모델: {alternative}
                - 특정 상황에서 교차 검증용
                
                **3단계 (고도화)**
                - 하이브리드 접근법 개발
                - 시장 상황별 모델 전환
                - 정기적 성과 모니터링
                """)
            
            # 상황별 권고사항
            st.markdown("#### 🎯 상황별 모델 선택 가이드")
            
            situation_guide = pd.DataFrame({
                '상황': [
                    '🏛️ 규제 보고', 
                    '📊 스트레스 테스트', 
                    '💼 일상 관리', 
                    '⚡ 즉시 개선 필요', 
                    '🔒 보수적 접근', 
                    '🚀 효율성 중시'
                ],
                '권장 모델': [
                    'Scenario 1' if scn1_total > scn2_total else 'Scenario 2',
                    'Scenario 1',  # 안정성이 중요
                    'Scenario 2',  # 효율성이 중요
                    recommended,
                    'Scenario 1',  # 보수적
                    'Scenario 2'   # 효율적
                ],
                '이유': [
                    '높은 종합 성능',
                    '극값 상황에서 안정성',
                    '운영 효율성과 단순함',
                    '우수한 전반적 성과',
                    '과대평가로 위험 완충',
                    '빠른 구현과 즉시 효과'
                ]
            })
            
            st.dataframe(situation_guide, use_container_width=True)
            
            # 모니터링 지표
            st.markdown("#### 📈 지속적 모니터링 지표")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **월별 점검**
                - 예측 오차 분포
                - 편향성 변화
                - 극값 성능
                """)
            
            with col2:
                st.markdown("""
                **분기별 평가**
                - 개선도 추이
                - 안정성 지표
                - 경쟁 모델 비교
                """)
            
            with col3:
                st.markdown("""
                **연간 재보정**
                - 파라미터 최적화
                - 새로운 경제지표 반영
                - 모델 구조 개선
                """)
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    📊 ECL 시나리오 분석 대시보드 | Built with Streamlit & Plotly<br>
    🔍 실시간 분석 • 📈 동적 시각화 • 🎯 인사이트 도출
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()