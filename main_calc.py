import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="ECL 분석 대시보드 (2010-2014)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .correlation-insight {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stage-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    """실제 데이터 로드"""
    try:
        # 기본 ECL 데이터
        basic_data = pd.read_csv('기대신용손실.csv', encoding='utf-8')
        # 시나리오 1 데이터
        scenario1_data = pd.read_csv('반영된_기대신용손실_scn1.csv', encoding='utf-8')
        # 시나리오 2 데이터  
        scenario2_data = pd.read_csv('반영된_기대신용손실_scn2.csv', encoding='utf-8')
        # 발생 신용손실 데이터
        actual_data = pd.read_csv('발생신용손실.csv', encoding='utf-8')
        
        return basic_data, scenario1_data, scenario2_data, actual_data
        
    except FileNotFoundError:
        # 파일이 없을 경우 샘플 데이터 생성
        return generate_sample_data()

@st.cache_data
def generate_sample_data():
    """샘플 데이터 생성 (파일이 없을 경우)"""
    np.random.seed(42)
    
    # 2010-2014년 데이터 생성
    years = []
    quarters = []
    for year in range(2010, 2015):
        for quarter in range(1, 5):
            years.append(year)
            quarters.append(quarter)
    
    n_periods = len(years)
    
    # 기본 ECL 데이터
    basic_data = pd.DataFrame({
        'YEAR': years,
        '부도확률(PD)': np.random.uniform(0.01, 0.05, n_periods),
        '부도_시_손실확률(LGD)': np.random.uniform(0.2, 0.8, n_periods),
        '부도_시_손실금액(EAD)(단위:십억원)': np.random.uniform(1000, 50000, n_periods),
        '기대신용손실(ECL)(단위:십억원)': np.random.uniform(10, 1000, n_periods),
        'Quarter': quarters
    })
    
    # 시나리오 데이터 생성
    scenario1_data = basic_data.copy()
    scenario1_data['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'] = basic_data['기대신용손실(ECL)(단위:십억원)'] * np.random.uniform(1.1, 1.5, n_periods)
    scenario1_data['경제성장률'] = np.random.uniform(0.02, 0.07, n_periods)
    scenario1_data['실업률'] = np.random.uniform(0.03, 0.05, n_periods)
    scenario1_data['기준금리'] = np.random.uniform(2.0, 3.5, n_periods)
    scenario1_data['현재경기판단CSI'] = np.random.uniform(70, 120, n_periods)
    scenario1_data['기업부도율'] = np.random.uniform(0.02, 0.04, n_periods)
    
    scenario2_data = basic_data.copy()
    scenario2_data['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'] = basic_data['기대신용손실(ECL)(단위:십억원)'] * np.random.uniform(0.8, 1.2, n_periods)
    
    # 발생 신용손실 데이터
    actual_data = pd.DataFrame({
        '발생신용손실(단위:백만원)': np.random.uniform(1000, 100000, n_periods)
    })
    
    return basic_data, scenario1_data, scenario2_data, actual_data

def classify_ecl_stage(ecl_value, percentile_33, percentile_67):
    """ECL 3단계 모델 분류"""
    if ecl_value <= percentile_33:
        return "1단계 (정상)"
    elif ecl_value <= percentile_67:
        return "2단계 (주의)"
    else:
        return "3단계 (부실)"

def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📊 ECL 분석 대시보드 (2010-2014)</h1>
        <p>Expected Credit Loss 모니터링 및 위험 분석 시스템</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    basic_data, scenario1_data, scenario2_data, actual_data = load_data()
    
    # 사이드바 - 분석 옵션
    st.sidebar.header("🎛️ 분석 옵션")
    
    # 연도 범위 선택
    available_years = sorted(basic_data['YEAR'].unique())
    year_range = st.sidebar.slider(
        "분석 연도 범위",
        min_value=min(available_years),
        max_value=max(available_years),
        value=(min(available_years), max(available_years)),
        step=1
    )
    
    # 분기 선택
    quarters_to_show = st.sidebar.multiselect(
        "표시할 분기",
        options=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        format_func=lambda x: f"{x}분기"
    )
    
    # 데이터 필터링
    basic_filtered = basic_data[
        (basic_data['YEAR'] >= year_range[0]) & 
        (basic_data['YEAR'] <= year_range[1])
    ].copy()
    
    if 'Quarter' in basic_filtered.columns:
        basic_filtered = basic_filtered[basic_filtered['Quarter'].isin(quarters_to_show)]
    
    scenario1_filtered = scenario1_data[
        (scenario1_data['YEAR'] >= year_range[0]) & 
        (scenario1_data['YEAR'] <= year_range[1])
    ].copy()
    
    scenario2_filtered = scenario2_data[
        (scenario2_data['YEAR'] >= year_range[0]) & 
        (scenario2_data['YEAR'] <= year_range[1])
    ].copy()
    
    # KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pd = basic_filtered['부도확률(PD)'].mean() * 100
        st.metric(
            "평균 연체율 (PD)", 
            f"{avg_pd:.2f}%"
        )
    
    with col2:
        avg_lgd = basic_filtered['부도_시_손실확률(LGD)'].mean() * 100
        st.metric(
            "평균 손실률 (LGD)", 
            f"{avg_lgd:.1f}%"
        )
    
    with col3:
        avg_ead = basic_filtered['부도_시_손실금액(EAD)(단위:십억원)'].mean() / 1000
        st.metric(
            "평균 대출잔액 (EAD)", 
            f"{avg_ead:.0f}K"
        )
    
    with col4:
        avg_ecl = basic_filtered['기대신용손실(ECL)(단위:십억원)'].mean() / 1000
        st.metric(
            "평균 기대손실 (ECL)", 
            f"{avg_ecl:.0f}K"
        )
    
    st.markdown("---")
    
    # 메인 차트 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 연도별 ECL 시계열 비교 (2010-2014)")
        
        # 연도별 ECL 평균 계산
        basic_yearly = basic_filtered.groupby('YEAR')['기대신용손실(ECL)(단위:십억원)'].mean()
        
        # 시나리오 데이터 처리
        if '경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)' in scenario1_filtered.columns:
            scenario1_yearly = scenario1_filtered.groupby('YEAR')['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'].mean()
        else:
            scenario1_yearly = basic_yearly * 1.2  # 샘플 데이터
            
        if '경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)' in scenario2_filtered.columns:
            scenario2_yearly = scenario2_filtered.groupby('YEAR')['경제지표_반영된_기대신용손실(Updated_ECL)(단위:십억원)'].mean()
        else:
            scenario2_yearly = basic_yearly * 0.9  # 샘플 데이터
        
        # 시계열 그래프 생성
        fig_timeseries = go.Figure()
        
        # 기본 ECL
        fig_timeseries.add_trace(
            go.Scatter(
                x=basic_yearly.index,
                y=basic_yearly.values,
                mode='lines+markers',
                name='기존 ECL',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            )
        )
        
        # 시나리오 1 ECL
        fig_timeseries.add_trace(
            go.Scatter(
                x=scenario1_yearly.index,
                y=scenario1_yearly.values,
                mode='lines+markers',
                name='Scenario 1 ECL',
                line=dict(color='#ff6b6b', width=3),
                marker=dict(size=8)
            )
        )
        
        # 시나리오 2 ECL
        fig_timeseries.add_trace(
            go.Scatter(
                x=scenario2_yearly.index,
                y=scenario2_yearly.values,
                mode='lines+markers',
                name='Scenario 2 ECL',
                line=dict(color='#4ecdc4', width=3),
                marker=dict(size=8)
            )
        )
        
        fig_timeseries.update_layout(
            title="연도별 ECL 변화 추이 비교",
            xaxis_title="연도",
            yaxis_title="ECL (십억원)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_timeseries, use_container_width=True)
    
    with col2:
        st.subheader("🎯 ECL 3단계 위험 분포")
        
        # ECL 3단계 모델 분류
        ecl_values = basic_filtered['기대신용손실(ECL)(단위:십억원)']
        percentile_33 = ecl_values.quantile(0.33)
        percentile_67 = ecl_values.quantile(0.67)
        
        stage_classification = ecl_values.apply(
            lambda x: classify_ecl_stage(x, percentile_33, percentile_67)
        )
        
        stage_counts = stage_classification.value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        fig_pie = px.pie(
            values=stage_counts.values, 
            names=stage_counts.index,
            color_discrete_sequence=colors,
            title="ECL 3단계 모델 분포"
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 3단계 모델 설명
        st.markdown("""
        <div class="stage-info">
            <h4>📋 ECL 3단계 모델</h4>
            <p><strong>1단계 (정상):</strong> 신용위험 미증가</p>
            <p><strong>2단계 (주의):</strong> 신용위험 유의한 증가</p>
            <p><strong>3단계 (부실):</strong> 신용손상 자산</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 최근 분기 현황
    st.subheader("📅 최근 분기 현황")
    latest_data = basic_filtered.iloc[-1] if len(basic_filtered) > 0 else basic_filtered.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PD", f"{latest_data['부도확률(PD)']*100:.2f}%")
    with col2:
        st.metric("LGD", f"{latest_data['부도_시_손실확률(LGD)']*100:.1f}%")
    with col3:
        st.metric("EAD", f"{latest_data['부도_시_손실금액(EAD)(단위:십억원)']:,.0f}억원")
    with col4:
        st.metric("ECL", f"{latest_data['기대신용손실(ECL)(단위:십억원)']:,.0f}억원")
    
    st.markdown("---")
    
    # 상관관계 분석
    st.subheader("🔗 경제지표 및 위험요소 상관관계 분석")
    
    # 상관관계 데이터 준비
    if len(scenario1_filtered) > 0:
        corr_data = scenario1_filtered[['부도확률(PD)', '부도_시_손실확률(LGD)', '부도_시_손실금액(EAD)(단위:십억원)']].copy()
        
        # 경제지표 추가 (있는 경우)
        economic_cols = ['경제성장률', '실업률', '기준금리', '현재경기판단CSI', '기업부도율']
        for col in economic_cols:
            if col in scenario1_filtered.columns:
                corr_data[col] = scenario1_filtered[col]
    else:
        # 샘플 상관관계 데이터
        corr_data = pd.DataFrame({
            'PD': np.random.uniform(0.01, 0.05, 20),
            'LGD': np.random.uniform(0.2, 0.8, 20),
            'EAD': np.random.uniform(1000, 50000, 20),
            '경제성장률': np.random.uniform(0.02, 0.07, 20),
            '실업률': np.random.uniform(0.03, 0.05, 20),
            '기준금리': np.random.uniform(2.0, 3.5, 20),
            '기업부도율': np.random.uniform(0.02, 0.04, 20),
            '경기판단CSI': np.random.uniform(70, 120, 20)
        })
    
    # 상관관계 매트릭스 계산
    correlation_matrix = corr_data.corr()
    
    # 히트맵 생성
    fig_corr = px.imshow(
        correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        color_continuous_scale='RdBu',
        aspect='auto',
        title="경제지표 및 위험요소 상관관계 히트맵",
        color_continuous_midpoint=0
    )
    
    # 상관계수 값 표시
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            fig_corr.add_annotation(
                x=j, y=i,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(value) > 0.5 else "black", size=10)
            )
    
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 상세 데이터 테이블
    st.subheader("📋 상세 데이터")
    
    # 최근 데이터 표시
    display_data = basic_filtered.tail(10).copy()
    if len(display_data) > 0:
        display_data['PD (%)'] = (display_data['부도확률(PD)'] * 100).round(2)
        display_data['LGD (%)'] = (display_data['부도_시_손실확률(LGD)'] * 100).round(1)
        display_data['EAD (억원)'] = display_data['부도_시_손실금액(EAD)(단위:십억원)'].round(0)
        display_data['ECL (억원)'] = display_data['기대신용손실(ECL)(단위:십억원)'].round(0)
        
        st.dataframe(
            display_data[['YEAR', 'PD (%)', 'LGD (%)', 'EAD (억원)', 'ECL (억원)']],
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()