import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="ECL 분석 대시보드",
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
    
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
        border: 2px solid #667eea;
        border-radius: 10px;
    }
    
    .quarter-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin: 0.1rem;
        background: #667eea;
        color: white;
        border-radius: 15px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 생성 함수
@st.cache_data
def generate_ecl_data():
    """ECL 데이터 생성 (원본 코드 구조 반영)"""
    # 2010년부터 2023년까지의 분기별 데이터
    dates = pd.date_range('2010-01-01', '2023-12-01', freq='QS')  # QS = Quarter Start
    np.random.seed(42)
    
    # 경제 사이클을 반영한 데이터 패턴
    n_periods = len(dates)
    time_trend = np.arange(n_periods)
    
    # PD (연체율) - 경제위기 시기에 높아지는 패턴
    base_pd = 0.025
    cycle_pd = 0.01 * np.sin(time_trend * 0.3) + 0.005 * np.sin(time_trend * 0.1)
    crisis_effect = np.where((time_trend > 8) & (time_trend < 16), 0.015, 0)  # 2012-2014 위기
    covid_effect = np.where(time_trend > 40, 0.01, 0)  # 2020년 이후 코로나 효과
    noise_pd = np.random.normal(0, 0.003, n_periods)
    pd_data = base_pd + cycle_pd + crisis_effect + covid_effect + noise_pd
    pd_data = np.clip(pd_data, 0.01, 0.08)
    
    # LGD (손실률) - PD와 양의 상관관계
    base_lgd = 0.45
    correlation_with_pd = 0.3 * (pd_data - base_pd/2)
    cycle_lgd = 0.08 * np.sin(time_trend * 0.25 + 1.5)
    noise_lgd = np.random.normal(0, 0.04, n_periods)
    lgd_data = base_lgd + correlation_with_pd + cycle_lgd + noise_lgd
    lgd_data = np.clip(lgd_data, 0.25, 0.7)
    
    # EAD (대출잔액) - 꾸준한 증가 추세 + 경기 사이클
    base_ead = 50000
    growth_trend = time_trend * 1200
    cycle_ead = 8000 * np.sin(time_trend * 0.2)
    recession_effect = np.where((time_trend > 8) & (time_trend < 16), -5000, 0)
    noise_ead = np.random.normal(0, 3000, n_periods)
    ead_data = base_ead + growth_trend + cycle_ead + recession_effect + noise_ead
    ead_data = np.clip(ead_data, 40000, 150000)
    
    # 분기 정보 추가
    quarters = []
    quarter_labels = []
    for date in dates:
        quarter = f"{date.year}Q{(date.month-1)//3 + 1}"
        quarters.append(quarter)
        quarter_labels.append(f"{date.year}년 {(date.month-1)//3 + 1}분기")
    
    # DataFrame 생성
    df = pd.DataFrame({
        'Date': dates,
        'Quarter': quarters,
        'Quarter_Label': quarter_labels,
        'Year': [d.year for d in dates],
        'Q': [(d.month-1)//3 + 1 for d in dates],
        'PD': pd_data,
        'LGD': lgd_data,
        'EAD': ead_data,
        'ECL': pd_data * lgd_data * ead_data
    })
    
    return df

# 상관관계 분석 함수
def analyze_correlations(df):
    """상관관계 상세 분석"""
    corr_matrix = df[['PD', 'LGD', 'EAD', 'ECL']].corr()
    
    # 시차 상관관계 분석 (lag correlation)
    lag_correlations = {}
    for lag in range(1, 5):
        if len(df) > lag:
            lag_corr = {}
            for col in ['LGD', 'EAD']:
                lag_corr[f'PD_lag{lag}_{col}'] = df['PD'].shift(lag).corr(df[col])
                lag_corr[f'{col}_lag{lag}_ECL'] = df[col].shift(lag).corr(df['ECL'])
            lag_correlations[f'lag_{lag}'] = lag_corr
    
    return corr_matrix, lag_correlations

# 메인 대시보드
def main():
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1>📊 ECL 분석 대시보드</h1>
        <p>Expected Credit Loss 모니터링 및 위험 분석 시스템 (분기별 분석)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 로드
    df = generate_ecl_data()
    
    # 사이드바
    st.sidebar.header("🎛️ 분석 옵션")
    
    # 연도 범위 선택
    year_range = st.sidebar.slider(
        "분석 연도 범위",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max())),
        step=1
    )
    
    # 분기 선택
    quarters_to_show = st.sidebar.multiselect(
        "표시할 분기",
        options=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        format_func=lambda x: f"{x}분기"
    )
    
    # 상관관계 분석 타입
    correlation_type = st.sidebar.selectbox(
        "상관관계 분석 방법",
        ["피어슨 상관계수", "스피어만 순위 상관계수", "시차 상관관계"]
    )
    
    # 데이터 필터링
    filtered_df = df[
        (df['Year'] >= year_range[0]) & 
        (df['Year'] <= year_range[1]) &
        (df['Q'].isin(quarters_to_show))
    ].copy()
    
    if filtered_df.empty:
        st.error("선택한 조건에 해당하는 데이터가 없습니다.")
        return
    
    # KPI 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_pd = filtered_df['PD'].mean() * 100
        st.metric(
            "평균 연체율 (PD)", 
            f"{avg_pd:.2f}%",
            delta=f"{(filtered_df['PD'].iloc[-1] - filtered_df['PD'].iloc[0])*100:.2f}%p"
        )
    
    with col2:
        avg_lgd = filtered_df['LGD'].mean() * 100
        st.metric(
            "평균 손실률 (LGD)", 
            f"{avg_lgd:.1f}%",
            delta=f"{(filtered_df['LGD'].iloc[-1] - filtered_df['LGD'].iloc[0])*100:.1f}%p"
        )
    
    with col3:
        avg_ead = filtered_df['EAD'].mean() / 1000
        st.metric(
            "평균 대출잔액 (EAD)", 
            f"{avg_ead:.0f}K",
            delta=f"{(filtered_df['EAD'].iloc[-1] - filtered_df['EAD'].iloc[0])/1000:.0f}K"
        )
    
    with col4:
        avg_ecl = filtered_df['ECL'].mean() / 1000
        st.metric(
            "평균 기대손실 (ECL)", 
            f"{avg_ecl:.0f}K",
            delta=f"{(filtered_df['ECL'].iloc[-1] - filtered_df['ECL'].iloc[0])/1000:.0f}K"
        )
    
    st.markdown("---")
    
    # 메인 차트 영역
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 분기별 ECL 시계열 변화")
        
        # ECL 시계열 차트 (분기별)
        fig_timeseries = go.Figure()
        
        # ECL 라인 추가
        fig_timeseries.add_trace(
            go.Scatter(
                x=filtered_df['Date'],
                y=filtered_df['ECL'],
                mode='lines+markers',
                name='ECL (기대신용손실)',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='white', line=dict(color='#667eea', width=2)),
                hovertemplate='<b>%{customdata}</b><br>ECL: %{y:,.0f}<extra></extra>',
                customdata=filtered_df['Quarter_Label']
            )
        )
        
        # 이동평균 추가
        if len(filtered_df) >= 4:
            ma_4 = filtered_df['ECL'].rolling(window=4, center=True).mean()
            fig_timeseries.add_trace(
                go.Scatter(
                    x=filtered_df['Date'],
                    y=ma_4,
                    mode='lines',
                    name='4분기 이동평균',
                    line=dict(color='#ff6b6b', width=2, dash='dash'),
                    hovertemplate='<b>%{customdata}</b><br>4분기 평균: %{y:,.0f}<extra></extra>',
                    customdata=filtered_df['Quarter_Label']
                )
            )
        
        # 평균선 추가
        mean_ecl = filtered_df['ECL'].mean()
        fig_timeseries.add_hline(
            y=mean_ecl, 
            line_dash="dot", 
            line_color="green",
            annotation_text=f"전체 평균: {mean_ecl:,.0f}"
        )
        
        fig_timeseries.update_layout(
            title="분기별 ECL 변화 추이",
            xaxis_title="분기",
            yaxis_title="ECL (기대신용손실)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_timeseries, use_container_width=True)
        
        # 분기별 세부 정보
        st.subheader("🔍 분기별 세부 분석")
        
        # 년도별 분기 평균
        quarterly_summary = filtered_df.groupby(['Year', 'Q']).agg({
            'PD': 'mean',
            'LGD': 'mean', 
            'EAD': 'mean',
            'ECL': 'mean'
        }).round(4)
        
        # 히트맵으로 표시
        pivot_data = quarterly_summary['ECL'].unstack(level='Q')
        
        fig_heatmap = px.imshow(
            pivot_data.values,
            x=[f"{q}분기" for q in pivot_data.columns],
            y=[str(int(y)) for y in pivot_data.index],
            color_continuous_scale='RdYlBu_r',
            aspect='auto',
            title="연도별 분기별 ECL 히트맵"
        )
        
        fig_heatmap.update_layout(height=300)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.subheader("🎯 위험도 분포")
        
        # ECL을 기준으로 위험도 분류
        ecl_percentiles = filtered_df['ECL'].quantile([0.33, 0.67])
        risk_categories = pd.cut(
            filtered_df['ECL'], 
            bins=[-np.inf, ecl_percentiles.iloc[0], ecl_percentiles.iloc[1], np.inf],
            labels=['저위험', '중위험', '고위험']
        )
        
        risk_counts = risk_categories.value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            color_discrete_sequence=colors,
            title="ECL 기준 위험도 분포"
        )
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # 최근 분기 정보
        st.subheader("📅 최근 분기 현황")
        latest_data = filtered_df.iloc[-1]
        
        st.markdown(f"""
        **{latest_data['Quarter_Label']}**
        - PD: {latest_data['PD']*100:.2f}%
        - LGD: {latest_data['LGD']*100:.1f}%  
        - EAD: {latest_data['EAD']:,.0f}원
        - ECL: {latest_data['ECL']:,.0f}원
        """)
    
    st.markdown("---")
    
    # 상관관계 분석 섹션
    st.subheader("🔗 PD, LGD, EAD 간 상관관계 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 상관관계 히트맵
        corr_matrix, lag_correlations = analyze_correlations(filtered_df)
        
        if correlation_type == "피어슨 상관계수":
            correlation_data = filtered_df[['PD', 'LGD', 'EAD', 'ECL']].corr()
        elif correlation_type == "스피어만 순위 상관계수":
            correlation_data = filtered_df[['PD', 'LGD', 'EAD', 'ECL']].corr(method='spearman')
        else:  # 시차 상관관계
            correlation_data = corr_matrix
        
        fig_corr = px.imshow(
            correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.index,
            color_continuous_scale='RdBu',
            aspect='auto',
            title=f"{correlation_type} 매트릭스",
            color_continuous_midpoint=0
        )
        
        # 상관계수 값 표시
        for i, row in enumerate(correlation_data.values):
            for j, value in enumerate(row):
                fig_corr.add_annotation(
                    x=j, y=i,
                    text=f"{value:.3f}",
                    showarrow=False,
                    font=dict(color="white" if abs(value) > 0.5 else "black", size=12)
                )
        
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # 상관관계 인사이트
        max_corr = correlation_data.abs().max().max()
        max_pair = correlation_data.abs().stack().idxmax()
        
        if max_pair[0] != max_pair[1]:  # 자기 자신과의 상관관계 제외
            st.markdown(f"""
            <div class="correlation-insight">
                <h4>🎯 주요 발견사항</h4>
                <p><strong>{max_pair[0]}</strong>와 <strong>{max_pair[1]}</strong> 간 
                상관계수가 <strong>{correlation_data.loc[max_pair]:.3f}</strong>로 가장 높습니다.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # PD vs LGD 산점도 (ECL 크기로 표현)
        fig_scatter = px.scatter(
            filtered_df,
            x='PD',
            y='LGD', 
            size='ECL',
            color='Year',
            hover_data={'Quarter_Label': True, 'ECL': ':,.0f'},
            title="PD vs LGD 관계 (점 크기: ECL, 색상: 연도)",
            labels={
                'PD': 'PD (연체율)',
                'LGD': 'LGD (손실률)',
                'Year': '연도'
            }
        )
        
        # 추세선 추가
        fig_scatter.add_trace(
            px.scatter(
                filtered_df, x='PD', y='LGD',
                trendline="ols"
            ).data[1]
        )
        
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 3D 상관관계 분석
        st.subheader("🌐 3차원 관계 분석")
        
        fig_3d = px.scatter_3d(
            filtered_df,
            x='PD',
            y='LGD',
            z='EAD', 
            color='ECL',
            size='ECL',
            hover_data={'Quarter_Label': True},
            title="PD-LGD-EAD 3차원 관계",
            labels={
                'PD': 'PD (연체율)',
                'LGD': 'LGD (손실률)', 
                'EAD': 'EAD (대출잔액)'
            },
            color_continuous_scale='Viridis'
        )
        
        fig_3d.update_layout(height=500, scene=dict(
            xaxis_title="PD (연체율)",
            yaxis_title="LGD (손실률)",
            zaxis_title="EAD (대출잔액)"
        ))
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # 상세 데이터 테이블
    st.subheader("📋 상세 데이터")
    
    # 최근 8분기 데이터 표시
    recent_data = filtered_df.tail(8)[['Quarter_Label', 'PD', 'LGD', 'EAD', 'ECL']].copy()
    recent_data['PD'] = (recent_data['PD'] * 100).round(2)
    recent_data['LGD'] = (recent_data['LGD'] * 100).round(1) 
    recent_data['EAD'] = recent_data['EAD'].round(0)
    recent_data['ECL'] = recent_data['ECL'].round(0)
    
    recent_data.columns = ['분기', 'PD (%)', 'LGD (%)', 'EAD (원)', 'ECL (원)']
    
    st.dataframe(
        recent_data,
        use_container_width=True,
        hide_index=True
    )
    
    # 시차 상관관계 정보 (선택시)
    if correlation_type == "시차 상관관계" and lag_correlations:
        st.subheader("⏱️ 시차 상관관계 분석")
        
        lag_df_data = []
        for lag, correlations in lag_correlations.items():
            for pair, corr_val in correlations.items():
                if not pd.isna(corr_val):
                    lag_num = lag.split('_')[1]
                    lag_df_data.append({
                        'Lag': f"{lag_num}분기",
                        'Variable Pair': pair.replace('_lag' + lag_num + '_', ' → '),
                        'Correlation': round(corr_val, 3)
                    })
        
        if lag_df_data:
            lag_df = pd.DataFrame(lag_df_data)
            st.dataframe(lag_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()