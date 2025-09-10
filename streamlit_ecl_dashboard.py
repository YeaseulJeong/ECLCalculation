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
    page_title="ECL 정확성 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .insight-box {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bfdbfe;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    try:
        # CSV 파일들을 읽어오는 부분 (실제 파일 경로에 맞게 수정 필요)
        # actual_loss = pd.read_csv('발생신용손실.csv')
        # original_ecl = pd.read_csv('기대신용손실.csv') 
        # updated_ecl = pd.read_csv('반영된_기대신용손실.csv')
        # macro_data = pd.read_csv('거시경제지표.csv')
        
        # 샘플 데이터 (실제 데이터로 대체 필요)
        yearly_data = pd.DataFrame({
            'year': [2010, 2011, 2012, 2013, 2014],
            'actual_loss': [50940043, 11656614, 768058, 953325, 676862],
            'original_ecl': [9251.61, 105.58, 104.28, 1997.65, 16.85],
            'updated_ecl': [42372.88, 496.01, 489.92, 37574.12, 79.71],
            'growth_rate': [0.07, 0.037, 0.025, 0.033, 0.032],
            'unemployment_rate': [0.037, 0.034, 0.032, 0.031, 0.035],
            'interest_rate': [2.5, 3.25, 2.75, 2.5, 2.0]
        })
        
        # 분기별 데이터
        quarterly_data = pd.DataFrame({
            'year': [2010]*4 + [2011]*4 + [2012]*4 + [2013]*4 + [2014]*4,
            'quarter': ['Q1', 'Q2', 'Q3', 'Q4']*5,
            'actual_loss': [87030, 101616, 30185947, 20665450, 9520497, 768058, 389194, 378865,
                           141533, 30487, 453843, 142195, 334835, 305197, 29636, 283557,
                           51278, 313292, 45295, 267997],
            'original_ecl': [324.141072, 152.438706, 8755.12078, 19.9136, 90.743832, 0.073024,
                           14.76003922, 0.00021366, 0.0012984, 0.69319125, 0.00044724, 103.5828218,
                           7860.541743, 79.907454, 15.29426, 41.9992524, 0.14292864, 16.831212,
                           7.80E-05, 0.0001314],
            'updated_ecl': [1522.84381, 716.1707044, 41132.34218, 93.55587773, 426.3226564,
                          0.343073297, 69.34398723, 0.001003794, 0.0061, 3.256674626,
                          0.002101174, 486.6413815, 36929.52968, 375.4123813, 71.85380436,
                          197.3162523, 0.671491562, 79.07454262, 0.000366263, 0.000617329]
        })
        
        return yearly_data, quarterly_data
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None, None

def calculate_metrics(data):
    """성능 메트릭 계산"""
    def mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    def accuracy_score(actual, predicted):
        return 100 - np.mean(np.abs((actual - predicted) / actual)) * 100
    
    original_mape = mape(data['actual_loss'], data['original_ecl'])
    updated_mape = mape(data['actual_loss'], data['updated_ecl'])
    
    original_accuracy = accuracy_score(data['actual_loss'], data['original_ecl'])
    updated_accuracy = accuracy_score(data['actual_loss'], data['updated_ecl'])
    
    improvement = original_mape - updated_mape
    
    return {
        'original_mape': original_mape,
        'updated_mape': updated_mape,
        'original_accuracy': original_accuracy,
        'updated_accuracy': updated_accuracy,
        'improvement': improvement
    }

def create_main_comparison_chart(data):
    """메인 비교 차트 생성"""
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # 발생신용손실 (Bar)
    fig.add_trace(
        go.Bar(
            x=data['year'],
            y=data['actual_loss'],
            name='발생신용손실',
            marker_color='#ef4444',
            yaxis='y'
        )
    )
    
    # 기존 ECL (Line)
    fig.add_trace(
        go.Scatter(
            x=data['year'],
            y=data['original_ecl'],
            mode='lines+markers',
            name='기존 ECL',
            line=dict(color='#3b82f6', width=3),
            yaxis='y2'
        )
    )
    
    # 개선된 ECL (Line)
    fig.add_trace(
        go.Scatter(
            x=data['year'],
            y=data['updated_ecl'],
            mode='lines+markers',
            name='개선된 ECL',
            line=dict(color='#10b981', width=3),
            yaxis='y2'
        )
    )
    
    fig.update_layout(
        title={
            'text': '연도별 신용손실 추이 비교',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='연도',
        height=500,
        hovermode='x unified'
    )
    
    # Y축 설정
    fig.update_yaxes(title_text="발생신용손실 (백만원)", secondary_y=False)
    fig.update_yaxes(title_text="ECL 예측값 (십억원)", secondary_y=True)
    
    return fig

def create_accuracy_comparison_chart(data):
    """정확도 비교 차트"""
    metrics = calculate_metrics(data)
    
    accuracy_data = pd.DataFrame({
        'year': data['year'],
        'original_accuracy': 100 - np.abs((data['actual_loss'] - data['original_ecl']) / data['actual_loss']) * 100,
        'updated_accuracy': 100 - np.abs((data['actual_loss'] - data['updated_ecl']) / data['actual_loss']) * 100
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=accuracy_data['year'],
        y=accuracy_data['original_accuracy'],
        name='기존 ECL 정확도',
        marker_color='#3b82f6',
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        x=accuracy_data['year'],
        y=accuracy_data['updated_accuracy'],
        name='개선된 ECL 정확도',
        marker_color='#10b981',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title={
            'text': '연도별 예측 정확도 비교',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='연도',
        yaxis_title='정확도 (%)',
        barmode='group',
        height=400
    )
    
    return fig

def create_macro_indicators_chart(data):
    """거시경제 지표 차트"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['year'],
        y=data['growth_rate'] * 100,
        mode='lines+markers',
        name='경제성장률',
        line=dict(color='#f59e0b', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['year'],
        y=data['unemployment_rate'] * 100,
        mode='lines+markers',
        name='실업률',
        line=dict(color='#ef4444', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['year'],
        y=data['interest_rate'],
        mode='lines+markers',
        name='기준금리',
        line=dict(color='#8b5cf6', width=3)
    ))
    
    fig.update_layout(
        title={
            'text': '거시경제 지표 추이',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='연도',
        yaxis_title='비율 (%)',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_error_analysis_chart(data):
    """오차 분석 차트"""
    original_error = np.abs(data['actual_loss'] - data['original_ecl'])
    updated_error = np.abs(data['actual_loss'] - data['updated_ecl'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['year'],
        y=original_error,
        name='기존 ECL 오차',
        marker_color='#3b82f6',
        offsetgroup=1
    ))
    
    fig.add_trace(go.Bar(
        x=data['year'],
        y=updated_error,
        name='개선된 ECL 오차',
        marker_color='#10b981',
        offsetgroup=2
    ))
    
    fig.update_layout(
        title={
            'text': '연도별 예측 오차 비교',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='연도',
        yaxis_title='절대 오차',
        barmode='group',
        height=400,
        yaxis=dict(type='log')  # 로그 스케일로 표시
    )
    
    return fig

def display_insights():
    """인사이트 표시"""
    st.markdown("""
    <div class="insight-box">
        <h3>📊 분석 결과 및 인사이트</h3>
        <ul>
            <li><strong>거시경제지표의 유효성</strong>: 경제성장률, 실업률, 기준금리를 반영한 ECL이 기존 모델보다 우수한 성능을 보임</li>
            <li><strong>2010년 금융위기 영향</strong>: 높은 신용손실 발생으로 예측의 어려움이 있었으나, 개선된 모델이 상대적으로 양호한 성과</li>
            <li><strong>경기 사이클 반영</strong>: 2011-2012년 경기 침체기와 2013-2014년 회복기의 패턴을 개선된 모델이 더 잘 포착</li>
            <li><strong>변동성 대응</strong>: 거시경제 변수를 통해 신용환경 변화에 더 민감하게 반응하는 예측 가능</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <h3>💡 ECL 모델 개선 방향</h3>
        <ul>
            <li><strong>추가 거시경제 변수</strong>: GDP 디플레이터, 환율, 주가지수, 부동산 가격지수 등 추가 고려</li>
            <li><strong>업종별 세분화</strong>: 업종별 특성을 반영한 차별화된 조정계수 적용</li>
            <li><strong>선행지표 활용</strong>: 경기선행지수, 소비자신뢰지수 등 미래 예측력 있는 지표 포함</li>
            <li><strong>머신러닝 기법</strong>: Random Forest, XGBoost 등 고도화된 알고리즘 적용</li>
            <li><strong>동적 조정</strong>: 시장 상황에 따른 실시간 가중치 조정 메커니즘 구축</li>
            <li><strong>스트레스 테스트</strong>: 극단적 시나리오에서의 모델 안정성 검증</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    """메인 함수"""
    # 헤더
    st.markdown('<h1 class="main-header">📊 ECL 정확성 분석 대시보드</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280;">발생신용손실 대비 기존 ECL과 거시경제지표 반영 ECL의 성능 비교</p>', unsafe_allow_html=True)
    
    # 데이터 로드
    yearly_data, quarterly_data = load_and_prepare_data()
    
    if yearly_data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 사이드바
    st.sidebar.header("분석 옵션")
    
    # 연도 필터
    year_range = st.sidebar.slider(
        "분석 연도 범위",
        min_value=int(yearly_data['year'].min()),
        max_value=int(yearly_data['year'].max()),
        value=(int(yearly_data['year'].min()), int(yearly_data['year'].max()))
    )
    
    # 데이터 필터링
    filtered_data = yearly_data[
        (yearly_data['year'] >= year_range[0]) & 
        (yearly_data['year'] <= year_range[1])
    ].copy()
    
    # 메트릭 계산
    metrics = calculate_metrics(filtered_data)
    
    # KPI 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="기존 ECL MAPE",
            value=f"{metrics['original_mape']:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="개선된 ECL MAPE",
            value=f"{metrics['updated_mape']:.1f}%",
            delta=f"{metrics['improvement']:+.1f}%p"
        )
    
    with col3:
        better_model = "개선된 ECL" if metrics['updated_mape'] < metrics['original_mape'] else "기존 ECL"
        st.metric(
            label="우수 모델",
            value=better_model,
            delta=None
        )
    
    with col4:
        improvement_pct = (metrics['improvement'] / metrics['original_mape']) * 100
        st.metric(
            label="개선율",
            value=f"{improvement_pct:.1f}%",
            delta=None
        )
    
    # 차트 표시
    st.subheader("📈 주요 분석 차트")
    
    # 메인 비교 차트
    st.plotly_chart(create_main_comparison_chart(filtered_data), use_container_width=True)
    
    # 2x2 차트 레이아웃
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_accuracy_comparison_chart(filtered_data), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_macro_indicators_chart(filtered_data), use_container_width=True)
    
    # 오차 분석 차트
    st.plotly_chart(create_error_analysis_chart(filtered_data), use_container_width=True)
    
    # 상세 데이터 테이블
    st.subheader("📋 상세 데이터")
    
    # 데이터 포맷팅
    display_data = filtered_data.copy()
    display_data['actual_loss'] = display_data['actual_loss'].apply(lambda x: f"{x:,.0f}")
    display_data['original_ecl'] = display_data['original_ecl'].apply(lambda x: f"{x:,.2f}")
    display_data['updated_ecl'] = display_data['updated_ecl'].apply(lambda x: f"{x:,.2f}")
    display_data['growth_rate'] = display_data['growth_rate'].apply(lambda x: f"{x*100:.1f}%")
    display_data['unemployment_rate'] = display_data['unemployment_rate'].apply(lambda x: f"{x*100:.1f}%")
    display_data['interest_rate'] = display_data['interest_rate'].apply(lambda x: f"{x:.2f}%")
    
    # 컬럼명 변경
    display_data.columns = ['연도', '발생신용손실(백만원)', '기존ECL(십억원)', '개선된ECL(십억원)', 
                          '경제성장률', '실업률', '기준금리']
    
    st.dataframe(display_data, use_container_width=True)
    
    # 인사이트 표시
    st.subheader("🔍 분석 인사이트")
    display_insights()
    
    # 다운로드 섹션
    st.subheader("💾 데이터 다운로드")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = filtered_data.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📊 분석 데이터 CSV",
            data=csv,
            file_name=f"ecl_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # 메트릭 데이터를 JSON으로 저장
        import json
        metrics_json = json.dumps(metrics, indent=2, ensure_ascii=False)
        st.download_button(
            label="📈 성능 메트릭 JSON",
            data=metrics_json,
            file_name=f"ecl_metrics_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    # 푸터
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #6b7280;">ECL 정확성 분석 대시보드 | 금융 리스크 관리</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# 실행 방법:
# 1. 필요한 라이브러리 설치: pip install streamlit pandas numpy plotly seaborn matplotlib
# 2. CSV 파일들을 같은 디렉토리에 저장
# 3. 터미널에서 실행: streamlit run ecl_dashboard.py