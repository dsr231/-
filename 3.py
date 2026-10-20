import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# 1. 页面基本配置与深度 UI 美化 (极简深色科技风)
# ==========================================
st.set_page_config(page_title="数据分析岗薪酬洞察", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* 1. 全局背景 */
    .stApp { background-color: #030a16; color: #c0d2e0; }

    /* 2. 隐藏顶部默认边距 */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }

    /* 3. 顶部科技感大标题 */
    .tech-header {
        text-align: center;
        background: linear-gradient(90deg, rgba(3,10,22,0) 0%, rgba(0,191,255,0.1) 50%, rgba(3,10,22,0) 100%);
        border-top: 1px solid rgba(0, 191, 255, 0.5);
        border-bottom: 1px solid rgba(0, 191, 255, 0.5);
        padding: 15px 0; margin-bottom: 20px;
    }
    .tech-header h1 {
        color: #00FFFF !important; font-size: 2.5rem !important;
        font-weight: 800; letter-spacing: 5px; margin: 0;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
    }

    /* 4. 科技感卡片容器 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(5, 16, 31, 0.5) !important;
        border: 1px solid rgba(0, 191, 255, 0.15) !important; 
        box-shadow: inset 0 0 20px rgba(0, 191, 255, 0.05) !important;
        border-radius: 4px !important; padding: 15px !important; margin-bottom: 15px;
    }

    /* 5. 指标卡 */
    [data-testid="stMetricLabel"] { color: #7f98b0 !important; font-weight: bold !important; }
    [data-testid="stMetricValue"] { color: #00E5FF !important; font-size: 2.2rem; font-weight: bold; text-shadow: 0 0 8px rgba(0,229,255,0.4); }

    /* 6. 输入组件标签 */
    [data-testid="stWidgetLabel"] p { color: #c0d2e0 !important; font-weight: 500 !important; font-size: 1.1rem; }

    /* 7. 多选框 */
    div[data-baseweb="select"] > div { background-color: #05101f !important; border: 1px solid #1a5b82 !important; }
    div[data-baseweb="select"] span { background-color: #0b2d4a !important; color: #00FFFF !important; }

    /* 8. 按钮 */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #005c97, #00BFFF) !important; 
        color: #ffffff !important; font-weight: bold; width: 100%;
        font-size: 1.2rem !important; padding: 10px !important;
        border: none !important; box-shadow: 0 0 10px rgba(0,191,255,0.5) !important;
    }

    /* 9. 结论框 */
    .insight-box {
        background-color: rgba(5, 16, 31, 0.8); padding: 20px; border-radius: 4px; 
        border-left: 4px solid #00E5FF; margin-bottom: 20px; 
        box-shadow: inset 0 0 10px rgba(0, 191, 255, 0.05); height: 100%;
    }
    .insight-box h4 { color: #00E5FF !important; margin-bottom: 15px; }
    .insight-box li { color: #a0b8cf; line-height: 1.8; font-size: 0.95rem; }
    .insight-box b { color: #ffffff; }

    /* 10. 明亮大 Tab */
    [data-testid="stTabs"] [data-baseweb="tab-list"] { 
        gap: 40px !important; border-bottom: none !important; 
        margin-bottom: 20px !important; justify-content: flex-start !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button { 
        height: auto !important; padding: 10px 0px !important; 
        background-color: transparent !important; border: none !important;
        border-bottom: 4px solid transparent !important; border-radius: 0 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="false"] div[data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important; font-size: 1.5rem !important; 
        font-weight: 500 !important; letter-spacing: 2px; margin: 0 !important; transition: all 0.3s ease;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button:hover div[data-testid="stMarkdownContainer"] p {
        color: #00E5FF !important; text-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] { 
        background-color: transparent !important; 
        border-bottom: 4px solid #00E5FF !important; 
        box-shadow: 0 4px 15px -5px rgba(0, 229, 255, 0.8) !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] div[data-testid="stMarkdownContainer"] p {
        color: #00E5FF !important; font-size: 1.5rem !important; 
        font-weight: 800 !important; letter-spacing: 2px; margin: 0 !important;
        text-shadow: 0 0 15px rgba(0, 229, 255, 0.6); 
    }

    /* 11. 图表入场动画 */
    @keyframes revealBottomUp {
        from { clip-path: inset(0 0 100% 0); opacity: 0; }
        to { clip-path: inset(0 0 0 0); opacity: 1; }
    }
    @keyframes revealLeftRight {
        from { clip-path: inset(0 100% 0 0); opacity: 0; }
        to { clip-path: inset(0 0 0 0); opacity: 1; }
    }

    /* 【终极优化】环形图切片依次扫过动画 */
    @keyframes pieSliceSweep {
        0% { opacity: 0; transform: rotate(-90deg) scale(0.8); }
        100% { opacity: 1; transform: rotate(0deg) scale(1); }
    }

    .js-plotly-plot:not(:has(.ytick text)):not(:has(.pielayer)) .barlayer { animation: revealBottomUp 0.8s cubic-bezier(0.25, 1, 0.5, 1) forwards; }
    .js-plotly-plot:has(.ytick text) .barlayer { animation: revealLeftRight 0.8s cubic-bezier(0.25, 1, 0.5, 1) forwards; }
    .js-plotly-plot .scatterlayer { animation: revealLeftRight 1.2s cubic-bezier(0.25, 1, 0.5, 1) forwards; }

    /* 环形图切片动态，延迟依次递增 */
    .js-plotly-plot .pielayer .slice {
        transform-origin: center;
        opacity: 0; 
        animation: pieSliceSweep 0.6s cubic-bezier(0.25, 1, 0.5, 1) forwards;
    }

    /* 为前 7 个切片设置错开的延迟时间 (覆盖所有学历分类) */
    .js-plotly-plot .pielayer .slice:nth-child(1) { animation-delay: 0.1s; }
    .js-plotly-plot .pielayer .slice:nth-child(2) { animation-delay: 0.4s; }
    .js-plotly-plot .pielayer .slice:nth-child(3) { animation-delay: 0.7s; }
    .js-plotly-plot .pielayer .slice:nth-child(4) { animation-delay: 1.0s; }
    .js-plotly-plot .pielayer .slice:nth-child(5) { animation-delay: 1.3s; }
    .js-plotly-plot .pielayer .slice:nth-child(6) { animation-delay: 1.6s; }
    .js-plotly-plot .pielayer .slice:nth-child(7) { animation-delay: 1.9s; }
    </style>
    """, unsafe_allow_html=True)


# ==========================================
# 2. 数据加载与深度清洗
# ==========================================
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv('boss_jobs.csv', encoding='gbk')
    except:
        df = pd.read_csv('boss_jobs.csv', encoding='gb18030')

    def parse_salary(s):
        if pd.isna(s): return np.nan
        s = str(s).lower().replace(' ', '')
        if '面议' in s: return np.nan
        if '/天' in s or '元/天' in s:
            nums = re.findall(r'\d+', s)
            if len(nums) >= 2:
                return ((float(nums[0]) + float(nums[1])) / 2 * 21.75) / 1000
        elif 'k' in s or '千' in s:
            nums = re.findall(r'\d+', s)
            if len(nums) >= 2:
                avg = (float(nums[0]) + float(nums[1])) / 2
                if '薪' in s:
                    multiplier = re.search(r'(\d+)薪', s)
                    if multiplier: avg = avg * (int(multiplier.group(1)) / 12)
                return avg
        return np.nan

    df['avg_salary'] = df['salary'].apply(parse_salary)

    def parse_requirements(req):
        req = str(req)
        edu = '其他'
        if '本科' in req:
            edu = '本科'
        elif '硕士' in req:
            edu = '硕士'
        elif '博士' in req:
            edu = '博士'
        elif '大专' in req:
            edu = '大专'
        elif '学历不限' in req or '不限' in req:
            edu = '学历不限'

        exp = '经验不限'
        if '应届' in req or '在校' in req:
            exp = '在校/应届'
        elif '1年以内' in req:
            exp = '1年以内'
        elif '1-3年' in req:
            exp = '1-3年'
        elif '3-5年' in req:
            exp = '3-5年'
        elif '5-10年' in req:
            exp = '5-10年'
        elif '10年以上' in req:
            exp = '10年以上'
        return pd.Series([edu, exp])

    df[['education', 'experience']] = df['requirements'].apply(parse_requirements)

    df['tags'] = df['tags'].fillna('').astype(str).str.replace('"', '').str.replace("'", "")
    df['tags'] = df['tags'].str.replace('，', ',')
    df = df.dropna(subset=['avg_salary'])

    bins = [0, 8, 15, 25, 40, 100]
    labels = ['0-8K', '8-15K', '15-25K', '25-40K', '>40K']
    df['salary_level'] = pd.cut(df['avg_salary'], bins=bins, labels=labels, right=False)

    city_coords = {
        '北京': [116.40, 39.90], '上海': [121.47, 31.23], '广州': [113.23, 23.16],
        '深圳': [114.07, 22.62], '杭州': [120.19, 30.26], '成都': [104.06, 30.67],
        '武汉': [114.31, 30.52], '南京': [118.78, 32.04], '西安': [108.95, 34.27],
        '长沙': [112.98, 28.19], '郑州': [113.65, 34.76], '重庆': [106.54, 29.59],
        '苏州': [120.62, 31.32], '天津': [117.20, 39.13], '厦门': [118.10, 24.46]
    }
    df['lng'] = df['address'].map(lambda x: city_coords.get(x, [np.nan, np.nan])[0])
    df['lat'] = df['address'].map(lambda x: city_coords.get(x, [np.nan, np.nan])[1])

    return df


df = load_and_clean_data()

# 全局原生 Hover 框配置
HOVER_LABEL_CONFIG = dict(
    bgcolor="rgba(5, 16, 31, 0.95)", bordercolor="#00FFFF",
    font=dict(color="#ffffff", size=14, family="Arial, sans-serif")
)

# ==========================================
# 3. 主页面与 Tab 划分
# ==========================================
st.markdown("""<div class="tech-header"><h1>📊 数据分析岗位招聘与薪酬洞察大屏</h1></div>""", unsafe_allow_html=True)

tab_overview, tab_interact, tab_conclusion, tab_predict = st.tabs([
    "行业大盘概览", "多维薪酬透视", "求职破局指南", "智能薪资预测"
])

# ------------------------------------------
# TAB 1: 行业大盘概览
# ------------------------------------------
with tab_overview:
    with st.container(border=True):
        st.markdown(f"**基于全量 {len(df)} 条数据分析岗位数据生成的行业大盘**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("有效岗位数", f"{len(df)} 个")
        col2.metric("薪资中位数", f"{df['avg_salary'].median():.1f} K")
        col3.metric("平均薪资", f"{df['avg_salary'].mean():.1f} K")
        col4.metric("覆盖行业数", f"{df['labels'].nunique()} 个")

    row1_col1, row1_col2 = st.columns([1, 1.2])
    with row1_col1:
        with st.container(border=True):
            st.subheader("📊 薪资档位分布")
            salary_counts = df['salary_level'].value_counts().reset_index()
            salary_counts.columns = ['薪资档位', '岗位数量']
            salary_counts['薪资档位'] = pd.Categorical(salary_counts['薪资档位'],
                                                       categories=['0-8K', '8-15K', '15-25K', '25-40K', '>40K'],
                                                       ordered=True)
            salary_counts = salary_counts.sort_values('薪资档位', ascending=True)
            fig_bar = px.bar(salary_counts, x='岗位数量', y='薪资档位', orientation='h', text='岗位数量',
                             template='plotly_dark', color='岗位数量', color_continuous_scale='Plasma')
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color='#c0d2e0'),
                                  margin=dict(t=30, b=0, l=0, r=0), coloraxis_showscale=False,
                                  hoverlabel=HOVER_LABEL_CONFIG)
            fig_bar.update_traces(hovertemplate='<b>%{y}</b><br>岗位数: %{x}<extra></extra>')
            st.plotly_chart(fig_bar, use_container_width=True)

    with row1_col2:
        with st.container(border=True):
            st.subheader("📈 全行业薪资分布")
            fig_dist = px.histogram(df, x='avg_salary', nbins=40, marginal='box', color_discrete_sequence=['#00E5FF'],
                                    template='plotly_dark')
            fig_dist.update_traces(marker_line_width=1, marker_line_color="#00BFFF",
                                   hovertemplate='薪资: %{x:.1f}K<br>岗位数: %{y}<extra></extra>',
                                   selector=dict(type='histogram'))
            fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   xaxis_title="平均月薪 (K)",
                                   yaxis_title="岗位数量", margin=dict(t=30, b=0, l=0, r=0), font=dict(color='#c0d2e0'),
                                   xaxis=dict(gridcolor='rgba(0, 191, 255, 0.1)'),
                                   yaxis=dict(gridcolor='rgba(0, 191, 255, 0.1)'),
                                   hoverlabel=HOVER_LABEL_CONFIG)
            st.plotly_chart(fig_dist, use_container_width=True)

    with st.container(border=True):
        st.subheader("🗺️ 全国数据分析岗城市热度地图")
        city_map_df = df.dropna(subset=['lng', 'lat']).groupby(['address', 'lng', 'lat']).agg(
            岗位数量=('avg_salary', 'count'), 平均薪资=('avg_salary', 'median')).reset_index()
        if not city_map_df.empty:
            fig_map = px.scatter_geo(city_map_df, lat='lat', lon='lng', size='岗位数量', color='平均薪资',
                                     hover_name='address', text='address',
                                     color_continuous_scale='Turbo', size_max=40,
                                     template='plotly_dark')
            fig_map.update_traces(
                hovertemplate='<b>%{text}</b><br>岗位数量: %{marker.size}<br>平均薪资: %{marker.color:.1f}K<extra></extra>',
                marker=dict(line=dict(width=1.5, color='white'))
            )
            fig_map.update_geos(
                visible=False, resolution=50, showcountries=True, countrycolor="#00BFFF",
                showland=True, landcolor="#0b2d4a", showocean=True, oceancolor="#030a16",
                showcoastlines=True, coastlinecolor="#00E5FF",
                showframe=False, lataxis_range=[15, 55], lonaxis_range=[70, 140]
            )
            fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  margin=dict(t=0, b=0, l=0, r=0), height=450, font=dict(color='#c0d2e0'),
                                  coloraxis_colorbar=dict(title="平均薪资(K)"),
                                  hoverlabel=HOVER_LABEL_CONFIG)
            st.plotly_chart(fig_map, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        with st.container(border=True):
            st.subheader("🏢 行业薪酬 TOP 10")
            ind_salary = df.groupby('labels')['avg_salary'].median().reset_index().sort_values(by='avg_salary',
                                                                                               ascending=False).head(10)
            fig_ind = px.bar(ind_salary, x='avg_salary', y='labels', orientation='h', text_auto='.1f',
                             template='plotly_dark', color='avg_salary', color_continuous_scale='Tealgrn')
            fig_ind.update_traces(hovertemplate='<b>%{y}</b><br>薪资中位数: %{x:.1f}K<extra></extra>')
            fig_ind.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', xaxis_title="薪资中位数(K)", font=dict(color='#c0d2e0'),
                                  hoverlabel=HOVER_LABEL_CONFIG)
            st.plotly_chart(fig_ind, use_container_width=True)

    with row2_col2:
        with st.container(border=True):
            st.subheader("⏳ 经验与薪酬成长曲线")
            exp_order = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
            temp_exp_df = df.copy()
            temp_exp_df['experience'] = pd.Categorical(temp_exp_df['experience'], categories=exp_order, ordered=True)
            exp_salary = temp_exp_df.groupby('experience')['avg_salary'].median().reset_index()

            fig_exp = px.line(exp_salary, x='experience', y='avg_salary', markers=True, template='plotly_dark',
                              color_discrete_sequence=['#FF007F'])
            fig_exp.update_traces(hovertemplate='<b>%{x}</b><br>薪资中位数: %{y:.1f}K<extra></extra>',
                                  line=dict(width=4),
                                  marker=dict(size=12, color='#FFD700', line=dict(width=2, color='white')))
            fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  yaxis_title="薪资中位数(K)",
                                  xaxis_title="", font=dict(color='#c0d2e0'),
                                  xaxis=dict(gridcolor='rgba(0, 191, 255, 0.1)'),
                                  yaxis=dict(gridcolor='rgba(0, 191, 255, 0.1)'),
                                  hoverlabel=HOVER_LABEL_CONFIG)
            st.plotly_chart(fig_exp, use_container_width=True)

    # 矩形树图
    with st.container(border=True):
        st.subheader("🌳 核心行业与城市岗位分布规模")
        top_inds = df['labels'].value_counts().head(8).index
        treemap_df = df[df['labels'].isin(top_inds)]
        treemap_data = treemap_df.groupby(['labels', 'address']).agg(
            岗位数量=('avg_salary', 'count'), 平均薪资=('avg_salary', 'median')).reset_index()

        fig_treemap = px.treemap(
            treemap_data, path=['labels', 'address'], values='岗位数量', color='平均薪资',
            color_continuous_scale='Turbo', template='plotly_dark', hover_data=['平均薪资']
        )
        fig_treemap.update_traces(
            hovertemplate='<b>%{label}</b><br>岗位数: %{value}<br>平均薪资: %{color:.1f}K<extra></extra>',
            # 【核心修改】只显示 label (即行业名或城市名)，不再显示括号里的数值和百分比
            textinfo="label",
            textfont=dict(size=14),
            marker=dict(line=dict(width=2, color='#030a16'))
        )
        fig_treemap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=10, b=10, l=10, r=10), height=450,
            coloraxis_colorbar=dict(title="平均薪资(K)"), hoverlabel=HOVER_LABEL_CONFIG
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

# ------------------------------------------
# TAB 2: 多维薪酬透视
# ------------------------------------------
with tab_interact:
    with st.container(border=True):
        st.markdown("### 🎯 锁定目标画像")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            city_list = sorted(df['address'].unique().tolist())
            selected_cities = st.multiselect("选择目标城市", city_list, default=city_list)
        with col_f2:
            edu_list = df['education'].unique().tolist()
            selected_edus = st.multiselect("选择学历要求", edu_list, default=edu_list)

        filtered_df = df[(df['address'].isin(selected_cities)) & (df['education'].isin(selected_edus))]
        st.markdown(f"当前筛选有效岗位样本：**{len(filtered_df)}** 条")

    row_sal1, row_sal2 = st.columns(2)
    with row_sal1:
        with st.container(border=True):
            st.markdown("**🎓 学历溢价对比**")
            if not filtered_df.empty:
                edu_order = ['其他', '大专', '学历不限', '本科', '硕士', '博士']
                existing_edus = [e for e in edu_order if e in filtered_df['education'].unique()]
                edu_salary = filtered_df.groupby('education')['avg_salary'].median().reset_index()
                edu_salary['education'] = pd.Categorical(edu_salary['education'], categories=existing_edus,
                                                         ordered=True)
                edu_salary = edu_salary.sort_values('education')
                fig_edu = px.bar(edu_salary, x='education', y='avg_salary', text_auto='.1f',
                                 template='plotly_dark', color='avg_salary', color_continuous_scale='Oranges')
                fig_edu.update_traces(hovertemplate='<b>%{x}</b><br>薪资中位数: %{y:.1f}K<extra></extra>')
                fig_edu.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis_title="", yaxis_title="薪资中位数(K)", font=dict(color='#c0d2e0'),
                                      coloraxis_showscale=False, hoverlabel=HOVER_LABEL_CONFIG)
                st.plotly_chart(fig_edu, use_container_width=True)
            else:
                st.info("当前筛选条件下无数据")

    with row_sal2:
        with st.container(border=True):
            st.markdown("**🍩 目标画像学历分布占比**")
            if not filtered_df.empty:
                edu_counts = filtered_df['education'].value_counts().reset_index()
                edu_counts.columns = ['学历', '人数']

                # 【核心修改 1】强制按人数升序排列 (博士 -> ... -> 本科)
                edu_counts = edu_counts.sort_values(by='人数', ascending=True)

                # 生成环形图
                fig_donut = px.pie(edu_counts, names='学历', values='人数', hole=0.6,
                                   color_discrete_sequence=px.colors.qualitative.Bold, template='plotly_dark')

                # 【核心修改 2】关闭自动排序，强制从 12 点钟方向顺时针画
                fig_donut.update_traces(
                    sort=False,  # 禁止重新排序
                    direction='clockwise',  # 顺时针画
                    rotation=0,  # 从 12 点方向开始
                    textposition='outside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>岗位数: %{value}<br>占比: %{percent}<extra></extra>',
                    marker=dict(line=dict(color='#030a16', width=2))
                )
                fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        margin=dict(t=30, b=30, l=0, r=0), showlegend=False,
                                        hoverlabel=HOVER_LABEL_CONFIG)

                # 【核心修改 3】绑定动态 key，确保筛选条件一变，动画就重新触发
                st.plotly_chart(fig_donut, use_container_width=True,
                                key=f"donut_{len(filtered_df)}_{len(selected_cities)}")

    with st.container(border=True):
        st.markdown("**🏙️ 目标城市薪酬排行 TOP 10**")
        if not filtered_df.empty:
            city_salary = filtered_df.groupby('address')['avg_salary'].median().reset_index().sort_values(
                by='avg_salary', ascending=False).head(10)
            fig_city = px.bar(city_salary, x='avg_salary', y='address', orientation='h', text_auto='.1f',
                              template='plotly_dark', color='avg_salary', color_continuous_scale='Blues')
            fig_city.update_traces(hovertemplate='<b>%{y}</b><br>薪资中位数: %{x:.1f}K<extra></extra>')
            fig_city.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', xaxis_title="薪资中位数(K)",
                                   font=dict(color='#c0d2e0'), coloraxis_showscale=False, hoverlabel=HOVER_LABEL_CONFIG)
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.info("当前筛选条件下无数据")

    with st.container(border=True):
        with st.expander("查看各城市数据分析岗薪酬区间（箱线图）"):
            if not filtered_df.empty:
                fig_box = px.box(filtered_df, x='address', y='avg_salary', color='address', template='plotly_dark',
                                 color_discrete_sequence=px.colors.qualitative.Bold)
                fig_box.update_traces(hovertemplate='<b>%{x}</b><br>薪资区间: %{y:.1f}K<extra></extra>')
                fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      xaxis_title="", yaxis_title="薪资区间(K)", font=dict(color='#c0d2e0'),
                                      showlegend=False, hoverlabel=HOVER_LABEL_CONFIG)
                st.plotly_chart(fig_box, use_container_width=True)

    # 桑基图
    with st.container(border=True):
        st.subheader("🌊 经验、学历与薪资档位的流动关系")
        if not filtered_df.empty:
            sankey_df = filtered_df.dropna(subset=['experience', 'education', 'salary_level'])
            exps = sankey_df['experience'].unique().tolist()
            edus = sankey_df['education'].unique().tolist()
            sals = sankey_df['salary_level'].unique().tolist()

            all_nodes = exps + edus + sals
            node_indices = {node: i for i, node in enumerate(all_nodes)}

            link_exp_edu = sankey_df.groupby(['experience', 'education']).size().reset_index(name='count')
            link_edu_sal = sankey_df.groupby(['education', 'salary_level']).size().reset_index(name='count')

            sources = [node_indices[x] for x in link_exp_edu['experience']] + [node_indices[x] for x in
                                                                               link_edu_sal['education']]
            targets = [node_indices[x] for x in link_exp_edu['education']] + [node_indices[x] for x in
                                                                              link_edu_sal['salary_level']]
            values = link_exp_edu['count'].tolist() + link_edu_sal['count'].tolist()

            node_colors = ['#00BFFF'] * len(exps) + ['#00E5FF'] * len(edus) + ['#FF7F0E'] * len(sals)

            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15, thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color=node_colors,
                    hovertemplate='<b>%{label}</b><br>岗位总数: %{value}<extra></extra>'
                ),
                link=dict(
                    source=sources, target=targets, value=values,
                    color="rgba(0, 191, 255, 0.2)",
                    hovertemplate='<b>%{source.label} ➔ %{target.label}</b><br>流动岗位数: %{value}<extra></extra>'
                )
            )])
            fig_sankey.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff', size=12), height=500, margin=dict(t=30, b=30, l=30, r=30),
                hoverlabel=HOVER_LABEL_CONFIG
            )
            st.plotly_chart(fig_sankey, use_container_width=True)
        else:
            st.info("当前筛选条件下无数据")

    row3_col1, row3_col2, row3_col3 = st.columns([1, 1, 1.2])
    df_tags = filtered_df.assign(tag=filtered_df['tags'].str.split(',')).explode('tag')
    df_tags = df_tags[df_tags['tag'].str.strip() != '']
    df_tags['tag'] = df_tags['tag'].str.strip()

    whitelist = ['Python', 'SQL', 'Excel', 'Hive', 'Spark', 'Hadoop', 'Tableau', 'Power BI',
                 'SAS', 'SPSS', 'R语言', 'R', 'MySQL', 'Oracle', 'Linux', 'HBase', 'Flink',
                 'Pandas', 'NumPy', 'Sklearn', 'Java', 'Scala', 'Shell', 'ETL', 'Kettle',
                 '机器学习', '数据挖掘', '数据仓库', '建模', '数据可视化', '深度学习', 'BI', 'AIGC']
    whitelist_lower = [w.lower() for w in whitelist]
    df_tags = df_tags[df_tags['tag'].str.lower().isin(whitelist_lower)]


    def format_skill(skill):
        skill_lower = skill.lower()
        for w in whitelist:
            if w.lower() == skill_lower: return w
        return skill


    if not df_tags.empty:
        df_tags['tag'] = df_tags['tag'].apply(format_skill)

        with row3_col1:
            with st.container(border=True):
                st.markdown("**🔥 技能需求 TOP 10**")
                top_tags = df_tags['tag'].value_counts().head(10).reset_index()
                top_tags.columns = ['技能', '需求频次']
                fig_skills = px.bar(top_tags, x='需求频次', y='技能', orientation='h',
                                    template='plotly_dark', color='需求频次', color_continuous_scale='Purples')
                fig_skills.update_traces(hovertemplate='<b>%{y}</b><br>出现次数: %{x}<extra></extra>')
                fig_skills.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                         plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#c0d2e0'),
                                         coloraxis_showscale=False, hoverlabel=HOVER_LABEL_CONFIG)
                st.plotly_chart(fig_skills, use_container_width=True)

        with row3_col2:
            with st.container(border=True):
                st.markdown("**🕸️ 核心城市技能偏好**")
                city_counts = filtered_df['address'].value_counts()
                top_3_cities = city_counts.head(min(3, len(city_counts))).index.tolist()
                top_6_tags = df_tags['tag'].value_counts().head(6).index.tolist()
                radar_df = df_tags[(df_tags['address'].isin(top_3_cities)) & (df_tags['tag'].isin(top_6_tags))]

                if not radar_df.empty:
                    radar_pivot = radar_df.groupby(['address', 'tag']).size().unstack(fill_value=0)
                    fig_radar = go.Figure()
                    colors = ['#00E5FF', '#FF007F', '#FFD700']
                    for i, city in enumerate(top_3_cities):
                        if city in radar_pivot.index:
                            fig_radar.add_trace(go.Scatterpolar(
                                r=radar_pivot.loc[city].values, theta=radar_pivot.columns,
                                fill='toself', name=city, opacity=0.3,
                                line=dict(color=colors[i % len(colors)], width=3), fillcolor=colors[i % len(colors)]
                            ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, gridcolor='rgba(0, 191, 255, 0.2)', linecolor='#00BFFF'),
                            angularaxis=dict(gridcolor='rgba(0, 191, 255, 0.2)'), bgcolor='rgba(0,0,0,0)'),
                        showlegend=True, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=30, l=30, r=30),
                        font=dict(color='#c0d2e0', size=10), hoverlabel=HOVER_LABEL_CONFIG
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

        with row3_col3:
            with st.container(border=True):
                st.markdown("**💎 掌握哪些技能最值钱？**")
                tag_counts = df_tags['tag'].value_counts()
                common_tags = tag_counts[tag_counts >= 3].index
                if not common_tags.empty:
                    tag_salary = df_tags[df_tags['tag'].isin(common_tags)].groupby('tag')[
                        'avg_salary'].median().reset_index().sort_values(by='avg_salary', ascending=False).head(10)
                    fig_tag_sal = px.bar(tag_salary, x='avg_salary', y='tag', orientation='h',
                                         text_auto='.1f', template='plotly_dark', color='avg_salary',
                                         color_continuous_scale='Bluyl')
                    fig_tag_sal.update_traces(hovertemplate='<b>%{y}</b><br>薪资中位数: %{x:.1f}K<extra></extra>')
                    fig_tag_sal.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                              plot_bgcolor='rgba(0,0,0,0)', xaxis_title="薪资中位数(K)",
                                              font=dict(color='#c0d2e0'), coloraxis_showscale=False,
                                              hoverlabel=HOVER_LABEL_CONFIG)
                    st.plotly_chart(fig_tag_sal, use_container_width=True)

# ------------------------------------------
# TAB 3: 求职破局指南
# ------------------------------------------
with tab_conclusion:
    st.subheader("💡 写给数据分析求职者的破局指南")
    st.markdown("基于全量数据分析岗位数据的宏观洞察，为您指明职业发展方向。")
    st.markdown("<br>", unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div style="background: rgba(5,16,31,0.8); padding: 15px; border-radius: 8px; border-left: 4px solid #FF007F; text-align: center; box-shadow: inset 0 0 10px rgba(255,0,127,0.1);">
            <p style="color: #a0b8cf; margin: 0; font-size: 1rem;">职业发展黄金期</p>
            <h2 style="color: #FF007F; margin: 5px 0 0 0; font-size: 2.5rem;">3-5年</h2>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown("""
        <div style="background: rgba(5,16,31,0.8); padding: 15px; border-radius: 8px; border-left: 4px solid #00E5FF; text-align: center; box-shadow: inset 0 0 10px rgba(0,229,255,0.1);">
            <p style="color: #a0b8cf; margin: 0; font-size: 1rem;">头部城市机会集中度</p>
            <h2 style="color: #00E5FF; margin: 5px 0 0 0; font-size: 2.5rem;">> 50%</h2>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown("""
        <div style="background: rgba(5,16,31,0.8); padding: 15px; border-radius: 8px; border-left: 4px solid #FFD700; text-align: center; box-shadow: inset 0 0 10px rgba(255,215,0,0.1);">
            <p style="color: #a0b8cf; margin: 0; font-size: 1rem;">最高学历薪资溢价</p>
            <h2 style="color: #FFD700; margin: 5px 0 0 0; font-size: 2.5rem;">46.7K</h2>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div style="background: rgba(5,16,31,0.8); padding: 15px; border-radius: 8px; border-left: 4px solid #00BFFF; text-align: center; box-shadow: inset 0 0 10px rgba(0,191,255,0.1);">
            <p style="color: #a0b8cf; margin: 0; font-size: 1rem;">核心高溢价技能</p>
            <h2 style="color: #00BFFF; margin: 5px 0 0 0; font-size: 2.5rem;">Python</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ins1, col_ins2 = st.columns(2)

    with col_ins1:
        st.markdown("""
        <div class="insight-box" style="margin-bottom: 15px;">
        <h4>🔍 行业宏观现状洞察</h4>
        <ul>
            <li style="margin-bottom: 10px;"><b>薪资结构：</b> 8K以下基础岗位存量最大，40K以上高薪岗位稀缺，行业薪资呈典型金字塔分布。</li>
            <li style="margin-bottom: 10px;"><b>地域集聚：</b> 京沪深杭占据绝对主导地位，呈现显著的一线城市溢价效应。</li>
            <li style="margin-bottom: 10px;"><b>经验壁垒：</b> 3-5年为薪资跃升的核心拐点，资深岗与应届生薪资差距呈倍数级放大。</li>
            <li style="margin-bottom: 10px;"><b>技能生态：</b> 复合型人才更具议价权，掌握大数据生态（Hive/Spark）及高级建模工具是提升核心竞争力的关键。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("**📊 薪资档位金字塔结构**")
            funnel_data = df['salary_level'].value_counts().reset_index()
            funnel_data.columns = ['薪资档位', '人数']
            funnel_data['薪资档位'] = pd.Categorical(funnel_data['薪资档位'],
                                                     categories=['0-8K', '8-15K', '15-25K', '25-40K', '>40K'],
                                                     ordered=True)
            funnel_data = funnel_data.sort_values('薪资档位', ascending=True)

            fig_funnel = px.funnel(
                funnel_data, x='人数', y='薪资档位',
                color='薪资档位',
                color_discrete_sequence=px.colors.sequential.Plasma_r,
                template='plotly_dark'
            )
            fig_funnel.update_traces(textinfo='value', hovertemplate='<b>%{y}</b><br>岗位数: %{x}<extra></extra>')
            fig_funnel.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=10, b=10, l=0, r=0), height=280, showlegend=False,
                font=dict(color='#c0d2e0'), hoverlabel=HOVER_LABEL_CONFIG
            )
            st.plotly_chart(fig_funnel, use_container_width=True)

    with col_ins2:
        st.markdown("""
        <div class="insight-box" style="margin-bottom: 15px;">
        <h4>🎯 职业战略规划建议</h4>
        <ul>
            <li style="margin-bottom: 10px;"><b>技术栈纵深：</b> 突破基础工具（Excel/SQL）局限，向底层大数据架构与上层BI可视化双向拓展。</li>
            <li style="margin-bottom: 10px;"><b>周期管理：</b> 牢牢把握3-5年职业发展窗口期，完成从“初级取数”向“高级业务分析/策略指导”的转型。</li>
            <li style="margin-bottom: 10px;"><b>区位选择：</b> 追求职业高度优选北深杭；注重综合生活质量与性价比，可聚焦蓉汉等新一线城市。</li>
            <li style="margin-bottom: 10px;"><b>学历提升：</b> 面对日益内卷的竞争环境，攻读硕博学位是突破职业天花板的有效战略投资。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("**📈 职业周期与薪资跃升拐点**")
            exp_order = ['在校/应届', '1年以内', '1-3年', '3-5年', '5-10年']
            temp_df_3 = df.copy()
            temp_df_3['experience'] = pd.Categorical(temp_df_3['experience'], categories=exp_order, ordered=True)
            exp_salary_plot = temp_df_3.groupby('experience')['avg_salary'].median().reset_index().dropna()

            fig_area = px.area(exp_salary_plot, x='experience', y='avg_salary', color_discrete_sequence=['#00E5FF'])
            fig_area.update_traces(
                line=dict(width=3, color='#00E5FF'), fillcolor='rgba(0, 229, 255, 0.15)',
                mode='lines+markers+text', text=exp_salary_plot['avg_salary'].apply(lambda x: f"{x:.1f}K"),
                textposition="top center", textfont=dict(color='#FFD700', size=12),
                hovertemplate='<b>%{x}</b><br>薪资中位数: %{y:.1f}K<extra></extra>'
            )
            fig_area.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(t=30, b=10, l=0, r=0), height=280,
                xaxis=dict(title="", gridcolor='rgba(0, 191, 255, 0.1)'),
                yaxis=dict(title="薪资中位数(K)", gridcolor='rgba(0, 191, 255, 0.1)'),
                font=dict(color='#c0d2e0'), hoverlabel=HOVER_LABEL_CONFIG
            )
            st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 求职策略建议")
    act_col1, act_col2, act_col3 = st.columns(3)
    with act_col1:
        st.markdown("""
        <div style="background: rgba(0, 191, 255, 0.05); border: 1px solid rgba(0, 191, 255, 0.2); padding: 20px; border-radius: 8px; height: 100%;">
            <h4 style="color: #00E5FF; margin-top: 0;">📝 简历包装：强调量化成果</h4>
            <p style="color: #c0d2e0; font-size: 0.95rem; line-height: 1.6;">
                避免空泛罗列技能。建议采用 STAR 法则，以数据量化产出。<br>
                <b>示例：</b>“利用 Python 处理百万级数据，通过 Tableau 构建业务看板，辅助定位转化漏斗痛点，驱动核心指标提升 X%。”
            </p>
        </div>
        """, unsafe_allow_html=True)
    with act_col2:
        st.markdown("""
        <div style="background: rgba(255, 0, 127, 0.05); border: 1px solid rgba(255, 0, 127, 0.2); padding: 20px; border-radius: 8px; height: 100%;">
            <h4 style="color: #FF007F; margin-top: 0;">💼 项目甄选：聚焦商业场景</h4>
            <p style="color: #c0d2e0; font-size: 0.95rem; line-height: 1.6;">
                规避同质化极高的基础教学案例（如泰坦尼克号预测）。<br>
                应重点展示<b>贴合实际业务流的实战项目</b>：如电商全链路分析、用户流失预警模型构建、A/B 测试设计与评估等。
            </p>
        </div>
        """, unsafe_allow_html=True)
    with act_col3:
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.05); border: 1px solid rgba(255, 215, 0, 0.2); padding: 20px; border-radius: 8px; height: 100%;">
            <h4 style="color: #FFD700; margin-top: 0;">🎯 面试表达：强化业务闭环</h4>
            <p style="color: #c0d2e0; font-size: 0.95rem; line-height: 1.6;">
                弱化纯技术工具的讨论，着重阐述<b>“如何通过数据驱动业务决策”</b>。<br>
                展现具备以业务目标为导向、形成“发现问题-提出假设-数据验证-策略落地”的闭环思维。
            </p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: 智能薪资预测
# ------------------------------------------
with tab_predict:
    st.subheader("🤖 数据分析岗薪资预测模型")
    st.markdown(f"基于 {len(df)} 条真实数据训练模型。输入你的求职画像，获取基于市场大数据的预期月薪参考。")
    st.divider()


    @st.cache_resource
    def train_model(data):
        if len(data) < 10: return None, None
        X = data[['education', 'experience', 'address', 'labels']]
        y = data['avg_salary']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['education', 'experience', 'address', 'labels'])])
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))])
        model.fit(X, y)

        ohe = model.named_steps['preprocessor'].named_transformers_['cat']
        feature_names = ohe.get_feature_names_out(['education', 'experience', 'address', 'labels'])
        importances = model.named_steps['regressor'].feature_importances_

        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

        def map_to_original(feat):
            for orig in ['education', 'experience', 'address', 'labels']:
                if feat.startswith(orig + '_'): return orig
            return feat

        imp_df['Original'] = imp_df['Feature'].apply(map_to_original)
        grouped_imp = imp_df.groupby('Original')['Importance'].sum().reset_index()
        grouped_imp['Original'] = grouped_imp['Original'].map(
            {'education': '学历', 'experience': '经验', 'address': '城市', 'labels': '行业'})
        grouped_imp = grouped_imp.sort_values(by='Importance', ascending=True)
        return model, grouped_imp


    model, feature_imp = train_model(df)

    if model:
        col_input, col_result = st.columns([1, 1.2])

        with col_input:
            with st.container(border=True):
                st.markdown("### ⚙️ 输入求职期望")
                with st.form("predict_form"):
                    p_col1, p_col2 = st.columns(2)
                    with p_col1:
                        pred_edu = st.selectbox("选择学历", df['education'].dropna().unique())
                        pred_city = st.selectbox("选择城市", df['address'].dropna().unique())
                    with p_col2:
                        pred_exp = st.selectbox("选择经验", df['experience'].dropna().unique())
                        pred_ind = st.selectbox("选择行业", df['labels'].dropna().unique())

                    st.markdown("<br>", unsafe_allow_html=True)
                    submit = st.form_submit_button("⚡ 开始测算薪资")

            with st.container(border=True):
                st.markdown("**📊 市场薪酬决定因素权重**")
                fig_imp = px.bar(feature_imp, x='Importance', y='Original', orientation='h',
                                 text_auto='.1%', template='plotly_dark', color='Importance',
                                 color_continuous_scale='Teal')
                fig_imp.update_traces(hovertemplate='<b>%{y}</b><br>影响权重: %{x:.1%}<extra></extra>')
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="", yaxis_title="", margin=dict(t=10, b=10, l=0, r=30),
                    height=220, coloraxis_showscale=False,
                    xaxis=dict(tickformat='.0%', gridcolor='rgba(0, 191, 255, 0.1)'), hoverlabel=HOVER_LABEL_CONFIG
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        with col_result:
            if submit:
                input_df = pd.DataFrame(
                    {'education': [pred_edu], 'experience': [pred_exp], 'address': [pred_city], 'labels': [pred_ind]})
                predicted_salary = model.predict(input_df)[0]

                filtered_bench = df[
                    (df['education'] == pred_edu) & (df['experience'] == pred_exp) & (df['address'] == pred_city)]

                with st.container(border=True):
                    st.markdown(f"""
                    <div style="background-color: rgba(0, 229, 255, 0.05); padding: 30px; border-radius: 8px; text-align: center; border: 1px solid #00BFFF; box-shadow: inset 0 0 20px rgba(0,191,255,0.1);">
                        <h4 style="color: #c0d2e0; margin-bottom: 10px;">大数据模型测算合理月薪</h4>
                        <h1 style="color: #00E5FF; font-size: 4.5rem; margin: 0; text-shadow: 0 0 20px rgba(0,229,255,0.6);">{predicted_salary:.1f} <span style="font-size: 2rem;">K</span></h1>
                        <p style="color: #7f98b0; margin-top: 15px;">供求职谈薪参考</p>
                    </div>
                    """, unsafe_allow_html=True)

                MIN_SAMPLE_SIZE = 10
                if len(filtered_bench) >= MIN_SAMPLE_SIZE:
                    better_ratio = (filtered_bench['avg_salary'] < predicted_salary).mean() * 100
                    avg_bench = filtered_bench['avg_salary'].median()

                    with st.container(border=True):
                        st.markdown("**🎯 同维度薪资竞争力分析**")
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number", value=better_ratio,
                                number={'suffix': "%", 'font': {'color': '#00E5FF', 'size': 36}},
                                title={'text': "击败同业者比例", 'font': {'color': '#c0d2e0', 'size': 14}},
                                gauge={
                                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#7f98b0"},
                                    'bar': {'color': "#00E5FF"}, 'bgcolor': "rgba(0,0,0,0)",
                                    'borderwidth': 2, 'bordercolor': "rgba(0, 191, 255, 0.3)",
                                    'steps': [{'range': [0, 50], 'color': "rgba(255, 0, 127, 0.2)"},
                                              {'range': [50, 100], 'color': "rgba(0, 229, 255, 0.2)"}]
                                }
                            ))
                            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                                    margin=dict(t=40, b=10, l=20, r=20), height=180)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        with c2:
                            st.markdown(f"""
                            <div style="padding-top: 20px;">
                                <p style="color: #a0b8cf; font-size: 0.9rem; margin-bottom: 5px;">市场薪资中位数</p>
                                <h3 style="color: #FFD700; margin-top: 0;">{avg_bench:.1f} K</h3>
                                <p style="color: #a0b8cf; font-size: 0.9rem; margin-bottom: 5px; margin-top: 15px;">对比样本池大小</p>
                                <h3 style="color: #e2e8f0; margin-top: 0;">{len(filtered_bench)} <span style="font-size: 1rem; color: #7f98b0;">个岗位</span></h3>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    with st.container(border=True):
                        st.markdown("**🎯 同维度薪资竞争力分析**")
                        st.info(f"""
                        💡 **市场样本不足提示**

                        当前您选择的画像在数据库中仅匹配到 **{len(filtered_bench)}** 个真实岗位。

                        由于样本量过小，计算“市场薪资中位数”和“击败比例”缺乏统计学意义。建议您参考上方模型测算结果，或切换为更通用的条件以获取对比数据。
                        """)
            else:
                with st.container(border=True):
                    st.markdown("""
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 400px; opacity: 0.6;">
                        <h1 style="font-size: 5rem; margin: 0;">🤖</h1>
                        <h3 style="color: #00E5FF; margin-top: 20px;">等待测算</h3>
                        <p style="color: #a0b8cf; text-align: center;">请在左侧输入你的求职期望画像，点击“开始测算”<br>获取基于全量大数据训练的薪资预测结果与市场竞争力分析。</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("数据集过少，无法训练模型。")
