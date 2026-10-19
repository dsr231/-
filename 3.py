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
# 1. 页面基本配置
# ==========================================
st.set_page_config(page_title="数据分析岗薪酬洞察", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #0b1426; color: #e2e8f0; }
    [data-testid="stMetricValue"] { color: #00FFAA; font-size: 1.8rem; }
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #00FFAA !important; color: #0b1426 !important; font-weight: bold; width: 100%;
    }
    .insight-box {
        background-color: #111c33; padding: 20px; border-radius: 10px; 
        border-left: 5px solid #00FFAA; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { color: #f8fafc !important; }
    /* 美化标签页 */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #1e293b; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre; background-color: transparent; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #111c33; color: #00FFAA !important; border-bottom: 3px solid #00FFAA; }
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
        if '本科' in req: edu = '本科'
        elif '硕士' in req: edu = '硕士'
        elif '博士' in req: edu = '博士'
        elif '大专' in req: edu = '大专'
        elif '学历不限' in req or '不限' in req: edu = '学历不限'

        exp = '经验不限'
        if '应届' in req or '在校' in req: exp = '在校/应届'
        elif '1年以内' in req: exp = '1年以内'
        elif '1-3年' in req: exp = '1-3年'
        elif '3-5年' in req: exp = '3-5年'
        elif '5-10年' in req: exp = '5-10年'
        elif '10年以上' in req: exp = '10年以上'
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

# ==========================================
# 3. 主页面与 Tab 划分
# ==========================================
st.title("📊 BOSS直聘 · 数据分析岗位招聘与薪酬洞察大屏")

tab_overview, tab_interact, tab_conclusion, tab_predict = st.tabs([
    "📈 行业大盘概览",
    "🎯 多维薪酬透视",
    "💡 求职破局指南",
    "🤖 智能薪资预测"
])

# ------------------------------------------
# TAB 1: 行业大盘概览 (全量数据，不筛选)
# ------------------------------------------
with tab_overview:
    st.markdown(f"**基于全量 {len(df)} 条数据分析岗位数据生成的行业大盘（不受筛选影响）**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("有效岗位数", f"{len(df)} 个")
    col2.metric("薪资中位数", f"{df['avg_salary'].median():.1f} K")
    col3.metric("平均薪资", f"{df['avg_salary'].mean():.1f} K")
    col4.metric("覆盖行业数", f"{df['labels'].nunique()} 个")
    st.divider()

    row1_col1, row1_col2 = st.columns([1, 1.2])
    with row1_col1:
        st.subheader("📊 数据分析岗薪资档位分布")
        salary_counts = df['salary_level'].value_counts().reset_index()
        salary_counts.columns = ['薪资档位', '岗位数量']
        salary_counts['薪资档位'] = pd.Categorical(salary_counts['薪资档位'],
                                                   categories=['0-8K', '8-15K', '15-25K', '25-40K', '>40K'],
                                                   ordered=True)
        salary_counts = salary_counts.sort_values('薪资档位', ascending=True)
        fig_bar = px.bar(salary_counts, x='岗位数量', y='薪资档位', orientation='h', text='岗位数量',
                         template='plotly_dark', color='岗位数量', color_continuous_scale='Plasma')
        fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                              margin=dict(t=30, b=0, l=0, r=0), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with row1_col2:
        st.subheader("📈 全行业薪资分布直方图与箱线图")
        fig_dist = px.histogram(df, x='avg_salary', nbins=40, marginal='box', color_discrete_sequence=['#00FFAA'],
                                template='plotly_dark')
        fig_dist.update_traces(hovertemplate='薪资: %{x:.1f}K<br>岗位数: %{y}<extra></extra>',
                               selector=dict(type='histogram'))
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="平均月薪 (K)",
                               yaxis_title="岗位数量", margin=dict(t=30, b=0, l=0, r=0), font=dict(color='#e2e8f0'),
                               xaxis=dict(gridcolor='#1e293b'), yaxis=dict(gridcolor='#1e293b'))
        st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("🗺️ 全国数据分析岗城市热度与薪酬地图")
    city_map_df = df.dropna(subset=['lng', 'lat']).groupby(['address', 'lng', 'lat']).agg(
        岗位数量=('avg_salary', 'count'), 平均薪资=('avg_salary', 'median')).reset_index()
    if not city_map_df.empty:
        fig_map = px.scatter_geo(city_map_df, lat='lat', lon='lng', size='岗位数量', color='平均薪资',
                                 hover_name='address', text='address',
                                 color_continuous_scale=px.colors.sequential.Plasma, size_max=35,
                                 template='plotly_dark')
        fig_map.update_traces(
            hovertemplate='<b>%{text}</b><br>岗位数量: %{marker.size}<br>平均薪资: %{marker.color:.1f}K<extra></extra>')
        fig_map.update_geos(visible=False, resolution=50, showcountries=True, countrycolor="#334155", showland=True,
                            landcolor="#1e293b", showocean=True, oceancolor="#0b1426", lataxis_range=[15, 55],
                            lonaxis_range=[70, 140])
        fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(t=0, b=0, l=0, r=0), height=450, font=dict(color='#e2e8f0'),
                              coloraxis_colorbar=dict(title="平均薪资(K)"))
        st.plotly_chart(fig_map, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.subheader("🏢 吸纳数据分析岗的行业薪酬 TOP 10")
        ind_salary = df.groupby('labels')['avg_salary'].median().reset_index().sort_values(by='avg_salary',
                                                                                           ascending=False).head(10)
        fig_ind = px.bar(ind_salary, x='avg_salary', y='labels', orientation='h', text_auto='.1f',
                         template='plotly_dark', color='avg_salary', color_continuous_scale='Teal')
        fig_ind.update_traces(hovertemplate='<b>%{y}</b><br>薪资中位数: %{x:.1f}K<extra></extra>')
        fig_ind.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)', xaxis_title="薪资中位数(K)", font=dict(color='#e2e8f0'))
        st.plotly_chart(fig_ind, use_container_width=True)

    with row2_col2:
        st.subheader("⏳ 数据分析师经验与薪酬成长曲线")
        exp_order = ['经验不限', '在校/应届', '1年以内', '1-3年', '3-5年', '5-10年', '10年以上']
        df['experience'] = pd.Categorical(df['experience'], categories=exp_order, ordered=True)
        exp_salary = df.groupby('experience')['avg_salary'].median().reset_index()
        fig_exp = px.line(exp_salary, x='experience', y='avg_salary', markers=True, template='plotly_dark',
                          color_discrete_sequence=['#00FFAA'])
        fig_exp.update_traces(hovertemplate='<b>%{x}</b><br>薪资中位数: %{y:.1f}K<extra></extra>', line=dict(width=3),
                              marker=dict(size=10))
        fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="薪资中位数(K)",
                              xaxis_title="", font=dict(color='#e2e8f0'), xaxis=dict(gridcolor='#1e293b'),
                              yaxis=dict(gridcolor='#1e293b'))
        st.plotly_chart(fig_exp, use_container_width=True)

# ------------------------------------------
# TAB 2: 多维薪酬透视 (带局部筛选)
# ------------------------------------------
with tab_interact:
    st.markdown("### 🎯 锁定目标画像")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        city_list = sorted(df['address'].unique().tolist())
        selected_cities = st.multiselect("选择目标城市", city_list, default=city_list)
    with col_f2:
        edu_list = df['education'].unique().tolist()
        selected_edus = st.multiselect("选择学历要求", edu_list, default=edu_list)

    # 生成筛选后的数据集
    filtered_df = df[(df['address'].isin(selected_cities)) & (df['education'].isin(selected_edus))]

    st.markdown(f"当前筛选有效岗位样本：**{len(filtered_df)}** 条")
    st.divider()

    st.subheader("💰 目标画像下的薪酬对比")
    row_sal1, row_sal2 = st.columns(2)

    with row_sal1:
        st.markdown("**🎓 数据分析岗学历溢价对比**")
        if not filtered_df.empty:
            edu_salary = filtered_df.groupby('education')['avg_salary'].median().reset_index().sort_values(
                by='avg_salary', ascending=False)
            fig_edu = px.bar(edu_salary, x='education', y='avg_salary', text_auto='.1f',
                             template='plotly_dark', color='avg_salary', color_continuous_scale='Reds')
            fig_edu.update_traces(hovertemplate='<b>%{x}</b><br>薪资中位数: %{y:.1f}K<extra></extra>')
            fig_edu.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  xaxis_title="", yaxis_title="薪资中位数(K)", font=dict(color='#e2e8f0'),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_edu, use_container_width=True)
        else:
            st.info("当前筛选条件下无数据")

    with row_sal2:
        st.markdown("**🏙️ 目标城市薪酬排行 TOP 10**")
        if not filtered_df.empty:
            city_salary = filtered_df.groupby('address')['avg_salary'].median().reset_index().sort_values(
                by='avg_salary', ascending=False).head(10)
            fig_city = px.bar(city_salary, x='avg_salary', y='address', orientation='h', text_auto='.1f',
                              template='plotly_dark', color='avg_salary', color_continuous_scale='Oranges')
            fig_city.update_traces(hovertemplate='<b>%{y}</b><br>薪资中位数: %{x:.1f}K<extra></extra>')
            fig_city.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='rgba(0,0,0,0)', xaxis_title="薪资中位数(K)",
                                   font=dict(color='#e2e8f0'),
                                   coloraxis_showscale=False)
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.info("当前筛选条件下无数据")

    with st.expander("查看各城市数据分析岗薪酬区间（箱线图）"):
        if not filtered_df.empty:
            fig_box = px.box(filtered_df, x='address', y='avg_salary', color='address', template='plotly_dark')
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  xaxis_title="", yaxis_title="薪资区间(K)", font=dict(color='#e2e8f0'),
                                  showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    st.subheader("🛠️ 目标岗位技能图谱与溢价")
    row3_col1, row3_col2, row3_col3 = st.columns([1, 1, 1.2])

    df_tags = filtered_df.assign(tag=filtered_df['tags'].str.split(',')).explode('tag')
    df_tags = df_tags[df_tags['tag'].str.strip() != '']
    df_tags['tag'] = df_tags['tag'].str.strip()

    whitelist = [
        'Python', 'SQL', 'Excel', 'Hive', 'Spark', 'Hadoop', 'Tableau', 'Power BI',
        'SAS', 'SPSS', 'R语言', 'R', 'MySQL', 'Oracle', 'Linux', 'HBase', 'Flink',
        'Pandas', 'NumPy', 'Sklearn', 'Java', 'Scala', 'Shell', 'ETL', 'Kettle',
        '机器学习', '数据挖掘', '数据仓库', '建模', '数据可视化', '深度学习', 'BI', 'AIGC'
    ]
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
            st.markdown("**🔥 数据分析师核心技能需求 TOP 10**")
            top_tags = df_tags['tag'].value_counts().head(10).reset_index()
            top_tags.columns = ['技能', '需求频次']
            fig_skills = px.bar(top_tags, x='需求频次', y='技能', orientation='h',
                                template='plotly_dark', color='需求频次', color_continuous_scale='Plasma')
            fig_skills.update_traces(hovertemplate='<b>%{y}</b><br>出现次数: %{x}<extra></extra>')
            fig_skills.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'),
                                     coloraxis_showscale=False)
            st.plotly_chart(fig_skills, use_container_width=True)

        with row3_col2:
            st.markdown("**🕸️ 核心城市技能需求偏好雷达**")
            top_3_cities = filtered_df['address'].value_counts().head(3).index.tolist()
            top_6_tags = df_tags['tag'].value_counts().head(6).index.tolist()
            radar_df = df_tags[(df_tags['address'].isin(top_3_cities)) & (df_tags['tag'].isin(top_6_tags))]

            if not radar_df.empty:
                radar_pivot = radar_df.groupby(['address', 'tag']).size().unstack(fill_value=0)
                fig_radar = go.Figure()
                colors = ['#00FFAA', '#FF4B4B', '#1F77B4']
                for i, city in enumerate(top_3_cities):
                    if city in radar_pivot.index:
                        fig_radar.add_trace(go.Scatterpolar(
                            r=radar_pivot.loc[city].values, theta=radar_pivot.columns,
                            fill='toself', name=city, opacity=0.2,
                            line=dict(color=colors[i % len(colors)], width=3), fillcolor=colors[i % len(colors)]
                        ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, gridcolor='#334155'), bgcolor='rgba(0,0,0,0)'),
                    showlegend=True, template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=30, b=30, l=30, r=30), font=dict(color='#e2e8f0', size=10)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("数据不足以生成雷达图")

        with row3_col3:
            st.markdown("**💎 掌握哪些技能最值钱？(高薪技能 TOP 10)**")
            tag_counts = df_tags['tag'].value_counts()
            common_tags = tag_counts[tag_counts >= 3].index
            if not common_tags.empty:
                tag_salary = df_tags[df_tags['tag'].isin(common_tags)].groupby('tag')[
                    'avg_salary'].median().reset_index().sort_values(by='avg_salary', ascending=False).head(10)
                fig_tag_sal = px.bar(tag_salary, x='avg_salary', y='tag', orientation='h',
                                     text_auto='.1f', template='plotly_dark', color='avg_salary',
                                     color_continuous_scale='Purples')
                fig_tag_sal.update_traces(hovertemplate='<b>%{y}</b><br>薪资中位数: %{x:.1f}K<extra></extra>')
                fig_tag_sal.update_layout(yaxis={'categoryorder': 'total ascending'}, paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)', xaxis_title="薪资中位数(K)",
                                          font=dict(color='#e2e8f0'),
                                          coloraxis_showscale=False)
                st.plotly_chart(fig_tag_sal, use_container_width=True)
            else:
                st.info("样本量不足，无法计算技能溢价")
    else:
        st.warning("当前筛选条件下的样本不足以进行技能分析。")

# ------------------------------------------
# TAB 3: 求职破局指南 (静态文本)
# ------------------------------------------
with tab_conclusion:
    st.subheader("💡 写给数据分析求职者的破局指南")
    st.markdown("基于全量数据分析岗位数据的宏观洞察，为你指明求职方向。")
    st.divider()

    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.markdown("""
        <div class="insight-box" style="height: 100%;">
        <h4>🔍 数据分析行业现状洞察</h4>
        <ul>
            <li style="margin-bottom: 10px;"><b>薪资分布：</b> 8K以下岗位占据大盘，而 40K 以上高薪属于极少数。这说明数据分析行业呈现金字塔结构。</li>
            <li style="margin-bottom: 10px;"><b>地域聚集：</b> 北京薪资遥遥领先，杭州、上海、深圳紧随其后，呈现明显的一线城市溢价。</li>
            <li style="margin-bottom: 10px;"><b>经验分水岭：</b> 3-5年是薪资跃升的黄金节点；5-10年经验的薪资是应届生/在校生的数倍。</li>
            <li style="margin-bottom: 10px;"><b>学历溢价：</b> 博士学历薪资断层领先，硕士显著优于本科。</li>
            <li style="margin-bottom: 10px;"><b>高薪技能组合：</b> 掌握大数据生态（Hive, Spark）和高级统计建模工具（SAS, Python）能显著提升薪资议价能力。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ins2:
        st.markdown("""
        <div class="insight-box" style="height: 100%;">
        <h4>🎯 数据分析师职业规划建议</h4>
        <ul>
            <li style="margin-bottom: 10px;"><b>技术栈拓展：</b> 不要局限于基础 Excel/SQL。向大数据（Hadoop、Spark）和 BI 可视化（Tableau、PowerBI）延伸。</li>
            <li style="margin-bottom: 10px;"><b>职业周期：</b> 重视 3-5 年的工作经验沉淀期，这是突破初级岗位、冲刺高薪的黄金阶段。</li>
            <li style="margin-bottom: 10px;"><b>地域选择：</b> 若追求薪资天花板，首选北上深杭；若注重生活性价比，可考虑成都、武汉等新一线城市。</li>
            <li style="margin-bottom: 10px;"><b>学历提升：</b> 在竞争激烈的数据分析赛道，攻读硕博学位是突破天花板的有效途径。</li>
            <li style="margin-bottom: 10px;"><b>行业选择：</b> 优先切入互联网、O2O、电子商务以及数字化程度高的金融行业。</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: 智能薪资预测
# ------------------------------------------
with tab_predict:
    st.subheader("🤖 数据分析岗薪资预测模型")
    st.markdown(f"基于 {len(df)} 条 BOSS直聘数据分析岗真实数据训练模型。输入你的求职画像，获取基于市场大数据的预期月薪参考。")
    st.divider()

    @st.cache_resource
    def train_model(data):
        if len(data) < 10: return None
        X = data[['education', 'experience', 'address', 'labels']]
        y = data['avg_salary']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['education', 'experience', 'address', 'labels'])
            ])
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
        model.fit(X, y)
        return model

    model = train_model(df)

    if model:
        col_input, col_result = st.columns([2, 1])
        with col_input:
            with st.form("predict_form"):
                st.markdown("<h4>⚙️ 输入你的求职期望画像</h4>", unsafe_allow_html=True)
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    pred_edu = st.selectbox("选择学历", df['education'].dropna().unique())
                    pred_city = st.selectbox("选择城市", df['address'].dropna().unique())
                with p_col2:
                    pred_exp = st.selectbox("选择经验", df['experience'].dropna().unique())
                    pred_ind = st.selectbox("选择行业", df['labels'].dropna().unique())

                submit = st.form_submit_button("⚡ 开始测算")

        with col_result:
            if submit:
                input_df = pd.DataFrame({
                    'education': [pred_edu],
                    'experience': [pred_exp],
                    'address': [pred_city],
                    'labels': [pred_ind]
                })
                predicted_salary = model.predict(input_df)[0]
                st.markdown(f"""
                <div style="background-color: #111c33; padding: 30px; border-radius: 10px; text-align: center; border: 2px solid #00FFAA; margin-top: 20px;">
                    <h4 style="color: #e2e8f0; margin-bottom: 15px;">模型测算合理月薪</h4>
                    <h1 style="color: #00FFAA; font-size: 4rem; margin: 0;">{predicted_salary:.1f} K</h1>
                    <p style="color: #94a3b8; margin-top: 15px;">供求职谈薪参考</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("数据集过少，无法训练模型。")