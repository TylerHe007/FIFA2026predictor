import streamlit as st
import torch
import numpy as np
import scipy.stats as stats
import pandas as pd
from predict import AdvancedSymmetricPredictor, ProfessionalSymmetricDCNet

st.set_page_config(
    page_title="FIFA 2026 预测模型",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main { background-color: #fafafa; color: #1d1d1f; }
    h1, h2, h3 { font-weight: 700 !important; color: #1d1d1f !important; letter-spacing: -0.02em; }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0f4c81 0%, #1d1d1f 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        height: 3.2em;
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(15,76,129,0.15);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15,76,129,0.25);
    }

    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #eeeeee;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-label { font-size: 13px; color: #86868b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #1d1d1f; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }

    .score-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #ffffff;
        padding: 14px 24px;
        border-radius: 12px;
        margin-bottom: 10px;
        border: 1px solid #f0f0f0;
        transition: all 0.2s ease;
    }
    .score-row:hover { background-color: #f9f9fb; transform: scale(1.01); }
    .score-badge { background-color: #1d1d1f; color: white; font-weight: 700; padding: 6px 16px; border-radius: 8px; font-size: 15px; min-width: 65px; text-align: center; }
    .score-prob { font-weight: 600; color: #0f4c81; font-size: 15px; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return AdvancedSymmetricPredictor('advanced_dixon_coles_model.pth')


try:
    predictor = load_predictor()
    selectable_teams = sorted(list(predictor.team_to_id.keys()))
except Exception as e:
    st.error(f"模型文件未就绪")
    st.stop()

# 3. 极简交互面板设计
st.title("⚽ 2026 FIFA 世界杯预测模型")
st.caption("深度学习神经网络模型，基于4.5万场国际男子足球比赛比分历史数据")
st.write("")

# 对阵双方精致微调
col_a, col_b = st.columns(2)
with col_a:
    home_team = st.selectbox("队伍 A (主队方位)", options=selectable_teams,
                             index=selectable_teams.index("Mexico") if "Mexico" in selectable_teams else 0)
with col_b:
    away_team = st.selectbox("队伍 B (客队方位)", options=selectable_teams,
                             index=selectable_teams.index("South Africa") if "South Africa" in selectable_teams else 0)

# 环境设置
venue_type = st.radio(
    "球场性质：",
    options=["中立球场模式", f"非中立球场模式 (主队 [{home_team}] 享有主场优势)"],
    index=0
)
is_neutral = True if "中立球场" in venue_type else False

st.write("")
start_predict = st.button("推理引擎 ")
st.write("")


if start_predict:
    if home_team == away_team:
        st.warning("相同队伍无法建立对阵拓扑，请重新选择对阵双方。")
    else:
        with st.spinner("神经网络正在计算攻防期望矩阵..."):

            # 提取两队底层的实力变量与近况元数据
            h_meta = predictor.latest_team_meta[home_team]
            a_meta = predictor.latest_team_meta[away_team]

            h_id = torch.tensor([predictor.team_to_id[home_team]], dtype=torch.long)
            a_id = torch.tensor([predictor.team_to_id[away_team]], dtype=torch.long)
            n_feat = torch.tensor([[1.0 if is_neutral else 0.0]], dtype=torch.float32)
            h_elo_t = torch.tensor([[h_meta['elo']]], dtype=torch.float32)
            a_elo_t = torch.tensor([[a_meta['elo']]], dtype=torch.float32)
            h_form_t = torch.tensor([h_meta['form']], dtype=torch.float32)
            a_form_t = torch.tensor([a_meta['form']], dtype=torch.float32)

            # 前向传播推理
            with torch.no_grad():
                pred = predictor.model(h_id, a_id, n_feat, h_elo_t, a_elo_t, h_form_t, a_form_t)
                lambda_h = torch.exp(pred[0, 0]).item()
                lambda_a = torch.exp(pred[0, 1]).item()
                rho = torch.tanh(pred[0, 2]).item() * 0.10

            # 构建 6x6 泊松关联概率空间
            max_goals = 5
            prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
            for x in range(max_goals + 1):
                for y in range(max_goals + 1):
                    p_x = stats.poisson.pmf(x, lambda_h)
                    p_y = stats.poisson.pmf(y, lambda_a)
                    tau = 1.0
                    if x == 0 and y == 0:
                        tau = 1.0 - lambda_h * lambda_a * rho
                    elif x == 1 and y == 0:
                        tau = 1.0 + lambda_a * rho
                    elif x == 0 and y == 1:
                        tau = 1.0 + lambda_h * rho
                    elif x == 1 and y == 1:
                        tau = 1.0 - rho
                    prob_matrix[x, y] = max(0.0, tau * p_x * p_y)

            prob_matrix /= np.sum(prob_matrix)

            # 大盘结局概率切片
            home_win = np.sum(np.tril(prob_matrix, -1))
            draw = np.sum(np.diag(prob_matrix))
            away_win = np.sum(np.triu(prob_matrix, 1))

            # 爆冷风险度量
            safe_probs = prob_matrix[prob_matrix > 1e-9]
            shannon_entropy = -np.sum(safe_probs * np.log2(safe_probs))


            st.write("### 📊 胜平负预测推理")

            hw_pct = f"{home_win * 100:.1f}%"
            dr_pct = f"{draw * 100:.1f}%"
            aw_pct = f"{away_win * 100:.1f}%"

            st.markdown(f"""
                <div style="display: flex; width: 100%; border-radius: 12px; overflow: hidden; height: 36px; margin: 15px 0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="width: {home_win * 100}%; background-color: #0f4c81; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 13px;">{home_team} 胜 {hw_pct}</div>
                    <div style="width: {draw * 100}%; background-color: #8e9aa6; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 13px;">平局 {dr_pct}</div>
                    <div style="width: {away_win * 100}%; background-color: #c8102e; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 13px;">{away_team} 胜 {aw_pct}</div>
                </div>
            """, unsafe_allow_html=True)


            st.write("")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">期望进球率 (Lambda)</div>
                    <div class="metric-value" style="color:#0f4c81;">{lambda_h:.2f} : {lambda_a:.2f}</div>
                </div>""", unsafe_allow_html=True)
            with meta_col2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">全局基准实力 (Elo)</div>
                    <div class="metric-value" style="font-size:22px;">{h_meta['elo']:.0f} vs {a_meta['elo']:.0f}</div>
                </div>""", unsafe_allow_html=True)
            with meta_col3:
                # 依据香农熵大小动态赋予风险评估标签
                risk_level = "低爆冷风险" if shannon_entropy < 3.8 else (
                    "中等爆冷风险" if shannon_entropy < 4.2 else "高爆冷风险")
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">爆冷风险指数 (Entropy)</div>
                    <div class="metric-value" style="font-size:20px; color:#c8102e;">{shannon_entropy:.2f} <span style="font-size:12px; font-weight:500;">({risk_level})</span></div>
                </div>""", unsafe_allow_html=True)


            st.write("")
            st.write("### 概率前 5 顺位比分预测 ")

            # 平铺并排序比分
            score_list = []
            for x in range(max_goals + 1):
                for y in range(max_goals + 1):
                    score_list.append((x, y, prob_matrix[x, y]))
            score_list.sort(key=lambda item: item[2], reverse=True)

            # 优雅渲染前 5 名比分行组件
            for rank, (h_g, a_g, prob) in enumerate(score_list[:5], 1):
                st.markdown(f"""
                    <div class="score-row">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="color: #86868b; font-weight: 700; font-size: 14px;">NO.{rank}</span>
                            <div class="score-badge">{h_g} — {a_g}</div>
                        </div>
                        <div class="score-prob">发生概率 {prob * 100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)