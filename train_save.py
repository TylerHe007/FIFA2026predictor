import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Any


# 1. 专业对称 Dixon-Coles 神经网络 (加入数值截断边界与不确定性 Dropout)
class ProfessionalSymmetricDCNet(nn.Module):
    def __init__(self, num_teams: int, embedding_dim: int = 16, dropout_p: float = 0.1):
        super(ProfessionalSymmetricDCNet, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)

        # 单个球队隐性核心实力网络：维度 16(emb) + 1(Elo) + 2(Form) = 19维
        # 加入 nn.Dropout 模块，不仅用于防止过拟合，还专门用于推理阶段的蒙特卡洛（MC）不确定性抽样
        self.team_net = nn.Sequential(
            nn.Linear(embedding_dim + 3, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 2)  # 分别输出无约束的原始潜在 [进攻力, 防守力]
        )

        # 主场优势参数 (标量，用 nn.Parameter 动态学习)
        self.home_advantage = nn.Parameter(torch.tensor([0.25]))

        # 对称低比分平局修正网络
        self.rho_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def get_team_stats(self, team_id: torch.Tensor, elo: torch.Tensor, form: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        emb = self.team_embedding(team_id)

        # 数值平衡层：
        # 1. Elo 积分除以 100 映射到 [15.0, 20.0] 区间
        # 2. 近期近况(Form)场均进失球除以 3.0 进行压缩缩放，使其与全局 Embedding 的方差量级保持齐次
        normalized_elo = elo / 100.0
        normalized_form = form / 3.0

        x = torch.cat([emb, normalized_elo, normalized_form], dim=1)
        out = self.team_net(x)
        return out[:, 0], out[:, 1]

    def forward(self, h_id: torch.Tensor, a_id: torch.Tensor, neutral: torch.Tensor,
                h_elo: torch.Tensor, a_elo: torch.Tensor, h_form: torch.Tensor, a_form: torch.Tensor) -> torch.Tensor:
        h_att, h_def = self.get_team_stats(h_id, h_elo, h_form)
        a_att, a_def = self.get_team_stats(a_id, a_elo, a_form)

        # 严格执行主客场空间对称逻辑
        h_adv_term = (1.0 - neutral) * self.home_advantage

        # 计算对数期望进球率
        log_lambda_h = h_att - a_def + h_adv_term.squeeze(1)
        log_lambda_a = a_att - h_def - h_adv_term.squeeze(1)

        # 避免因指数爆炸导致后序 exp 出现 NaN 或 Inf。范围 [-3.0, 3.0] 意味着期望进球在 0.05 到 20 之间
        log_lambda_h = torch.clamp(log_lambda_h, min=-3.0, max=3.0)
        log_lambda_a = torch.clamp(log_lambda_a, min=-3.0, max=3.0)

        # 提取双方攻防两端绝对差异，作为低比分依赖度 rho 的无序对称输入
        diff = torch.cat([torch.abs(h_att - a_att).unsqueeze(1), torch.abs(h_def - a_def).unsqueeze(1)], dim=1)
        rho_raw = self.rho_net(diff).squeeze(1)

        return torch.stack([log_lambda_h, log_lambda_a, rho_raw], dim=1)


# 2. 带有对称输入的 PyTorch Dataset
class SymmetricFootballDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.h_id = torch.tensor(dataframe['home_id'].values, dtype=torch.long)
        self.a_id = torch.tensor(dataframe['away_id'].values, dtype=torch.long)
        self.neutral = torch.tensor(dataframe['neutral'].values.astype(int), dtype=torch.float32).unsqueeze(1)
        self.h_elo = torch.tensor(dataframe['elo_h'].values, dtype=torch.float32).unsqueeze(1)
        self.a_elo = torch.tensor(dataframe['elo_a'].values, dtype=torch.float32).unsqueeze(1)
        self.h_form = torch.tensor(dataframe[['roll_h_sc', 'roll_h_con']].values, dtype=torch.float32)
        self.a_form = torch.tensor(dataframe[['roll_a_sc', 'roll_a_con']].values, dtype=torch.float32)
        self.scores = torch.tensor(dataframe[['home_score', 'away_score']].values, dtype=torch.float32)
        self.weights = torch.tensor(dataframe['sample_weight'].values, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.h_id)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (self.h_id[idx], self.a_id[idx], self.neutral[idx], self.h_elo[idx], self.a_elo[idx],
                self.h_form[idx], self.a_form[idx], self.scores[idx], self.weights[idx])


# 3. 稳健型 Dixon-Coles 损失函数 (增强极端情况下的数值保护)
class ProfessionalDixonColesLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super(ProfessionalDixonColesLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_lh, log_la = pred[:, 0], pred[:, 1]

        # 严格使用 tanh 进行截断映射，将 rho 锁死在物理上有安全理论边界的 [-0.10, 0.10]
        rho = torch.tanh(pred[:, 2]) * 0.10

        lh = torch.exp(log_lh)
        la = torch.exp(log_la)
        x, y = targets[:, 0], targets[:, 1]

        # 基础泊松负对数似然损失计算
        base_nll = -(x * log_lh - lh) - (y * log_la - la)

        # Dixon-Coles 低比分(0或1球)相互依赖依赖修正项计算
        tau = torch.ones_like(lh)
        m00 = (x == 0) & (y == 0)
        m10 = (x == 1) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m11 = (x == 1) & (y == 1)

        tau = torch.where(m00, 1.0 - lh * la * rho, tau)
        tau = torch.where(m10, 1.0 + la * rho, tau)
        tau = torch.where(m01, 1.0 + lh * rho, tau)
        tau = torch.where(m11, 1.0 - rho, tau)

        # 引入极其严格的极小值约束 eps，防止极限情况下模型在特定低比分上由于激进优化导致 tau 变成负数或 0
        log_tau = torch.log(torch.clamp(tau, min=self.eps))

        return base_nll - log_tau


# 4. 主训练流程 (引入余弦退火学习率、动态学习率调度与梯度剪切)
def main_train():
    csv_path = 'results.csv'
    save_path = 'advanced_dixon_coles_model.pth'

    print(">> [1/5] 加载历史国际比赛数据")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未在当前目录下找到数据集: {csv_path}")

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 彻底隔离未赛与已赛行
    train_df = df.dropna(subset=['home_score', 'away_score']).copy()

    print(">> [2/5] 构建全局图连通关联矩阵并序列化全图动态 Elo 积分")
    all_teams = sorted(list(set(df['home_team'].unique()).union(set(df['away_team'].unique()))))
    team_to_id = {team: idx for idx, team in enumerate(all_teams)}
    train_df['home_id'] = train_df['home_team'].map(team_to_id)
    train_df['away_id'] = train_df['away_team'].map(team_to_id)

    # 全图动态滑窗数据结构
    elo_dict = {t: 1500.0 for t in all_teams}
    team_history = {}
    elo_h, elo_a = [], []
    roll_h_sc, roll_h_con, roll_a_sc, roll_a_con = [], [], [], []

    for idx, row in train_df.iterrows():
        h, a = row['home_team'], row['away_team']

        # 无泄漏时序特征读取
        elo_h.append(elo_dict[h])
        elo_a.append(elo_dict[a])

        roll_h_sc.append(
            np.mean(team_history[h], axis=0)[0] if h in team_history and len(team_history[h]) > 0 else 1.35)
        roll_h_con.append(
            np.mean(team_history[h], axis=0)[1] if h in team_history and len(team_history[h]) > 0 else 1.35)
        roll_a_sc.append(
            np.mean(team_history[a], axis=0)[0] if a in team_history and len(team_history[a]) > 0 else 1.35)
        roll_a_con.append(
            np.mean(team_history[a], axis=0)[1] if a in team_history and len(team_history[a]) > 0 else 1.35)

        # 全图基于动态赛事等级(K)与净胜球乘子(G)的自适应 Elo 更新逻辑
        hs, as_ = float(row['home_score']), float(row['away_score'])
        dr = elo_dict[h] - elo_dict[a] + (100.0 if not row['neutral'] else 0.0)
        e_h = 1.0 / (10.0 ** (-dr / 400.0) + 1.0)
        sa = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)

        k = 40 if row['tournament'] == 'FIFA World Cup' else 20
        g_mult = 1.0 if abs(hs - as_) <= 1 else (1.5 if abs(hs - as_) == 2 else (11.0 + abs(hs - as_)) / 8.0)

        elo_delta = k * g_mult * (sa - e_h)
        elo_dict[h] += elo_delta
        elo_dict[a] -= elo_delta

        # 滚动维护最近5场基础技战术状态队列
        if h not in team_history: team_history[h] = []
        if a not in team_history: team_history[a] = []
        team_history[h].append([hs, as_])
        team_history[a].append([as_, hs])
        if len(team_history[h]) > 5: team_history[h].pop(0)
        if len(team_history[a]) > 5: team_history[a].pop(0)

    train_df['elo_h'] = elo_h
    train_df['elo_a'] = elo_a
    train_df['roll_h_sc'] = roll_h_sc
    train_df['roll_h_con'] = roll_h_con
    train_df['roll_a_sc'] = roll_a_sc
    train_df['roll_a_con'] = roll_a_con

    # 固化历史终点线的原生浮点变量字典 (安全跨版本反序列化防报错)
    latest_team_meta = {}
    for team in all_teams:
        form = np.mean(team_history[team], axis=0) if team in team_history and len(team_history[team]) > 0 else [1.35,
                                                                                                                 1.35]
        latest_team_meta[team] = {
            "elo": float(elo_dict[team]),
            "form": [float(form[0]), float(form[1])]
        }

    # 时序衰减及赛事、主客场复合权重优化
    max_date = train_df['date'].max()
    train_df['years_ago'] = (max_date - train_df['date']).dt.days / 365.25
    train_df['weight_time'] = np.exp(-0.1 * train_df['years_ago'])
    train_df['weight_tournament'] = train_df['tournament'].apply(lambda x: 2.5 if x == 'FIFA World Cup' else 1.0)
    train_df['weight_home_away'] = train_df['neutral'].apply(lambda x: 1.0 if x else 1.1)
    train_df['sample_weight'] = train_df['weight_time'] * train_df['weight_tournament'] * train_df['weight_home_away']

    # >> [3/5] 数据加载器
    dataset = SymmetricFootballDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # >> [4/5] 建模与工业级自适应调度策略配置
    model = ProfessionalSymmetricDCNet(num_teams=len(team_to_id), embedding_dim=16, dropout_p=0.1)
    criterion = ProfessionalDixonColesLoss()

    # 采用带有更强正则化防过拟合的 AdamW 优化器，配合同步学习率衰减
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    # 引入余弦退火学习率调度器 (Cosine Annealing LR)，让学习率随 Epoch 呈余弦平滑下降，确保收敛到全局极小值
    epochs = 200
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f">> [5/5] 开始执行高阶深度神经网络训练 (共 {epochs} 轮)...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for h_id, a_id, neutral, h_elo, a_elo, h_form, a_form, scores, weights in dataloader:
            optimizer.zero_grad()

            preds = model(h_id, a_id, neutral, h_elo, a_elo, h_form, a_form)
            loss_element = criterion(preds, scores)

            # 样本级动态乘子加权
            weighted_loss = (loss_element * weights).mean()
            weighted_loss.backward()

            # 彻底杜绝由于客制化复杂负对数似然损失中对数计算引发的任何反向传播梯度爆炸 (NaN)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += weighted_loss.item()

        # 每一个 Epoch 结束，调度器平滑缩减学习率
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss / len(dataloader):.4f} | LR: {current_lr:.6f}")

    # 持久化存储
    torch.save({
        'model_state_dict': model.state_dict(),
        'team_to_id': team_to_id,
        'latest_team_meta': latest_team_meta,
        'num_teams': len(team_to_id)
    }, save_path)
    print(f"\n>> 模型训练成功，保存至: {save_path}")


if __name__ == "__main__":
    main_train()