import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import os

class ProfessionalSymmetricDCNet(nn.Module):
    def __init__(self, num_teams, embedding_dim=16, dropout_p=0.1):
        super(ProfessionalSymmetricDCNet, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)

        self.team_net = nn.Sequential(
            nn.Linear(embedding_dim + 3, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(32, 2)
        )
        self.home_advantage = nn.Parameter(torch.tensor([0.25]))
        self.rho_net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))

    def get_team_stats(self, team_id, elo, form):
        emb = self.team_embedding(team_id)

        normalized_elo = elo / 100.0
        normalized_form = form / 3.0

        x = torch.cat([emb, normalized_elo, normalized_form], dim=1)
        out = self.team_net(x)
        return out[:, 0], out[:, 1]

    def forward(self, h_id, a_id, neutral, h_elo, a_elo, h_form, a_form):
        h_att, h_def = self.get_team_stats(h_id, h_elo, h_form)
        a_att, a_def = self.get_team_stats(a_id, a_elo, a_form)

        h_adv_term = (1.0 - neutral) * self.home_advantage

        log_lambda_h = h_att - a_def + h_adv_term.squeeze(1)
        log_lambda_a = a_att - h_def - h_adv_term.squeeze(1)

        log_lambda_h = torch.clamp(log_lambda_h, min=-3.0, max=3.0)
        log_lambda_a = torch.clamp(log_lambda_a, min=-3.0, max=3.0)

        diff = torch.cat([torch.abs(h_att - a_att).unsqueeze(1), torch.abs(h_def - a_def).unsqueeze(1)], dim=1)
        rho_raw = self.rho_net(diff).squeeze(1)

        return torch.stack([log_lambda_h, log_lambda_a, rho_raw], dim=1)


class AdvancedSymmetricPredictor:
    def __init__(self, model_path='advanced_dixon_coles_model.pth'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到指定的训练模型权重文件: {model_path}")

        # 保持对安全机制的非纯权重防御加载机制
        checkpoint = torch.load(model_path, weights_only=False)
        self.team_to_id = checkpoint['team_to_id']
        self.latest_team_meta = checkpoint['latest_team_meta']

        self.model = ProfessionalSymmetricDCNet(num_teams=checkpoint['num_teams'])
        self.model.load_state_dict(
            checkpoint['checkpoint'] if 'checkpoint' in checkpoint else checkpoint['model_state_dict'])
        self.model.eval()
        print("模型已就绪\n")

    def predict_match(self, home_team, away_team, neutral=True, max_goals=5):
        if home_team not in self.team_to_id or away_team not in self.team_to_id:
            print(f" 错误: 目标球队 [{home_team}] 或 [{away_team}] 缺失历史交锋特征，无法进行量化评估。")
            return None

        h_meta = self.latest_team_meta[home_team]
        a_meta = self.latest_team_meta[away_team]

        # 特征流转换为张量
        h_id = torch.tensor([self.team_to_id[home_team]], dtype=torch.long)
        a_id = torch.tensor([self.team_to_id[away_team]], dtype=torch.long)
        n_feat = torch.tensor([[1.0 if neutral else 0.0]], dtype=torch.float32)
        h_elo_t = torch.tensor([[h_meta['elo']]], dtype=torch.float32)
        a_elo_t = torch.tensor([[a_meta['elo']]], dtype=torch.float32)
        h_form_t = torch.tensor([h_meta['form']], dtype=torch.float32)
        a_form_t = torch.tensor([a_meta['form']], dtype=torch.float32)

        # 执行前向推理
        with torch.no_grad():
            pred = self.model(h_id, a_id, n_feat, h_elo_t, a_elo_t, h_form_t, a_form_t)
            lambda_h = torch.exp(pred[0, 0]).item()
            lambda_a = torch.exp(pred[0, 1]).item()
            rho = torch.tanh(pred[0, 2]).item() * 0.10

        # 生成双向 Dixon-Coles 联合概率分布比分矩阵
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

        # 归一化矩阵全域概率
        prob_matrix /= np.sum(prob_matrix)

        safe_probs = prob_matrix[prob_matrix > 1e-9]
        shannon_entropy = -np.sum(safe_probs * np.log2(safe_probs))

        home_win = np.sum(np.tril(prob_matrix, -1))
        draw = np.sum(np.diag(prob_matrix))
        away_win = np.sum(np.triu(prob_matrix, 1))

        score_list = []
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                score_list.append((x, y, prob_matrix[x, y]))

        score_list.sort(key=lambda item: item[2], reverse=True)
        top_5_scores = score_list[:5]

        print(f" {home_team} VS {away_team}")
        print(f" 场地: {'中立球场' if neutral else '非中立球场'}")
        print(f" Elo: {home_team}({h_meta['elo']:.1f}) | {away_team}({a_meta['elo']:.1f})")
        print(f" 期望进球率: {home_team}={lambda_h:.3f} | {away_team}={lambda_a:.3f}")
        print(f" 信息熵: {shannon_entropy:.3f} bits")
        print(f" 结局预测:")
        print(f"  [{home_team}] 胜出率: {home_win * 100:.2f}%")
        print(f"  常规时间平局: {draw * 100:.2f}%")
        print(f"  [{away_team}] 胜出率: {away_win * 100:.2f}%")
        print(f" 高概率比分:")
        for rank, (h_g, a_g, prob) in enumerate(top_5_scores, 1):
            print(f"  [第 {rank} 顺位] 比分 {h_g} - {a_g} | 概率: {prob * 100:.2f}%")

        return top_5_scores


if __name__ == "__main__":
    predictor = AdvancedSymmetricPredictor()
    predictor.predict_match("China", "Japan", neutral=True)