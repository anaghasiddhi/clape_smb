import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0.0, num_classes=2, num_dim=2):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, num_dim))

    def forward(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))         # [B*L, D]
        targets = targets.view(-1).long()                 # [B*L]

        mask = targets != -1
        inputs = inputs[mask]
        targets = targets[mask]

        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)

        num_centers = self.centers.size(0)
        targets = torch.clamp(targets, 0, num_centers - 1)

        N = inputs.size(0)
        centers_batch = self.centers[targets]             # [N, D]

        inputs_exp = inputs.unsqueeze(1).expand(N, N, -1)
        centers_exp = centers_batch.unsqueeze(0).expand(N, N, -1)
        dist = torch.sqrt(torch.clamp(((inputs_exp - centers_exp) ** 2).sum(dim=2), min=1e-12))

        targets_exp = targets.unsqueeze(1)
        pos_mask = targets_exp.eq(targets_exp.T)

        dist_ap = []
        dist_an = []
        for i in range(N):
            pos = dist[i][pos_mask[i]]
            neg = dist[i][~pos_mask[i]]
            dist_ap.append(pos.max().unsqueeze(0) if pos.numel() > 0 else torch.zeros(1, device=dist.device))
            dist_an.append(neg.min().unsqueeze(0) if neg.numel() > 0 else torch.zeros(1, device=dist.device))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha if alpha is not None else 1.0
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))
        targets = targets.view(-1).long()

        mask = targets != -1
        inputs = inputs[mask]
        targets = targets[mask]

        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device)

        C = inputs.size(1)
        targets = torch.clamp(targets, 0, C-1)
        one_hot = F.one_hot(targets, C).float()
        probs = F.softmax(inputs, dim=1)
        pt = (probs * one_hot).sum(dim=1)

        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-9)
        return loss.mean()

class CrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # inputs: [B, L, C], targets: [B, L]
        B, L, C = inputs.shape
        inputs = inputs.view(-1, C)
        targets = targets.view(-1)
        return self.loss_fn(inputs, targets)

