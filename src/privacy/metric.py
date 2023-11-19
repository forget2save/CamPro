import torch
import math


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class CMC:
    def __init__(self, gallery, probe, label) -> None:
        gallery_size = gallery.shape[0]
        probe_size = probe.shape[0]
        device = gallery.device
        # NOTE: calculate cosine similarity matrix via l2 norm + dot product
        gallery = l2_norm(gallery)
        probe = l2_norm(probe)
        probe_score = torch.mm(probe, gallery.T)
        # retrieve score of correct pairs
        indices = torch.arange(probe_size, dtype=torch.long, device=device)
        correct = probe_score[indices, label]
        # sorted scores (probe, gallery)
        values = torch.sort(probe_score)[0]
        self.probe_rank = gallery_size - torch.searchsorted(
            values, correct.unsqueeze(-1)
        ).squeeze(-1)
        self.max_rank = gallery_size

    def __call__(self, rank):
        return (self.probe_rank <= rank).float().mean().item()

    def cmc_curve(self):
        ranks = torch.unique(
            torch.exp(torch.linspace(0, math.log(self.max_rank), steps=100))
            .round()
            .type(torch.long)
        )
        probs = [self.__call__(rank) for rank in ranks]
        ranks = list(map(float, ranks))
        return ranks, probs
