import torch
import numpy as np


def ctc_alignment(log_probs, targets, blank=0):
    """
    Args:
    - log_probs(torch.FloatTensor): (T, C) where T = input length, C = number of classes (including blank)
    - targets(torch.LongTensor): (S) where S = target length
    - blank(int, optional): blank label. Default: 0

    Examples:
    >>> T = 50      # Input sequence length
    >>> C = 20      # Number of classes (including blank)
    >>> S = 15      # Target sequence length
    >>>
    >>> log_probs = torch.randn(T, C).log_softmax(1)
    >>> targets = torch.randint(low=1, high=C, size=(S,), dtype=torch.long)
    >>> results = ctc_alignment(log_probs, targets)

    Reference:
    A. Vaglio et al.: MULTILINGUAL LYRICS-TO-AUDIO ALIGNMENT:
    https://program.ismir2020.net/static/final_papers/101.pdf
    """

    # the input length must be greater than or equal to the target length
    assert log_probs.shape[0] >= len(targets)

    # add uniform noise
    log_probs = log_probs.exp()
    log_probs += torch.rand_like(log_probs) * (1e-10 - 1e-11) + 1e-11
    log_probs = log_probs.log()

    # add blank at the beginning, end, and between every unit
    _targets = torch.ones([len(targets) * 2 + 1], device=log_probs.device, dtype=torch.long) * blank
    _targets[1::2] = targets

    # [target lengths, time]
    none_value = np.inf
    alphas = torch.ones([len(_targets), log_probs.shape[0]], device=log_probs.device) * none_value
    betas = torch.ones([len(_targets), log_probs.shape[0]], device=log_probs.device) * none_value

    # initialize
    alphas[0, 0] = log_probs[0, _targets[0]]
    alphas[1, 0] = log_probs[0, _targets[1]]

    # - inf means zero prob
    alphas[2:, 0] = -np.inf

    # compute value function
    def value(s, j):
        # avoid minus indexing
        if s < 0 or j < 0:
            return torch.ones(1, device=log_probs.device)[0] * -np.inf

        # prevent duplicate calculations
        if alphas[s, j].item() != none_value:
            return alphas[s, j]

        # self-loop or one-step
        if _targets[s].item() in [blank, _targets[s - 2].item()]:
            candidates = torch.stack([value(s - t, j - 1) for t in [0, 1]])
        # self-loop or one-step or two-step
        else:
            candidates = torch.stack([value(s - t, j - 1) for t in [0, 1, 2]])

        # log prob summation
        alphas[s, j] = torch.max(candidates) + log_probs[j, _targets[s]]

        # record step
        betas[s, j] = torch.argmax(candidates).item()
        return alphas[s, j]

    # forward recursion for best alignment compute
    last_candidate = torch.stack([value(len(_targets) - 1 - t, log_probs.shape[0] - 1) for t in [0, 1]])

    # init
    align_idx = torch.zeros([log_probs.shape[0]], device=log_probs.device, dtype=torch.long)
    align_idx[-1] = len(_targets) - 1 - torch.argmax(last_candidate).long().item()

    # inverse recursion
    for i in range(1, len(align_idx)):
        align_idx[-i - 1] = align_idx[-i] - betas[align_idx[-i].long().item(), -i]

    # alignment results
    align_result = _targets[align_idx.cpu().tolist()]

    return align_result
