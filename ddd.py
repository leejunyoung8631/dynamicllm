import torch
import torch.nn.functional as F

def sample_gumbel(shape, device='cpu', eps=1e-10):
    """
    Sample from Gumbel(0, 1)
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.0, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    (i.e., Gumbel(0,1) + logits, then softmax).
    """
    gumbel_noise = sample_gumbel(logits.size(), device=logits.device, eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax function.
    Args:
        logits: shape [batch_size, ..., n_classes]
        temperature: float, controls how 'hard' or 'soft' the distribution is
        hard: if True, output a one-hot vector on forward pass (straight-through).
              if False, output the soft sample.
    Returns:
        [batch_size, ..., n_classes] sample from the Gumbel-Softmax distribution.
    """
    # 1. Sample from Gumbel-Softmax
    y_soft = gumbel_softmax_sample(logits, temperature)

    if not hard:
        # Just return the soft sample (fully differentiable)
        return y_soft

    # 2. If we want a "hard" one-hot vector on forward pass, do argmax
    index = torch.argmax(y_soft, dim=-1)  # shape: [batch_size, ...]
    y_hard = F.one_hot(index, num_classes=logits.size(-1)).float()
    # shape: [batch_size, ..., n_classes]

    # 3. Straight-through trick:
    # - Forward pass uses y_hard
    # - Backward pass uses y_soft (so gradients still flow)
    y = (y_hard - y_soft).detach() + y_soft
    return y