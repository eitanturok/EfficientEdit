import torch
from torch.nn import functional as F

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature == 0:
        return logits
    else:
        logits = logits / temperature
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=1)
    return probs


def sample(probs : torch.Tensor, num_samples: int = 1):

    idx_next = torch.multinomial(probs, num_samples=num_samples)
    if (idx_next.item() == 0):
        print("Warning: Sampled idx is zero, retrying...")
        idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


class Timer:
    def __init__(self):
        self.prefill_times = []
        self.decode_times = []
    def start(self):
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        self.start_time.record()
    def stop_prefill(self):
        self.end_time.record()
        torch.cuda.synchronize()
        self.prefill_times.append(self.start_time.elapsed_time(self.end_time) / 1000) # convert to seconds
    def stop_decode(self):
        self.end_time.record()
        torch.cuda.synchronize()
        self.decode_times.append(self.start_time.elapsed_time(self.end_time) / 1000) # convert to seconds
    def __repr__(self):
        total_prefill = sum(self.prefill_times)
        total_decode = sum(self.decode_times)
        return f"<Timer> Prefill time: {total_prefill:.2f} ms, Decode time: {total_decode:.2f} ms"
    def to_dict(self):
        total_prefill = sum(self.prefill_times)
        total_decode = sum(self.decode_times)
        total_time = total_prefill + total_decode
        return {"total_time": total_time, "prefill_time": total_prefill, "decode_time": total_decode}
