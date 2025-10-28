from collections import defaultdict
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
        self.times = defaultdict(list)
        self.n_tokens = defaultdict(int)

    def start(self, key):
        torch.cuda.synchronize()  # Ensure previous ops are done
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        self.times[key].append((start_time, end_time))

    def stop(self, key, n_tokens:int=0):
        start_time, end_time = self.times[key][-1]
        end_time.record()
        torch.cuda.synchronize()  # Wait for all ops to complete
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # convert to seconds
        self.times[key][-1] = (elapsed_time)
        if n_tokens: self.n_tokens[key] += n_tokens

    def __repr__(self):
        result = []
        for name, times in self.times.items():
            # Filter out only the completed timings
            completed = [t for t in times if isinstance(t, float)]
            if completed:
                result.append(f"{name}: {sum(completed):.4f}s")
        return ', '.join(result)

    def to_dict(self):
        result = {}
        total_time, total_n_tokens = 0.0, 0
        for name, times in self.times.items():
            # Filter out only the float values (completed timings)
            completed_times = [t for t in times if isinstance(t, float)]
            if completed_times:
                result[f"time_{name}"] = sum(completed_times)
                if "_forward" not in name:
                    result[f"num_forwards_{name}"] = len(completed_times)
                    result[f"num_tokens_{name}"] = self.n_tokens[name]
                    result[f"throughput_{name}"] = self.n_tokens[name] / sum(completed_times) if sum(completed_times) > 0 else 0.0
                    total_time += sum(completed_times)
                    total_n_tokens += self.n_tokens[name]
        result[f"time"] = total_time
        result[f"n_tokens"] = total_n_tokens
        result[f"throughput"] = total_n_tokens / total_time if total_time > 0 else 0.0
        return result
