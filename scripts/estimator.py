import os
import torch
import torch.nn.functional as F


class BaseEstimator:
    """Default estimator."""
    def __repr__(self):
        return "Base"

    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        return 0.0

    @property
    def higher_is_better(self) -> bool:
        return True


class RandomEstimator(BaseEstimator):
    def __init__(self):
        self._rng = torch.Generator()

    def __repr__(self):
        return "Random"
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        seed = int.from_bytes(os.urandom(8), "little")
        self._rng.manual_seed(seed)
        return torch.rand(1, generator=self._rng, device=logits.device)


class MSP(BaseEstimator):
    """
    Calculates the uncertainty score using Maximum Sequence Probability (MSP).

    This method computes the total log-probability of a given sequence (`input_ids`)
    based on the model's `logits`. It's a common baseline for uncertainty
    estimation, where a higher score indicates higher model confidence.

    The score is calculated by summing the log-softmax probabilities of the
    actual tokens in the input sequence.
    """
    def __repr__(self):
        return "MSP"

    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        return float(logprobs.sum().detach().cpu())

    @property
    def higher_is_better(self) -> bool:
        return True
    
class MTE(BaseEstimator):
    def __repr__(self):
        return "MTE"
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        logprobs: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Mean Token Entropy"""
        if logits.dim() == 3:
            logits = logits.squeeze(0)

        all_log_probs = F.log_softmax(logits, dim=-1)
        token_entropies = -torch.sum(torch.exp(all_log_probs) * all_log_probs, dim=-1)

        return token_entropies.mean()

    @property
    def higher_is_better(self) -> bool:
        return False


ESTIMATOR_MAP = {
    "None": BaseEstimator(),
    "Random": RandomEstimator(),
    "MSP": MSP(),
    "MTE": MTE(),
}
