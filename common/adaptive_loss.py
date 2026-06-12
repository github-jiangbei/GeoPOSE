import torch


class EMALossNormalizer:
    """
    EMA-based auxiliary loss normalizer.

    The base weight remains the user's explicit hyperparameter. EMA is used
    as a protective normalizer for auxiliary losses: the base weight is the
    maximum intended strength, and normalization only down-weights an
    auxiliary term when its EMA grows above its reference. This avoids the
    failure mode where decreasing auxiliary losses are rewarded by larger
    weights and quickly saturate at max_scale.
    """

    def __init__(self, enabled=True, decay=0.99, eps=1e-6, min_scale=0.5, max_scale=1.0):
        self.enabled = bool(enabled)
        self.decay = float(decay)
        self.eps = float(eps)
        self.min_scale = None if min_scale is None or min_scale <= 0 else float(min_scale)
        self.max_scale = None if max_scale is None or max_scale <= 0 else float(max_scale)
        self.ema = {}
        self.reference_ema = {}
        self.last_effective_weights = {}

    def _update_ema(self, name, loss):
        value = loss.detach()
        if value.dim() != 0:
            value = value.mean()

        previous = self.ema.get(name)
        if previous is None:
            updated = value
        else:
            previous = previous.to(device=value.device, dtype=value.dtype)
            updated = self.decay * previous + (1.0 - self.decay) * value

        self.ema[name] = updated.detach()
        return self.ema[name]

    def weighted(self, name, loss, base_weight):
        if base_weight <= 0:
            self.last_effective_weights[name] = 0.0
            return loss.new_zeros(())

        if not self.enabled:
            self.last_effective_weights[name] = float(base_weight)
            return base_weight * loss

        ema = self._update_ema(name, loss)
        reference = self.reference_ema.get(name)
        if reference is None:
            reference = ema.detach()
            self.reference_ema[name] = reference

        with torch.no_grad():
            reference = reference.to(device=ema.device, dtype=ema.dtype)
            scale = torch.minimum(reference / (ema + self.eps), torch.ones_like(ema))
            if self.min_scale is not None or self.max_scale is not None:
                min_scale = self.min_scale if self.min_scale is not None else -float('inf')
                max_scale = self.max_scale if self.max_scale is not None else float('inf')
                scale = torch.clamp(scale, min=min_scale, max=max_scale)
            effective_weight = base_weight * scale
            self.last_effective_weights[name] = float(effective_weight.detach().cpu())

        return effective_weight * loss

    def state_dict(self):
        return {
            'ema': {name: value.detach().cpu() for name, value in self.ema.items()},
            'reference_ema': {name: value.detach().cpu() for name, value in self.reference_ema.items()},
            'last_effective_weights': dict(self.last_effective_weights),
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.ema = {
            name: value.detach()
            for name, value in state_dict.get('ema', {}).items()
        }
        loaded_reference = state_dict.get('reference_ema')
        if loaded_reference is None:
            loaded_reference = state_dict.get('ema', {})
        self.reference_ema = {
            name: value.detach()
            for name, value in loaded_reference.items()
        }
        self.last_effective_weights = dict(state_dict.get('last_effective_weights', {}))
