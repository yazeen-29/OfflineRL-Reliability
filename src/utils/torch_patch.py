import torch


def apply_torch_load_patch():
    """
    Patch torch.load so d3rlpy checkpoints created with older
    serialization behavior can be loaded under PyTorch 2.6+.
    """
    if not getattr(torch.load, "_is_patched", False):
        _original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_load(*args, **kwargs)

        patched_load._is_patched = True
        torch.load = patched_load