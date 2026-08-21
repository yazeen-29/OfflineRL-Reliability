def apply_torch_load_patch():
    """
    Compatibility hook.

    d3rlpy 2.8.1 full .d3 checkpoints are loaded through
    d3rlpy.load_learnable(), so no torch.load monkey patch
    is required for the project's current checkpoint format.
    """
    return None