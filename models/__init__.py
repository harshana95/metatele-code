import importlib

_ONESTEP_MODEL_MODULES = [
    'models.OneStepDiffusion.onedataset_model',
    'models.OneStepDiffusion.twodataset_model',
]


def create_model(opt, logger):
    """Create model from explicit OneStepDiffusion module list."""
    model_type = opt['model_type']

    model_cls = None
    for module_name in _ONESTEP_MODEL_MODULES:
        module = importlib.import_module(module_name)
        cls_ = getattr(module, model_type, None)
        if cls_ is not None:
            model_cls = cls_
            break

    if model_cls is None:
        raise ValueError(f"Model class '{model_type}' not found in OneStepDiffusion modules.")
    return model_cls(opt, logger)
