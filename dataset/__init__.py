from dataset.hf_dataset import HuggingFaceDataset

__all__ = ['create_dataset']


def create_dataset(dataset_opt):
    """Create dataset from explicit class registry."""
    dataset_type = dataset_opt['type']

    registry = {
        'HuggingFaceDataset': HuggingFaceDataset,
    }

    dataset_cls = registry.get(dataset_type)
    if dataset_cls is None:
        raise ValueError(f"Dataset class '{dataset_type}' not found.")
    return dataset_cls(dataset_opt)
