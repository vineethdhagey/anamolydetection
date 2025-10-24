from datasets import load_dataset
import pandas as pd

def load_fd002_dataset(split='train'):
    """
    Load NASA Turbofan FD002 dataset from Hugging Face.
    Args:
        split: 'train', 'valid', or 'test'
    Returns:
        df: pandas DataFrame
    """
    if split not in ['train', 'valid', 'test']:
        raise ValueError("split must be 'train', 'valid', or 'test'")
    dataset = load_dataset("LucasThil/nasa_turbofan_degradation_FD002", split=split)
    df = pd.DataFrame(dataset)
    df = df.sort_values(by=["unit_number", "time_cycles"]).reset_index(drop=True)
    return df


def simulate_stream(df, batch_size=1):
    """
    Simulate streaming of data in batches.
    Args:
        df: pandas DataFrame
        batch_size: int, number of rows per batch
    Yields:
        batch: pandas DataFrame
    """
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size].copy()
