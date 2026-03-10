"""Data Loader for GPT-2 Training."""
import os
import grain.python as grain

from datasets import load_dataset, load_from_disk, Dataset
from huggingface_hub import hf_hub_download
from jax import numpy as jnp

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATASET = "roneneldan/TinyStories"

def download_dataset(dataset_name: str = DEFAULT_DATASET, split: str = "train+validation"):
    """Downloads a dataset from Hugging Face and saves to disk."""
    folder_name = dataset_name.replace("/", "_")
    save_path = os.path.join(os.getenv("DATA_PATH", "./data"), folder_name)
    dataset = load_dataset(dataset_name, split=split)
    dataset.save_to_disk(save_path)
    return save_path


def download_tinystories_v2():
    """Downloads TinyStories V2 (GPT-4 only) text files and saves as a HF dataset."""
    data_path = os.getenv("DATA_PATH", "./data")
    save_path = os.path.join(data_path, "TinyStoriesV2")

    file_paths = []
    for filename in ["TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"]:
        file_paths.append(hf_hub_download(
            repo_id="roneneldan/TinyStories",
            filename=filename,
            repo_type="dataset",
            cache_dir=data_path,
        ))

    def story_generator():
        for fp in file_paths:
            current_story = []
            with open(fp, "r") as f:
                for line in f:
                    if "<|endoftext|>" in line:
                        text = "".join(current_story).strip()
                        if text:
                            yield {"text": text}
                        current_story = []
                    else:
                        current_story.append(line)
                text = "".join(current_story).strip()
                if text:
                    yield {"text": text}

    dataset = Dataset.from_generator(story_generator)
    dataset.save_to_disk(save_path)
    print(f"TinyStories V2: {len(dataset):,} examples saved to {save_path}")
    return save_path


def dataset_summary(dataset_name: str = DEFAULT_DATASET):
    """Loads a downloaded dataset from disk and prints its summary."""
    folder_name = dataset_name.replace("/", "_")
    load_path = os.path.join(os.getenv("DATA_PATH", "./data"), folder_name)

    dataset = load_from_disk(load_path)

    print(f"Dataset:    {dataset_name}")
    print(f"Path:       {load_path}")
    print(f"Examples:   {len(dataset):,}")
    print(f"Columns:    {dataset.column_names}")
    print(f"Features:   {dataset.features}")
    print(f"Disk size:  {dataset.dataset_size / (1024**2):.2f} MB")

    print("\nSample (first 3):")
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        for key, value in example.items():
            text = str(value)[:150]
            print(f"  [{i}] {key}: {text}")

    return dataset


if __name__ == "__main__":
    download_tinystories_v2()
    dataset_summary("TinyStoriesV2")