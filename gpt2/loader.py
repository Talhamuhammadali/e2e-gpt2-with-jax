"""Data Loader for GPT-2 Training."""
import os
import grain.python as grain

from datasets import load_dataset, load_from_disk
from jax import numpy as jnp

from dotenv import load_dotenv

load_dotenv()

def download_tiny_stories():
    """ Downloads the 'tinystories' dataset from Hugging Face. 
        Use to download dataset once and save to disk for faster loading in future runs.
    """
    dataset = load_dataset("roneneldan/TinyStories", split="train+validation")
    dataset.save_to_disk(os.path.join(os.getenv("DATA_PATH", "./data"), "tiny_stories"))
    return "success" 

if __name__ == "__main__":
    result = download_tiny_stories()
    print(result)
    