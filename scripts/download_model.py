"""Download model weights via ModelScope.

Usage:
  python scripts/download_model.py --model gemma2-2b --dest /path/to/models
"""

import argparse
import os

MODEL_REGISTRY = {
    "gemma2-2b":  "AI-ModelScope/gemma-2-2b",
    "gemma2-9b":  "AI-ModelScope/gemma-2-9b",
    "gemma2-27b": "AI-ModelScope/gemma-2-27b",
    "llama2-7b":  "shakechen/Llama-2-7b-hf",
    "llama2-13b": "shakechen/Llama-2-13b-hf",
    "llama3-8b":  "LLM-Research/Meta-Llama-3-8B",
}


def main():
    parser = argparse.ArgumentParser(description="Download model via ModelScope")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--dest", default="./models", help="Destination directory")
    args = parser.parse_args()

    model_id = MODEL_REGISTRY[args.model]
    dest = os.path.join(args.dest, args.model)
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading {args.model} ({model_id}) to {dest}")

    from modelscope import snapshot_download
    snapshot_download(model_id, cache_dir=dest)
    print(f"Done: {dest}")


if __name__ == "__main__":
    main()
