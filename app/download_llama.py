import os
from huggingface_hub import snapshot_download


def download_model(repo_id: str = "meta-llama/Llama-2-13b-chat-hf"):
    token = os.environ.get("HF_TOKEN")
    print(f"Starting download for {repo_id}...")
    path = snapshot_download(repo_id=repo_id, token=token, allow_patterns=["*"], force_download=False)
    print("Downloaded to:", path)
    return path


if __name__ == "__main__":
    download_model()
