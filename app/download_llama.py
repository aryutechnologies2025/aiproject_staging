from huggingface_hub import snapshot_download
import os

repo_id = "meta-llama/Llama-2-13b-chat-hf"
token = os.environ.get("HF_TOKEN")  # set above

print("Starting download ... this may take a while.")
path = snapshot_download(repo_id=repo_id, token=token, allow_patterns=["*"], force_download=False)
print("Downloaded to:", path)

# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="meta-llama/Llama-2-70b-chat-hf",
#     local_dir="./llama70b",
#     resume_download=True,
#     token="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"  # your HF token
# )
