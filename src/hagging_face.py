from huggingface_hub import login, upload_file

login()

repo_id = "Gopina/maize-disease-classifier"

files = [
    ("model.onnx",      "model.onnx"),
    ("model_int8.onnx", "model_int8.onnx"),
    ("model_meta.json", "model_meta.json"),
    ("hf_README.md",    "README.md"),       # becomes the model card
]

for local, remote in files:
    upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo_id, repo_type="model")
    print(f"✅  Uploaded {remote}")

print(f"\n🎉  https://huggingface.co/{repo_id}")