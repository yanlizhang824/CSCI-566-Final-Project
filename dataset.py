import kagglehub

# Download latest version
path = kagglehub.dataset_download("gnurtqh/cmu-mosei")

print("Path to dataset files:", path)