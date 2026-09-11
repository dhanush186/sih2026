import kagglehub

path = kagglehub.dataset_download(
    "mohammedabdeldayem/the-fake-or-real-dataset"
)

print("Path to dataset files:", path)
