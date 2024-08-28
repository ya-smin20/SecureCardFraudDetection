import os

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_model(filepath):
    import joblib
    return joblib.load(filepath)
