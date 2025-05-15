import os

# Script location
script_dir = os.path.dirname(__file__)
print(f"Script Directory: {script_dir}")

# Project root
project_root = os.path.abspath(os.path.join(script_dir, '..'))
print(f"Project Root: {project_root}")

# Model path
model_path = os.path.join(project_root, 'sound_classification_output', 'best_model.pth')
print(f"Model Path: {model_path}")
print(f"Model Exists: {os.path.exists(model_path)}")