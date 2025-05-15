import os
import sys

def print_path_info():
    # Current working directory
    print("Current Working Directory:")
    print(os.getcwd())
    print()

    # Script location
    print("Script Location:")
    print(__file__)
    print()

    # Sys argv
    print("Sys Argv:")
    print(sys.argv[0])
    print()

    # Project root attempts
    print("Project Root Attempts:")
    print("1. dirname(__file__):")
    print(os.path.dirname(__file__))
    print()

    print("2. dirname of dirname(__file__):")
    print(os.path.dirname(os.path.dirname(__file__)))
    print()

    print("3. Absolute path to project root:")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(project_root)
    print()

    # Model path attempts
    print("Model Path Attempts:")
    print("1. Relative to current directory:")
    print(os.path.abspath(os.path.join(os.getcwd(), '..', 'sound_classification_output', 'best_model.pth')))
    print()

    print("2. Relative to script directory:")
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sound_classification_output', 'best_model.pth')))
    print()

if __name__ == '__main__':
    print_path_info()