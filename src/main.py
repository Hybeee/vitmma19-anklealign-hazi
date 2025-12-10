import torch
import os

def main():
    print("Hello World from Docker container!")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

    for dir in os.listdir("data"):
        print(dir)

if __name__ == "__main__":
    main()