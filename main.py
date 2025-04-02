import argparse
import os

def run_train():
    print("🚀 Starting pretraining on CINIC-10...")
    os.system("python train.py")

def run_finetune():
    print("🔧 Starting fine-tuning on CIFAR-10...")
    os.system("python finetune.py")

def run_scratch():
    print("🧪 Starting training from scratch on CIFAR-10...")
    os.system("python scratch.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified entry for GoogLeNet CINIC10 → CIFAR10")
    parser.add_argument('--mode', type=str, choices=['train', 'finetune', 'scratch'], required=True,
                        help="Select mode: 'train' (CINIC-10), 'finetune' (CIFAR-10), or 'scratch'")
    
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "finetune":
        run_finetune()
    elif args.mode == "scratch":
        run_scratch()
      
