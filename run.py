import os
import subprocess
import sys


def main():
    # 添加当前目录到Python路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 检查模型是否存在
    model_path = 'models/generator_final.pth'

    if not os.path.exists(model_path):
        print("Model not found. Training model first...")
        # 运行训练脚本
        result = subprocess.run([sys.executable, "train.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Error during training:", result.stderr)

        # 检查训练是否成功
        if not os.path.exists(model_path):
            print("Training failed. Please check the error messages above.")
            return

    # 运行生成脚本
    print("Generating and visualizing 3D shape...")
    result = subprocess.run([sys.executable, "generate.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error during generation:", result.stderr)


if __name__ == "__main__":
    main()