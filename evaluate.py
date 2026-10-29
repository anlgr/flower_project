import argparse
import os
import subprocess
import sys
import time
import warnings

try:
    import timm
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError:
    _venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Scripts", "python.exe")
    if os.path.exists(_venv_py) and os.path.normcase(sys.executable) != os.path.normcase(_venv_py):
        sys.exit(subprocess.call([_venv_py] + sys.argv))
    raise

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="tf_efficientnetv2_b0.in1k")
    parser.add_argument("--weights", type=str, default="runs/effv2b0/best_weights.pth")
    parser.add_argument("--data-dir", type=str, default="flower_data/test")
    parser.add_argument("--export-path", type=str, default="runs/effv2b0/model_submission.pt")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    model = timm.create_model(args.arch, pretrained=False, num_classes=102)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        traced = torch.jit.freeze(torch.jit.trace(model, dummy))
        torch.jit.save(traced, args.export_path)

    size_mb = os.path.getsize(args.export_path) / (1024 * 1024)
    size_score = min(100.0, (35.0 / size_mb) * 100.0)

    ts_model = torch.jit.load(args.export_path, map_location=device)
    ts_model.eval()
    ts_model = torch.jit.optimize_for_inference(ts_model)
    with torch.inference_mode():
        for _ in range(20): _ = ts_model(dummy)
        times = []
        for _ in range(200):
            t0 = time.perf_counter(); _ = ts_model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)
    avg_ms = sum(times) / len(times)
    speed_score = min(100.0, (50.0 / avg_ms) * 100.0)

    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loader = DataLoader(datasets.ImageFolder(args.data_dir, transform=eval_tf), batch_size=32)
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            correct += (ts_model(x).argmax(dim=1) == y).sum().item(); total += y.size(0)
    acc = (correct / total) * 100.0
    final = (0.60 * acc) + (0.25 * speed_score) + (0.15 * size_score)
    print(f"Acc: {acc:.2f}% | Hız: {avg_ms:.2f} ms ({speed_score:.1f}p) | Boyut: {size_mb:.2f} MB ({size_score:.1f}p) | FİNAL: {final:.2f}")


if __name__ == "__main__":
    main()

