"""Local hardware microbenchmark. Random tensors are NOT recommendation data."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/environment.json')
    args = parser.parse_args()
    torch.manual_seed(42)
    available = torch.backends.mps.is_available()
    device = torch.device('mps' if available else 'cpu')
    torch.set_num_threads(4)
    model = torch.nn.Sequential(torch.nn.Linear(1024, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 128)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(128, 1024, device=device)
    target = torch.randn(128, 128, device=device)

    def sync():
        if available:
            torch.mps.synchronize()

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(5):
        step()
    sync()
    start = time.perf_counter()
    for _ in range(50):
        loss = step()
    sync()
    train_ms = (time.perf_counter() - start) * 1000 / 50
    model.eval()
    with torch.inference_mode():
        sync()
        start = time.perf_counter()
        for _ in range(100):
            prediction = model(x)
        sync()
    result = {
        'purpose': 'random_tensor_hardware_probe_not_recommendation_experiment',
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'device': str(device),
        'mps_available': available, 'cpu_threads': 4,
        'batch_size': 128, 'input_dim': 1024, 'output_dim': 128,
        'train_steps': 50, 'warmup_steps': 5, 'train_step_ms': train_ms,
        'inference_batch_ms': (time.perf_counter() - start) * 1000 / 100,
        'finite_loss': bool(torch.isfinite(loss).item()),
        'prediction_shape': list(prediction.shape),
        'mps_current_allocated_bytes': torch.mps.current_allocated_memory() if available else None,
        'mps_driver_allocated_bytes': torch.mps.driver_allocated_memory() if available else None,
        'limitations': 'Not a real-data training/evaluation/inference closure; memory is a final snapshot, not peak; latency is not service latency.'
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
