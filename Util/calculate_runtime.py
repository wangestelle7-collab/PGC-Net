import torch
import time




def measure_runtime(model, input_args, device, num_runs=100):
    model = model.to(device)
    input_args = [arg.to(device) for arg in input_args]  # 输入数据移到目标设备

    # 预热（消除首次运行的初始化开销，如GPU kernel编译）
    with torch.no_grad():
        model(*input_args)
    if device.type == "cuda":
        torch.cuda.synchronize()  # GPU同步，确保计时准确

    # 多次运行取平均
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            model(*input_args)
            if device.type == "cuda":
                torch.cuda.synchronize()  # 每次迭代后同步GPU
    end = time.time()

    avg_time = (end - start) / num_runs
    return avg_time