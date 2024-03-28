import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from cosmic_conn.inference_cr import init_model
from inference_cr import MODEL_VERSON


def worker_init():
    """
    Ignore SIGINT in worker processes to allow main process to handle KeyboardInterrupt.
    """
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)


def model_inference(inputs):
    model, input_slice = inputs
    with torch.no_grad():
        output = model(input_slice)
    return output


def parallel_inference(model, input_tensor, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    # Assuming the model and data are small enough to be easily pickled
    # otherwise consider another way of distributing computation
    # Splitting the original tensor into sub-tensors
    split_size = int(np.ceil(input_tensor.size(0) / num_processes))
    inputs = [
        (model, input_tensor[i : i + split_size])
        for i in range(0, input_tensor.size(0), split_size)
    ]

    with Pool(processes=num_processes, initializer=worker_init) as pool:
        results = pool.map(model_inference, inputs)

    output = torch.cat(results, dim=0)
    return output


if __name__ == "__main__":

    num_processes = 16
    input_tensor = torch.randn(32, 1, 256, 256).float()

    # Load model
    model = init_model("ground_imaging")
    model.eval()

    # Disable gradients for inference
    torch.set_grad_enabled(False)

    # Execute the parallel inference
    output = parallel_inference(model, input_tensor, num_processes=num_processes)
    print(f"Output shape: {output.shape}")

    # compare the output with the original model
    output_original = model(input_tensor).cpu()
    assert torch.allclose(output, output_original)

    # compare the speed of parallel inference with single process
    import time

    start = time.time()
    output_single = model(input_tensor).cpu()
    end = time.time()

    print(f"Single process inference time: {end - start:.2f} seconds")

    start = time.time()
    output_parallel = parallel_inference(model, input_tensor).cpu()
    end = time.time()

    print(f"Parallel inference time: {end - start:.2f} seconds")

    # Single process inference time: 2.94 seconds
    # Parallel inference time: 3.48 seconds
