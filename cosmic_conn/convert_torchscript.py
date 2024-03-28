# this file converts the cosmic-conn_ground_imaging model to torchscript

import torch
from cosmic_conn.inference_cr import init_model

# import model version
from inference_cr import MODEL_VERSON

# fake input of size 1x1x1024x1024
# image = hdul[ext].data.astype("float32")
input = torch.randn(32, 1, 256, 256).float()
out_name = f"ground_imaging_torchscript_{MODEL_VERSON}.pt"

# load model
model = init_model("ground_imaging")
model.eval()

# convert model to torchscript
traced_cell = torch.jit.trace(model, input)
traced_cell.save(out_name)

print(f"Model saved as {out_name}")

# verify if the traced model outputs the same result
output = model(input)
output_traced = traced_cell(input)
assert torch.allclose(output, output_traced)

print("Traced model output matches the original model output")

# run each model 100 times to compare the speed
import time

# load traced model
traced_cell = torch.jit.load(out_name)
n = 100

start = time.time()
for _ in range(n):
    model(input)
end = time.time()
print(f"Pytorch model: {end - start:.2f} seconds")

start = time.time()
for _ in range(n):
    traced_cell(input)
end = time.time()
print(f"Traced model: {end - start:.2f} seconds")

# no significant difference in speed if input is single image
# input = torch.randn(1, 1, 256, 256).float()
# Pytorch model: 10.59 seconds
# Traced model: 10.57 seconds

# same, with a batch size of 32, the difference is negligible
# input = torch.randn(32, 1, 256, 256).float()
# Pytorch model: 331.22 seconds
# Traced model: 329.16 seconds
