# -*- coding: utf-8 -*-

"""
In this file, we convert the Cosmic-CoNN model to TorchScript
for production deployment at LCO.
CY Xu (cxu@ucsb.edu)
"""
import os, psutil
process = psutil.Process(os.getpid())

print("--> before model init:", process.memory_info().rss / 10**9, "GB")  # in bytes 

import torch
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
from time import perf_counter

from cosmic_conn.inference_cr import init_model
from cosmic_conn.dl_framework.cosmic_conn import Cosmic_CoNN


def timer(f,*args):
    start = perf_counter()
    f(*args)
    print("--> after inference with timer:", process.memory_info().rss / 10**9, "GB")  # in bytes 
    return (perf_counter() - start)

# trace model
dummy_input = torch.randn(1, 1, 1024, 1024, dtype=torch.float32)
# dummy_input = torch.randn(1, 1, 512, 512, dtype=torch.float32)
# dummy_input = torch.ones(1, 1, 256, 256, dtype=torch.float32)

# init model
model = init_model("ground_imaging")

# trace model
# traced_model = torch.jit.trace(model, dummy_input)
# traced_model.save('traced_ground_imaging.pt')

# script model
# scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model))
# scripted_model.save('scripted_ground_imaging.pt')

# load previous
# model = torch.jit.load('scripted_ground_imaging.pt')
# model = torch.jit.load('traced_ground_imaging.pt')

# print("--> after model init:", process.memory_info().rss / 10**9, "GB")  # in bytes 

# read a real image
# from astropy.io import fits
# test_file = "/Users/cyxu/astro/test_data/coj1m011-fa12-20190309-0034-e91_5frms_aligned.fits"
# test_file = "/Users/cyxu/astro/test_data/ogg0m406-kb27-20210713-0233-e91.fits.fz"

# with fits.open(test_file) as hdul:
#     image = hdul['SCI'].data.astype("float32")[:1000, :1000]
#     dummy_input = torch.tensor(image, dtype=torch.float32).view(1, 1, image.shape[0], image.shape[1])

# print("--> after fit read:", process.memory_info().rss / 10**9, "GB")  # in bytes 

# test
# model_output = model(dummy_input)
# scripted_output = scripted_model(dummy_input)
# traced_output = traced_model(dummy_input)

n = 5
# print(f"eager model time ({n} average): ", np.mean([timer(model,dummy_input) for _ in range(n)]))
# print(f"compiled model time ({n} average): ", np.mean([timer(scripted_model,dummy_input) for _ in range(n)]))
# print(f"traced model time ({n} average): ", np.mean([timer(traced_model,dummy_input) for _ in range(n)]))


# print(torch.sum(model_output - scripted_output))
# print(torch.sum(model_output - traced_output))

# breakpoint()

# print(traced_model.graph)
# print(traced_model.code)

# Pytorch profiler
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(dummy_input)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         model(dummy_input)

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

# profile with tensorboard

prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)
prof.start()
model(dummy_input)
prof.step()
model(dummy_input)
prof.step()
model(dummy_input)
prof.step()
model(dummy_input)
prof.step()
model(dummy_input)
prof.step()
prof.stop()