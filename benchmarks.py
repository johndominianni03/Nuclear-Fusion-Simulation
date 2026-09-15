"""Multi-core and GPU HPC benchmark sweep, split out of main.py (step 14).

Imports diagnostics and physics_engine; nothing from the reactor modules.
"""

import numpy as np
import time
import torch

from physics_engine import (
    HPCPhysicsAccelerator,
    vectorized_boris_push_numba_fallback,
)
import diagnostics

# =======================================================
# MULTI-CORE & GPU HPC BENCHMARK
# =======================================================
def run_hpc_benchmark(cfg):
    print("==================================================")
    print("   MULTI-CORE & GPU HPC BENCHMARK RUN    ")
    print("==================================================")
    
    cpu_times = []
    gpu_times = []
    
    q, m, dt = cfg.e_charge, cfg.m_deuterium, cfg.reactor_dt
    hpc_engine = HPCPhysicsAccelerator(cfg.HPC_DEVICE)
    
    for num_particles in cfg.BENCHMARK_PARTICLE_COUNTS:
        print(f"[{num_particles:,} Particles] Generating Tensors and Caches...")
        pos_arr = np.random.rand(num_particles, 3).astype(np.float32)
        vel_arr = np.random.rand(num_particles, 3).astype(np.float32)
        B_arr = np.ones((num_particles, 3), dtype=np.float32) * cfg.B0
        E_arr = np.random.rand(num_particles, 3).astype(np.float32)
        
        pos_cpu, vel_cpu = pos_arr.copy(), vel_arr.copy()
        vectorized_boris_push_numba_fallback(pos_cpu[:10], vel_cpu[:10], q, m, B_arr[:10], E_arr[:10], dt)
        
        start_cpu = time.time()
        for _ in range(cfg.BENCHMARK_STEPS):
            vectorized_boris_push_numba_fallback(pos_cpu, vel_cpu, q, m, B_arr, E_arr, dt)
        cpu_duration = time.time() - start_cpu
        cpu_times.append(cpu_duration)
        
        if cfg.HPC_DEVICE.type != "cpu":
            pos_tensor = torch.tensor(pos_arr, device=cfg.HPC_DEVICE)
            vel_tensor = torch.tensor(vel_arr, device=cfg.HPC_DEVICE)
            B_tensor = torch.tensor(B_arr, device=cfg.HPC_DEVICE)
            E_tensor = torch.tensor(E_arr, device=cfg.HPC_DEVICE)
            
            # Warm up at the FULL particle count: the compiled push specializes per exact
            # shape, so a 10-particle warmup would leave the real shape to compile inside
            # the timed loop below and inflate gpu_duration.
            hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            if cfg.HPC_DEVICE.type == "cuda": torch.cuda.synchronize()
            if cfg.HPC_DEVICE.type == "mps": torch.mps.synchronize()
            
            start_gpu = time.time()
            for _ in range(cfg.BENCHMARK_STEPS):
                pos_tensor, vel_tensor = hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            
            if cfg.HPC_DEVICE.type == "mps": torch.mps.synchronize() 
            elif cfg.HPC_DEVICE.type == "cuda": torch.cuda.synchronize()
                
            gpu_duration = time.time() - start_gpu
            gpu_times.append(gpu_duration)
        else:
            gpu_times.append(None)
            
        gpu_str = f"{gpu_duration:.4f}s" if gpu_times[-1] is not None else "N/A"
        print(f"  -> CPU Parallel Time: {cpu_duration:.4f}s | Apple Metal Time: {gpu_str}")
        
    print("[SYSTEM] HPC Benchmark Complete! Handing off to diagnostics...")
    diagnostics.plot_hpc_benchmark(cfg.BENCHMARK_PARTICLE_COUNTS, cpu_times, gpu_times)
