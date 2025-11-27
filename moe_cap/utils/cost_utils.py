# GPU hardware costs in FLOPs (approximate market prices as of late 2024/early 2025)
# Based on NVIDIA specifications and market availability
GPU_COST_DICT = {
    # A100 Series - Ampere Architecture
    "NVIDIA-A100-PCIe-40GB": 9000,       # ~$8,000-$10,000 - PCIe variant
    "NVIDIA-A100-PCIe-80GB": 19000,      # ~$18,000-$20,000 - PCIe variant, 80GB memory
    "NVIDIA-A100-SXM4-40GB": 12000,      # ~$10,000-$12,000 - SXM4 variant, 40GB memory
    "NVIDIA-A100-SXM4-80GB": 19000,      # ~$18,000-$20,000 - SXM4 for DGX systems
    
    # H100 Series - Hopper Architecture  
    "NVIDIA-H100-PCIe-80GB": 28000,      # ~$25,000-$30,000 - PCIe variant
    
    # H200 Series - Enhanced Hopper
    "NVIDIA-H200-141GB": 35000,          # ~$30,000-$40,000 - 141GB HBM3e memory
    "NVIDIA-H100-80GB-HBM3": 38000,     # ~$35,000-$40,000 - HBM3 variant
    
    # RTX Professional/Workstation Series
    "NVIDIA-RTX-A5000-24GB": 2400,       # ~$2,250-$2,500 - Professional workstation GPU
    "NVIDIA-RTX-A6000-48GB": 4500,       # ~$4,500 - High-end workstation GPU
}

GPU_TDP_DICT = {
    # A100 Series - Ampere Architecture
    "NVIDIA-A100-PCIe-40GB": 250,    # 250W TDP - PCIe variant
    "NVIDIA-A100-PCIe-80GB": 300,    # 300W TDP - PCIe variant with more memory
    "NVIDIA-A100-SXM4-40GB": 400,    # 400W TDP - SXM4 variant
    "NVIDIA-A100-SXM4-80GB": 400,    # 400W TDP - SXM4 variant (same as 40GB)
    
    # H100 Series - Hopper Architecture  
    "NVIDIA-H100-PCIe-80GB": 350,    # 350W TDP - PCIe variant
    
    # H200 Series - Enhanced Hopper
    "NVIDIA-H200-141GB": 350,        # 350W TDP - PCIe variant
    "NVIDIA-H100-80GB-HBM3": 700,   # 700W TDP - HBM3 variant (configurable 350-700W)
    
    # RTX Professional/Workstation Series
    "NVIDIA-RTX-A5000-24GB": 230,    # 230W TDP - Professional workstation
    "NVIDIA-RTX-A6000-48GB": 300,    # 300W TDP - High-end workstation
}

CPU_COST = 7214.00
CPU_TDP = 350
ENERGY_COST = 0.15 #$/KWh
SCALE_RATE_OTHER_COST_CAPITAL = 1.2
SCALE_RATE_OTHER_COST_ENERGY = 1.25

def calculate_cost(e2e_latency, gpu_type, num_gpus):
    """Calculate cost based on GPU type and latency.
    
    Args:
        e2e_latency: End-to-end latency in seconds
        gpu_type: GPU type string (must match a key in GPU_COST_DICT)
        num_gpus: Number of GPUs used
        
    Returns:
        Calculated cost, or None if GPU type not recognized
    """
    if gpu_type not in GPU_COST_DICT:
        return None
    
    gpu_capital = GPU_COST_DICT[gpu_type]
    gpu_power = GPU_TDP_DICT[gpu_type]

    capital_cost = gpu_capital * num_gpus + CPU_COST
    energy_cost = (gpu_power * num_gpus + CPU_TDP) / 1000 * ENERGY_COST * e2e_latency / 3600
    cost = energy_cost * SCALE_RATE_OTHER_COST_CAPITAL + capital_cost * SCALE_RATE_OTHER_COST_ENERGY
    return cost
