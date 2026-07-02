import torch
from safetensors.torch import load_file
import os

# 这里改成你的 checkpoint 路径
ckpt_dir = "/Users/weilongxuan/codes/codi/outputs/codi_gpt2_cross_attn"  # 先放一个占位
# 如果在服务器上，用这个路径:
# ckpt_dir = "/inspire/qb-ilm2/project/neosmosis/weilongxuan-2531080168/codi/checkpoints/gpt2/gsm8k-aug/lca/gpt2-gsm8k-aug-contrastive/gpt2/ep_40/lr_0.003/seed_11"

print(f"Checking checkpoint in: {ckpt_dir}")

# 加载 checkpoint
try:
    state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    print("Loaded safetensors")
except Exception as e:
    print(f"Failed to load safetensors: {e}")
    try:
        state_dict = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
        print("Loaded bin")
    except Exception as e2:
        print(f"Failed to load bin: {e2}")
        exit(1)

print(f"\nTotal keys: {len(state_dict)}")

# 检查每个 tensor 是否有 NaN 或 Inf
has_nan = False
has_inf = False
nan_keys = []
inf_keys = []

for k, v in state_dict.items():
    if torch.is_tensor(v):
        if torch.isnan(v).any():
            print(f"[NaN] {k}: shape={v.shape}, dtype={v.dtype}")
            has_nan = True
            nan_keys.append(k)
        if torch.isinf(v).any():
            print(f"[Inf] {k}: shape={v.shape}, dtype={v.dtype}")
            has_inf = True
            inf_keys.append(k)

print(f"\nSummary:")
print(f"Has NaN: {has_nan}, {len(nan_keys)} keys")
print(f"Has Inf: {has_inf}, {len(inf_keys)} keys")

if nan_keys:
    print(f"\nNaN keys: {nan_keys}")
if inf_keys:
    print(f"\nInf keys: {inf_keys}")

# 检查几个关键层
print(f"\nChecking key layers:")
key_names = ["wte", "prj", "lora"]
for name in key_names:
    matching_keys = [k for k in state_dict.keys() if name in k.lower()]
    if matching_keys:
        print(f"\n{name.upper()} layers:")
        for k in matching_keys[:5]:  # 只看前5个
            v = state_dict[k]
            if torch.is_tensor(v):
                print(f"  {k}: mean={v.float().mean().item():.6f}, std={v.float().std().item():.6f}, "
                      f"min={v.float().min().item():.6f}, max={v.float().max().item():.6f}")
