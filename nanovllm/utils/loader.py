import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))


def load_from_hf_model(target_model: nn.Module, hf_model: nn.Module):
    """
    Load weights from a HuggingFace transformer model to the custom model.
    
    Args:
        target_model: model for inference engine
        hf_model: model from HuggingFace modeling
    """
    import torch
    import torch.nn as nn

    # Get device from HF model
    device = next(hf_model.parameters()).device

    # Track which parameters have been loaded
    loaded_params = set()
    missing_params = []

    # Build a new state dict with materialized tensors
    new_state_dict = {}

    # Process each parameter in the target model
    for name, param in target_model.named_parameters():
        with torch.no_grad():
            # Handle fused qkv_proj
            if 'qkv_proj.weight' in name:
                base_name = name.replace('qkv_proj.weight', '')
                q_name = base_name + 'q_proj.weight'
                k_name = base_name + 'k_proj.weight'
                v_name = base_name + 'v_proj.weight'

                try:
                    q_weight = hf_model.get_parameter(q_name)
                    k_weight = hf_model.get_parameter(k_name)
                    v_weight = hf_model.get_parameter(v_name)

                    # Concatenate Q, K, V weights along dim 0 (output dimension)
                    qkv_weight = torch.cat(
                        [q_weight, k_weight, v_weight], dim=0)

                    # Convert to target dtype and appropriate device
                    if param.is_meta:
                        # Materialize on the HF model's device
                        new_state_dict[name] = qkv_weight.to(
                            device=device, dtype=param.dtype)
                    else:
                        # Keep on parameter's current device
                        new_state_dict[name] = qkv_weight.to(
                            device=param.device, dtype=param.dtype)

                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    # Only keep existing non-meta parameters
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            # Handle fused gate_up_proj (non-MoE)
            elif 'gate_up_proj.weight' in name and 'experts.' not in name:
                base_name = name.replace('gate_up_proj.weight', '')
                if "llada" in hf_model.__class__.__name__.lower():
                    gate_name = base_name + 'ff_proj.weight'
                else:
                    gate_name = base_name + 'gate_proj.weight'
                up_name = base_name + 'up_proj.weight'

                try:
                    gate_weight = hf_model.get_parameter(gate_name)
                    up_weight = hf_model.get_parameter(up_name)

                    # Concatenate gate and up weights along dim 0 (output dimension)
                    gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                    # Convert to target dtype and appropriate device
                    if param.is_meta:
                        # Materialize on the HF model's device
                        new_state_dict[name] = gate_up_weight.to(
                            device=device, dtype=param.dtype)
                    else:
                        # Keep on parameter's current device
                        new_state_dict[name] = gate_up_weight.to(
                            device=param.device, dtype=param.dtype)

                    loaded_params.add(name)
                except AttributeError:
                    missing_params.append(name)
                    # Only keep existing non-meta parameters
                    if not param.is_meta:
                        new_state_dict[name] = param.data

            # Handle regular parameters (direct mapping)
            else:
                try:
                    hf_param = hf_model.get_parameter(name)

                    # Convert to target dtype and appropriate device
                    if param.is_meta:
                        # Materialize on the HF model's device
                        new_state_dict[name] = hf_param.to(
                            device=device, dtype=param.dtype)
                    else:
                        # Keep on parameter's current device
                        new_state_dict[name] = hf_param.to(
                            device=param.device, dtype=param.dtype)

                    loaded_params.add(name)
                except AttributeError:
                    # Try without model prefix if not found
                    if name.startswith('model.'):
                        try:
                            hf_param = hf_model.get_parameter(name[6:])

                            # Convert to target dtype and appropriate device
                            if param.is_meta:
                                # Materialize on the HF model's device
                                new_state_dict[name] = hf_param.to(
                                    device=device, dtype=param.dtype)
                            else:
                                # Keep on parameter's current device
                                new_state_dict[name] = hf_param.to(
                                    device=param.device, dtype=param.dtype)

                            loaded_params.add(name)
                        except AttributeError:
                            missing_params.append(name)
                            # Only keep existing non-meta parameters
                            if not param.is_meta:
                                new_state_dict[name] = param.data
                    else:
                        missing_params.append(name)
                        # Only keep existing non-meta parameters
                        if not param.is_meta:
                            new_state_dict[name] = param.data

    # Load the new state dict into the target model
    # Use strict=False to handle any mismatches gracefully
    target_model.load_state_dict(new_state_dict, assign=True)

    # Disable gradients for all parameters (inference mode)
    for param in target_model.parameters():
        param.requires_grad_(False)

    # Report loading status
    # print(
    #     f"Successfully loaded {len(loaded_params)}/{len(list(target_model.named_parameters()))} parameters")
    if missing_params:
        print(
            f"Warning: Could not find {len(missing_params)} parameters in HF model:")
        for param in missing_params[:10]:  # Show first 10
            print(f"  - {param}")
        if len(missing_params) > 10:
            print(f"  ... and {len(missing_params) - 10} more")

    # Check if there are still meta parameters after loading
    meta_params = [name for name,
                   param in target_model.named_parameters() if param.is_meta]
    if meta_params:
        print(
            f"ERROR: {len(meta_params)} parameters are still on meta device after loading!")
        for param in meta_params[:5]:
            print(f"  - {param}")
        if len(meta_params) > 5:
            print(f"  ... and {len(meta_params) - 5} more")
        raise RuntimeError(
            f"Failed to materialize {len(meta_params)} meta parameters")
