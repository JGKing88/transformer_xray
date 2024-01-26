import numpy as np
import torch

def register_prehook(layer_num, tokens, perturbation_func, device):
    def prehook(module, input):
        if tokens == "all":
            iter_tokens = range(input[0].shape[1])
        else:
            iter_tokens = tokens 
        for token in iter_tokens:
            detached_input = input[0][:, token, :].detach().cpu().numpy()
            pertubation = perturbation_func(detached_input, layer_num, token)
            if not isinstance(pertubation, torch.Tensor):
                pertubation = torch.tensor(pertubation, dtype=torch.float32)
            pertubation = pertubation.to(device)
            input[0][:, token, :] += pertubation
            pertubation.detach_()
        return input
    return prehook

def register_hooks(layer, layer_num, perturbation_info, device):
    for (location, tokens, perturbation_func) in perturbation_info:
        if location == "before_mlp":
            layer.mlp.c_fc.register_forward_pre_hook(register_prehook(layer_num, tokens, perturbation_func, device))
        elif location == "before_attn":
            layer.attn.c_attn.register_forward_pre_hook(register_prehook(layer_num, tokens, perturbation_func, device))
        else:
            raise ValueError("Invalid hook location")

def register_pertubation_hooks(model, perturbation_dict, device):
    """
    perturbation_dict: dictionary of the form {layer: [(hook_locations, tokens, perturbation_function)]}. If layer is "all", then hooks will be registered for all layers.
        hook_location types:
                before_mlp: the residual before the MLP and after layer norm
                before_attn: the residual before the attention layer and after layer norm
    """
    for layer_num, perturbation_info in perturbation_dict.items():
        if layer_num == "all":
            for layer, model_layer in enumerate(model.transformer.h):
                register_hooks(model_layer, layer, perturbation_info, device)
        else:
            register_hooks(model.transformer.h[layer_num], layer_num, perturbation_info, device)