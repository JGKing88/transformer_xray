

class ActivationRecorder():
    def __init__(self, model, hook_locations):
        """
        hook_locations: dictionary of the form {layer: [hook_locations]}. If layer is "all", then hooks will be registered for all layers.
            hook_location types:
                before_mlp: the residual before the MLP and after layer norm
                before_attn: the residual before the attention layer and after layer norm
        model: the model to register hooks on
        """
        self.hook_locations = hook_locations
        self.model = model
        activations = {}
        for layer in hook_locations.keys():
            if layer == "all":
                for layer in range(len(model.transformer.h)):
                    activations[layer] = {hook_location: [] for hook_location in hook_locations["all"]}
            else:
                activations[layer] = {hook_location: [] for hook_location in hook_locations[layer]}
        self.activations = activations
    
    def hook_wrapper(self, layer, location):
        def hook_function(module, input, output):
            self.activations[layer][location].append(output.detach().cpu())
        return hook_function
    
    def register_hooks(self, model_layer, layer_num, hook_locations):
        for location in hook_locations:
            if location == "before_mlp":
                model_layer.ln_2.register_forward_hook(self.hook_wrapper(layer_num, location))
            elif location == "before_attn":
                model_layer.ln_1.register_forward_hook(self.hook_wrapper(layer_num, location))
            else:
                raise ValueError("Invalid hook location")
    
    def register_recording_hooks(self):
        for layer_num, hook_locations in self.hook_locations.items():
            if layer_num == "all":
                for layer, model_layer in enumerate(self.model.transformer.h):
                    self.register_hooks(model_layer, layer, hook_locations)
            else:
                self.register_hooks(self.model.transformer.h[layer_num], layer_num, hook_locations)
    
    def get_activations(self):
        return self.activations