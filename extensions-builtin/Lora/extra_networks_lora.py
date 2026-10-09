import math

from modules import extra_networks, shared, torchao_weight_quant
import networks


class FatalLoraPreparationError(RuntimeError):
    """Raised when proceeding would generate with unsupported/stale quantized LoRA state."""


class ExtraNetworkLora(extra_networks.ExtraNetwork):
    def __init__(self):
        super().__init__('lora')

    remove_symbols = str.maketrans('', '', ":,")

    def activate(self, p, params_list):
        additional = shared.opts.sd_lora

        if additional != "None" and additional in networks.available_networks and not any(x for x in params_list if x.items[0] == additional):
            p.all_prompts = [x + f"<lora:{additional}:{shared.opts.extra_networks_default_multiplier}>" for x in p.all_prompts]
            params_list.append(extra_networks.ExtraNetworkParams(items=[additional, shared.opts.extra_networks_default_multiplier]))

        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        try:
            for params in params_list:
                assert params.items

                names.append(params.positional[0])

                te_multiplier = float(params.positional[1]) if len(params.positional) > 1 else 1.0
                te_multiplier = float(params.named.get("te", te_multiplier))

                unet_multiplier = float(params.positional[2]) if len(params.positional) > 2 else te_multiplier
                unet_multiplier = float(params.named.get("unet", unet_multiplier))

                dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
                dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim

                if not (math.isfinite(te_multiplier) and math.isfinite(unet_multiplier)):
                    raise ValueError(f"non-finite multiplier in <lora:{':'.join(params.items)}>")

                te_multipliers.append(te_multiplier)
                unet_multipliers.append(unet_multiplier)
                dyn_dims.append(dyn_dim)

            networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
        except Exception as e:
            # Name the failing step; extra_networks.activate lets it fail the request (generating without the requested
            # LoRAs is never an option).
            raise FatalLoraPreparationError(f"LoRA activation failed: {e}") from e
        for backend in torchao_weight_quant.BACKENDS.values():
            if not networks.prepare_quant_active_config(backend):
                error = getattr(shared.sd_model, f"network_{backend.name}_prepare_error", f"{backend.label} LoRA preparation failed")
                p.comment(f"{backend.label} LoRA preparation failed; generation stopped to avoid slow per-step fallback. {error}")
                raise FatalLoraPreparationError(error)

        # loaded_networks holds the requested networks in request order. Infotext keys on the names this request
        # used: an unchanged applied state (same sources and multipliers) keeps the published networks, whose
        # mentioned_name is the name of the request that published them (another alias of the same file).
        is_hr_pass = getattr(p, "is_hr_pass", False)
        if not is_hr_pass or not hasattr(p, "lora_errors"):
            p.lora_errors = {}
        for name, item in zip(names, networks.loaded_networks):
            if item.unmatched_keys:
                p.lora_errors[name.translate(self.remove_symbols)] = f"{len(item.unmatched_keys)} unmatched keys"
        if p.lora_errors:
            p.extra_generation_params["Lora errors"] = ', '.join(f'{k}: {v}' for k, v in p.lora_errors.items())

        if shared.opts.lora_add_hashes_to_infotext:
            if not is_hr_pass or not hasattr(p, "lora_hashes"):
                p.lora_hashes = {}

            for name, item in zip(names, networks.loaded_networks):
                if item.network_on_disk.shorthash:
                    p.lora_hashes[name.translate(self.remove_symbols)] = item.network_on_disk.shorthash

            if p.lora_hashes:
                p.extra_generation_params["Lora hashes"] = ', '.join(f'{k}: {v}' for k, v in p.lora_hashes.items())

    def deactivate(self, p):
        # Retain the physically published state across requests. The next activation
        # always calls load_networks, including for an empty desired state, so real
        # semantic changes and clear still reconcile before sampling.
        pass
