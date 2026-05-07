from __future__ import annotations
import gradio as gr
import logging
import os
import re

import lora_patches
import network
import network_lora
import network_glora
import network_hada
import network_ia3
import network_lokr
import network_full
import network_norm
import network_oft

import torch
from typing import Union

from modules import shared, devices, sd_models, errors, scripts, sd_hijack, mxfp8_config, nvfp4_config
import modules.textual_inversion.textual_inversion as textual_inversion
import modules.models.sd3.mmdit

from lora_logger import logger

module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
    network_oft.ModuleTypeOFT(),
]


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key


def assign_network_names_to_compvis_modules(sd_model):
    network_layer_mapping = {}

    if shared.sd_model.is_sdxl:
        for i, embedder in enumerate(shared.sd_model.conditioner.embedders):
            if not hasattr(embedder, 'wrapped'):
                continue

            for name, module in embedder.wrapped.named_modules():
                network_name = f'{i}_{name.replace(".", "_")}'
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name
    else:
        cond_stage_model = getattr(shared.sd_model.cond_stage_model, 'wrapped', shared.sd_model.cond_stage_model)

        for name, module in cond_stage_model.named_modules():
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name

    for name, module in shared.sd_model.model.named_modules():
        network_name = name.replace(".", "_")
        network_layer_mapping[network_name] = module
        module.network_layer_name = network_name

    sd_model.network_layer_mapping = network_layer_mapping


class BundledTIHash(str):
    def __init__(self, hash_str):
        self.hash = hash_str

    def __str__(self):
        return self.hash if shared.opts.lora_bundled_ti_to_infotext else ''


def load_network(name, network_on_disk):
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)

    sd = sd_models.read_state_dict(network_on_disk.filename)

    # this should not be needed but is here as an emergency fix for an unknown error people are experiencing in 1.2.0
    if not hasattr(shared.sd_model, 'network_layer_mapping'):
        assign_network_names_to_compvis_modules(shared.sd_model)

    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in shared.sd_model.network_layer_mapping
    if hasattr(shared.sd_model, 'diffusers_weight_map'):
        diffusers_weight_map = shared.sd_model.diffusers_weight_map
    elif hasattr(shared.sd_model, 'diffusers_weight_mapping'):
        diffusers_weight_map = {}
        for k, v in shared.sd_model.diffusers_weight_mapping():
            diffusers_weight_map[k] = v
        shared.sd_model.diffusers_weight_map = diffusers_weight_map
    else:
        diffusers_weight_map = None

    matched_networks = {}
    bundle_embeddings = {}

    for key_network, weight in sd.items():

        if diffusers_weight_map:
            key_network_without_network_parts, network_name, network_weight = key_network.rsplit(".", 2)
            network_part = network_name + '.' + network_weight
        else:
            key_network_without_network_parts, _, network_part = key_network.partition(".")

        if key_network_without_network_parts == "bundle_emb":
            emb_name, vec_name = network_part.split(".", 1)
            emb_dict = bundle_embeddings.get(emb_name, {})
            if vec_name.split('.')[0] == 'string_to_param':
                _, k2 = vec_name.split('.', 1)
                emb_dict['string_to_param'] = {k2: weight}
            else:
                emb_dict[vec_name] = weight
            bundle_embeddings[emb_name] = emb_dict

        if diffusers_weight_map:
            key = diffusers_weight_map.get(key_network_without_network_parts, key_network_without_network_parts)
        else:
            key = convert_diffusers_name_to_compvis(key_network_without_network_parts, is_sd2)

        sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.network_layer_mapping.get(m.group(1), None)

        # SDXL loras seem to already have correct compvis keys, so only need to replace "lora_unet" with "diffusion_model"
        if sd_module is None and "lora_unet" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("lora_unet", "diffusion_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        elif sd_module is None and "lora_te1_text_model" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("lora_te1_text_model", "0_transformer_text_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

            # some SD1 Loras also have correct compvis keys
            if sd_module is None:
                key = key_network_without_network_parts.replace("lora_te1_text_model", "transformer_text_model")
                sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        # kohya_ss OFT module
        elif sd_module is None and "oft_unet" in key_network_without_network_parts:
            key = key_network_without_network_parts.replace("oft_unet", "diffusion_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        # KohakuBlueLeaf OFT module
        if sd_module is None and "oft_diag" in key:
            key = key_network_without_network_parts.replace("lora_unet", "diffusion_model")
            key = key_network_without_network_parts.replace("lora_te1_text_model", "0_transformer_text_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        if sd_module is None:
            keys_failed_to_match[key_network] = key
            continue

        if key not in matched_networks:
            matched_networks[key] = network.NetworkWeights(network_key=key_network, sd_key=key, w={}, sd_module=sd_module)

        matched_networks[key].w[network_part] = weight

    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                break

        if net_module is None:
            raise AssertionError(f"Could not find a module type (out of {', '.join([x.__class__.__name__ for x in module_types])}) that would accept those keys: {', '.join(weights.w)}")

        net.modules[key] = net_module

    embeddings = {}
    for emb_name, data in bundle_embeddings.items():
        embedding = textual_inversion.create_embedding_from_data(data, emb_name, filename=network_on_disk.filename + "/" + emb_name)
        embedding.loaded = None
        embedding.shorthash = BundledTIHash(name)
        embeddings[emb_name] = embedding

    net.bundle_embeddings = embeddings

    if keys_failed_to_match:
        logging.debug(f"Network {network_on_disk.filename} didn't match keys: {keys_failed_to_match}")

    return net


def purge_networks_from_memory():
    while len(networks_in_memory) > shared.opts.lora_in_memory_limit and len(networks_in_memory) > 0:
        name = next(iter(networks_in_memory))
        networks_in_memory.pop(name, None)

    devices.torch_gc()


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    emb_db = sd_hijack.model_hijack.embedding_db
    already_loaded = {}

    for net in loaded_networks:
        if net.name in names:
            already_loaded[net.name] = net
        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded:
                emb_db.register_embedding_by_name(None, shared.sd_model, emb_name)

    loaded_networks.clear()

    unavailable_networks = []
    for name in names:
        if name.lower() in forbidden_network_aliases and available_networks.get(name) is None:
            unavailable_networks.append(name)
        elif available_network_aliases.get(name) is None:
            unavailable_networks.append(name)

    if unavailable_networks:
        update_available_networks_by_names(unavailable_networks)

    networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()

        networks_on_disk = [available_networks.get(name, None) if name.lower() in forbidden_network_aliases else available_network_aliases.get(name, None) for name in names]

    failed_to_load_networks = []

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = already_loaded.get(name, None)

        if network_on_disk is not None:
            if net is None:
                net = networks_in_memory.get(name)

            if net is None or os.path.getmtime(network_on_disk.filename) > net.mtime:
                try:
                    net = load_network(name, network_on_disk)

                    networks_in_memory.pop(name, None)
                    networks_in_memory[name] = net
                except Exception as e:
                    errors.display(e, f"loading network {network_on_disk.filename}")
                    continue

            net.mentioned_name = name

            network_on_disk.read_hash()

        if net is None:
            failed_to_load_networks.append(name)
            logging.info(f"Couldn't find network with name {name}")
            continue

        net.te_multiplier = te_multipliers[i] if te_multipliers else 1.0
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else 1.0
        net.dyn_dim = dyn_dims[i] if dyn_dims else 1.0
        loaded_networks.append(net)

        for emb_name, embedding in net.bundle_embeddings.items():
            if embedding.loaded is None and emb_name in emb_db.word_embeddings:
                logger.warning(
                    f'Skip bundle embedding: "{emb_name}"'
                    ' as it was already loaded from embeddings folder'
                )
                continue

            embedding.loaded = False
            if emb_db.expected_shape == -1 or emb_db.expected_shape == embedding.shape:
                embedding.loaded = True
                emb_db.register_embedding(embedding, shared.sd_model)
            else:
                emb_db.skipped_embeddings[name] = embedding

    if failed_to_load_networks:
        lora_not_found_message = f'Lora not found: {", ".join(failed_to_load_networks)}'
        sd_hijack.model_hijack.comments.append(lora_not_found_message)
        if shared.opts.lora_not_found_warning_console:
            print(f'\n{lora_not_found_message}\n')
        if shared.opts.lora_not_found_gradio_warning:
            gr.Warning(lora_not_found_message)

    purge_networks_from_memory()


def allowed_layer_without_weight(layer):
    if isinstance(layer, torch.nn.LayerNorm) and not layer.elementwise_affine:
        return True

    return False


def store_weights_backup(weight):
    if weight is None:
        return None

    return weight.to(devices.cpu, copy=True)


def restore_weights_backup(obj, field, weight):
    if weight is None:
        setattr(obj, field, None)
        return

    getattr(obj, field).copy_(weight)


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)

    if weights_backup is None and bias_backup is None:
        return

    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            restore_weights_backup(self, 'in_proj_weight', weights_backup[0])
            restore_weights_backup(self.out_proj, 'weight', weights_backup[1])
        else:
            restore_weights_backup(self, 'weight', weights_backup)

    if isinstance(self, torch.nn.MultiheadAttention):
        restore_weights_backup(self.out_proj, 'bias', bias_backup)
    else:
        restore_weights_backup(self, 'bias', bias_backup)


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores original weights from backup and alters weights according to networks.
    """

    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return

    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)

    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != ():
        if current_names != () and not allowed_layer_without_weight(self):
            raise RuntimeError(f"{network_layer_name} - no backup weights found and current weights are not unchanged")

        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (store_weights_backup(self.in_proj_weight), store_weights_backup(self.out_proj.weight))
        else:
            weights_backup = store_weights_backup(self.weight)

        self.network_weights_backup = weights_backup

    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None and wanted_names != ():
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_backup = store_weights_backup(self.out_proj.bias)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = store_weights_backup(self.bias)
        else:
            bias_backup = None

        # Unlike weight which always has value, some modules don't have bias.
        # Only report if bias is not None and current bias are not unchanged.
        if bias_backup is not None and current_names != ():
            raise RuntimeError("no backup bias found and current bias are not unchanged")

        self.network_bias_backup = bias_backup

    if current_names != wanted_names:
        network_restore_weights_from_backup(self)

        for net in loaded_networks:
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight') and not isinstance(module, modules.models.sd3.mmdit.QkvLinear):
                try:
                    with torch.no_grad():
                        if getattr(self, 'fp16_weight', None) is None:
                            weight = self.weight
                            bias = self.bias
                        else:
                            weight = self.fp16_weight.clone().to(self.weight.device)
                            bias = getattr(self, 'fp16_bias', None)
                            if bias is not None:
                                bias = bias.clone().to(self.bias.device)
                        updown, ex_bias = module.calc_updown(weight)

                        if len(weight.shape) == 4 and weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))

                        self.weight.copy_((weight.to(dtype=updown.dtype) + updown).to(dtype=self.weight.dtype))
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias).to(self.weight.dtype)
                            else:
                                self.bias.copy_((bias + ex_bias).to(dtype=self.bias.dtype))
                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.in_proj_weight.chunk(3, 0)
                        updown_q, _ = module_q.calc_updown(qw)
                        updown_k, _ = module_k.calc_updown(kw)
                        updown_v, _ = module_v.calc_updown(vw)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight)

                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                    if ex_bias is not None:
                        if self.out_proj.bias is None:
                            self.out_proj.bias = torch.nn.Parameter(ex_bias)
                        else:
                            self.out_proj.bias += ex_bias

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if isinstance(self, modules.models.sd3.mmdit.QkvLinear) and module_q and module_k and module_v:
                try:
                    with torch.no_grad():
                        # Send "real" orig_weight into MHA's lora module
                        qw, kw, vw = self.weight.chunk(3, 0)
                        updown_q, _ = module_q.calc_updown(qw)
                        updown_k, _ = module_k.calc_updown(kw)
                        updown_v, _ = module_v.calc_updown(vw)
                        del qw, kw, vw
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        self.weight += updown_qkv

                except RuntimeError as e:
                    logging.debug(f"Network {net.name} layer {network_layer_name}: {e}")
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

                continue

            if module is None:
                continue

            logging.debug(f"Network {net.name} layer {network_layer_name}: couldn't find supported operation")
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1

        self.network_current_names = wanted_names


def network_forward(org_module, input, original_forward):
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """

    if len(loaded_networks) == 0:
        return original_forward(org_module, input)

    input = devices.cond_cast_unet(input)

    network_restore_weights_from_backup(org_module)
    network_reset_cached_weight(org_module)

    y = original_forward(org_module, input)

    network_layer_name = getattr(org_module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue

        y = module.forward(input, y)

    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None
    self.network_bias_backup = None


def is_mxfp8_weight(weight):
    return type(weight).__name__ == "MXTensor" and type(weight).__module__.startswith("torchao.")


def network_mxfp8_wanted_names():
    return tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)


def network_mxfp8_active_config_signature():
    """Return the currently active in-memory MXFP8+LoRA weight signature.

    This intentionally describes only the one active configuration, not a disk
    cache key for every possible LoRA permutation. If any field that changes
    effective weights changes, eager preparation invalidates and rebuilds the
    model's in-memory prepared state.
    """

    checkpoint_info = getattr(shared.sd_model, "sd_checkpoint_info", None)
    checkpoint_key = (
        getattr(checkpoint_info, "filename", None),
        getattr(checkpoint_info, "hash", None),
        getattr(checkpoint_info, "sha256", None),
    )
    coverage = tuple(sorted(getattr(shared.opts, "mxfp8_linear_coverage", ()) or ()))
    config_name = mxfp8_config.CONFIG_NAME

    loras = []
    for net in loaded_networks:
        network_on_disk = getattr(net, "network_on_disk", None)
        filename = getattr(network_on_disk, "filename", None)
        try:
            mtime = os.path.getmtime(filename) if filename else None
        except OSError:
            mtime = None
        loras.append((
            net.name,
            net.te_multiplier,
            net.unet_multiplier,
            net.dyn_dim,
            filename,
            getattr(network_on_disk, "shorthash", None),
            mtime,
        ))

    return (checkpoint_key, coverage, config_name, tuple(loras))


def network_mxfp8_mark_model_unprepared(model=None):
    model = model or getattr(shared, "sd_model", None)
    if model is None:
        return
    for attr in (
        "network_mxfp8_active_config_signature",
        "network_mxfp8_prepare_stats",
        "network_mxfp8_prepare_error",
    ):
        try:
            delattr(model, attr)
        except Exception:
            pass


def network_mxfp8_is_model_prepared(model=None):
    model = model or getattr(shared, "sd_model", None)
    if model is None:
        return False
    signature = getattr(model, "network_mxfp8_active_config_signature", None)
    return signature is not None and signature == network_mxfp8_active_config_signature()


network_mxfp8_missing = object()


def network_mxfp8_snapshot_state(module):
    return (
        module,
        module._parameters.get("weight"),
        module._parameters.get("bias"),
        getattr(module, "network_current_names", network_mxfp8_missing),
        getattr(module, "network_mxfp8_merged_lora_applied", network_mxfp8_missing),
    )


def network_mxfp8_restore_attr(module, attr, value):
    if value is network_mxfp8_missing:
        try:
            delattr(module, attr)
        except Exception:
            pass
    else:
        setattr(module, attr, value)


def network_mxfp8_restore_state(snapshot):
    module, weight, bias, current_names, merged_lora = snapshot
    if weight is not None:
        module._parameters["weight"] = weight
    if bias is not None:
        module._parameters["bias"] = bias
    elif "bias" in module._parameters:
        module._parameters["bias"] = None
    network_mxfp8_restore_attr(module, "network_current_names", current_names)
    network_mxfp8_restore_attr(module, "network_mxfp8_merged_lora_applied", merged_lora)


def prepare_mxfp8_active_config():
    """Prepare the one active in-memory MXFP8+LoRA weight configuration.

    This is deliberately a model-level transaction. The hot Linear.forward()
    path must not merge LoRAs, scan active LoRAs, or quantize weights during
    sampling. For the current checkpoint + coverage + active LoRA signature,
    this rebuilds managed Linear layers from immutable BF16 master weights,
    applies active LoRA deltas once, quantizes the final effective weights to
    MXFP8 once, and marks the model prepared.
    """

    model = getattr(shared, "sd_model", None)
    if model is None:
        return False

    if not getattr(devices, "mxfp8", False):
        network_mxfp8_mark_model_unprepared(model)
        return True

    managed_modules = [(fqn, module) for fqn, module in model.named_modules() if getattr(module, "network_mxfp8_base_weight", None) is not None]
    if not managed_modules:
        network_mxfp8_mark_model_unprepared(model)
        return True

    signature = network_mxfp8_active_config_signature()
    if getattr(model, "network_mxfp8_active_config_signature", None) == signature:
        return True

    network_mxfp8_mark_model_unprepared(model)

    wanted_names = network_mxfp8_wanted_names()
    from torchao.quantization import quantize_
    quantize_config = mxfp8_config.get_mxfp8_config()
    mxfp8_config.validate_kernel_preference(quantize_config)
    quantize_fn = quantize_

    prepared = 0
    quantized = 0
    untouched = 0
    failed = 0
    failures = []
    snapshots = []

    for fqn, module in managed_modules:
        snapshots.append(network_mxfp8_snapshot_state(module))
        if network_apply_mxfp8_merged_lora(module, quantize_config=quantize_config, quantize_fn=quantize_fn):
            prepared += 1
            if is_mxfp8_weight(getattr(module, "weight", None)):
                quantized += 1
            else:
                untouched += 1
            module.network_current_names = wanted_names
        else:
            failed += 1
            failures.append(getattr(module, "network_layer_name", fqn))

    stats = {
        "signature": signature,
        "prepared_linear": prepared,
        "quantized_linear": quantized,
        "untouched_linear": untouched,
        "failed_linear": failed,
        "failed_layers": failures[:50],
        "active_lora_count": len(loaded_networks),
        "mxfp8_linear_coverage": sorted(getattr(shared.opts, "mxfp8_linear_coverage", ()) or ()),
    }
    model.network_mxfp8_prepare_stats = stats

    if failed == 0:
        model.network_mxfp8_active_config_signature = signature
        try:
            delattr(model, "network_mxfp8_prepare_error")
        except Exception:
            pass
        if prepared:
            print(
                "Prepared active MXFP8 LoRA config: "
                f"prepared {prepared} Linear, quantized {quantized}, "
                f"untouched {untouched}, "
                f"LoRAs {len(loaded_networks)}",
                flush=True,
            )
        return True

    for snapshot in reversed(snapshots):
        network_mxfp8_restore_state(snapshot)

    message = f"failed to prepare active MXFP8 LoRA config for {failed} Linear modules: {failures[:10]}"
    model.network_mxfp8_prepare_error = message
    logging.warning(message)
    return False


def network_mxfp8_lora_ops_for_layer(self, network_layer_name):
    """Return LoRA operations for an MXFP8-managed Linear.

    This mirrors A1111's normal direct-module and SD3 QkvLinear split q/k/v
    handling closely enough that MXFP8-managed layers do not silently mark a
    LoRA set current while skipping a supported split projection mutation.
    """
    ops = []
    unsupported = []
    for net in loaded_networks:
        module = net.modules.get(network_layer_name, None)
        if module is not None:
            ops.append(("direct", net, module))
            continue

        module_q = net.modules.get(network_layer_name + "_q_proj", None)
        module_k = net.modules.get(network_layer_name + "_k_proj", None)
        module_v = net.modules.get(network_layer_name + "_v_proj", None)
        module_out = net.modules.get(network_layer_name + "_out_proj", None)

        if isinstance(self, modules.models.sd3.mmdit.QkvLinear) and module_q and module_k and module_v and module_out is None:
            ops.append(("qkv", net, (module_q, module_k, module_v)))
            continue

        if module_q or module_k or module_v or module_out:
            unsupported.append((net, tuple(name for name, value in (("q_proj", module_q), ("k_proj", module_k), ("v_proj", module_v), ("out_proj", module_out)) if value is not None)))

    return ops, unsupported


def network_apply_mxfp8_merged_lora(self, quantize_config=None, quantize_fn=None):
    network_layer_name = getattr(self, 'network_layer_name', None)
    base_weight = getattr(self, 'network_mxfp8_base_weight', None)
    if network_layer_name is None or base_weight is None:
        return False

    wanted_names = network_mxfp8_wanted_names()
    ops_for_layer, unsupported_ops = network_mxfp8_lora_ops_for_layer(self, network_layer_name)

    original_snapshot = network_mxfp8_snapshot_state(self)

    try:
        if unsupported_ops:
            details = ", ".join(f"{net.name}:{'/'.join(parts)}" for net, parts in unsupported_ops)
            raise RuntimeError(f"unsupported MXFP8 LoRA split projection target(s): {details}")

        with torch.no_grad():
            weight = base_weight.to(device=devices.device, dtype=torch.bfloat16)
            base_bias = getattr(self, 'network_mxfp8_base_bias', None)
            bias = base_bias.to(device=devices.device, dtype=torch.bfloat16) if base_bias is not None else None

            for op_kind, net, payload in ops_for_layer:
                if op_kind == "direct":
                    module = payload
                    updown, ex_bias = module.calc_updown(weight)
                    if len(weight.shape) == 4 and weight.shape[1] == 9:
                        updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
                    weight = (weight.to(dtype=updown.dtype) + updown).to(dtype=torch.bfloat16)
                    if ex_bias is not None:
                        bias = ex_bias.to(device=devices.device, dtype=torch.bfloat16) if bias is None else (bias + ex_bias).to(dtype=torch.bfloat16)
                    continue

                if op_kind == "qkv":
                    module_q, module_k, module_v = payload
                    qw, kw, vw = weight.chunk(3, 0)
                    updown_q, _ = module_q.calc_updown(qw)
                    updown_k, _ = module_k.calc_updown(kw)
                    updown_v, _ = module_v.calc_updown(vw)
                    del qw, kw, vw
                    updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                    weight = (weight.to(dtype=updown_qkv.dtype) + updown_qkv).to(dtype=torch.bfloat16)
                    continue

                raise RuntimeError(f"unsupported MXFP8 LoRA operation kind: {op_kind}")

            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            if bias is not None:
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            elif self.bias is not None:
                self.bias = None

            if quantize_config is None or quantize_fn is None:
                from torchao.quantization import quantize_
                quantize_config = mxfp8_config.get_mxfp8_config()
                mxfp8_config.validate_kernel_preference(quantize_config)
                quantize_fn = quantize_
            quantize_fn(self, quantize_config, filter_fn=lambda module, fqn: module is self, device=devices.device)
            self.network_current_names = wanted_names
            self.network_mxfp8_merged_lora_applied = bool(ops_for_layer)
            return True
    except Exception as e:
        # Restore the pre-call MXFP8/base state before reporting preparation
        # failure so callers never see partially merged effective weights.
        network_mxfp8_restore_state(original_snapshot)
        logging.debug(f"Network {network_layer_name}: MXFP8 merged LoRA failed: {e}", exc_info=True)
        for op_kind, net, _payload in ops_for_layer:
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
        for net, _parts in unsupported_ops:
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
        return False


def is_nvfp4_weight(weight):
    return type(weight).__name__ == "NVFP4Tensor" and type(weight).__module__.startswith("torchao.")


def network_nvfp4_wanted_names():
    return tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)


def network_nvfp4_active_config_signature():
    """Return the currently active in-memory NVFP4+LoRA weight signature.

    This intentionally describes only the one active configuration, not a disk
    cache key for every possible LoRA permutation. If any field that changes
    effective weights changes, eager preparation invalidates and rebuilds the
    model's in-memory prepared state.
    """

    checkpoint_info = getattr(shared.sd_model, "sd_checkpoint_info", None)
    checkpoint_key = (
        getattr(checkpoint_info, "filename", None),
        getattr(checkpoint_info, "hash", None),
        getattr(checkpoint_info, "sha256", None),
    )
    coverage = tuple(sorted(getattr(shared.opts, "nvfp4_linear_coverage", ()) or ()))
    config_name = nvfp4_config.CONFIG_NAME

    loras = []
    for net in loaded_networks:
        network_on_disk = getattr(net, "network_on_disk", None)
        filename = getattr(network_on_disk, "filename", None)
        try:
            mtime = os.path.getmtime(filename) if filename else None
        except OSError:
            mtime = None
        loras.append((
            net.name,
            net.te_multiplier,
            net.unet_multiplier,
            net.dyn_dim,
            filename,
            getattr(network_on_disk, "shorthash", None),
            mtime,
        ))

    return (checkpoint_key, coverage, config_name, tuple(loras))


def network_nvfp4_mark_model_unprepared(model=None):
    model = model or getattr(shared, "sd_model", None)
    if model is None:
        return
    for attr in (
        "network_nvfp4_active_config_signature",
        "network_nvfp4_prepare_stats",
        "network_nvfp4_prepare_error",
    ):
        try:
            delattr(model, attr)
        except Exception:
            pass


def network_nvfp4_is_model_prepared(model=None):
    model = model or getattr(shared, "sd_model", None)
    if model is None:
        return False
    signature = getattr(model, "network_nvfp4_active_config_signature", None)
    return signature is not None and signature == network_nvfp4_active_config_signature()


network_nvfp4_missing = object()


def network_nvfp4_snapshot_state(module):
    return (
        module,
        module._parameters.get("weight"),
        module._parameters.get("bias"),
        getattr(module, "network_current_names", network_nvfp4_missing),
        getattr(module, "network_nvfp4_merged_lora_applied", network_nvfp4_missing),
    )


def network_nvfp4_restore_attr(module, attr, value):
    if value is network_nvfp4_missing:
        try:
            delattr(module, attr)
        except Exception:
            pass
    else:
        setattr(module, attr, value)


def network_nvfp4_restore_state(snapshot):
    module, weight, bias, current_names, merged_lora = snapshot
    if weight is not None:
        module._parameters["weight"] = weight
    if bias is not None:
        module._parameters["bias"] = bias
    elif "bias" in module._parameters:
        module._parameters["bias"] = None
    network_nvfp4_restore_attr(module, "network_current_names", current_names)
    network_nvfp4_restore_attr(module, "network_nvfp4_merged_lora_applied", merged_lora)


def prepare_nvfp4_active_config():
    """Prepare the one active in-memory NVFP4+LoRA weight configuration.

    This is deliberately a model-level transaction. The hot Linear.forward()
    path must not merge LoRAs, scan active LoRAs, or quantize weights during
    sampling. For the current checkpoint + coverage + active LoRA signature,
    this rebuilds managed Linear layers from immutable BF16 master weights,
    applies active LoRA deltas once, quantizes the final effective weights to
    NVFP4 once, and marks the model prepared.
    """

    model = getattr(shared, "sd_model", None)
    if model is None:
        return False

    if not getattr(devices, "nvfp4", False):
        network_nvfp4_mark_model_unprepared(model)
        return True

    managed_modules = [(fqn, module) for fqn, module in model.named_modules() if getattr(module, "network_nvfp4_base_weight", None) is not None]
    if not managed_modules:
        network_nvfp4_mark_model_unprepared(model)
        return True

    signature = network_nvfp4_active_config_signature()
    if getattr(model, "network_nvfp4_active_config_signature", None) == signature:
        return True

    network_nvfp4_mark_model_unprepared(model)

    wanted_names = network_nvfp4_wanted_names()
    from torchao.quantization import quantize_
    quantize_config = nvfp4_config.get_nvfp4_config()
    nvfp4_config.validate_config(quantize_config)
    quantize_fn = quantize_

    prepared = 0
    quantized = 0
    untouched = 0
    failed = 0
    failures = []
    snapshots = []

    for fqn, module in managed_modules:
        snapshots.append(network_nvfp4_snapshot_state(module))
        if network_apply_nvfp4_merged_lora(module, quantize_config=quantize_config, quantize_fn=quantize_fn):
            prepared += 1
            if is_nvfp4_weight(getattr(module, "weight", None)):
                quantized += 1
            else:
                untouched += 1
            module.network_current_names = wanted_names
        else:
            failed += 1
            failures.append(getattr(module, "network_layer_name", fqn))

    stats = {
        "signature": signature,
        "prepared_linear": prepared,
        "quantized_linear": quantized,
        "untouched_linear": untouched,
        "failed_linear": failed,
        "failed_layers": failures[:50],
        "active_lora_count": len(loaded_networks),
        "nvfp4_linear_coverage": sorted(getattr(shared.opts, "nvfp4_linear_coverage", ()) or ()),
    }
    model.network_nvfp4_prepare_stats = stats

    if failed == 0:
        model.network_nvfp4_active_config_signature = signature
        try:
            delattr(model, "network_nvfp4_prepare_error")
        except Exception:
            pass
        if prepared:
            print(
                "Prepared active NVFP4 LoRA config: "
                f"prepared {prepared} Linear, quantized {quantized}, "
                f"untouched {untouched}, "
                f"LoRAs {len(loaded_networks)}",
                flush=True,
            )
        return True

    for snapshot in reversed(snapshots):
        network_nvfp4_restore_state(snapshot)

    message = f"failed to prepare active NVFP4 LoRA config for {failed} Linear modules: {failures[:10]}"
    model.network_nvfp4_prepare_error = message
    logging.warning(message)
    return False


def network_nvfp4_lora_ops_for_layer(self, network_layer_name):
    """Return LoRA operations for an NVFP4-managed Linear.

    This mirrors A1111's normal direct-module and SD3 QkvLinear split q/k/v
    handling closely enough that NVFP4-managed layers do not silently mark a
    LoRA set current while skipping a supported split projection mutation.
    """
    ops = []
    unsupported = []
    for net in loaded_networks:
        module = net.modules.get(network_layer_name, None)
        if module is not None:
            ops.append(("direct", net, module))
            continue

        module_q = net.modules.get(network_layer_name + "_q_proj", None)
        module_k = net.modules.get(network_layer_name + "_k_proj", None)
        module_v = net.modules.get(network_layer_name + "_v_proj", None)
        module_out = net.modules.get(network_layer_name + "_out_proj", None)

        if isinstance(self, modules.models.sd3.mmdit.QkvLinear) and module_q and module_k and module_v and module_out is None:
            ops.append(("qkv", net, (module_q, module_k, module_v)))
            continue

        if module_q or module_k or module_v or module_out:
            unsupported.append((net, tuple(name for name, value in (("q_proj", module_q), ("k_proj", module_k), ("v_proj", module_v), ("out_proj", module_out)) if value is not None)))

    return ops, unsupported


def network_apply_nvfp4_merged_lora(self, quantize_config=None, quantize_fn=None):
    network_layer_name = getattr(self, 'network_layer_name', None)
    base_weight = getattr(self, 'network_nvfp4_base_weight', None)
    if network_layer_name is None or base_weight is None:
        return False

    wanted_names = network_nvfp4_wanted_names()
    ops_for_layer, unsupported_ops = network_nvfp4_lora_ops_for_layer(self, network_layer_name)

    original_snapshot = network_nvfp4_snapshot_state(self)

    try:
        if unsupported_ops:
            details = ", ".join(f"{net.name}:{'/'.join(parts)}" for net, parts in unsupported_ops)
            raise RuntimeError(f"unsupported NVFP4 LoRA split projection target(s): {details}")

        with torch.no_grad():
            weight = base_weight.to(device=devices.device, dtype=torch.bfloat16)
            base_bias = getattr(self, 'network_nvfp4_base_bias', None)
            bias = base_bias.to(device=devices.device, dtype=torch.bfloat16) if base_bias is not None else None

            for op_kind, net, payload in ops_for_layer:
                if op_kind == "direct":
                    module = payload
                    updown, ex_bias = module.calc_updown(weight)
                    if len(weight.shape) == 4 and weight.shape[1] == 9:
                        updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))
                    weight = (weight.to(dtype=updown.dtype) + updown).to(dtype=torch.bfloat16)
                    if ex_bias is not None:
                        bias = ex_bias.to(device=devices.device, dtype=torch.bfloat16) if bias is None else (bias + ex_bias).to(dtype=torch.bfloat16)
                    continue

                if op_kind == "qkv":
                    module_q, module_k, module_v = payload
                    qw, kw, vw = weight.chunk(3, 0)
                    updown_q, _ = module_q.calc_updown(qw)
                    updown_k, _ = module_k.calc_updown(kw)
                    updown_v, _ = module_v.calc_updown(vw)
                    del qw, kw, vw
                    updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                    weight = (weight.to(dtype=updown_qkv.dtype) + updown_qkv).to(dtype=torch.bfloat16)
                    continue

                raise RuntimeError(f"unsupported NVFP4 LoRA operation kind: {op_kind}")

            self.weight = torch.nn.Parameter(weight, requires_grad=False)
            if bias is not None:
                self.bias = torch.nn.Parameter(bias, requires_grad=False)
            elif self.bias is not None:
                self.bias = None

            if quantize_config is None or quantize_fn is None:
                from torchao.quantization import quantize_
                quantize_config = nvfp4_config.get_nvfp4_config()
                nvfp4_config.validate_config(quantize_config)
                quantize_fn = quantize_
            quantize_fn(self, quantize_config, filter_fn=lambda module, fqn: module is self, device=devices.device)
            self.network_current_names = wanted_names
            self.network_nvfp4_merged_lora_applied = bool(ops_for_layer)
            return True
    except Exception as e:
        # Restore the pre-call NVFP4/base state before reporting preparation
        # failure so callers never see partially merged effective weights.
        network_nvfp4_restore_state(original_snapshot)
        logging.debug(f"Network {network_layer_name}: NVFP4 merged LoRA failed: {e}", exc_info=True)
        for op_kind, net, _payload in ops_for_layer:
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
        for net, _parts in unsupported_ops:
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
        return False

def network_Linear_forward(self, input):
    if getattr(self, 'network_mxfp8_base_weight', None) is not None:
        # MXFP8-managed LoRA weights must be prepared once per active config,
        # outside the sampling hot path. If a caller reaches forward before
        # ExtraNetworkLora.activate() prepared the model, prepare the whole
        # model once here; never fall back to per-layer functional LoRA in this
        # path because that reintroduces LoRA-count-sensitive step time.
        model = getattr(shared, "sd_model", None)
        if not network_mxfp8_is_model_prepared(model):
            if not prepare_mxfp8_active_config():
                message = getattr(model, "network_mxfp8_prepare_error", "MXFP8 LoRA active config is not prepared")
                raise RuntimeError(message)
        return originals.Linear_forward(self, input)

    if getattr(self, 'network_nvfp4_base_weight', None) is not None:
        # NVFP4 mirrors the MXFP8 active-config transaction: merge active LoRA
        # deltas into BF16 master weights once, quantize once, and keep the
        # sampling hot path free of per-step LoRA merging/quantization.
        model = getattr(shared, "sd_model", None)
        if not network_nvfp4_is_model_prepared(model):
            if not prepare_nvfp4_active_config():
                message = getattr(model, "network_nvfp4_prepare_error", "NVFP4 LoRA active config is not prepared")
                raise RuntimeError(message)
        return originals.Linear_forward(self, input)

    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Linear_forward)

    network_apply_weights(self)

    return originals.Linear_forward(self, input)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Linear_load_state_dict(self, *args, **kwargs)


def network_Conv2d_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.Conv2d_forward)

    network_apply_weights(self)

    return originals.Conv2d_forward(self, input)


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


def network_GroupNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.GroupNorm_forward)

    network_apply_weights(self)

    return originals.GroupNorm_forward(self, input)


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


def network_LayerNorm_forward(self, input):
    if shared.opts.lora_functional:
        return network_forward(self, input, originals.LayerNorm_forward)

    network_apply_weights(self)

    return originals.LayerNorm_forward(self, input)


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)

    return originals.MultiheadAttention_forward(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)

    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


def process_network_files(names: list[str] | None = None):
    candidates = list(shared.walk_files(shared.cmd_opts.lora_dir, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    candidates += list(shared.walk_files(shared.cmd_opts.lyco_dir_backcompat, allowed_extensions=[".pt", ".ckpt", ".safetensors"]))
    candidates = [x for x in candidates if not sd_models.mxfp8_model_cache.is_mxfp8_cache_path(x)]
    candidates = [x for x in candidates if not sd_models.nvfp4_model_cache.is_nvfp4_cache_path(x)]
    for filename in candidates:
        if os.path.isdir(filename):
            continue
        name = os.path.splitext(os.path.basename(filename))[0]
        # if names is provided, only load networks with names in the list
        if names and name not in names:
            continue
        try:
            entry = network.NetworkOnDisk(name, filename)
        except OSError:  # should catch FileNotFoundError and PermissionError etc.
            errors.report(f"Failed to load network {name} from {filename}", exc_info=True)
            continue

        available_networks[name] = entry

        if entry.alias in available_network_aliases:
            forbidden_network_aliases[entry.alias.lower()] = 1

        available_network_aliases[name] = entry
        available_network_aliases[entry.alias] = entry


def update_available_networks_by_names(names: list[str]):
    process_network_files(names)


def list_available_networks():
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})

    os.makedirs(shared.cmd_opts.lora_dir, exist_ok=True)

    process_network_files()


re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def infotext_pasted(infotext, params):
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything

    added = []

    for k in params:
        if not k.startswith("AddNet Model "):
            continue

        num = k[13:]

        if params.get("AddNet Module " + num) != "LoRA":
            continue

        name = params.get("AddNet Model " + num)
        if name is None:
            continue

        m = re_network_name.match(name)
        if m:
            name = m.group(1)

        multiplier = params.get("AddNet Weight A " + num, "1.0")

        added.append(f"<lora:{name}:{multiplier}>")

    if added:
        params["Prompt"] += "\n" + "".join(added)


originals: lora_patches.LoraPatches = None

extra_network_lora = None

available_networks = {}
available_network_aliases = {}
loaded_networks = []
loaded_bundle_embeddings = {}
networks_in_memory = {}
available_network_hash_lookup = {}
forbidden_network_aliases = {}

list_available_networks()
