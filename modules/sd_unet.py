import sys

import torch.nn

from modules import script_callbacks, shared, devices, sd_unet_row_memo

unet_options = []
current_unet_option = None
current_unet = None
original_forward = None  # not used, only left temporarily for compatibility

def list_unets():
    new_unets = script_callbacks.list_unets_callback()

    unet_options.clear()
    unet_options.extend(new_unets)


def get_unet_option(option=None):
    option = option or shared.opts.sd_unet

    if option == "None":
        return None

    if option == "Automatic":
        name = shared.sd_model.sd_checkpoint_info.model_name

        options = [x for x in unet_options if x.model_name == name]

        option = options[0].label if options else "None"

    return next(iter([x for x in unet_options if x.label == option]), None)


def apply_unet(option=None):
    global current_unet_option
    global current_unet

    new_option = get_unet_option(option)
    if new_option == current_unet_option:
        return

    if current_unet is not None:
        print(f"Dectivating unet: {current_unet.option.label}")
        current_unet.deactivate()

    current_unet_option = new_option
    if current_unet_option is None:
        current_unet = None

        if not shared.sd_model.lowvram:
            shared.sd_model.model.diffusion_model.to(devices.device)

        return

    shared.sd_model.model.diffusion_model.to(devices.cpu)
    devices.torch_gc()

    current_unet = current_unet_option.create_unet()
    current_unet.option = current_unet_option
    print(f"Activating unet: {current_unet.option.label}")
    current_unet.activate()


class SdUnetOption:
    model_name = None
    """name of related checkpoint - this option will be selected automatically for unet if the name of checkpoint matches this"""

    label = None
    """name of the unet in UI"""

    def create_unet(self):
        """returns SdUnet object to be used as a Unet instead of built-in unet when making pictures"""
        raise NotImplementedError()


class SdUnet(torch.nn.Module):
    def forward(self, x, timesteps, context, *args, **kwargs):
        raise NotImplementedError()

    def activate(self):
        pass

    def deactivate(self):
        pass


def create_unet_forward(original_forward):
    # Only the sgm (SDXL) UNet records/replays encoder rows for PAG (modules/sd_unet_row_memo.py).
    sgm_openaimodel = sys.modules.get(original_forward.__module__) if original_forward.__module__ == "sgm.modules.diffusionmodules.openaimodel" else None

    def UNetModel_forward(self, x, timesteps=None, context=None, *args, **kwargs):
        if current_unet is not None:
            return current_unet.forward(x, timesteps, context, *args, **kwargs)

        memo_slot = sd_unet_row_memo.claim() if sgm_openaimodel is not None else None
        if memo_slot is not None and not args and set(kwargs) <= {"y"}:
            return sgm_forward_with_row_memo(sgm_openaimodel, self, memo_slot, x, timesteps, context, **kwargs)

        return original_forward(self, x, timesteps, context, *args, **kwargs)

    return UNetModel_forward


def sgm_forward_with_row_memo(openaimodel, unet, memo_slot, x, timesteps=None, context=None, y=None):
    """sgm ``UNetModel.forward`` for a call recorded for, or replayed by, PAG's hidden pass.

    Op-for-op the generative-models forward (openaimodel.py), so a recorded main-pass call is bitwise
    unchanged; it only keeps the encoder outputs. A replay of the call's rows resumes at the middle block,
    where PAG's perturbation starts, from those outputs. ``th`` and ``timestep_embedding`` are looked up on
    the module per call because the WebUI hijacks replace them there.
    """
    owner = ("sgm", unet)
    prefix = sd_unet_row_memo.replay_prefix(memo_slot, owner, x, context, y) if memo_slot.replay else None
    if prefix is not None:
        m = x.shape[0]
        hs = [sd_unet_row_memo.rows_of(t, prefix.rows, m) for t in prefix.state["hs"]]
        emb = sd_unet_row_memo.rows_of(prefix.state["emb"], prefix.rows, m)
        h = hs[-1]
    else:
        assert (y is not None) == (unet.num_classes is not None), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = openaimodel.timestep_embedding(timesteps, unet.model_channels, repeat_only=False)
        emb = unet.time_embed(t_emb)

        if unet.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + unet.label_emb(y)

        h = x
        for module in unet.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        # Hypertile draws tile layouts per wrapped encoder layer call; a replay must keep those draws.
        if sd_unet_row_memo.can_record(memo_slot, x) and not sd_unet_row_memo.hypertile_unet_enabled(getattr(shared.sd_model, "model", None)):
            sd_unet_row_memo.attach_prefix(memo_slot, sd_unet_row_memo.Prefix(owner, x, context, y, {"hs": list(hs), "emb": emb}))

    h = unet.middle_block(h, emb, context)
    for module in unet.output_blocks:
        h = openaimodel.th.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context)
    h = h.type(x.dtype)

    return unet.out(h)

