#!/usr/bin/env python3
"""Patch ControlNet hook lifecycle invariants."""

from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
init_old = '''        self.current_uc_indices = []
        self.current_c_indices = []
        self.is_in_high_res_fix = False
'''
init_new = '''        self.current_uc_indices = []
        self.current_c_indices = []
        self.is_in_high_res_fix = False
        self._forward_hook_installed = False
'''
forward_old = (
    "        def forward_webui(*args, **kwargs):\n"
    "            # webui will handle other compoments \n"
    "            try:\n"
)
forward_new = (
    "        def forward_webui(*args, **kwargs):\n"
    "            # webui will handle other compoments \n"
    "            # A stale hook must behave exactly like inactive ControlNet.  This\n"
    "            # also closes the tiny restore/reconfigure window in API workers.\n"
    "            if outer.control_params is None:\n"
    "                return outer.original_forward(*args, **kwargs)\n"
    "            try:\n"
)
hook_install_old = '''        model._original_forward = model.forward
        outer.original_forward = model.forward
        model.forward = forward_webui.__get__(model, UNetModel)
'''
hook_install_new = '''        original_forward = getattr(model, "_original_forward", None)
        if original_forward is not None:
            # The UNet is already wrapped by a ControlNet hook. Reusing the
            # baseline forward avoids self-chaining forward_webui wrappers when
            # process hooks are applied repeatedly before postprocess restores.
            outer.original_forward = original_forward
            outer._forward_hook_installed = False
        else:
            model._original_forward = model.forward
            outer.original_forward = model.forward
            model.forward = forward_webui.__get__(model, UNetModel)
            outer._forward_hook_installed = True
'''
restore_old = '''    def restore(self):
        scripts.script_callbacks.remove_callbacks_for_function(self.guidance_schedule_handler)
        self.control_params = None

        if self.model is not None:
            if hasattr(self.model, "_original_forward"):
                self.model.forward = self.model._original_forward
                del self.model._original_forward
'''
restore_new = '''    def restore(self):
        scripts.script_callbacks.remove_callbacks_for_function(self.guidance_schedule_handler)

        if self.model is not None:
            if self._forward_hook_installed and hasattr(self.model, "_original_forward"):
                self.model.forward = self.model._original_forward
                del self.model._original_forward
            self._forward_hook_installed = False

        # Clear parameters only after the UNet hook is detached.  A warmed API
        # process can otherwise leave forward_webui installed with no params,
        # making the next generation fail before ControlNet is reconfigured.
        self.control_params = None
'''
changed = []
if init_new not in source:
    if init_old not in source:
        raise SystemExit(f"unsupported ControlNet hook init implementation: {path}")
    source = source.replace(init_old, init_new, 1)
    changed.append("hook owner state")
if forward_new not in source:
    if forward_old not in source:
        raise SystemExit(f"unsupported ControlNet forward_webui implementation: {path}")
    source = source.replace(forward_old, forward_new, 1)
    changed.append("inactive-hook guard")
if hook_install_new not in source:
    if hook_install_old not in source:
        raise SystemExit(f"unsupported ControlNet hook install implementation: {path}")
    source = source.replace(hook_install_old, hook_install_new, 1)
    changed.append("idempotent hook install")
if restore_new not in source:
    if restore_old not in source:
        raise SystemExit(f"unsupported ControlNet hook restore implementation: {path}")
    source = source.replace(restore_old, restore_new, 1)
    changed.append("restore ordering")
if changed:
    path.write_text(source, encoding="utf-8")
    print(f"Patched ControlNet {', '.join(changed)}: {path}")
else:
    print(f"ControlNet lifecycle patch already present: {path}")
