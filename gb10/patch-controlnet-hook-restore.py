#!/usr/bin/env python3
"""Patch ControlNet inactive-hook fallback and restore ordering."""

from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
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
            if hasattr(self.model, "_original_forward"):
                self.model.forward = self.model._original_forward
                del self.model._original_forward

        # Clear parameters only after the UNet hook is detached.  A warmed API
        # process can otherwise leave forward_webui installed with no params,
        # making the next generation fail before ControlNet is reconfigured.
        self.control_params = None
'''
changed = []
if forward_new not in source:
    if forward_old not in source:
        raise SystemExit(f"unsupported ControlNet forward_webui implementation: {path}")
    source = source.replace(forward_old, forward_new, 1)
    changed.append("inactive-hook guard")
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
