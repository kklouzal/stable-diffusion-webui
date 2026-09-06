#!/usr/bin/env python3
"""Patch and verify ControlNet UNet hook ownership invariants."""
from __future__ import annotations

import argparse
from pathlib import Path

OWNER_ATTR = "_controlnet_forward_hook_owner"
BASELINE_ATTR = "_controlnet_forward_hook_baseline"
MARKER = "OPENCLAW_CONTROLNET_FORWARD_OWNER_V2"

INIT_ORIGINAL = '''        self.current_uc_indices = []
        self.current_c_indices = []
        self.is_in_high_res_fix = False
'''
INIT_PATCHED = '''        self.current_uc_indices = []
        self.current_c_indices = []
        self.is_in_high_res_fix = False
        # OPENCLAW_CONTROLNET_FORWARD_OWNER_V2: this instance owns only the
        # wrapper carrying this opaque token. Never restore another owner.
        self._forward_hook_owner_token = object()
        self._forward_hook_wrapper = None
'''

FORWARD_ORIGINAL = '''        def forward_webui(*args, **kwargs):
            # webui will handle other compoments
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                return forward(*args, **kwargs)
'''
FORWARD_OLD_PATCH = '''        def forward_webui(*args, **kwargs):
            # webui will handle other compoments
            # A stale hook must behave exactly like inactive ControlNet.  This
            # also closes the tiny restore/reconfigure window in API workers.
            if outer.control_params is None:
                return outer.original_forward(*args, **kwargs)
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                return forward(*args, **kwargs)
'''
FORWARD_PATCHED = '''        def forward_webui(self, x, timesteps=None, context=None, y=None, **kwargs):
            # webui will handle other compoments
            # Disabled/no-unit execution calls the captured bound baseline
            # without forwarding the wrapper's bound self a second time.
            if not outer.control_params:
                return outer.original_forward(x, timesteps=timesteps, context=context, y=y, **kwargs)
            try:
                if shared.cmd_opts.lowvram:
                    lowvram.send_everything_to_cpu()
                return forward(self, x, timesteps=timesteps, context=context, y=y, **kwargs)
'''

INSTALL_ORIGINAL = '''        model._original_forward = model.forward
        outer.original_forward = model.forward
        model.forward = forward_webui.__get__(model, UNetModel)
'''
INSTALL_OLD_PATCH = '''        original_forward = getattr(model, "_original_forward", None)
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
INSTALL_PATCHED = '''        active_owner = getattr(model, "_controlnet_forward_hook_owner", None)
        active_wrapper = getattr(model, "_controlnet_forward_hook_wrapper", None)
        if active_owner is outer._forward_hook_owner_token:
            # hook() may be called twice for one owner, but never wrap or
            # overwrite a callable another actor installed above our wrapper.
            if model.forward is not active_wrapper:
                raise RuntimeError("ControlNet UNet forward changed while this hook still owns it")
            outer.original_forward = model._controlnet_forward_hook_baseline
            outer._forward_hook_wrapper = active_wrapper
        elif active_owner is not None:
            raise RuntimeError("ControlNet UNet forward hook is owned by another live hook")
        else:
            outer.original_forward = model.forward
            wrapper = forward_webui.__get__(model, UNetModel)
            model._controlnet_forward_hook_baseline = outer.original_forward
            model._controlnet_forward_hook_owner = outer._forward_hook_owner_token
            model._controlnet_forward_hook_wrapper = wrapper
            outer._forward_hook_wrapper = wrapper
            model.forward = wrapper
'''

RESTORE_ORIGINAL = '''    def restore(self):
        scripts.script_callbacks.remove_callbacks_for_function(self.guidance_schedule_handler)
        self.control_params = None

        if self.model is not None:
            if hasattr(self.model, "_original_forward"):
                self.model.forward = self.model._original_forward
                del self.model._original_forward
'''
RESTORE_OLD_PATCH = '''    def restore(self):
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
RESTORE_LIVE_PARTIAL = '''    def restore(self):
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
RESTORE_PATCHED = '''    def restore(self):
        scripts.script_callbacks.remove_callbacks_for_function(self.guidance_schedule_handler)

        model = self.model
        if model is not None and getattr(model, "_controlnet_forward_hook_owner", None) is self._forward_hook_owner_token:
            wrapper = getattr(model, "_controlnet_forward_hook_wrapper", None)
            # Only detach the wrapper installed by this owner. If another actor
            # changed model.forward, leave that live callable untouched.
            if model.forward is wrapper and wrapper is self._forward_hook_wrapper:
                model.forward = model._controlnet_forward_hook_baseline
                del model._controlnet_forward_hook_baseline
                del model._controlnet_forward_hook_owner
                del model._controlnet_forward_hook_wrapper
        self._forward_hook_wrapper = None
        self.control_params = None
'''


def restore_is_patched(source: str) -> bool:
    """Accept the owner-aware restore block with extension-specific cleanup."""
    return all(fragment in source for fragment in (
        "    def restore(self):",
        'getattr(model, "_controlnet_forward_hook_owner", None) is self._forward_hook_owner_token',
        "model._controlnet_forward_hook_baseline",
        "model._controlnet_forward_hook_wrapper",
        "self._forward_hook_wrapper = None",
        "self.control_params = None",
    ))


def replace_one(source: str, old_options: tuple[str, ...], new: str, label: str) -> tuple[str, bool]:
    if new in source:
        return source, False
    matches = [old for old in old_options if old in source]
    if len(matches) != 1:
        raise SystemExit(f"unsupported or partial ControlNet {label} implementation")
    return source.replace(matches[0], new, 1), True


def verify(source: str, path: Path) -> None:
    required = (INIT_PATCHED, FORWARD_PATCHED, INSTALL_PATCHED)
    if not all(block in source for block in required) or not restore_is_patched(source):
        raise SystemExit(f"ControlNet lifecycle verification failed (partial markers): {path}")
    forbidden = ("model._original_forward = model.forward", "self._forward_hook_installed")
    if any(text in source for text in forbidden):
        raise SystemExit(f"ControlNet lifecycle verification failed (legacy ownership): {path}")
    if source.count(MARKER) != 1:
        raise SystemExit(f"ControlNet lifecycle verification failed (marker count): {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.path.read_text(encoding="utf-8")
    if not args.check:
        changes = []
        for old, new, label in (
            ((INIT_ORIGINAL,), INIT_PATCHED, "owner state"),
            (
                (
                    FORWARD_ORIGINAL,
                    FORWARD_ORIGINAL.replace("compoments\n", "compoments \n"),
                    FORWARD_OLD_PATCH,
                    FORWARD_OLD_PATCH.replace("compoments\n", "compoments \n"),
                ),
                FORWARD_PATCHED,
                "inactive guard",
            ),
            ((INSTALL_ORIGINAL, INSTALL_OLD_PATCH), INSTALL_PATCHED, "install"),
            ((RESTORE_ORIGINAL, RESTORE_OLD_PATCH, RESTORE_LIVE_PARTIAL), RESTORE_PATCHED, "restore"),
        ):
            if label == "restore" and restore_is_patched(source):
                continue
            source, changed = replace_one(source, old, new, label)
            if changed:
                changes.append(label)
        if changes:
            args.path.write_text(source, encoding="utf-8")
            print(f"Patched ControlNet {', '.join(changes)}: {args.path}")
    verify(source, args.path)
    print(f"ControlNet lifecycle verified: {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
