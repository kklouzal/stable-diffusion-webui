#!/usr/bin/env python3
"""Patch and verify Detail Daemon: a daemon whose schedule would drive the model sigma to <= 0 fails the request.

Exact ORIGINAL -> PATCHED text blocks on gb10/patchlib.py's contract; ORIGINAL is upstream muerrilla/sd-webui-detail-daemon
master 1947999.
- denoiser_callback() scales the sigma the model sees each step by a per-daemon factor built from the step's schedule
  value s (multiplier 0.1): 1 - 0.1*s on the cond rows ("cond"), 1 + 0.1*s on the uncond rows ("uncond") or
  1 - 0.1*s*cfg_scale on every row ("both"). Detail Amount is an unbounded number field, so amount*cfg >= 10 in "both"
  (amount >= 10 in "cond", <= -10 in "uncond") makes the factor <= 0: the model is then called with sigma <= 0
  (sigma_to_t takes its log) and the image is garbage or NaN without any error.
- The schedule is built once per pass at its first denoiser call, from that pass's step count (unknown in process()).
  Right there, before the daemon touches a sigma, every factor of the schedule is computed with the same float64
  arithmetic the step applies; any factor that is not > 0 (NaN included) raises ValueError, which fails the request
  (modules/script_callbacks.py propagates output-changing callback errors). A request whose factors are all positive
  runs the upstream code unchanged.
"""
from __future__ import annotations

from pathlib import Path

from patchlib import Block, apply_blocks, parse_cli

LABEL = "Detail Daemon"
TARGET_RELATIVE = Path("scripts") / "detail_daemon.py"
MARKER = "OPENCLAW_DETAIL_DAEMON_SIGMA_GUARD_V1"

BLOCKS = [
    Block(
        "sigma factor guard",
        "            if daemon['schedule'] is None:                \n"
        "                daemon['schedule'] = self.make_schedule(actual_steps, **daemon['schedule_params'])\n",
        "            if daemon['schedule'] is None:                \n"
        "                schedule = self.make_schedule(actual_steps, **daemon['schedule_params'])\n"
        f"                # {MARKER} (gb10/patch-detail-daemon.py): every sigma factor below must stay > 0, or the\n"
        "                # model is called with sigma <= 0. Same float64 arithmetic as the per-step code.\n"
        "                multipliers = schedule * daemon['multiplier']\n"
        "                if mode == \"cond\":\n"
        "                    factors = 1 - multipliers\n"
        "                elif mode == \"uncond\":\n"
        "                    factors = 1 + multipliers\n"
        "                else:\n"
        "                    factors = 1 - multipliers * self.cfg_scale\n"
        "                invalid = np.flatnonzero(~(factors > 0))\n"
        "                if invalid.size:\n"
        "                    first = int(invalid[0])\n"
        "                    applied = \"0.1 x schedule\" if mode in (\"cond\", \"uncond\") else f\"0.1 x schedule x CFG {self.cfg_scale}\"\n"
        "                    raise ValueError(\n"
        "                        f\"Detail Daemon: {name} ({mode} mode, amount {daemon['schedule_params']['amount']}) scales the \"\n"
        "                        f\"model sigma by {factors[first]:.6g} at step {first}/{actual_steps - 1}: the sigma would be <= 0. \"\n"
        "                        f\"{applied} must stay {'above -1' if mode == 'uncond' else 'below 1'} at every step; lower the amount.\"\n"
        "                    )\n"
        "                daemon['schedule'] = schedule\n",
        sentinel=MARKER,
    ),
]


def main() -> int:
    args = parse_cli(__doc__.splitlines()[0])
    target = args.path / TARGET_RELATIVE if args.path.is_dir() else args.path
    if apply_blocks({target: BLOCKS}, label=LABEL, check=args.check):
        print(f"Patched Detail Daemon sigma guard: {target}")
    else:
        print(f"Detail Daemon sigma guard verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
