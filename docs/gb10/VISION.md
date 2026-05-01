# GB10 A1111 fork runtime Vision

## Core goal

Operate the GB10 as a reliable, user-worthy AUTOMATIC1111 image-generation appliance for Schwi.

The baseline is no longer theoretical. The project now has a working containerized A1111 bring-up with a reachable web UI and successful image generation.

## Product posture

Optimize for:

- practical usability
- faithful A1111 behavior
- explicit and reviewable build/runtime behavior
- durable host-mounted user data
- understandable upgrades
- minimal host pollution outside the intentional storage root

Do not optimize primarily for:

- theoretical image minimalism
- container cleverness for its own sake
- aggressive purity that makes ordinary use worse

## Source-of-truth posture

Prefer:

- official NVIDIA base images
- official AUTOMATIC1111 upstream source
- explicit pinned image tags
- explicit documented companion-repo pins
- explicit documented storage layout
- repo-visible launch/bootstrap behavior

## Runtime vision from this baseline

The current baseline proves the container can:

- start cleanly
- serve the UI
- load a model
- generate an image

The next level of quality is operational polish:

- better default launch/tuning choices for the GB10
- explicit model ownership/download policy
- restart/rebuild confidence
- extension/plugin layering discipline
- optional acceleration improvements when they are genuinely worth the complexity

## Build doctrine

- build on the GB10 itself
- keep builder clutter in builder stages
- keep the runtime image understandable
- prefer reproducibility over premature slimming

## Runtime doctrine

User-owned state should remain host-mounted and durable:

- checkpoints
- VAEs and helper models
- LoRAs
- embeddings
- extensions
- outputs
- configs

The image should remain replaceable without threatening those user-owned surfaces.

## Evaluation doctrine

Judge future work by questions like:

- Does A1111 feel good to use on the GB10?
- Is the runtime behavior legible and controllable?
- Are updates understandable instead of magical?
- Do persistence and restarts behave exactly the way Schwi would expect?
- Do performance tweaks actually improve real use enough to justify their complexity?

## Current forward path

From the working baseline, the sensible frontier is:

- tune
- validate persistence and restart behavior
- reduce noise and accidental downloads where appropriate
- evaluate optional accelerators carefully
- pin upstream more tightly later if a more fixed posture becomes desirable
