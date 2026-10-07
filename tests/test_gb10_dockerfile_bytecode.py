from pathlib import Path

COMPILE = "python -m compileall -q -f -j 0 --invalidation-mode timestamp webui.py launch.py modules scripts repositories extensions-builtin"


def test_runtime_stage_precompiles_app_bytecode_before_handing_it_to_the_runtime_user():
    dockerfile = Path("Dockerfile").read_text(encoding="utf8")
    runtime = dockerfile[dockerfile.index("FROM torch-base AS runtime"):]

    compile_at = runtime.index(COMPILE)
    # Compiled after the app tree (and the patched companion repositories) is final in this stage ...
    assert runtime.index("COPY --from=source /opt/build/stable-diffusion-webui /opt/stable-diffusion-webui") < compile_at
    # ... from the app root, and in the same layer as the chown that gives the runtime user the pycs.
    step = runtime[runtime.rindex("RUN ", 0, compile_at):runtime.index("\n\n", compile_at)]
    assert step.index("&& cd /opt/stable-diffusion-webui") < step.index(COMPILE) < step.index("&& chown -R a1111:a1111 /opt/stable-diffusion-webui /home/a1111")
    # Nothing after that layer rewrites app sources (which would invalidate the timestamp pycs).
    assert "RUN " not in runtime[compile_at:]
