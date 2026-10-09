import ipaddress
import json
import os
import signal
import sys
import re

from starlette.middleware.gzip import GZipMiddleware, IdentityResponder

from modules.timer import startup_timer


def server_name():
    from modules.shared_cmd_options import cmd_opts

    if cmd_opts.server_name:
        return cmd_opts.server_name
    else:
        return "0.0.0.0" if cmd_opts.listen else None


def fix_torch_version():
    import torch

    # Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
    if ".dev" in torch.__version__ or "+git" in torch.__version__:
        torch.__long_version__ = torch.__version__
        torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

def configure_torch_threads():
    """
    Sizes torch's CPU thread pools to the CPUs this process may run on. Call right after importing torch,
    before any parallel work: the inter-op size can only be set once, before that pool starts.

    torch sizes its intra-op pool from the machine's core count, not the affinity mask (20 threads on the
    10-CPU GB10 container cpuset). Nothing here launches inter-op work, so that pool is kept small.
    An explicit OMP_NUM_THREADS/MKL_NUM_THREADS is the operator's choice and is left alone.
    """
    import torch

    if "OMP_NUM_THREADS" in os.environ or "MKL_NUM_THREADS" in os.environ:
        return

    cpus = len(os.sched_getaffinity(0))
    torch.set_num_threads(cpus)
    torch.set_num_interop_threads(min(cpus, 2))


def fix_asyncio_event_loop_policy():
    """
        The default `asyncio` event loop policy only automatically creates
        event loops in the main threads. Other threads must create event
        loops explicitly or `asyncio.get_event_loop` (and therefore
        `.IOLoop.current`) will fail. Installing this policy allows event
        loops to be created automatically on any thread, matching the
        behavior of Tornado versions prior to 5.0 (or 5.0 on Python 2).
    """

    import asyncio

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        # "Any thread" and "selector" should be orthogonal, but there's not a clean
        # interface for composing policies so pick the right base.
        _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
    else:
        _BasePolicy = asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
        """Event loop policy that allows loop creation on any thread.
        Usage::

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        """

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                return super().get_event_loop()
            except (RuntimeError, AssertionError):
                # This was an AssertionError in python 3.4.2 (which ships with debian jessie)
                # and changed to a RuntimeError in 3.4.3.
                # "There is no current event loop in thread %r"
                loop = self.new_event_loop()
                self.set_event_loop(loop)
                return loop

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def restore_config_state_file():
    from modules import shared, config_states

    config_state_file = shared.opts.restore_config_state_file
    if config_state_file == "":
        return

    shared.opts.restore_config_state_file = ""
    shared.opts.save(shared.config_filename)

    if os.path.isfile(config_state_file):
        print(f"*** About to restore extension state from file: {config_state_file}")
        with open(config_state_file, "r", encoding="utf-8") as f:
            config_state = json.load(f)
            config_states.restore_extension_config(config_state)
        startup_timer.record("restore extension config")
    elif config_state_file:
        print(f"!!! Config state backup not found: {config_state_file}")


def validate_tls_options():
    from modules.shared_cmd_options import cmd_opts

    if not (cmd_opts.tls_keyfile and cmd_opts.tls_certfile):
        return

    try:
        if not os.path.exists(cmd_opts.tls_keyfile):
            print("Invalid path to TLS keyfile given")
        if not os.path.exists(cmd_opts.tls_certfile):
            print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
    except TypeError:
        cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
        print("TLS setup invalid, running webui without TLS")
    else:
        print("Running with TLS")
    startup_timer.record("TLS")


def dumpstacks():
    import threading
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append(f"\n# Thread: {id2name.get(threadId, '')}({threadId})")
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append(f"""File: "{filename}", line {lineno}, in {name}""")
            if line:
                code.append("  " + line.strip())

    print("\n".join(code))


def configure_sigint_handler():
    # make the program just exit at ctrl+c without waiting for anything

    from modules import shared

    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')

        if shared.opts.dump_stacks_on_signal:
            dumpstacks()

        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
        signal.signal(signal.SIGINT, sigint_handler)


def configure_opts_onchange():
    from modules import shared, sd_models, sd_vae, sd_hijack
    from modules.call_queue import wrap_queued_call

    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_overrides_per_model_preferences", wrap_queued_call(lambda: sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("cross_attention_optimization", wrap_queued_call(lambda: sd_hijack.model_hijack.redo_hijack(shared.sd_model)), call=False)
    shared.opts.onchange("fp8_storage", wrap_queued_call(lambda: sd_models.reload_model_weights()), call=False)
    force_reload = wrap_queued_call(lambda: sd_models.reload_model_weights(forced_reload=True))
    for key in ("mxfp8_storage", "mxfp8_linear_coverage", "nvfp4_storage", "nvfp4_linear_coverage", "cache_fp16_weight"):
        shared.opts.onchange(key, force_reload, call=False)
    startup_timer.record("opts onchange")


def is_loopback_client(client) -> bool:
    """True when an ASGI scope["client"] (host, port) pair is a loopback peer (127.0.0.0/8 or ::1)."""
    if client is None:
        return False
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False


class LoopbackIdentityGZipMiddleware(GZipMiddleware):
    """
    Starlette's GZipMiddleware, except that loopback clients are answered as if they had not offered gzip.

    Generation responses are mostly base64 PNG, which deflate barely shrinks (ratio ~0.76) at ~20 ms per MB on the
    request path, and on loopback there is no network to save. Identity is always an acceptable content-coding,
    so the decoded body is unchanged; LAN clients keep gzip.
    """

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and is_loopback_client(scope.get("client")):
            responder = IdentityResponder(self.app, self.minimum_size, exclude_content_types=self.exclude_content_types)
            await responder(scope, receive, send)
            return

        await super().__call__(scope, receive, send)


def setup_middleware(app):
    app.middleware_stack = None  # reset current middleware to allow modifying user provided list
    app.add_middleware(LoopbackIdentityGZipMiddleware, minimum_size=1000)
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    from starlette.middleware.cors import CORSMiddleware
    from modules.shared_cmd_options import cmd_opts

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(',')
    if cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex
    app.add_middleware(CORSMiddleware, **cors_options)

