import datetime
import logging
import time

from modules import errors, shared, devices

log = logging.getLogger(__name__)


class State:
    skipped = False
    interrupted = False
    stopping_generation = False
    job = ""
    job_no = 0
    job_count = 0
    processing_has_refined_job_count = False
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    _current_latent = None
    current_image = None
    current_image_sampling_step = 0
    # Set by a /progress poll (request_current_image), consumed by the sampler thread (current_latent's setter). A plain
    # attribute: a racing poll can only cost one decode more or less.
    current_image_requested = False
    textinfo = None
    time_start = None

    def skip(self):
        self.skipped = True
        log.info("Received skip request")

    def interrupt(self):
        self.interrupted = True
        log.info("Received interrupt request")

    def nextjob(self):
        if shared.opts.live_previews_enable and shared.opts.show_progress_every_n_steps == -1:
            self.do_set_current_image()

        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "stopping_generation": self.stopping_generation,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }

        return obj

    def begin(self, job: str = "(unknown)"):
        self.sampling_step = 0
        self.time_start = time.time()
        self.job_count = -1
        self.processing_has_refined_job_count = False
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.current_latent = None
        self.current_image = None
        self.current_image_sampling_step = 0
        self.current_image_requested = False
        self.skipped = False
        self.interrupted = False
        self.stopping_generation = False
        self.textinfo = None
        self.job = job
        devices.torch_gc()
        log.info("Starting job %s", job)

    def end(self):
        duration = time.time() - self.time_start
        log.info("Ending job %s (%.2f seconds)", self.job, duration)
        self.job = ""
        self.job_count = 0

        # The one per-job release of cached device memory. Generation phases (VAE encode,
        # sampling, hires, decode) deliberately do not empty the cache: the caching allocator
        # serves each phase from blocks the previous phase freed, so the in-job reserved
        # high-water mark stays at the largest phase's need (up to allocator fragmentation),
        # and a failed device allocation already frees the cache and retries. Emptying between
        # phases only forced device-synchronizing frees and re-faulting of multi-GB segments.
        # Returning memory between jobs (here) is kept because unified memory is shared with
        # other processes.
        devices.torch_gc()

    @property
    def current_latent(self):
        return self._current_latent

    @current_latent.setter
    def current_latent(self, latent):
        """The sampler thread stores each step's preview latent here (sd_samplers_common.store_latent). When a /progress
        poll asked for a preview (request_current_image) and enough sampling steps have been made since the last one,
        it is decoded here, on the sampler thread, between denoiser calls.

        Previews are decoded on the GPU (VAE or approximation). Decoded on the polling thread, they ran concurrently
        with sampling and could poison a CUDA graph capture in progress (the U-Net/VAE graphs and torch.compile's
        reduce-overhead graphs): every device operation stays on the generating thread instead, under its inference
        mode and autocast. Without parallel_processing_allowed (lowvram, training) store_latent itself decodes at the
        preview period, so requests are not served here."""
        self._current_latent = latent
        if latent is None or not self.current_image_requested or not shared.parallel_processing_allowed:
            return

        period = shared.opts.show_progress_every_n_steps
        if shared.opts.live_previews_enable and period != -1 and self.sampling_step - self.current_image_sampling_step >= period:
            self.current_image_requested = False
            self.do_set_current_image()

    def request_current_image(self):
        """Ask the sampler thread for a live preview: the next stored latent due by the preview period is decoded into
        current_image (see current_latent). Called by /progress, which returns the current_image already produced."""
        self.current_image_requested = True

    def do_set_current_image(self):
        if self.current_latent is None:
            return

        import modules.sd_samplers

        try:
            if shared.opts.show_progress_grid:
                self.assign_current_image(modules.sd_samplers.samples_to_image_grid(self.current_latent))
            else:
                self.assign_current_image(modules.sd_samplers.sample_to_image(self.current_latent))

            self.current_image_sampling_step = self.sampling_step

        except Exception:
            # when switching models during generation, VAE would be on CPU, so creating an image will fail.
            # we silently ignore this error
            errors.record_exception()

    def assign_current_image(self, image):
        if shared.opts.live_previews_image_format == 'jpeg' and image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        self.current_image = image
