"""train_embedding / train_hypernetwork on CPU with fake models: failures reach the caller with every global restored,
the target file survives a failed run, interrupts still save, previews leave the training RNG alone, and the loop
control keeps the step/epoch/checkpoint/preview schedule of the pre-refactor loops (test-local oracle below)."""

import glob
import html
import json
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import devices, errors, hashes, images, processing, sd_hijack_checkpoint, sd_models  # noqa: E402
from modules.hypernetworks import hypernetwork as hn  # noqa: E402
from modules.textual_inversion import dataset, textual_inversion as ti  # noqa: E402
from modules.textual_inversion.learn_schedule import LearnRateScheduler  # noqa: E402

NAME = "probe"
WIDTH = 8  # conditioning width, and the hypernetwork layer size
KINDS = ["embedding", "hypernetwork"]


class FakeDataset:
    def __init__(self, length, batch_size, gradient_step, latent_sampling_method, **_):
        # PersonalizedBase clamps both to the dataset size
        self.length = length
        self.batch_size = min(batch_size, length)
        self.gradient_step = min(gradient_step, length // self.batch_size)
        self.latent_sampling_method = latent_sampling_method

    def __len__(self):
        return self.length


class FakeLoader:
    """Each iteration is one epoch of len(ds) // batch_size batches; a batch's latent value is epoch * 100 + index."""

    def __init__(self, ds, batch_size, **_):
        self.ds = ds
        self.batch_size = batch_size
        self.epoch = 0

    def __iter__(self):
        epoch, self.epoch = self.epoch, self.epoch + 1
        for k in range(len(self.ds) // self.batch_size):
            label = epoch * 100 + k
            yield SimpleNamespace(
                latent_sample=torch.full((self.batch_size, 4, 2, 2), float(label)),
                cond_text=[f"prompt {label}"] * self.batch_size,
                cond=[torch.full((1, WIDTH), 0.1 * (k + 1)) for _ in range(self.batch_size)],
                weight=None,
            )


class FakeStage:
    def __init__(self, encode=None):
        self.encode = encode
        self.on_device = True

    def to(self, device):
        self.on_device = device is not devices.cpu
        return self

    def __call__(self, texts):
        return self.encode(texts)


class FakeState:
    def __init__(self):
        self.interrupted = False
        self.job = self.textinfo = None
        self.job_count = self.job_no = 0
        self.current_images = []

    def assign_current_image(self, image):
        self.current_images.append(image)


class TrainingEnv:
    def __init__(self, monkeypatch, tmp_path):
        self.tmp = tmp_path
        self.events = []
        self.tb_images = []
        self.tb_loss_steps = []
        self.loss_files = set()
        self.noise = []
        self.reports = []
        self.checkpointing = []
        self.p_kwargs = []
        self.ppa_seen = set()
        self.fail_at_step = None
        self.interrupt_at_step = None
        self.length = 4
        self.embedding = None

        self.data_root = tmp_path / "data"
        self.data_root.mkdir()
        (self.data_root / "image.png").write_bytes(b"")
        template = tmp_path / "template.txt"
        template.write_text("[name]\n")
        for kind in KINDS:
            (tmp_path / kind).mkdir()

        self.opts = SimpleNamespace(
            unload_models_when_training=True, training_enable_tensorboard=False, training_tensorboard_save_images=False,
            pin_memory=False, save_training_settings_to_txt=False, training_image_repeats_per_epoch=1,
            save_optimizer_state=False, training_write_csv_every=0, samples_format="png", print_hypernet_extra=False,
        )
        self.word_embeddings = {}
        self.sd_model = SimpleNamespace(
            cond_stage_model=FakeStage(lambda texts: self.embedding.vec.sum(dim=0).expand(len(texts), 3, WIDTH)),
            first_stage_model=FakeStage(),
            model=SimpleNamespace(conditioning_key="crossattn"),
            forward=self.forward,
        )
        save_with_identity = ti.save_with_identity

        def recording_save(trained, checkpoint, name, filename, **attributes):
            self.events.append(("save", name, os.path.basename(filename), trained.step))
            save_with_identity(trained, checkpoint, name, filename, **attributes)

        env = self

        class FakeTxt2Img:
            def __init__(self, **kwargs):
                env.p_kwargs.append(kwargs)
                self.prompt = ""
                self.seed = -1

            def close(self):
                pass

        class FakeWriter:
            def add_scalar(self, tag, scalar_value, global_step):
                if tag == "Loss/train":
                    env.tb_loss_steps.append(global_step)

            def add_image(self, tag, img_tensor, global_step):
                env.tb_images.append((tag, global_step))

        def process_images(p):
            # processing reseeds the global generator from the job seed and draws its noise from it
            torch.manual_seed(p.seed if p.seed != -1 else 4321)
            torch.randn(16)
            return SimpleNamespace(images=[Image.new("RGB", (256, 256), (90, 120, 200))], infotexts=["infotext"])

        def save_image(image, path, basename, seed, prompt, extension, info, p=None, forced_filename=None, save_to_dirs=None):
            self.events.append(("preview", forced_filename, self.trained.step))
            return os.path.join(path, f"{forced_filename}.{extension}"), None

        def write_loss(log_directory, filename, step, epoch_len, values):
            self.loss_files.add(filename)
            self.events.append(("loss", step, epoch_len, values["learn_rate"]))

        mp = monkeypatch
        mp.setattr(shared, "opts", self.opts)
        mp.setattr(shared, "state", FakeState())
        mp.setattr(shared, "parallel_processing_allowed", True)
        mp.setattr(shared, "hypernetworks", {NAME: self.target("hypernetwork")})
        mp.setattr(shared, "loaded_hypernetworks", [])
        mp.setattr(shared.cmd_opts, "embeddings_dir", str(tmp_path / "embedding"))
        mp.setattr(shared.cmd_opts, "hypernetwork_dir", str(tmp_path / "hypernetwork"))
        mp.setattr(sd_models, "model_data", SimpleNamespace(get_sd_model=lambda: self.sd_model))
        mp.setattr(sd_models, "select_checkpoint", lambda: SimpleNamespace(model_name="fake-model", shorthash="0123456789"))
        mp.setattr(devices, "device", torch.device("cpu"))  # a distinct object, so FakeStage can tell it from devices.cpu
        mp.setattr(devices, "autocast", lambda disable=False: nullcontext())
        mp.setattr(dataset, "PersonalizedBase", lambda **kwargs: FakeDataset(self.length, **kwargs))
        mp.setattr(dataset, "PersonalizedDataLoader", FakeLoader)
        mp.setattr(processing, "StableDiffusionProcessingTxt2Img", FakeTxt2Img)
        mp.setattr(processing, "process_images", process_images)
        mp.setattr(images, "save_image", save_image)
        mp.setattr(hashes, "sha256", lambda *args, **kwargs: None)
        mp.setattr(errors, "report", lambda message, exc_info=False: self.reports.append(message))
        mp.setattr(sd_hijack_checkpoint, "add", lambda: self.checkpointing.append("add"))
        mp.setattr(sd_hijack_checkpoint, "remove", lambda: self.checkpointing.append("remove"))
        mp.setattr(torch.nn.utils, "clip_grad_value_", lambda parameters, clip_value: self.events.append(("clip", clip_value)))
        mp.setattr(ti, "sd_hijack", SimpleNamespace(model_hijack=SimpleNamespace(embedding_db=SimpleNamespace(word_embeddings=self.word_embeddings))))
        mp.setattr(ti, "sd_samplers", SimpleNamespace(samplers_map={"euler": "Euler"}))
        mp.setattr(ti, "save_with_identity", recording_save)
        mp.setattr(ti, "write_loss", write_loss)
        mp.setattr(ti, "tensorboard_setup", lambda log_directory: FakeWriter())
        mp.setitem(ti.textual_inversion_templates, "template.txt", ti.TextualInversionTemplate("template.txt", str(template)))
        assert devices.device is not devices.cpu

    def target(self, kind):
        return str(self.tmp / kind / f"{NAME}.pt")

    @property
    def trained(self):
        return self.embedding if self.embedding is not None else shared.loaded_hypernetworks[0]

    def forward(self, x, cond):
        step = self.trained.step
        self.events.append(("batch", int(x.flatten()[0]), step))
        self.ppa_seen.add(shared.parallel_processing_allowed)
        if step == self.fail_at_step:
            raise RuntimeError("boom")
        if step == self.interrupt_at_step:
            shared.state.interrupted = True

        noise = torch.randn(4)  # training noise and timesteps come from the global generator
        self.noise.append(noise.tolist())
        if self.embedding is not None:
            out = cond
        else:
            context_k, context_v = hn.apply_hypernetworks(shared.loaded_hypernetworks, cond)
            out = context_k + context_v
        return (((out - 1) ** 2).mean() + 0 * noise.sum(),)

    def prepare(self, kind, initial_step=0):
        """Write the model's target file, as the API trains an existing embedding or hypernetwork."""
        torch.manual_seed(0)
        if kind == "embedding":
            self.embedding = ti.Embedding(torch.zeros(2, WIDTH), NAME, step=initial_step)
            self.embedding.save(self.target(kind))
            self.word_embeddings[NAME] = self.embedding
        else:
            hypernetwork = hn.Hypernetwork(name=NAME, enable_sizes=[WIDTH], layer_structure=[1, 2, 1], activation_func="relu", weight_init="Normal")
            hypernetwork.step = initial_step
            hypernetwork.save(self.target(kind))

    def train(self, kind, *, length=4, batch_size=1, gradient_step=1, steps=6, learn_rate="0.005", clip_grad_value="0.5", save_every=0, create_image_every=0, preview_from_txt2img=False):
        self.length = length
        args = dict(
            id_task="task", learn_rate=learn_rate, batch_size=batch_size, gradient_step=gradient_step, data_root=str(self.data_root),
            log_directory=str(self.tmp / "log"), training_width=64, training_height=64, varsize=False, steps=steps, clip_grad_mode="value",
            clip_grad_value=clip_grad_value, shuffle_tags=False, tag_drop_out=0, latent_sampling_method="once", use_weight=False,
            create_image_every=create_image_every, template_filename="template.txt", preview_from_txt2img=preview_from_txt2img,
            preview_prompt="a preview", preview_negative_prompt="", preview_steps=4, preview_sampler_name="Euler", preview_cfg_scale=7.0,
            preview_seed=1234, preview_width=64, preview_height=64,
        )
        if kind == "embedding":
            return ti.train_embedding(embedding_name=NAME, save_embedding_every=save_every, save_image_with_stored_embedding=True, **args)
        return hn.train_hypernetwork(hypernetwork_name=NAME, save_hypernetwork_every=save_every, **args)


@pytest.fixture
def env(monkeypatch, tmp_path):
    # The API runs training in a worker thread, where autograd is on; earlier tests in this process may have left the
    # (thread-local) grad mode off.
    with torch.enable_grad():
        yield TrainingEnv(monkeypatch, tmp_path)


def old_schedule(kind, *, length, batch_size, gradient_step, steps, initial_step, learn_rate, clip_grad_value, save_every, create_image_every):
    """The loop control of both trainers at c98dc893 (before the shared helpers), with the model taken out.

    Returns the expected (events, tensorboard images, image-embedding files) of a run with previews from txt2img,
    tensorboard on and save_image_with_stored_embedding on.
    """
    batch_size = min(batch_size, length)
    gradient_step = min(gradient_step, length // batch_size)
    steps_per_epoch = length // batch_size // gradient_step
    max_steps_per_epoch = length // batch_size - (length // batch_size) % gradient_step
    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    optimizer = SimpleNamespace(param_groups=[{}])
    events, tb_images, embed_images = [], [], []
    step = initial_step
    embedding_yet_to_be_embedded = False
    epoch = 0

    for _ in range((steps - initial_step) * gradient_step):
        if scheduler.finished:
            break
        for j in range(length // batch_size):
            # works as a drop_last=True for gradient accumulation
            if j == max_steps_per_epoch:
                break
            scheduler.apply(optimizer, step)
            if scheduler.finished:
                break
            clip_grad_sched.step(step)

            events.append(("batch", epoch * 100 + j, step))

            if (j + 1) % gradient_step != 0:
                continue

            events.append(("clip", clip_grad_sched.learn_rate))
            step += 1
            steps_done = step + 1
            epoch_num = step // steps_per_epoch

            if save_every and steps_done % save_every == 0:
                events.append(("save", f"{NAME}-{steps_done}" if kind == "embedding" else NAME, f"{NAME}-{steps_done}.pt", step))
                embedding_yet_to_be_embedded = True

            if kind == "hypernetwork":
                epoch_num = step // length  # its tensorboard block reuses epoch_num

            events.append(("loss", step, steps_per_epoch, scheduler.learn_rate))

            if create_image_every and steps_done % create_image_every == 0:
                events.append(("preview", f"{NAME}-{steps_done}", step))
                tb_images.append((f"Validation at epoch {epoch_num}", step))
                if kind == "embedding" and embedding_yet_to_be_embedded:
                    embed_images.append(f"{NAME}-{steps_done}.png")
                    embedding_yet_to_be_embedded = False
        epoch += 1

    events.append(("save", NAME, f"{NAME}.pt", step))
    return events, tb_images, embed_images


SCHEDULES = {
    "accumulate-2-drop-last": dict(length=7, batch_size=2, gradient_step=2, steps=9, initial_step=0, learn_rate="0.005:3, 0.002:6, 0.001", clip_grad_value="0.5:4, 0.25", save_every=2, create_image_every=3),
    "resume-lr-schedule-ends-early": dict(length=5, batch_size=1, gradient_step=3, steps=8, initial_step=2, learn_rate="0.005:5", clip_grad_value="0.1", save_every=3, create_image_every=2),
    "no-accumulation": dict(length=4, batch_size=1, gradient_step=1, steps=6, initial_step=0, learn_rate="0.01", clip_grad_value="0.3", save_every=4, create_image_every=1),
}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_loop_control_keeps_the_previous_schedule(env, kind, schedule):
    config = SCHEDULES[schedule]
    env.opts.training_enable_tensorboard = True
    env.opts.training_tensorboard_save_images = True
    env.opts.save_training_settings_to_txt = True
    env.prepare(kind, initial_step=config["initial_step"])
    run = {key: value for key, value in config.items() if key != "initial_step"}

    trained, filename = env.train(kind, preview_from_txt2img=True, **run)

    events, tb_images, embed_images = old_schedule(kind, **config)
    assert env.events == events
    assert env.tb_images == tb_images
    final_step = events[-1][-1]
    assert trained.step == final_step
    assert filename == env.target(kind)
    assert torch.load(filename)["step"] == final_step
    assert shared.state.job_no == final_step
    assert env.loss_files == {"textual_inversion_loss.csv" if kind == "embedding" else "hypernetwork_loss.csv"}
    assert env.tb_loss_steps == ([] if kind == "embedding" else [event[1] for event in events if event[0] == "loss"])
    assert env.p_kwargs == [{"sd_model": env.sd_model, "do_not_save_grid": True, "do_not_save_samples": True, "do_not_reload_embeddings" if kind == "embedding" else "disable_extra_networks": True}] * len(tb_images)
    assert env.reports == []
    assert env.checkpointing == ["add", "remove"]
    assert shared.parallel_processing_allowed is True
    embed_files = [os.path.basename(path) for path in glob.glob(str(env.tmp / "log" / "*" / NAME / "image_embeddings" / "*"))]
    assert sorted(embed_files) == sorted(embed_images)

    (settings_file,) = glob.glob(str(env.tmp / "log" / "*" / NAME / "settings-*.json"))
    with open(settings_file) as file:
        settings = json.load(file)
    del settings["datetime"]
    expected = {
        "batch_size": config["batch_size"], "clip_grad_mode": "value", "clip_grad_value": config["clip_grad_value"],
        "create_image_every": config["create_image_every"], "data_root": str(env.data_root), "gradient_step": config["gradient_step"],
        "initial_step": config["initial_step"], "latent_sampling_method": "once", "learn_rate": config["learn_rate"],
        "log_directory": os.path.dirname(settings_file), "model_hash": "0123456789", "model_name": "fake-model",
        "num_of_dataset_images": config["length"], "steps": config["steps"], "template_file": str(env.tmp / "template.txt"),
        "training_height": 64, "training_width": 64, "preview_cfg_scale": 7.0, "preview_height": 64, "preview_negative_prompt": "",
        "preview_prompt": "a preview", "preview_seed": 1234, "preview_steps": 4, "preview_width": 64,
    }
    if kind == "embedding":
        expected |= {"embedding_name": NAME, "num_vectors_per_token": 2, "save_embedding_every": config["save_every"], "save_image_with_stored_embedding": True}
    else:
        expected |= {"hypernetwork_name": NAME, "save_hypernetwork_every": config["save_every"], "activation_func": "relu", "add_layer_norm": False, "layer_structure": [1, 2, 1], "use_dropout": False, "weight_init": "Normal"}
    assert settings == expected


@pytest.mark.parametrize("kind", KINDS)
def test_failure_mid_loop_reaches_the_caller_and_restores_everything(env, kind):
    env.prepare(kind)
    with open(env.target(kind), "rb") as file:
        target_before = file.read()
    env.fail_at_step = 3

    with pytest.raises(RuntimeError, match="boom"):
        env.train(kind, steps=8, save_every=2, create_image_every=2)

    with open(env.target(kind), "rb") as file:
        assert file.read() == target_before
    assert env.ppa_seen == {False}
    assert shared.parallel_processing_allowed is True
    assert env.sd_model.first_stage_model.on_device and env.sd_model.cond_stage_model.on_device
    assert env.checkpointing == ["add", "remove"]
    assert env.reports == ["Error training embedding" if kind == "embedding" else "Exception in training hypernetwork"]
    assert [event for event in env.events if event[0] == "save" and event[2] == f"{NAME}.pt"] == []
    if kind == "hypernetwork":
        hypernetwork = shared.loaded_hypernetworks[0]
        assert not any(parameter.requires_grad for parameter in hypernetwork.weights())
        assert not any(layer.training for layers in hypernetwork.layers.values() for layer in layers)


def test_hypernetwork_final_save_failure_reaches_the_caller_and_restores_everything(env, monkeypatch):
    env.prepare("hypernetwork")
    with open(env.target("hypernetwork"), "rb") as file:
        target_before = file.read()
    save = hn.Hypernetwork.save

    def failing_final_save(hypernetwork, filename):
        if filename == env.target("hypernetwork"):
            raise OSError("disk full")
        save(hypernetwork, filename)

    monkeypatch.setattr(hn.Hypernetwork, "save", failing_final_save)
    env.opts.save_optimizer_state = True

    with pytest.raises(OSError, match="disk full"):
        env.train("hypernetwork", steps=4)

    with open(env.target("hypernetwork"), "rb") as file:
        assert file.read() == target_before
    hypernetwork = shared.loaded_hypernetworks[0]
    assert hypernetwork.optimizer_state_dict is None
    assert (hypernetwork.name, hypernetwork.sd_checkpoint, hypernetwork.sd_checkpoint_name) == (NAME, None, None)
    assert shared.parallel_processing_allowed is True
    assert env.sd_model.first_stage_model.on_device and env.sd_model.cond_stage_model.on_device
    assert env.checkpointing == ["add", "remove"]
    assert env.reports == ["Exception in training hypernetwork"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("gradient_step", [1, 2])
def test_interrupt_ends_the_run_normally_and_saves(env, kind, gradient_step):
    env.prepare(kind)
    env.interrupt_at_step = 3

    trained, filename = env.train(kind, steps=8, gradient_step=gradient_step)

    # the interrupted batch finishes (and its optimizer step when it closes an accumulation window); then the run ends
    assert trained.step == (4 if gradient_step == 1 else 3)
    assert filename == env.target(kind)
    assert torch.load(filename)["step"] == trained.step
    assert env.events[-1] == ("save", NAME, f"{NAME}.pt", trained.step)
    assert env.reports == []
    assert shared.parallel_processing_allowed is True


@pytest.mark.parametrize("kind", KINDS)
def test_previews_do_not_disturb_the_training_rng(env, kind):
    env.prepare(kind)
    env.train(kind, steps=6, create_image_every=0)
    without_previews = env.noise
    env.noise = []
    env.prepare(kind)

    env.train(kind, steps=6, create_image_every=1, preview_from_txt2img=True)  # a fixed preview seed every step

    assert len(env.noise) == 6
    assert env.noise == without_previews


def test_textinfo_keeps_the_previous_markup():
    loss_step, steps_done, prompt, saved_file, saved_image = 0.125, 7, "a <cat> & dog", "/log/probe-6.pt", "/log/probe-7.png, prompt: a"
    previous = {
        "embedding": f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(prompt)}<br/>
Last saved embedding: {html.escape(saved_file)}<br/>
Last saved image: {html.escape(saved_image)}<br/>
</p>
""",
        "hypernetwork": f"""
<p>
Loss: {loss_step:.7f}<br/>
Step: {steps_done}<br/>
Last prompt: {html.escape(prompt)}<br/>
Last saved hypernetwork: {html.escape(saved_file)}<br/>
Last saved image: {html.escape(saved_image)}<br/>
</p>
""",
    }

    for kind, markup in previous.items():
        assert ti.training_textinfo(loss_step, steps_done, prompt, kind, saved_file, saved_image) == markup
