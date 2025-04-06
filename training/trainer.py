from contextlib import contextmanager
from pathlib import Path
from einops import rearrange
from beartype import beartype

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


from einops import rearrange
from tqdm import tqdm

from models.genie_redux import GenieRedux, GenieReduxGuided
from models.lam import LatentActionModel
from models.tokenizer import Tokenizer
from training.optimizer import get_optimizer, LinearWarmup_CosineAnnealing

from data.data import (
    video_tensor_to_gif,
    video_tensor_to_pil_images,
    EnvironmentDataset as Dataset,
)


from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

import wandb

from training.evaluation import Evaluator

import torch

from PIL import Image

from utils.utils import debug


# helpers


def exists(val):
    return val is not None


def cycle(dl):
    """
    Create an infinite iterator over a DataLoader.

    Args:
        dl (DataLoader): The DataLoader to cycle through.

    Yields:
        dict: A batch of data with rearranged input frames.
    """
    while True:
        for data in dl:
            data["input_frames"] = rearrange(
                data["input_frames"], "b f c h w -> b c f h w"
            )
            yield data


def noop(*args, **kwargs):
    pass


def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def yes_or_no(question):
    answer = input(f"{question} (y/n) ")
    return answer.lower() in ("yes", "y")


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


# Define a conditional context manager
@contextmanager
def conditional_with(context_manager, condition):
    if condition:
        # If condition is true, enter the context manager
        with context_manager:
            yield
    else:
        # If condition is false, do nothing
        yield


# main trainer class
@beartype
class Trainer(nn.Module):
    """
    A trainer class for various models including Tokenizer, LatentActionModel, Genie, and GroundTruthActionDynamics.

    This class handles the training loop, evaluation, logging, and saving of models.
    """

    def __init__(
        self,
        model: Tokenizer | LatentActionModel | GenieRedux | GenieReduxGuided,
        batch_size,
        dataset: tuple[Dataset, Dataset] | Dataset,
        num_train_steps=100000,
        accelerator: Accelerator | None = None,
        num_frames=16,
        sample_num_frames=15,
        lr=3e-4,
        adam_betas=(0.9, 0.99),
        grad_accum_every=1,
        wd=0.0,
        max_grad_norm=0.5,
        wandb_mode="disabled",
        wandb_project="GenieRedux",
        wandb_name="",
        wandb_dpath="./wandb",
        wandb_log_every=50,
        linear_warmup_start_factor=0.1,
        linear_warmup_total_iters=100,
        cosine_annealing_T_max=1000000,
        cosine_annealing_eta_min=1e-5,
        save_results_every=1000,
        save_model_every=1000,
        save_dpath="./results",
        valid_frac=0.10,
        apply_grad_penalty_every=4,
        save_model=True,
        max_valid_size=128,
        validate_every=1000,
        train_config={},
    ):
        """
        Initialize the Trainer.

        Args:
            model: The model to train (Tokenizer, LatentActionModel, Genie, or GroundTruthActionDynamics).
            num_train_steps (int): Total number of training steps.
            batch_size (int): Batch size for training.
            dataset: The dataset(s) to use for training and validation.
            num_frames (int): Number of frames to use in training.
            sample_num_frames (int): Number of frames to use when sampling.
            lr (float): Learning rate.
            adam_betas (tuple): Beta parameters for Adam optimizer.
            grad_accum_every (int): Number of steps to accumulate gradients.
            wd (float): Weight decay.
            max_grad_norm (float): Maximum gradient norm for clipping.
            wandb_mode (str): Mode for Weights & Biases logging.
            wandb_project (str): Weights & Biases project name.
            wandb_name (str): Weights & Biases run name.
            wandb_dpath (str): Directory for Weights & Biases files.
            wandb_log_every (int): Log to Weights & Biases every N steps.
            linear_warmup_start_factor (float): Start factor for linear warmup.
            linear_warmup_total_iters (int): Total iterations for linear warmup.
            cosine_annealing_T_max (int): T_max for cosine annealing.
            cosine_annealing_eta_min (float): Minimum learning rate for cosine annealing.
            save_results_every (int): Save results every N steps.
            save_model_every (int): Save model every N steps.
            results_dpath (str): Folder to save results.
            valid_frac (float): Fraction of data to use for validation.
            apply_grad_penalty_every (int): Apply gradient penalty every N steps.
            save_model (bool): Whether to save the model.
            max_valid_size (int): Maximum size of validation dataset.
            validate_every (int): Validate every N steps.
        """
        super().__init__()

        # Initialize instance variables
        self.image_size = model.image_size
        self.save_model = save_model
        self.validate_every = validate_every
        self.wandb_log_every = wandb_log_every
        self.num_frames = num_frames
        self.sample_num_frames = sample_num_frames

        # Create config dictionary for logging
        config = {}
        arguments = locals()
        for key in arguments.keys():
            if key not in ["self", "config", "__class__", "model"]:
                config[key] = arguments[key]

        # Add the model config to the wandb config
        config["model_config"] = model.config

        # Determine the type of model
        self.is_genie = isinstance(model, GenieRedux | GenieReduxGuided)
        self.is_lam = isinstance(model, LatentActionModel)

        if accelerator is not None:
            self.accelerator = accelerator
        else:
            # Set up the accelerator for distributed training

            kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(
                cpu=False,
                log_with="wandb",
                kwargs_handlers=[kwargs],
                mixed_precision="bf16",
                gradient_accumulation_steps=grad_accum_every,
            )

        self.evaluator = Evaluator(device=self.accelerator.device)

        # Print config if main process
        if self.accelerator.is_main_process:
            print("config\n")
            print(config)
        self.wandb_mode = wandb_mode
        self.model = model
        self.model.wandb_mode = wandb_mode

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # Set up optimizer and scheduler
        self.optim = get_optimizer(
            model.trainable_parameters(), lr=lr, wd=wd, betas=adam_betas
        )
        self.scheduler_optim = LinearWarmup_CosineAnnealing(
            optimizer=self.optim,
            linear_warmup_start_factor=linear_warmup_start_factor,
            linear_warmup_total_iters=linear_warmup_total_iters,
            cosine_annealing_T_max=cosine_annealing_T_max,
            cosine_annealing_eta_min=cosine_annealing_eta_min,
        )

        self.max_grad_norm = max_grad_norm

        # Set up dataloaders
        if isinstance(dataset, tuple):
            self.ds, self.valid_ds = dataset
        else:
            # Split the dataset into train and validation
            ds_size = len(dataset)
            indices = list(range(ds_size))
            split_index = int(ds_size * valid_frac)
            self.valid_ds = Subset(dataset, indices[:split_index])
            self.ds = Subset(dataset, indices[split_index:])

        self.valid_ds = Subset(
            self.valid_ds, list(range(min(len(self.valid_ds), max_valid_size)))
        )
        self.max_valid_size = max_valid_size
        self.dl = DataLoader(
            self.ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=max_valid_size // 4,
            shuffle=False,
            num_workers=4,
        )

        # Prepare with accelerator
        (self.model, self.optim, self.dl, self.valid_dl, self.evaluator) = (
            self.accelerator.prepare(
                self.model, self.optim, self.dl, self.valid_dl, self.evaluator
            )
        )

        # Set up inference function
        if self.is_genie:
            if self.is_distributed:
                self.inference = self.model.module.sample
            else:
                self.inference = self.model.sample
        else:
            self.inference = self.model

        self.dl_iter = cycle(self.dl)

        # Set up full batch validation dataloader
        full_batch_valid_dl = DataLoader(
            self.valid_ds,
            batch_size=max_valid_size,
            shuffle=False,
            num_workers=4,
        )
        full_batch_valid_dl = self.accelerator.prepare(full_batch_valid_dl)
        self.valid_dl_iter = cycle(full_batch_valid_dl)

        self.valid_data_to_log = {}

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.checkpoint_dpath = save_dpath
        if wandb_name == "":
            wandb_name = self.checkpoint_dpath.name

        self.step = 1

        self.load_log_data()

        # Print training information
        self.print(f"Batch size : {batch_size}")
        self.print(f"Grad accum every : {grad_accum_every}")
        self.print(f"Acctual batch size : {batch_size * grad_accum_every}")
        self.print(f"Number of train steps : {num_train_steps}")

        config["train config"] = train_config

        # Initialize wandb
        self.accelerator.init_trackers(
            project_name=wandb_project,
            config=config,
            init_kwargs={
                "wandb": {
                    "mode": self.wandb_mode,
                    "config": config,
                    "name": wandb_name,
                    "dir": wandb_dpath,
                }
            },
        )

    def load_log_data(self):
        """
        Load and prepare validation data for logging.
        """
        # Divide self.max_valid_size into 4 quarters
        q0, q1, q2, q3 = (
            0,
            self.max_valid_size // 4 - 1,
            self.max_valid_size // 2 - 1,
            3 * self.max_valid_size // 4 - 1,
        )

        valid_data_to_log = next(self.valid_dl_iter)
        self.valid_data_to_log["input_frames"] = valid_data_to_log["input_frames"][
            [q0, q1, q2, q3]
        ]
        self.valid_data_to_log["actions"] = valid_data_to_log["actions"][
            [q0, q1, q2, q3]
        ]

    def save(self, milestone):
        """
        Save the model, optimizer, and scaler states.

        Args:
            milestone: The current milestone (usually the step number).
        """
        if not self.accelerator.is_local_main_process:
            return

        model = self.accelerator.unwrap_model(self.model)

        data = {
            "step": self.step,
            "model": model.state_dict(),
            "optim": self.optim.state_dict(),
            "scaler": (
                self.accelerator.scaler.state_dict()
                if exists(self.accelerator.scaler)
                else None
            ),
        }

        torch.save(data, f"{self.checkpoint_dpath}/model-{milestone}.pt")

    def load(self, path, weights_only=False):
        """
        Load the model, optimizer, and scaler states from a file.

        Args:
            path (str): The path to the saved model file.
        """

        data = torch.load(path, map_location=torch.device("cpu"))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        if not weights_only:
            self.step = data["step"]
            self.optim.load_state_dict(data["optim"])

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

        del data

        self.model = self.accelerator.prepare(model)

    def print(self, msg, *args, **kwargs):
        """
        Print a message using the accelerator's print method.

        Args:
            msg: The message to print.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.accelerator.print(msg, *args, **kwargs)

    @property
    def device(self):
        """Get the device of the accelerator."""
        return self.accelerator.device

    @property
    def is_distributed(self):
        """Check if the training is distributed."""
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    @property
    def is_main(self):
        """Check if this is the main process."""
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        """Check if this is the local main process."""
        return self.accelerator.is_local_main_process

    def train_step(self, *args, **kwargs):
        """
        Perform a single training step.

        Returns:
            dict: A dictionary containing the training logs.
        """
        step = self.step
        apply_grad_penalty = not (step % self.apply_grad_penalty_every)

        self.model.train()

        # Initialize logs and total loss
        logs = {}
        total_loss = 0.0

        # Perform gradient accumulation
        for i in range(self.grad_accum_every):
            # Get next batch of data
            videos = next(self.dl_iter)
            actions = videos["actions"]
            videos = videos["input_frames"]

            accelerator_tracker_dict = None
            if step % self.wandb_log_every == 0:
                accelerator_tracker_dict = {
                    "tracker": self.accelerator,
                    "step": step,
                    "train": True,
                }

            # Compute loss with automatic mixed precision
            with conditional_with(
                self.accelerator.no_sync(self.model), self.grad_accum_every != i + 1
            ):
                with self.accelerator.autocast():
                    loss = self.model(
                        videos=videos,
                        actions=actions,
                        apply_grad_penalty=apply_grad_penalty,
                        accelerator_tracker_dict=accelerator_tracker_dict,
                    )

                # Backward pass
                self.accelerator.backward(loss / self.grad_accum_every)

            # Clip gradients if specified
            if exists(self.max_grad_norm) and self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            total_loss += loss / self.grad_accum_every

        # Optimizer step
        self.optim.step()
        self.optim.zero_grad()

        # Learning rate scheduler step
        self.scheduler_optim.step(self.step)

        # Log training loss
        accum_log(logs, {"train_loss": total_loss.item()})

        # Log learning rate and Loss
        if step % self.wandb_log_every == 0:
            self.accelerator.log({"Train loss": total_loss.item()}, step=step)
            self.accelerator.log({"lr": self.optim.param_groups[0]["lr"]}, step=step)

        # Validation
        if not (step % self.validate_every):
            self.model.eval()
            accelerator_tracker_dict = {
                "tracker": self.accelerator,
                "step": step,
                "train": False,
            }
            psnrs = []
            with torch.no_grad():
                for videos in self.valid_dl:
                    actions = videos["actions"]
                    videos = videos["input_frames"]
                    videos = rearrange(videos, "b f c h w -> b c f h w")
                    first_frames = videos[:, :, :1]
                    loss = self.model(
                        videos=videos,
                        actions=actions,
                        step=step,
                        accelerator_tracker_dict=accelerator_tracker_dict,
                    )

                    if self.is_genie:
                        actions = actions.argmax(dim=-1)

                    # Generate reconstructions
                    recons = self.inference(
                        videos=videos,
                        actions=actions,
                        return_recons_only=True,
                        step=step,
                        num_frames=self.sample_num_frames,
                        prime_frames=first_frames,
                    )

                    if self.is_lam or self.is_genie:
                        videos = videos[:, :, 1:]

                    # Clamp values to [0, 1] range
                    videos = torch.clamp(videos, 0.0, 1.0)
                    recons = torch.clamp(recons, 0.0, 1.0)

                    # Update FID and PSNR metrics
                    self.evaluator.fid_update_batch(videos, recons)
                    psnrs.append(self.evaluator.psnr(videos, recons))

            # Compute final metrics
            fid_score = self.evaluator.fid()
            psnr_score = torch.tensor(psnrs).mean().item()
            self.evaluator.reset()

            # Gather metrics from all processes
            fid_score = torch.tensor(fid_score).to(self.device)
            psnr_score = torch.tensor(psnr_score).to(self.device)
            loss = self.accelerator.gather(loss)
            fid_score = self.accelerator.gather(fid_score)
            psnr_score = self.accelerator.gather(psnr_score)

            self.accelerator.wait_for_everyone()

            # Compute mean of gathered metrics
            loss = loss.mean()
            fid_score = fid_score.mean()
            psnr_score = psnr_score.mean()

            # Print validation results
            self.print("\n============================")
            self.print("Loss on validation set : ", loss)
            self.print("FID score on validation set : ", fid_score)
            self.print("PSNR score on validation set : ", psnr_score)
            self.print("============================\n")

            # Log validation results
            accum_log(
                logs,
                {
                    "valid_loss": loss.item(),
                    "valid_fid_score": fid_score,
                    "valid_psnr_score": psnr_score,
                },
            )

            self.accelerator.log(
                {
                    "Validation loss": loss.item(),
                    "Validation FID score": fid_score,
                    "Validation PSNR score": psnr_score,
                },
                step=step,
            )

            self.print(f"{step}: Validation loss: {logs['valid_loss']}")

        # Sample and save results periodically
        if not (step % self.save_results_every):
            videos = self.valid_data_to_log["input_frames"]
            actions = self.valid_data_to_log["actions"]
            first_frames = videos[:, :, :1]

            if self.is_genie:
                actions = actions.argmax(dim=-1)

            # Generate reconstructions
            with torch.no_grad():
                recons = self.inference(
                    videos=videos,
                    actions=actions,
                    prime_frames=first_frames,
                    num_frames=self.sample_num_frames,
                    return_recons_only=True,
                )

            # Save reconstructed videos
            sampled_videos_path = f"{self.checkpoint_dpath}/samples.{step}"
            sampled_videos_path = Path(sampled_videos_path)
            (sampled_videos_path).mkdir(parents=True, exist_ok=True)

            for i, recons_frames in enumerate(recons.unbind(dim=0)):
                if self.is_lam or self.is_genie:
                    recons_frames = torch.cat([first_frames[i], recons_frames], dim=1)
                orig_frames = videos[i]

                if self.accelerator.is_local_main_process:
                    video_tensor_to_gif(
                        recons_frames.cpu(),
                        str(sampled_videos_path / f"{i}.gif"),
                    )

                    # Convert tensors to PIL images
                    recon_frames = video_tensor_to_pil_images(
                        recons_frames.cpu(), only_first_image=False
                    )
                    orig_frames = video_tensor_to_pil_images(
                        orig_frames.cpu(), only_first_image=False
                    )

                    # Combine original and reconstructed frames
                    combined_height = orig_frames.height + recon_frames.height
                    combined_image = Image.new(
                        "RGB", (orig_frames.width, combined_height)
                    )
                    combined_image.paste(orig_frames, (0, 0))
                    combined_image.paste(recon_frames, (0, orig_frames.height))

                    # Log combined image to WandB
                    self.accelerator.log(
                        {
                            f"Image {i}": [
                                wandb.Image(
                                    combined_image,
                                    caption="Original vs Reconstructed Video",
                                ),
                            ]
                        },
                        step=self.step,
                    )

            self.print(f"{step}: saving to {str(self.checkpoint_dpath)}")

        # Save model checkpoint
        if self.is_main and not (step % self.save_model_every):
            save_path = Path(str(self.checkpoint_dpath))
            save_path.mkdir(parents=True, exist_ok=True)
            self.save(self.step)

            self.print(f"{step}: saving model to {str(save_path)}")

        self.step += 1
        return logs

    def train(self, *args, **kwargs):
        """
        Main training loop.
        """
        with tqdm(
            initial=self.step, total=self.num_train_steps, disable=not self.is_main
        ) as pbar:
            while self.step <= self.num_train_steps:
                # Perform a single training step
                logs = self.train_step(*args, **kwargs)

                # Update progress bar
                pbar.set_description(f"loss: {logs['train_loss']:.4f}")
                pbar.update(1)

        self.print("training complete")
