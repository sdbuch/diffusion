#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports

import dataclasses
import os
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib import patches
from numpy.random import default_rng
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from denoisers.gmm import GMMEqualVarianceDenoiser
from samplers.sde import BasicOUSampler
from util.configs import ExperimentConfig, OptimizerConfig
from util.types_custom import OptimizerType


def test_sampling() -> None:
    # Logging
    wandb.init(project="gmm_test", name="gmm_test_sampling")

    # Parameters
    input_size = (1, 2, 1)
    num_clusters = 3
    # Denoiser
    model = GMMEqualVarianceDenoiser(input_size, num_clusters, init_variance=1e-6)
    # Perform diffusion model sampling.
    # sampler params
    device = torch.device("cpu")
    min_time = torch.tensor(0.01)
    num_samples = 100
    debias = True
    rng = default_rng()
    seed = int(rng.integers(0, high=2**32))

    # The sampler might be written for a different parameterization of the SDE
    # > discretizations need not be invariant to time reparameterization...?
    # - actually the main issue seems to be invariance of integrators
    # - euler-maruyama integrator is fine, it seems...
    # - but in general it need not work...
    score = lambda x, t: model.score(x, t)
    sampler = BasicOUSampler(
        dimension=input_size,
        score_estimator=score,
        min_time=min_time,
        num_points=num_samples,  # 2*int(base_num_points * log(dim)),
        device=device,
        seed=seed,
    )
    snapshot, times = sampler.sample_trajectory(num_samples, debias=debias)

    # visualize the samples
    # create a matplotlib scatterplot of the generated samples, show clusters
    filenames = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for idx in range(len(times)):
            fig, ax = plt.subplots()
            samples, time = snapshot[idx, ...], times[idx]
            samples_np = model.flatten(samples).detach().numpy()
            plt.scatter(
                samples_np[:, 0], samples_np[:, 1], label="final samples", alpha=0.5
            )
            # plot covariance ellipsoids on top of the scatter
            means, variances = (model.get_means(), model.get_variances())
            means_t, variances_t = (model.get_means(time), model.get_variances(time))
            plot_cov_ellipses(means, variances, legend_str="(t=0)", color_str="blue")
            plot_cov_ellipses(
                means_t, variances_t, legend_str=f"(t={time:.3f})", color_str="red"
            )
            plt.legend(loc="upper right")
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            # Save each frame as an image file
            filename = os.path.join(temp_dir, f"frame_{idx:04d}.png")
            plt.savefig(filename)
            plt.close(fig)
            filenames.append(filename)
        # Create a video file from the images
        video_filename = os.path.join(temp_dir, "animation.mp4")
        imageio.mimsave(video_filename, [imageio.imread(f) for f in filenames], fps=10)

        # Log the video to wandb
        wandb.log({"video": wandb.Video(video_filename, fps=10, format="mp4")})

    wandb.finish()


# TODO: test learning
def test_learning(config: ExperimentConfig) -> None:
    wandb.init(
        project="gmm_test", name="gmm_test_learning", config=dataclasses.asdict(config)
    )
    # Parameters
    input_size = (1, 2, 1)
    num_clusters_gt = 3
    num_clusters = 32
    variance_gt = 0.35**2
    num_samples = 32
    noise_upsampling_rate = 1000
    time_to_train_at = torch.log(torch.tensor(1 - variance_gt)) / -2
    device = torch.device(config.device_str)
    generator = torch.Generator(device=device)
    if config.seed is not None:
        generator.manual_seed(config.seed)
    else:
        generator.seed()
    if config.batch_size is None:
        # full-batch operation
        batch_size = num_samples
    else:
        batch_size = config.batch_size
    # ground-truth Denoiser
    model_gt = GMMEqualVarianceDenoiser(
        input_size, num_clusters_gt, init_variance=variance_gt
    )
    for param in model_gt.parameters():
        param.requires_grad = False
    # data: get some samples from the ground-truth model
    t = torch.tensor(0.0)
    train_data = model_gt.generate_samples(num_samples, t)
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    # Set up denoiser training.
    model = GMMEqualVarianceDenoiser(
        input_size, num_clusters, init_variance=config.noise_level
    )
    model.to(device)
    wandb.watch(model, log_freq=config.num_epochs // 10)

    # # Configure the optimizer!
    optimizer = config.optimizer.algorithm.value(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    # Training loop involves 'online sgd': adding indep. random noise to each minibatch
    # We will fix just one time to train at.
    scale, _ = model_gt.get_time_scaling(time_to_train_at)
    variance = 1 - scale**2
    loss_fn = nn.MSELoss(reduction="sum")  # it takes a mean on all dims by default
    # loss_fn = nn.MSELoss()
    model.train()
    for epoch in tqdm(range(config.num_epochs), desc="Epoch", position=0):
        for batch, (X,) in enumerate(
            tqdm(train_dataloader, desc="Batch", position=1, leave=False)
        ):
            tiled_X = X.repeat(noise_upsampling_rate, 1, 1, 1)
            noisy_X = scale * tiled_X + torch.sqrt(variance) * torch.randn(
                tiled_X.shape, device=device, generator=generator
            )
            denoised = model(noisy_X, time_to_train_at)
            loss = loss_fn(tiled_X, denoised) / batch_size / noise_upsampling_rate

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log(
                {"batch MSE": loss.item(), "variance": model.standard_dev.detach().item()**2}
            )

    # visualize the learned model.
    # create a matplotlib scatterplot of the generated samples, show clusters
    fig, ax = plt.subplots()
    x_np = model_gt.flatten(train_data).detach().numpy()
    noisy_x = scale * train_data + torch.sqrt(variance) * torch.randn(
        train_data.shape, device=device, generator=generator
    )
    noisy_x_np = model_gt.flatten(noisy_x).detach().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], label="gt samples", alpha=0.5)
    plt.scatter(noisy_x_np[:, 0], noisy_x_np[:, 1], label="noisy gt samples", alpha=0.5)
    # plot covariance ellipsoids on top of the scatter
    means, variances = (model_gt.get_means(), model_gt.get_variances())
    means_learned, variances_learned = (model.get_means(), model.get_variances())
    plot_cov_ellipses(means, variances, legend_str="ground truth", color_str="blue")
    plot_cov_ellipses(
        means_learned, variances_learned, legend_str=f"learned", color_str="red"
    )
    plt.legend()
    frames = []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close(fig)
        frames.append(wandb.Image(tmpfile.name))

    # Log the images to wandb as an animation
    wandb.log(
        {
            "image": frames[0],
            "learned means": means_learned,
            "learned variances": variances_learned,
        }
    )
    wandb.finish()


def test_working() -> None:
    wandb.init(project="gmm_test", name="gmm_test_working")
    # Parameters
    input_size = (1, 2, 1)
    num_clusters = 6
    # Denoiser
    model = GMMEqualVarianceDenoiser(input_size, num_clusters, init_variance=1e-6)
    # data: get some samples
    num_samples = 100
    t = torch.tensor(0.01)
    x = model.generate_samples(num_samples, t)
    # denoise the samples
    y = model(x, t)
    # visualize the samples
    # create a matplotlib scatterplot of the generated samples, show clusters
    fig, ax = plt.subplots()
    x_np = model.flatten(x).detach().numpy()
    y_np = model.flatten(y).detach().numpy()
    plt.scatter(x_np[:, 0], x_np[:, 1], label="noisy samples", alpha=0.5)
    plt.scatter(y_np[:, 0], y_np[:, 1], label="denoised samples", alpha=0.5)
    # plot covariance ellipsoids on top of the scatter
    means, variances = (model.get_means(), model.get_variances())
    means_t, variances_t = (model.get_means(t), model.get_variances(t))
    plot_cov_ellipses(means, variances, legend_str="(t=0)", color_str="blue")
    plot_cov_ellipses(means_t, variances_t, legend_str=f"(t={t:.3f})", color_str="red")
    plt.legend()
    frames = []
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        plt.savefig(tmpfile.name)
        plt.close(fig)
        frames.append(wandb.Image(tmpfile.name))

    # Log the images to wandb as an animation
    wandb.log({"animation": frames})
    wandb.finish()


def plot_cov_ellipses(
    means,
    variances,
    legend_str="",
    color_str="red",
    linestyle_str="dashed",
):
    num_clusters = variances.shape[0]
    # Get the current axes, or create a new one if none exists
    ax = plt.gca()
    for i in range(num_clusters):
        # Construct the covariance matrix based on variances
        cov = variances[i] * torch.eye(2)
        mean = means[i].detach().numpy()
        # Calculate the eigenvalues and eigenvectors
        eigvals, eigvecs = torch.linalg.eigh(
            cov
        )  # Use eigh since covariance matrices are symmetric
        eigvals = torch.real(eigvals)
        eigvecs = torch.real(eigvecs)
        # Width and height of the ellipse are proportional to sqrt of the eigenvalues
        # (width/height are 'diameters', not 'radii'... 2* is 'one sigma', 4* 'two sigma', etc)
        width, height = 4 * torch.sqrt(eigvals).detach().numpy()
        # Calculate the angle of the ellipse
        angle = torch.atan2(eigvecs[0, 1], eigvecs[0, 0]).item() * 180 / 3.14159
        # Plot the ellipse
        ell = patches.Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color_str,
            facecolor="none",
            linestyle=linestyle_str,
            label=f"cluster {i}" + legend_str,
        )
        # Add the ellipse to the axes
        ax.add_patch(ell)
    # Ensure the plot limits are updated to accommodate the ellipses
    plt.autoscale()


if __name__ == "__main__":
    # print("Testing that it works.")
    # test_working()
    print("Testing that it can sample.")
    test_sampling()
    # print("Testing that it can learn.")
    # test_learning(
    #     ExperimentConfig(
    #         device_str="cpu",
    #         batch_size=None,
    #         num_epochs=10000,
    #         optimizer=OptimizerConfig(
    #             algorithm=OptimizerType.ADAM, learning_rate=1e-3, weight_decay=0.0
    #         ),
    #     )
    # )
