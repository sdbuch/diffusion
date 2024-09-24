#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports

import dataclasses
import math
import os
import tempfile

import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from einops import repeat
from matplotlib import patches
from numpy.random import default_rng
from scipy.special import gamma
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from denoisers.gmm import GMMEqualVarianceDenoiser
from samplers.sde import BasicOUSampler
from util.configs import ExperimentConfig, OptimizerConfig
from util.types_custom import OptimizerType


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
            means, variances = (model.flatten(model.get_means()), model.get_variances())
            means_t, variances_t = (
                model.flatten(model.get_means(time)),
                model.get_variances(time),
            )
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


def test_learning(config: ExperimentConfig) -> None:
    wandb.init(
        project="gmm_test", name="gmm_test_learning", config=dataclasses.asdict(config)
    )
    # Parameters
    input_size = (1, 6, 1)
    num_clusters_gt = 1
    num_samples = math.prod(input_size) ** 2
    num_clusters = num_samples**2
    variance_gt = 0.5**2
    noise_upsampling_rate = 1000
    time_to_train_at = torch.log(torch.tensor(1 - variance_gt)) / -2 / 8

    # Hack some config logging since the ExperimentConfig is not set up right for this experiment...
    wandb.log(
        {
            "input_size": input_size,
            "num_clusters_gt": num_clusters_gt,
            "num_clusters": num_clusters,
            "variance_gt": variance_gt,
            "num_samples": num_samples,
            "noise_upsampling_rate": noise_upsampling_rate,
            "time_to_train_at": time_to_train_at,
        }
    )

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
    train_data, _, __ = model_gt.generate_samples(num_samples, t)
    model_mem = GMMEqualVarianceDenoiser(
        input_size, num_samples, init_variance=0.0, init_means=train_data
    )
    for param in model_mem.parameters():
        param.requires_grad = False
    train_data = train_data.to(device)
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    # Set up denoiser training.
    model = GMMEqualVarianceDenoiser(
        input_size, num_clusters, init_variance=config.noise_level
    )
    wandb.watch(model, log_freq=config.num_epochs // 10)

    # # Configure the optimizer!
    optimizer = config.optimizer.algorithm.value(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    # Training loop involves 'online sgd': adding indep. random noise to each minibatch
    # We will fix just one time to train at.
    model.to(device)
    model_gt.to(device)
    model_mem.to(device)
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
            denoised = model(noisy_X, time_to_train_at, compute_scores=True)
            denoised_mem = model_mem(noisy_X, time_to_train_at)
            denoised_gen = model_gt(noisy_X, time_to_train_at)
            loss_mem = (
                torch.sum((tiled_X - denoised_mem) ** 2)
                / batch_size
                / noise_upsampling_rate
            )
            loss_gen = (
                torch.sum((tiled_X - denoised_gen) ** 2)
                / batch_size
                / noise_upsampling_rate
            )
            loss = loss_fn(tiled_X, denoised) / batch_size / noise_upsampling_rate

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log(
                {
                    "batch MSE": loss.item(),
                    "memorization MSE": loss_mem.item(),
                    "generalization MSE": loss_gen.item(),
                    "variance": model.standard_dev.detach().item() ** 2,
                    "one sparse score": model.one_sparse_score,
                    "uniform score": model.uniform_score,
                }
            )

    # visualize the gt model, if we're in low-dims.
    # create a matplotlib scatterplot of the generated samples, show clusters
    model_gt.to("cpu")
    model.to("cpu")
    train_data = train_data.to("cpu")
    if math.prod(input_size) == 2:
        fig, ax = plt.subplots()
        x_np = model_gt.flatten(train_data).detach().numpy()
        noisy_x = (
            scale * train_data
            + torch.sqrt(variance)
            * torch.randn(train_data.shape, device=device, generator=generator).cpu()
        )
        noisy_x_np = model_gt.flatten(noisy_x).detach().numpy()
        plt.scatter(x_np[:, 0], x_np[:, 1], label="gt samples", alpha=0.5)
        plt.scatter(
            noisy_x_np[:, 0], noisy_x_np[:, 1], label="noisy gt samples", alpha=0.5
        )
        # plot covariance ellipsoids on top of the scatter
        means, variances = (
            model_gt.flatten(model_gt.get_means()),
            model_gt.get_variances(),
        )
        means_learned, variances_learned = (
            model.flatten(model.get_means()),
            model.get_variances(),
        )
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
    x, _ = model.generate_samples(num_samples, t)
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
    means, variances = (model.flatten(model.get_means()), model.get_variances())
    means_t, variances_t = (model.flatten(model.get_means(t)), model.get_variances(t))
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


def test_loss_values(config: ExperimentConfig) -> None:
    wandb.init(
        project="gmm_test",
        name="gmm_test_loss_values",
        config=dataclasses.asdict(config),
    )
    # Parameters
    input_size = (1, 8, 1)
    num_clusters_gt = 1
    variance_gt = 0.25**2
    # num_samples = 8 * math.prod(input_size) ** 2
    num_samples = 128
    noise_upsampling_rate = 200
    variance_to_time = lambda variance: torch.log(torch.tensor(1 - variance)) / -2
    time_for_gt_variance = variance_to_time(variance_gt)
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
    model_gen = GMMEqualVarianceDenoiser(
        input_size, num_clusters_gt, init_variance=variance_gt
    )
    for param in model_gen.parameters():
        param.requires_grad = False
    # data: get some samples from the ground-truth model
    t = torch.tensor(0.0)
    train_data, train_data_means, train_data_class_idxs = model_gen.generate_samples(
        num_samples, t
    )
    # train_dataset = TensorDataset(train_data)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
    # memorizing denoiser
    model_mem = GMMEqualVarianceDenoiser(
        input_size, num_samples, init_variance=0.0, init_means=train_data
    )
    for param in model_mem.parameters():
        param.requires_grad = False
    model_gen.to(device)
    model_mem.to(device)
    train_data = train_data.to(device)

    # Loop, calculate loss values.
    num_times = 4
    times = torch.logspace(
        2 * torch.log(time_for_gt_variance),
        torch.log(time_for_gt_variance) / 2,
        num_times,
        base=math.exp(1.0),
    )
    for idx_time in tqdm(range(num_times), desc="Time", position=0):
        cur_time = times[idx_time]
        scale, variance_t = model_gen.get_time_scaling(cur_time)
        variance = 1 - scale**2

        # starting M at num_clusters_gt runs into "coupon collector problem" issues,
        # where we might have a random subset of indices that doesn't cover all clusters.
        # so we start at a larger value, which tries to reduce the prob. this happens
        starting_M = 1 + math.ceil(
            3 * num_clusters_gt * math.log(num_clusters_gt)
        )  # prob.\ <= 1/9

        for idx in tqdm(
            range(starting_M, num_samples + 1),
            desc="Epoch",
            position=1,
            leave=False,
        ):
            shuffled_idxs = torch.randperm(num_samples)
            random_sample_idxs = shuffled_idxs[:idx]
            random_sample_idxs_complement = shuffled_idxs[idx:]
            random_data_sample = train_data[random_sample_idxs, ...].detach().clone()
            random_data_sample_complement = (
                train_data[random_sample_idxs_complement, ...].detach().clone()
            )
            model_pmem = GMMEqualVarianceDenoiser(
                input_size,
                int(idx),
                init_variance=0.0,
                # init_means=random_data_sample_means,
                init_means=random_data_sample,
            )
            for param in model_pmem.parameters():
                param.requires_grad = False
            model_pmem.to(device)
            tiled_X = repeat(
                train_data, "b c h w -> (r b) c h w", r=noise_upsampling_rate
            )
            noisy_X = scale * tiled_X + torch.sqrt(variance) * torch.randn(
                tiled_X.shape, device=device, generator=generator
            )
            denoised_mem = model_mem(noisy_X, cur_time)
            denoised_pmem = model_pmem(noisy_X, cur_time)
            denoised_gen = model_gen(noisy_X, cur_time)
            loss_mem = (
                torch.sum((tiled_X - denoised_mem) ** 2)
                / num_samples
                / noise_upsampling_rate
            )
            loss_pmem = (
                torch.sum((tiled_X - denoised_pmem) ** 2)
                / num_samples
                / noise_upsampling_rate
            )
            loss_gen = (
                torch.sum((tiled_X - denoised_gen) ** 2)
                / num_samples
                / noise_upsampling_rate
            )
            dim = math.prod(input_size)
            est_loss_gen = dim * variance_gt * variance / variance_t + loss_mem
            # # This formula is **wrong**
            # est_loss_pmem = (
            #     math.prod(input_size) * variance_gt * (1 - idx / num_samples) + loss_mem
            # )
            # # This formula is also **wrong**, but a bit closer
            # est_loss_pmem = (
            #     math.prod(input_size)
            #     * variance_gt
            #     * (1 - idx / num_samples)
            #     * (1 + num_clusters_gt / idx)
            #     + loss_mem
            # )
            # this formula is right! but it's not really closed-form enough
            in_sample_out_sample_dists = (
                torch.cdist(
                    random_data_sample_complement.flatten(start_dim=1).unsqueeze(0),
                    random_data_sample.flatten(start_dim=1).unsqueeze(0),
                )[0, ...]
                ** 2
            )
            est_loss_pmem = (
                in_sample_out_sample_dists.min(dim=1).values.sum() / num_samples
            )
            # est_loss_pmem = (
            #     math.prod(input_size)
            #     * variance_gt
            #     * (1 - idx / num_samples)
            #     * num_clusters_gt
            #     / idx
            #     / 4
            #     + loss_mem
            # )
            # est_loss_pmem = (
            #     2
            #     * gamma(dim / 2 + 1) ** (2 / dim)
            #     * gamma(2 / dim + 1)
            #     * (dim / (dim - 2)) ** (dim / 2)
            #     * variance_gt
            #     * (1 - idx / num_samples)
            #     * (num_clusters_gt / idx) ** (2 / dim)
            #     + loss_mem
            # )
            # est_loss_pmem = loss_pmem

            wandb.log(
                {
                    "memorization MSE": loss_mem.item(),
                    "partial memorization MSE": loss_pmem.item(),
                    "est. partial memorization MSE": est_loss_pmem.item(),
                    "generalizing MSE": loss_gen.item(),
                    "est. generalizing MSE": est_loss_gen.item(),
                    # "gen-to-mem gap": loss_gen.item() - loss_mem.item(),
                    "gen-to-pmem gap": loss_gen.item() - loss_pmem.item(),
                }
            )

    # visualize the gt model, if we're in low-dims.
    # create a matplotlib scatterplot of the generated samples, show clusters
    model_gen.to("cpu")
    if math.prod(input_size) == 2:
        fig, ax = plt.subplots()
        x_np = model_gen.flatten(train_data).detach().cpu().numpy()
        scale, _ = model_gen.get_time_scaling(time_for_gt_variance)
        variance = 1 - scale**2
        noisy_x = scale * train_data + torch.sqrt(variance) * torch.randn(
            train_data.shape, device=device, generator=generator
        )
        noisy_x_np = model_gen.flatten(noisy_x).detach().cpu().numpy()
        plt.scatter(x_np[:, 0], x_np[:, 1], label="gt samples", alpha=0.5)
        plt.scatter(
            noisy_x_np[:, 0], noisy_x_np[:, 1], label="noisy gt samples", alpha=0.5
        )
        # plot covariance ellipsoids on top of the scatter
        means, variances = (
            model_gen.flatten(model_gen.get_means()),
            model_gen.get_variances(),
        )
        means_t, variances_t = (
            model_gen.flatten(model_gen.get_means(time_for_gt_variance)),
            model_gen.get_variances(time_for_gt_variance),
        )
        plot_cov_ellipses(means, variances, legend_str=" gt", color_str="blue")
        plot_cov_ellipses(
            means_t,
            variances_t,
            legend_str=f" gt (time {time_for_gt_variance:0.2f})",
            color_str="red",
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
            }
        )
    wandb.finish()


if __name__ == "__main__":
    # print("Testing that it works.")
    # test_working()
    # print("Testing that it can sample.")
    # test_sampling()
    print("Testing that it can learn.")
    test_learning(
        ExperimentConfig(
            device_str="cuda",
            batch_size=None,
            num_epochs=10000,
            optimizer=OptimizerConfig(
                algorithm=OptimizerType.ADAM, learning_rate=1e-3, weight_decay=0.0
            ),
        )
    )
    # print("Testing the values of losses.")
    # test_loss_values(
    #     ExperimentConfig(
    #         device_str="cuda",
    #         batch_size=None,
    #     )
    # )
