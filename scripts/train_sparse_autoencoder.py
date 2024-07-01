#!/usr/bin/env python3
import argparse
import functools
from collections import defaultdict
import operator
from typing import Callable
import os
import pickle

import torch
from openretina.neuron_data_io import make_final_responses
from openretina.models.autoencoder import SparsityMSELoss, Autoencoder
from openretina.utils.h5_handling import load_h5_into_dict
from openretina.hoefling_2024.configs import model_config
from openretina.hoefling_2024.models import SFB3d_core_SxF3d_readout
from openretina.hoefling_2024.data_io import (
    get_chirp_dataloaders,
    get_mb_dataloaders,
    natmov_dataloaders_v2,
)
from openretina.cyclers import LongCycler


def parse_args():
    parser = argparse.ArgumentParser(description="Model training")

    parser.add_argument("--data_folder", type=str, help="Path to the base data folder", default="/Data/fd_export")
    parser.add_argument("--save_folder", type=str, help="Path were to save outputs", default=".")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--datasets",
        type=str,
        default="natural",
        help="Underscore separated list of datasets, " "e.g. 'natural', 'chirp', 'mb', or 'natural_mb'",
    )

    return parser.parse_args()

def main(
    data_folder: str,
    save_folder: str,
    device: str,
    datasets: str,
) -> None:
    dataset_names_list = datasets.split("_")
    for name in dataset_names_list:
        if name not in {"natural", "chirp", "mb"}:
            raise ValueError(f"Unsupported dataset name {name}")

    movies_path = os.path.join(data_folder, "2024-01-11_movies_dict_8c18928.pkl")
    with open(movies_path, "rb") as f:
        movies_dict = pickle.load(f)

    data_path_responses = os.path.join(data_folder, "2024-03-28_neuron_data_responses_484c12d_djimaging.h5")
    responses = load_h5_into_dict(data_path_responses)

    dataloader_list = []

    dataloader_name_to_function: dict[str, Callable] = {
        "chirp": get_chirp_dataloaders,
        "mb": get_mb_dataloaders,
        "natural": functools.partial(natmov_dataloaders_v2, movies_dictionary=movies_dict, seed=1000),
    }
    for dataset_name in dataset_names_list:
        data_dict = make_final_responses(responses, response_type=dataset_name)  # type: ignore
        dataloader_fn = dataloader_name_to_function[dataset_name]
        dataloader = dataloader_fn(data_dict, train_chunk_size=100)
        dataloader_list.append(dataloader)

    def get_joint_dataloader(dataloader_list: list, set_name: str):
        dict_list = [d[set_name] for d in dataloader_list if set_name in d]
        if len(dict_list) == 0:
            print(f"Warn: Using training data for {set_name=}")
            dict_list = [dataloader_list[0]["train"]]

        dataloader = functools.reduce(operator.or_, dict_list)
        return dataloader

    joint_dataloaders = {
        "train": get_joint_dataloader(dataloader_list, "train"),
        "validation": get_joint_dataloader(dataloader_list, "validation"),
        "test": get_joint_dataloader(dataloader_list, "test"),
    }
    print("Initialized dataloaders")

    model = SFB3d_core_SxF3d_readout(**model_config, dataloaders=joint_dataloaders, seed=42)  # type: ignore
    model.to(device)
    print("Init model")

    # generate model outputs
    # We currently generate outputs for each readout key for each training example
    # This likely results in duplicate examples, as the training examples for each readout key are
    # the same or at least similar.
    outputs_model = []
    readout_keys_list = model.readout.readout_keys()
    for batch_no, (_, data) in enumerate(LongCycler(joint_dataloaders["train"])):
        all_activations_list = []
        for readout_key in readout_keys_list:
            with torch.no_grad():
                activations = model.forward(data.inputs, readout_key)
                all_activations_list.append(activations)
        all_activations = torch.cat(all_activations_list, dim=-1)
        outputs_model.append(all_activations)

    # How to treat activations across different session?
    # - Train independent autoencoders?
    # - Same autoencoder but with zero weights for neurons not in that session?
    # - Just sample all data, or is input data the same (ignore outputs, feed input through all data_keys)
    sparsity_mse_loss = SparsityMSELoss(sparsity_factor=0.1)
    num_model_neurons = outputs_model[0].shape[-1]
    sparse_autoencoder = Autoencoder(num_model_neurons, 1000, sparsity_mse_loss)

    optimizer = sparse_autoencoder.configure_optimizers()
    loss_array = []
    for i, activations in enumerate(outputs_model):
        optimizer.zero_grad()
        batch = (activations, "foo")
        loss = sparse_autoencoder.training_step(batch, i)
        loss.backward()
        optimizer.step()
        print(f"{float(loss):.5f}")
    avg_loss = sum(loss_array) / len(loss_array)
    print(f"Avg loss after {len(loss_array)} iterations: {avg_loss:.5f}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))