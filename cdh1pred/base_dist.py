"""
In parts from https://github.com/KatherLab/STAMP/blob/main/stamp/modeling/marugoto/transformer/base.py
"""

from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, EarlyStoppingCallback,
    MixedPrecision, AMPMode, OptimWrapper
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .data import make_dist_dataset
from .model_dist import TransMILDist


def train_dist(
        bags,
        targets,
        valid_idxs,
        n_epoch: int = 32,
        patience: int = 8,
        path=None,
        batch_size: int = 64,
        cores: int = 8,
        plot: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    target_enc, targs = targets
    train_ds = make_dist_dataset(
        bags=bags[~valid_idxs],
        target_enc=target_enc,
        targs=targs[~valid_idxs],
        bag_size=512)

    valid_ds = make_dist_dataset(
        bags=bags[valid_idxs],
        target_enc=target_enc,
        targs=targs[valid_idxs],
        bag_size=None)

    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=cores,
        drop_last=len(train_ds) > batch_size,
        device=device, pin_memory=device.type == "cuda"
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=cores,
        device=device, pin_memory=device.type == "cuda"
    )
    batch = train_dl.one_batch()
    feature_dim = batch[0].shape[-1]

    # for binary classification num_classes=2
    model = TransMILDist(
        num_classes=len(target_enc.categories_[0]), input_dim=feature_dim,
        dim=4, depth=2, heads=2, k=10, mlp_dim=512, dropout=.0
    ) #dim=512, heads=8

    # model = torch.compile(model)
    model.to(device)
    print(f"Model: {model}", end=" ")
    print(f"[Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}]")

    # weigh inversely to class occurrences
    counts = pd.Series(targs[~valid_idxs]).value_counts()
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32, device=device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl, device=device)

    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        opt_func=partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=[RocAuc()],
        path=path,
    )

    cbs = [
        SaveModelCallback(monitor='valid_loss', fname=f'best_valid'),
        EarlyStoppingCallback(monitor='valid_loss', patience=patience),
        CSVLogger(),
    ]
    learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)

    # Plot training and validation losses as well as learning rate schedule
    if plot:
        path_plots = path / "plots"
        path_plots.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_loss()
        plt.savefig(path_plots / 'losses_plot.png')
        plt.close()

        learn.recorder.plot_sched()
        plt.savefig(path_plots / 'lr_scheduler.png')
        plt.close()

    return learn


def deploy_dist(
        test_df,
        learn,
        target_label=None,
        device=torch.device('cpu')
):
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'
    # assert (len(add_label)
    #        == (n := len(learn.dls.train.dataset._datasets[-2]._datasets))), \
    #    f'not enough additional feature labels: expected {n}, got {len(add_label)}'
    if target_label is None:
        target_label = learn.target_label

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]

    test_ds = make_dist_dataset(
        bags=test_df.slide_path.values,
        target_enc=target_enc,
        targs=test_df[target_label].values,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1,
        device=device, pin_memory=device.type == "cuda")

    # removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}': patient_preds[:, i]
           for i, cat in enumerate(categories)}})

    # calculate loss
    patient_preds = patient_preds_df[[
        f'{target_label}_{cat}' for cat in categories]].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs),
        reduction='none')

    patient_preds_df['pred'] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        *(f'{target_label}_{cat}' for cat in categories),
        'loss']]
    patient_preds_df = patient_preds_df.sort_values(by='loss')

    return patient_preds_df
