import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

from off_moo_baselines.data import spearman_correlation, tkwargs
from off_moo_baselines.util.pcgrad import PCGrad


def get_trainer(train_mode):
    if train_mode.lower() == "gradnorm":
        trainer = End2EndGradNormTrainer
    elif train_mode.lower() == "pcgrad":
        trainer = End2EndPcGradTrainer
    else:
        trainer = End2EndBaseTrainer
    return trainer


class End2EndBaseTrainer:
    def __init__(self, forward_model, config):
        self.config = config

        if config["data_pruning"] and not isinstance(
            config["data_preserved_ratio"], float
        ):
            config["data_preserved_ratio"] = 0.2

        self.forward_lr = config["forward_lr"]
        self.n_epochs = config["n_epochs"]

        self.use_wandb = config["use_wandb"]
        self.forward_model = forward_model

        self.forward_opt = Adam(forward_model.parameters(), lr=config["forward_lr"])
        # Add cosine annealing scheduler
        self.forward_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.forward_opt, T_max=self.n_epochs, eta_min=0  # minimum learning rate
        )

        self.train_criterion = (
            lambda yhat, y: torch.sum(torch.mean((yhat - y) ** 2, dim=1))
            * (1 / config["data_preserved_ratio"])
            if config["data_pruning"]
            else torch.sum(torch.mean((yhat - y) ** 2, dim=1))
        )
        self.mse_criterion = nn.MSELoss()

        self.save_ckpt_metric = config["save_ckpt_metric"]

    def _evaluate_performance(
        self, statistics, epoch, train_loader, val_loader, test_loader
    ):
        self.forward_model.eval()
        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            for (
                batch_x,
                batch_y,
            ) in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.forward_model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_mse = self.mse_criterion(outputs_all, y_all)
            train_corr = spearman_correlation(outputs_all, y_all)

            statistics["train/mse"] = train_mse.item()
            for i in range(self.n_obj):
                statistics[f"train/rank_corr_{i + 1}"] = train_corr[i].item()

            print(
                "Epoch [{}/{}], MSE: {:}".format(
                    epoch + 1, self.n_epochs, train_mse.item()
                )
            )

        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
            outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.forward_model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))

            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)

            statistics["valid/mse"] = val_mse.item()
            for i in range(self.n_obj):
                statistics[f"valid/rank_corr_{i + 1}"] = val_corr[i].item()

            val_corr_avg = torch.mean(val_corr).item()
            print("Valid MSE: {:}".format(val_mse.item()))
            print("Valid rank_corr: {:}".format(val_corr_avg))

            if len(test_loader) != 0:
                y_all = torch.zeros((0, self.n_obj)).to(**tkwargs)
                outputs_all = torch.zeros((0, self.n_obj)).to(**tkwargs)

                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(**tkwargs)
                    batch_y = batch_y.to(**tkwargs)

                    y_all = torch.cat((y_all, batch_y), dim=0)
                    outputs = self.forward_model(batch_x)
                    outputs_all = torch.cat((outputs_all, outputs))

                test_mse = self.mse_criterion(outputs_all, y_all)
                test_corr = spearman_correlation(outputs_all, y_all)

                statistics["test/mse"] = test_mse.item()
                for i in range(self.n_obj):
                    statistics[f"test/rank_corr_{i + 1}"] = test_corr[i].item()

                print("Test MSE: {:}".format(test_mse.item()))

            if self.save_ckpt_metric.lower() == "mse":
                if val_mse.item() < self.min_mse:
                    print("🌸 New best epoch! 🌸")
                    self.min_mse = val_mse.item()
                    self.forward_model.save(val_mse=self.min_mse)
            elif self.save_ckpt_metric.lower() == "rank_corr":
                if val_corr_avg > self.max_rank_corr:
                    print("🌸 New best epoch! 🌸")
                    self.max_rank_corr = val_corr_avg
                    self.forward_model.save(val_rank_corr=val_corr_avg)
            else:
                raise NotImplementedError
        return statistics

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):
        if not retrain_model and os.path.exists(self.forward_model.save_path):
            self.forward_model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        self.max_rank_corr = -1.0
        statistics = {}

        for epoch in range(self.n_epochs):
            self.forward_model.train()

            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                self.forward_opt.zero_grad()
                outputs = self.forward_model(batch_x)
                loss = self.train_criterion(outputs, batch_y)
                losses.append(loss.item() / batch_x.size(0))
                loss.backward()
                self.forward_opt.step()

            # Step the scheduler after each epoch
            self.forward_scheduler.step()

            statistics["train/loss/mean"] = np.array(losses).mean()
            statistics["train/loss/std"] = np.array(losses).std()
            statistics["train/loss/max"] = np.array(losses).max()

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            # Update learning rate statistics
            statistics["train/lr"] = self.forward_scheduler.get_last_lr()[0]

            if self.use_wandb:
                statistics["train_epoch"] = epoch
                wandb.log(statistics)


class End2EndGradNormTrainer(End2EndBaseTrainer):
    def __init__(self, forward_model, config):
        super(End2EndGradNormTrainer, self).__init__(forward_model, config)
        self.alpha = config["gradient_alpha"]
        self.weight_lr = config["weight_lr"]
        self.weight_lr_decay = config["weight_lr_decay"]

        self.norm_layer = forward_model.layers[-1]
        self.forward_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.forward_opt, 
            T_max=self.n_epochs,
            eta_min=0
        )

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):
        def update_weight_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if not retrain_model and os.path.exists(self.forward_model.save_path):
            self.forward_model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        iters = 0
        self.min_mse = float("inf")
        self.max_rank_corr = -1.
        statistics = {}

        for epoch in range(self.n_epochs):
            gradnorm_losses = []
            weighted_losses = []
            task_losses = []

            self.forward_model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                outputs = self.forward_model(batch_x)

                losses = []
                for i in range(batch_y.shape[1]):
                    task_loss = self.mse_criterion(batch_y[:, i].float(), outputs[:, i].float())
                    losses.append(task_loss)
                losses = torch.stack(losses)  # [n_tasks]

                if iters == 0:
                    weights = torch.ones(self.n_obj, device=batch_x.device)
                    weights = torch.nn.Parameter(weights)
                    self.weight_opt = torch.optim.Adam([weights], lr=self.weight_lr)
                    l0 = losses.detach()

                weighted_loss = torch.sum(weights * losses)
                
                self.forward_opt.zero_grad()
                weighted_loss.backward(retain_graph=True)

                norms = []
                for i in range(self.n_obj):
                    grad_i = torch.autograd.grad(
                        losses[i], 
                        self.norm_layer.parameters(),
                        create_graph=True,  
                        retain_graph=True
                    )[0]
                    norms.append(torch.sqrt(torch.sum(grad_i ** 2)))
                norms = torch.stack(norms)  # [n_tasks]

                loss_ratio = losses.detach() / l0
                inverse_train_rate = loss_ratio / loss_ratio.mean()

                grad_norm_mean = norms.mean()
                target_norms = grad_norm_mean * (inverse_train_rate ** self.alpha)

                gradnorm_loss = torch.sum(torch.abs(norms - target_norms.detach()))

                self.weight_opt.zero_grad()
                gradnorm_loss.backward()
                self.weight_opt.step()

                with torch.no_grad():
                    weights.data = (weights.data / weights.data.sum() * self.n_obj)

                gradnorm_losses.append(gradnorm_loss.item())
                weighted_losses.append(weighted_loss.item())
                task_losses.append([l.item() for l in losses])

                self.forward_opt.step()
                iters += 1

            self.forward_scheduler.step()
            self.weight_lr *= self.weight_lr_decay
            update_weight_lr(self.weight_opt, self.weight_lr)

            statistics["train/gradnorm_loss/mean"] = np.mean(gradnorm_losses)
            statistics["train/weighted_loss/mean"] = np.mean(weighted_losses)
            
            task_losses = np.array(task_losses)
            for i in range(self.n_obj):
                statistics[f"train/task_{i}_loss/mean"] = task_losses[:, i].mean()
                statistics[f"train/task_{i}_weight"] = weights[i].item()

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            statistics["train/lr"] = self.forward_scheduler.get_last_lr()[0]
            statistics["train/weight_lr"] = self.weight_lr

            if self.use_wandb:
                statistics["train_epoch"] = epoch
                wandb.log(statistics)


class End2EndPcGradTrainer(End2EndBaseTrainer):
    def __init__(self, forward_model, config):
        super(End2EndPcGradTrainer, self).__init__(forward_model, config)
        self.forward_opt = PCGrad(self.forward_opt)

    def launch(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        retrain_model: bool = True,
    ):
        if not retrain_model and os.path.exists(self.forward_model.save_path):
            self.forward_model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_mse = float("inf")
        self.max_rank_corr = -1.
        statistics = {}

        for epoch in range(self.n_epochs):
            self.forward_model.train()

            losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(**tkwargs)
                batch_y = batch_y.to(**tkwargs)
                if self.n_obj is None:
                    self.n_obj = batch_y.shape[1]

                self.forward_opt.zero_grad()
                outputs = self.forward_model(batch_x)
                loss = []
                for i in range(batch_y.shape[1]):
                    loss.append(
                        self.mse_criterion(batch_y[:, i].float(), outputs[:, i].float())
                        * (
                            1 / self.config["data_preserved_ratio"]
                            if self.config["data_pruning"]
                            else 1
                        )
                    )
                assert len(loss) == self.n_obj

                losses.append(
                    np.array([single_loss.item() for single_loss in loss]).mean()
                    / batch_x.size(0)
                )
                self.forward_opt.pc_backward(loss)
                self.forward_opt.step()

            statistics["train/loss/mean"] = np.array(losses).mean()
            statistics["train/loss/std"] = np.array(losses).std()
            statistics["train/loss/max"] = np.array(losses).max()

            self._evaluate_performance(
                statistics, epoch, train_loader, val_loader, test_loader
            )

            if self.use_wandb:
                statistics["train_epoch"] = epoch
                wandb.log(statistics)
