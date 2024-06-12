import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os
from utils import plot_and_save_mse
from reweight import sigmoid_reweighting
from tqdm import tqdm
import time

class ConservativeObjectiveModel(nn.Module):
    def __init__(self,forward_model,input_shape,args,
                 forward_model_lr=0.003, alpha=0.1,
                 alpha_lr=0.01, overestimation_limit=0.5,
                 particle_lr=0.05, particle_gradient_steps=50,
                 entropy_coefficient=0.0, noise_std=0.0) -> None:
        
        super(ConservativeObjectiveModel, self).__init__()
        
        self.forward_model = forward_model
        self.forward_model_optimizer = torch.optim.Adam(forward_model.parameters(), lr=forward_model_lr)

        alpha = torch.tensor(alpha)
        self.log_alpha = torch.nn.Parameter(torch.log(alpha))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.overestimation_limit = overestimation_limit
        self.particle_lr = particle_lr * np.sqrt(np.prod(input_shape))
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

        self.args = args
        
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def obtain_x_neg(self, x, steps, **kwargs):

        # gradient ascent on the conservatism
        def gradient_step(xt):

            # shuffle the designs for calculating entropy
            indices = torch.randperm(xt.size(0))
            shuffled_xt = xt[indices]

            # entropy using the gaussian kernel
            entropy = torch.mean((xt - shuffled_xt) ** 2)

            # the predicted score according to the forward model
            score = self.forward_model(xt, **kwargs)

            # the conservatism of the current set of particles
            losses = self.entropy_coefficient * entropy + score

            # calculate gradients for each element separately
            # print(losses.shape)
            # grads = []
            # for i, loss in enumerate(losses):
            #     grad = torch.autograd.grad(loss, xt, create_graph=True)[0][i]
            #     # print('111', grad)
            #     grads.append(grad)
            grads = torch.autograd.grad(outputs=losses, inputs=xt, grad_outputs=torch.ones_like(losses))

            # print(torch.stack(grads).shape)
            # print('xt.shape:', xt.shape)
            # update the particles to maximize the conservatism
            with torch.no_grad():
                xt.data = xt.data - self.particle_lr * grads[0].detach()
                xt.detach_()
                if xt.grad is not None:
                    xt.grad.zero_()
            # print('xt.shape:', xt.shape)
            return xt.detach()

        xt = torch.tensor(x, requires_grad=True)
        
        for n_iter in range(steps):
            xt = gradient_step(xt)
            xt.requires_grad = True
        return xt
    
    def train_step(self, x, y):
        # corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x)
        x, y = Variable(x, requires_grad=True), Variable(y, requires_grad=False)

        # calculate the prediction error and accuracy of the model
        d_pos = self.forward_model(x)
        mse = F.mse_loss(d_pos.float(), y.float())

        # calculate negative samples starting from the dataset
        x_neg = self.obtain_x_neg(x, self.particle_gradient_steps)
        # print(x_neg)
        # calculate the prediction error and accuracy of the model
        d_neg = self.forward_model(x_neg)
        overestimation = d_pos[:, 0] - d_neg[:, 0]

        # build a lagrangian for dual descent
        alpha_loss = (self.alpha * self.overestimation_limit -
                    self.alpha * overestimation)

        # loss that combines maximum likelihood with a constraint
        model_loss = mse + self.alpha * overestimation
        if self.args.train_data_mode == 'onlybest_1':
            model_loss = model_loss * (1 / 0.2)
        total_loss = model_loss.mean()
        alpha_loss = alpha_loss.mean()

        # calculate gradients using the model
        alpha_grads = torch.autograd.grad(alpha_loss, self.log_alpha, retain_graph=True)[0]
        model_grads = torch.autograd.grad(total_loss, self.forward_model.parameters())

        # take gradient steps on the model
        with torch.no_grad():
            self.log_alpha.grad = alpha_grads
            self.alpha_opt.step()
            self.alpha_opt.zero_grad()

            for param, grad in zip(self.forward_model.parameters(), model_grads):
                param.grad = grad
            self.forward_model_optimizer.step()
            self.forward_model_optimizer.zero_grad()

def com_train_one_model(
        model, x, y, x_test, y_test, device,
        retrain = False,
        lr = 0.001,
        lr_decay = 0.98,
        n_epochs = 3,
        batch_size = 128,
        split_ratio = 0.9,
):
    print('here')
    if not retrain and os.path.exists(model.save_path):
        checkpoint = torch.load(model.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mse_loss = checkpoint['mse_loss']
        print(f"Successfully load trained model from {model.save_path} with MSE loss = {model.mse_loss}")
        return 
    
    batch_size = 128
    n_epochs = 200
    
    # def update_lr(optimizer, lr):
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr
    
    model = model.to(device)
    y = y.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x).to(dtype=torch.float32)
    if isinstance(y, np.ndarray):
        y = torch.Tensor(y).to(dtype=torch.float32)
    if isinstance(x_test, np.ndarray):
        x_test = torch.Tensor(x_test).to(dtype=torch.float32)
    if isinstance(y_test, np.ndarray):
        y_test = torch.Tensor(y_test).to(dtype=torch.float32)

    com_trainer = ConservativeObjectiveModel(forward_model=model, input_shape=x.shape[1:], args=model.args)

    tensor_dataset = TensorDataset(x, y)
    lengths = [int(split_ratio*len(tensor_dataset)), len(tensor_dataset)-int(split_ratio*len(tensor_dataset))]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)

    x_torch, y_torch = train_dataset[:]
    if model.args.reweight_mode == 'sigmoid':
        weights = sigmoid_reweighting(y_torch, quantile=model.args.sigmoid_quantile)
    else:
        weights = torch.ones(len(x_torch))
        
    train_dataset = TensorDataset(x_torch, y_torch, weights)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f"Begin to train No.{model.which_obj} models")

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if model.args.train_mode == 'onlybest_1':
        criterion_sum_mse = lambda yhat, y, w: torch.sum(w * torch.mean((yhat-y)**2, dim=1)) * (1 / 0.2)
    else:
        criterion_sum_mse = lambda yhat, y, w: torch.sum(w * torch.mean((yhat-y)**2, dim=1))

    total_step = len(train_dataloader)

    train_mse = []
    val_mse = []
    test_mse = []
    epochs = list(range(n_epochs))
    
    min_loss = np.PINF

    for epoch in range(n_epochs):
        for i, (batch_x, batch_y, batch_w) in enumerate(train_dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            com_trainer.train_step(batch_x, batch_y)

        with torch.no_grad():
            y_all = torch.zeros((0, 1)).to(device)
            outputs_all = torch.zeros((0, 1)).to(device)
            for batch_x, batch_y, _ in train_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_loss = criterion(outputs_all, y_all)
            print ('Epoch [{}/{}], Loss: {:}'
                .format(epoch+1, n_epochs, train_loss.item()))
            
            train_mse.append(train_loss.item())
        
        # lr *= lr_decay
        # update_lr(optimizer, lr)

        with torch.no_grad():
            y_all = torch.zeros((0, 1)).to(device)
            outputs_all = torch.zeros((0, 1)).to(device)

            for batch_x, batch_y in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))
            
            val_loss = criterion(outputs_all, y_all)
            val_mse.append(val_loss.item())

            print('Validataion MSE is: {}'.format(val_loss.item()))

            plot_and_save_mse(model, val_mse, 'val')

            y_all = torch.zeros((0, 1)).to(device)
            outputs_all = torch.zeros((0, 1)).to(device)

            for batch_x, batch_y in test_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))
            
            test_loss = criterion(outputs_all, y_all)
            test_mse.append(test_loss.item())
            print('test MSE is: {}'.format(test_loss.item()))

            if val_loss.item() < min_loss:
                min_loss = val_loss.item()
                model = model.to('cpu')
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'mse_loss': val_loss.item(),
                    'test_loss': test_loss.item()
                }
                torch.save(checkpoint, model.save_path)
                model = model.to(device)

            plot_and_save_mse(model, test_mse, 'test')
    

    