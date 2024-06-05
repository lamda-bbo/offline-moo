import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from torch.utils.data import TensorDataset, DataLoader, random_split
from reweight import sigmoid_reweighting
from utils import plot_and_save_mse

class IOMModel(nn.Module):

    def __init__(self, g, discriminator_model, mmd_param, rep_model, rep_model_lr, 
                 forward_model, forward_model_lr=0.001, alpha=1.0, 
                 alpha_lr=0.01, overestimation_limit=0.5, particle_lr=0.05, 
                 particle_gradient_steps=50, entropy_coefficient=0.9, 
                 noise_std=0.0):

        super(IOMModel, self).__init__()

        # Initialize the models
        self.discriminator_model = discriminator_model
        self.rep_model = rep_model
        self.forward_model = forward_model

        # Initialize the optimizers
        self.forward_model_opt = optim.Adam(self.forward_model.parameters(), lr=forward_model_lr)
        self.rep_model_opt = optim.Adam(self.rep_model.parameters(), lr=rep_model_lr)
        self.discriminator_model_opt = optim.Adam(self.discriminator_model.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        # Initialize the alpha parameter and its optimizer
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha).float()))
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Set other attributes
        self.mmd_param = mmd_param
        self.overestimation_limit = overestimation_limit
        self.particle_lr = particle_lr
        self.particle_gradient_steps = particle_gradient_steps
        self.entropy_coefficient = entropy_coefficient
        self.noise_std = noise_std

        self.new_sample_size = 128

        # Variables for the particles (equivalent to tf.Variable)
        self.register_parameter('g', nn.Parameter(g.clone().detach()))
        self.register_parameter('g0', nn.Parameter(g.clone().detach()))

        self.epoch = 0
        # self.task = task

    def alpha(self):
        return self.log_alpha.exp()
    
    def optimize(self, x, steps, **kwargs):
        # gradient ascent on the conservatism
        for _ in range(steps):
            x.requires_grad_(True)

            # shuffle the designs for calculating entropy
            indices = torch.randperm(x.size(0))
            shuffled_xt = x[indices]

            # entropy using the gaussian kernel
            entropy = torch.mean((x - shuffled_xt) ** 2)

            # the predicted score according to the forward model
            with torch.no_grad():  # rep_model is used in inference mode, no need to track gradients
                xt_rep = self.rep_model(x)
                xt_rep = xt_rep / (torch.sqrt(torch.sum(xt_rep ** 2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
            score = self.forward_model(xt_rep, **kwargs)

            # the conservatism of the current set of particles
            losses = self.entropy_coefficient * entropy + score  # negative because we need gradient ascent
            
            grads = torch.autograd.grad(outputs=losses, inputs=x, grad_outputs=torch.ones_like(losses))

            # update the particles to maximize the conservatism
            with torch.no_grad():  # updates should not be tracked by autograd
                x.data = x.data - self.particle_lr * grads[0].detach()
                x.detach_()  # stop tracking history
                if x.grad is not None:
                    x.grad.zero_()  # reset gradients for the next iteration

        return x
    
    def train_step(self, x, y):
        # Corrupt the inputs with noise
        x = x + self.noise_std * torch.randn_like(x)

        statistics = dict()

        # Forward pass and compute gradients for rep_model and forward_model
        rep_x = self.rep_model(x)
        rep_x = rep_x / (rep_x.norm(dim=-1, keepdim=True) + 1e-6)

        d_pos_rep = self.forward_model(rep_x)
        mse = F.mse_loss(y, d_pos_rep)
        # statistics['train/mse_L2'] = mse.item()
        # mse_l1 = F.l1_loss(y, d_pos_rep)
        # statistics['train/mse_L1'] = mse_l1.item()

        # Evaluate how correct the rank of the model predictions are
        # rank_corr = spearman(y[:, 0], d_pos_rep[:, 0])
        # statistics['train/rank_corr'] = rank_corr

        # Calculate negative samples starting from the dataset
        x_neg = self.optimize(self.g, 1)
        self.g.data = x_neg.data

        # Log the task score for this set of x every 50 epochs
        # if self.epoch % 50 == 0:
        #     score_here = self.task.predict(self.g)
        #     score_here = self.task.denormalize_y(score_here)
        #     statistics['train/score_g_max'] = score_here

        # statistics['train/distance_from_start'] = (self.g - self.g0).norm().mean().item()
        x_neg = x_neg[:x.shape[0]]

        # Calculate the prediction error and accuracy of the model
        rep_x_neg = self.rep_model(x_neg)
        rep_x_neg = rep_x_neg / (rep_x_neg.norm(dim=-1, keepdim=True) + 1e-6)

        d_neg_rep = self.forward_model(rep_x_neg)
        overestimation = d_neg_rep[:, 0] - d_pos_rep[:, 0]
        # statistics['train/overestimation'] = overestimation.mean().item()
        # statistics['train/prediction'] = d_neg_rep.mean().item()

        # Build a Lagrangian for dual descent
        alpha_loss = self.alpha() * self.overestimation_limit - self.alpha() * overestimation.mean()
        statistics['train/alpha'] = self.alpha().item()

        mmd = F.mse_loss(rep_x.mean(dim=0), rep_x_neg.mean(dim=0))
        statistics['train/mmd'] = mmd.item()

        # mmd_before_rep = F.mse_loss(x.mean(dim=0), x_neg.mean(dim=0))
        # statistics['train/distance_before_rep'] = mmd_before_rep.item()

        # GAN loss
        valid = torch.ones(rep_x.shape[0], 1, device=x.device)
        fake = torch.zeros(rep_x.shape[0], 1, device=x.device)

        # Discriminator predictions
        dis_rep_x = self.discriminator_model(rep_x.detach()).view(rep_x.shape[0])
        dis_rep_x_neg = self.discriminator_model(rep_x_neg).view(rep_x.shape[0])

        real_loss = F.mse_loss(dis_rep_x, valid.squeeze())
        fake_loss = F.mse_loss(dis_rep_x_neg, fake.squeeze())
        d_loss = (real_loss + fake_loss) / 2
        # statistics['train/d_loss'] = d_loss.item()
        # statistics['train/real_loss'] = real_loss.item()
        # statistics['train/fake_loss'] = fake_loss.item()

        # statistics['train/square_dif_x_neg'] = F.mse_loss(rep_x, rep_x_neg).mean().item()

        # Accuracy real and fake
        # truth_pos = (self.discriminator_model(rep_x) >= 0.5).float().mean().item()
        # statistics['train/accuracy_real'] = truth_pos

        # truth_neg = (self.discriminator_model(rep_x_neg) < 0.5).float().mean().item()
        # statistics['train/accuracy_fake'] = truth_neg

        mmd_param = self.mmd_param

        model_loss1 = mse - d_loss * mmd_param
        total_loss1 = model_loss1.mean()
        statistics['train/loss1'] = total_loss1.item()

        model_loss2 = mse - d_loss * mmd_param
        total_loss2 = model_loss2.mean()
        # statistics['train/loss2```python
        # # continue from above code
        # statistics['train/loss2'] = total_loss2.item()

        # Backward pass and optimize
        self.rep_model_opt.zero_grad()
        self.forward_model_opt.zero_grad()
        self.discriminator_model_opt.zero_grad()

        # Backpropagation for different losses
        total_loss1.backward(retain_graph=True)
        self.forward_model_opt.step()

        total_loss2.backward(retain_graph=True)
        self.rep_model_opt.step()

        d_loss.backward(retain_graph=True)
        self.discriminator_model_opt.step()

        # Update alpha (dual variable)
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # return statistics
    
def iom_train_one_model(
        models, x, y, x_test, y_test, device,
        retrain = False,
        split_ratio = 0.9
):
    def check_models(models):
        for model in models.values():
            if not os.path.exists(model.save_path):
                return False
        return True
    
    if not retrain and check_models(models):
        for model in models.values():
            checkpoint = torch.load(model.save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.mse_loss = checkpoint['mse_loss']
            print(f"Successfully load trained model from {model.save_path} with MSE loss = {model.mse_loss}")
        return 
    
    for model in models.values():
        model.to(device)

    all_kwargs = dict(
        in_latent_space=False,
        particle_lr=0.05,
        particle_train_gradient_steps=50,
        particle_evaluate_gradient_steps=50,
        particle_entropy_coefficient=0.0,
        forward_model_activations=['relu', 'relu'],
        forward_model_hidden_size=2048,
        forward_model_final_tanh=False,
        forward_model_lr=0.0003,
        forward_model_alpha=0.1,
        forward_model_alpha_lr=0.01,
        forward_model_overestimation_limit=0.5,
        forward_model_noise_std=0.0,
        forward_model_batch_size=128,
        forward_model_val_size=200,
        forward_model_epochs=300,
        evaluation_samples=128,
        fast=False,
        latent_space_size=[128,1],
        rep_model_activations=['relu', 'relu'],
        rep_model_lr=0.0003,
        rep_model_hidden_size=2048,
        noise_input = [1, 10],
        mmd_param = 2
    )

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

    batch_size = 128

    # com_trainer = ConservativeObjectiveModel(forward_model=model, input_shape=x.shape[1:], args=model.args)

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

    top_k_indices = torch.argsort(y.squeeze())[:model.args.num_solutions]
    from copy import deepcopy
    initial_x = deepcopy(x[top_k_indices])

    trainer_kwargs = dict(
        g=initial_x.to(device),
        discriminator_model=models['DiscriminatorModel'],
        rep_model=models['RepModel'],
        rep_model_lr=0.0003,
        forward_model=models['ForwardModel'],
        forward_model_lr=0.0003,
        alpha=0.1,
        alpha_lr=0.01,
        overestimation_limit=0.5,
        particle_lr=0.05,
        particle_gradient_steps=50,
        entropy_coefficient=0.0,
        mmd_param = 2
    )

    iom_trainer = IOMModel(**trainer_kwargs)


    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    if model.args.train_mode == 'onlybest_1':
        criterion_sum_mse = lambda yhat, y, w: torch.sum(w * torch.mean((yhat-y)**2, dim=1)) * (1 / 0.2)
    else:
        criterion_sum_mse = lambda yhat, y, w: torch.sum(w * torch.mean((yhat-y)**2, dim=1))

    total_step = len(train_dataloader)

    n_epochs = 300

    train_mse = []
    val_mse = []
    test_mse = []
    epochs = list(range(n_epochs))
    
    min_loss = np.PINF

    def calc_output(input):
        rep_model = models['RepModel']
        main_model = models['ForwardModel']

        x_rep = rep_model(input)
        x_rep = x_rep / (torch.sqrt(torch.sum(x_rep ** 2, dim=-1, keepdim=True) + 1e-6) + 1e-6)
        return main_model(x_rep)

    for epoch in range(n_epochs):
        for i, (batch_x, batch_y, batch_w) in enumerate(train_dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            iom_trainer.train_step(batch_x, batch_y)

        with torch.no_grad():
            y_all = torch.zeros((0, 1)).to(device)
            outputs_all = torch.zeros((0, 1)).to(device)
            for batch_x, batch_y, _ in train_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = calc_output(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_loss = criterion(outputs_all, y_all)
            print ('Epoch [{}/{}], Loss: {:}'
                .format(epoch+1, n_epochs, train_loss.item()))
            
            train_mse.append(train_loss.item())

        with torch.no_grad():
            y_all = torch.zeros((0, 1)).to(device)
            outputs_all = torch.zeros((0, 1)).to(device)

            for batch_x, batch_y in val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = calc_output(batch_x)
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
                outputs = calc_output(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))
            
            test_loss = criterion(outputs_all, y_all)
            test_mse.append(test_loss.item())
            print('test MSE is: {}'.format(test_loss.item()))

            if val_loss.item() < min_loss:
                for model in models.values():
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