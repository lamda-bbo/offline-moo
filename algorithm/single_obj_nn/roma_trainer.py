import os 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader, random_split

from utils import plot_and_save_mse 
from reweight import sigmoid_reweighting
from .roma_model import DoubleheadModel

class ROMATrainer(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        perturb_fn,
        is_discrete,
        sol_x,
        sol_y,
        device,
        # sol_x_opt,
        # mu_x,
        # st_x,
        coef_stddev,
        temp_model,
        steps_per_update,
        inner_lr=1e-3,
        region=2.,
        max_x=None,
        max_y=1.,
        lr=1.,
        alpha=1.
    ):
        super(ROMATrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.perturb_fn = perturb_fn
        self.is_discrete = is_discrete
        self.init_sol_x = sol_x.to(device)
        self.sol_x = sol_x.clone().detach().to(device).requires_grad_(True)
        # self.sol_x_opt = sol_x_opt
        self.sol_x_opt = torch.optim.Adam([self.sol_x], lr = 3e-3)
        self.coef_stddev = coef_stddev
        self.sol_x_samples = sol_x.size(0)
        self.inner_lr = inner_lr
        self.temp_model = temp_model.to(device)
        self.prev_sol_x = sol_x.clone().detach().to(device)
        self.steps_per_update = steps_per_update
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.region = region
        self.max_x = max_x
        self.max_y = max_y
        self.sol_y = sol_y.clone().detach().to(device).requires_grad_(True)
        self.temp_x = sol_x.clone().detach().requires_grad_(True)
        # self.mu_x = mu_x
        # self.st_x = st_x
        self.lr = lr
        self.alpha = alpha
        self.device = device
    
    def get_sol_x(self):
        return self.sol_x
    
    @torch.enable_grad()
    def train_step(self, x, y):
        self.model.train()
        if self.perturb_fn is not None:
            x = self.perturb_fn(x)
            
        self.temp_model.load_state_dict(self.model.state_dict())

        x = self.perturb_fn(x)

        for _ in range(self.steps_per_update):
            # mean, std = self.model(x)
            # d = torch.distributions.Normal(mean, std)
            d = self.model.get_distribution(x)
            temp_d = self.temp_model.get_distribution(x)

            loss_total = d.log_prob(y).mean()
            if self.model.args.train_data_mode == 'onlybest_1':
                loss_total = loss_total * (1 / 0.2)
            self.optimizer.zero_grad() 
            loss_total.backward() 

            for param, temp_param in zip(self.model.parameters(), self.temp_model.parameters()):
                if param.grad is not None and param.grad.size(0) > 1:
                    grad_norm = param.grad.norm(2)
                    param_norm = temp_param.data.norm(2)
                    normalized_grad = param.grad * (param_norm / grad_norm)
                    # normalized_grad = param.grad
                    # print(self.inner_lr / self.steps_per_update, normalized_grad)
                    param.data -= (self.inner_lr / self.steps_per_update) * normalized_grad
            # assert 0
        perturbations = {
            name: param - temp_param for (name, param), temp_param in zip(
                self.model.named_parameters(), self.temp_model.parameters())
        }

        # Outer optimization
        # Calculate the statistics for the unperturbed model
        d = self.model.get_distribution(x)
        temp_d = self.temp_model.get_distribution(x)

        loss_nll = -d.log_prob(y).mean()
        # rank_correlation = self.spearman(y[:, 0], mean[:, 0])

        # Take gradient steps on the model using the perturbations
        self.optimizer.zero_grad()
        loss_nll.backward()
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Clip gradients to prevent exploding gradients
                param.grad = torch.clamp(param.grad, -1.0, 1.0)
        self.optimizer.step()

        for param, (name, perturb) in zip(self.model.parameters(), perturbations.items()):
            param.data -= perturbations[name]

        return {'loss': loss_nll.item()}


    def valid_step(self, x, y):
        if self.perturb_fn is not None:
            x = self.perturb_fn(x)

        d = self.model.get_distribution(x)
        loss_nll = -d.log_prob(y).mean() 

        return {'loss': loss_nll.item()}
    
    def init_step(self):
        self.temp_model.load_state_dict(self.model.state_dict())

    @torch.enable_grad() 
    def fix_step(self):
        self.temp_model.to(self.device)
        # weight perturbation (current, 1 step)
        for _ in range(self.steps_per_update * 5):
            self.temp_model.train()  # Set the model to training mode
            inp = self.sol_x.to(self.device).requires_grad_(True)
            prev_inp = self.init_sol_x.to(self.device).requires_grad_(True)
            sol_d = self.temp_model.get_distribution(inp)
            prev_sol_d = self.temp_model.get_distribution(prev_inp)
            loss_sol_x = sol_d.mean - self.coef_stddev * torch.log(sol_d.stddev)
            
            # Backpropagate the loss
            self.optimizer.zero_grad()  # Clear gradients
            loss_sol_x.sum().backward(retain_graph=True)  # Compute gradients
            
            sol_x_grad = self.sol_x.grad
            loss_pessimism_gradnorm = sol_x_grad.view(self.sol_x_samples, -1).norm(dim=1)
            loss_pessimism_con = -sol_d.log_prob(self.sol_y)
            loss_total = loss_pessimism_gradnorm.mean() + self.alpha * loss_pessimism_con.mean()
            if self.model.args.train_data_mode == 'onlybest_1':
                loss_total = loss_total * (1 / 0.2)
            
            # Compute gradients with respect to the total loss
            self.optimizer.zero_grad()  # Clear gradients
            loss_total.backward()  # Compute gradients

            # Update the weights
            for temp_param, origin_param in zip(self.temp_model.parameters(), self.model.parameters()):
                if temp_param.grad is not None and temp_param.grad.numel() > 1:  # Equivalent to tf.shape(grad)[0] > 1
                    grad = temp_param.grad * (origin_param.norm() / (temp_param.grad.norm() + 1e-6))
                    temp_param.data -= self.inner_lr / self.steps_per_update / 5 * grad

        # Calculate pessimism loss for the statistics
        self.temp_model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients for this part
            inp = self.sol_x
            prev_inp = self.prev_sol_x
            sol_d = self.temp_model.get_distribution(inp)
            prev_sol_d = self.temp_model.get_distribution(prev_inp)

            loss_pessimism = (sol_d.mean - self.coef_stddev * torch.log(sol_d.stddev) - 
                            prev_sol_d.mean - self.coef_stddev * torch.log(prev_sol_d.stddev)) ** 2
            loss_total = loss_pessimism.mean()

        statistics = {"loss": loss_total.item()}  # Convert to Python scalar with .item()

        return statistics
    
    def update_step(self):
        statistics = {}
        # In PyTorch, you can directly assign values to tensors with [:] or .data
        self.prev_sol_x.data.copy_(self.sol_x.data)
        
        # Set the model to evaluation mode
        self.temp_model.eval()
        self.model.eval()
        
        # No need for explicit GradientTape in PyTorch
        inp = self.sol_x.to(self.device)
        prev_inp = self.init_sol_x.to(self.device)
        
        d = self.temp_model.get_distribution(inp)
        prev_sol_d = self.model.get_distribution(prev_inp)
        
        loss_pessimism = (d.mean - self.coef_stddev * torch.log(d.stddev) -
                        prev_sol_d.mean - self.coef_stddev * torch.log(prev_sol_d.stddev)) ** 2

        loss = (-(d.mean - self.coef_stddev * torch.log(d.stddev)) +
                (1. / (2. * self.region)) * loss_pessimism) * (-1)
        # Times -1 since it is a minimization problem as offline MBO is maxmization

        if self.model.args.train_data_mode == 'onlybest_1':
            loss = loss * (1 / 0.2)

        # Manually zero the gradients after updating weights
        if self.sol_x.grad is not None:
            self.sol_x.grad.detach_()
            self.sol_x.grad.zero_()
        
        # Backward pass to compute the gradient
        loss.sum().backward(retain_graph=True)
        self.sol_x_opt.step()

        # No need for apply_gradients in PyTorch, we manually update the tensor
        # with torch.no_grad():  # Disable gradient calculation for this update
        #     sol_x_grad_norm = self.sol_x.grad.view(self.sol_x_samples, -1).norm(dim=1)
        #     self.sol_x += self.lr * self.sol_x.grad  # Assume self.lr is the learning rate for sol_x
            # + since minimization problem

        # Get new distribution after the update
        new_d = self.temp_model.get_distribution(self.sol_x)
        self.sol_y.data.copy_(new_d.mean)

        # Compute the travelled distance and update statistics
        travelled = torch.norm(self.sol_x - self.init_sol_x) / self.sol_x.size(0)  # Assuming sol_x is 2D: [samples, features]
        
        statistics["travelled"] = travelled.item()  # Convert to Python scalar with .item()

        return statistics


def roma_train_one_model(
        model, x, y, x_test, y_test, args,
        device,
        retrain = False,
        batch_size = 128,
        split_ratio = 0.9
):
    print('Training RoMA...')
    if not retrain and os.path.exists(model.save_path):
        checkpoint = torch.load(model.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mse_loss = checkpoint['mse_loss']
        print(f"Successfully load trained model from {model.save_path} with MSE loss = {model.mse_loss}")
        return 
    
    
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

    def is_discrete(env_name):
        return env_name.startswith('motsp') or env_name.startswith('mocvrp') \
                    or env_name.startswith('mokp') or env_name in ['regex', 'rfp', 'zinc'] or env_name.startswith(('c10mop', 'in1kmop'))
    
    best_indices = torch.argsort(y.flatten())[:128]

    if not is_discrete(model.args.env_name):
        perturb_fn = lambda x: x + 0.2 * torch.randn_like(x)
    else:
        perturb_fn = lambda x: x

    kwargs = {
            "model": model,
            "is_discrete": is_discrete(model.args.env_name),
            "optimizer": torch.optim.Adam(model.parameters(), lr=0.001),
            "perturb_fn": perturb_fn,
            "sol_x": x[best_indices],
            "sol_y": y[best_indices],
            # "sol_x_opt": torch.optim.Adam(lr=3e-3),
            "steps_per_update": 20,
            "temp_model": DoubleheadModel(input_size=len(x[0]), 
                                          args = model.args,
                                          idx = model.which_obj,
                                          hidden_size=64).to(device),
            "coef_stddev": 0.0,
            "inner_lr": 5e-3,
            "max_y": np.amax(y.detach().cpu().numpy()),
            "region": 4.,
            "alpha": 1.,
            "lr": 2e-3,
            "device": torch.device('cuda' if torch.cuda.is_available else 'cpu')
        }
    
    warmup_epochs = 50
    updates = 500

    min_loss = np.PINF

    roma_trainer = ROMATrainer(**kwargs)

    for epoch in range(warmup_epochs):
        train_losses = []
        for x_batch, y_batch, w_batch in train_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            res = roma_trainer.train_step(x_batch, y_batch)
            train_losses.append(res['loss'])
            # assert 0, res
        train_loss = np.array(train_losses).mean()
        print ('Epoch [{}/{}], Loss: {:}'
                .format(epoch+1, warmup_epochs, train_loss))

        val_losses = [] 
        for x_batch, y_batch in val_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            res = roma_trainer.valid_step(x_batch, y_batch)
            # assert 0, res
            val_losses.append(res['loss'])
        
        val_loss = np.array(val_losses).mean() 
        print('Valid Loss is: {}'.format(val_loss))

        y_all = torch.zeros((0, 1)).to(device)
        outputs_all = torch.zeros((0, 1)).to(device)

        for x_batch, y_batch in test_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            d = model.get_distribution(x_batch)
            outputs = d.mean
            y_all = torch.cat((y_all, y_batch), dim=0)
            outputs_all = torch.cat((outputs_all, outputs))

        mse_func = nn.MSELoss() 
        test_mse = mse_func(y_all, outputs_all)
        print('test MSE is: {}'.format(test_mse.item()))

        if val_loss < min_loss:
            min_loss = val_loss
            model = model.to('cpu')
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'mse_loss': val_loss,
            }
            torch.save(checkpoint, model.save_path)
            model = model.to(device)

    step = 0
    update = 0

    while update < updates:
        roma_trainer.init_step() 
        res = roma_trainer.fix_step()
        print ('Update [{}/{}], Fix loss: {:}'
                .format(update+1, updates, res['loss']))
        
        res = roma_trainer.update_step() 
        print ('Travelled: {:}'
                .format(res['travelled']))
        
        update += 1 
        if update + 1 > updates:
            break 

    model = model.to('cpu')
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'mse_loss': val_loss,
    }
    torch.save(checkpoint, model.save_path)
