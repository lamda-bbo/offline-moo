import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os
from utils import plot_and_save_mse
from reweight import sigmoid_reweighting

def train_model(
        model, x, y, x_test, y_test, device,
        retrain = False,
        lr = 0.001,
        lr_decay = 0.98,
        n_epochs = 800,
        batch_size = 32,
        split_ratio = 0.9
):
    print(x.shape, y.shape)
    if not retrain and os.path.exists(model.save_path):
        checkpoint = torch.load(model.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mse_loss = checkpoint['mse_loss']
        print(f"Successfully load trained model from {model.save_path} with MSE loss = {model.mse_loss}")
        return 
    
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    model = model.to(device)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    if isinstance(x_test, np.ndarray):
        x_test = torch.from_numpy(x_test)
    if isinstance(y_test, np.ndarray):
        y_test = torch.from_numpy(y_test)
        
    tensor_dataset = TensorDataset(x, y)
    lengths = [int(split_ratio*len(tensor_dataset)), len(tensor_dataset)-int(split_ratio*len(tensor_dataset))]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)
    
    x_torch, y_torch = train_dataset[:]
    if model.args.reweight_mode == 'sigmoid':
        weights = sigmoid_reweighting(y_torch, quantile=model.args.sigmoid_quantile)
    else:
        weights = torch.ones(len(x_torch))
        
    train_dataset = TensorDataset(x_torch, y_torch, weights)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"Begin to train model ")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            batch_w = batch_w.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion_sum_mse(outputs, batch_y, batch_w)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_all = torch.zeros((0, model.n_obj)).to(device)
            outputs_all = torch.zeros((0, model.n_obj)).to(device)
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
        
        lr *= lr_decay
        update_lr(optimizer, lr)

        with torch.no_grad():
            y_all = torch.zeros((0, model.n_obj)).to(device)
            outputs_all = torch.zeros((0, model.n_obj)).to(device)

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

            y_all = torch.zeros((0, model.n_obj)).to(device)
            outputs_all = torch.zeros((0, model.n_obj)).to(device)

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
    