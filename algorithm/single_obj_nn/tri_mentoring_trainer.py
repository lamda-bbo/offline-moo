import torch 
import numpy as np 
import os 
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)
import higher
from copy import deepcopy
from reweight import sigmoid_reweighting
from utils import set_seed, plot_and_save_mse
from algorithm.single_obj_nn.tri_mentoring_utils import * 

def tri_mentoring_train_one_model(
        model, x, y, x_test, y_test, device,
        retrain = False,
        lr = 0.1,
        weight_decay = 0.0,
        n_epochs = 3,
        batch_size = 128,
        split_ratio = 0.9
):
    if not retrain and os.path.exists(model.save_path):
        checkpoint = torch.load(model.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mse_loss = checkpoint['mse_loss']
        print(f"Successfully load trained model from {model.save_path} with MSE loss = {model.mse_loss}")
        return 
    
    set_seed(model.seed)
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

    if model.args.train_data_mode == 'onlybest_1':
        criterion_sum_mse = lambda yhat, y, w: torch.sum(w * torch.mean((yhat-y)**2, dim=1)) * (1 / 0.2)
    else:
        criterion_sum_mse = lambda yhat, y, w: torch.sum(w * torch.mean((yhat-y)**2, dim=1))

    criterion = nn.MSELoss()

    # T = int(train_L / args.bs) + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # begin training
    best_pcc = -1
    train_mse = []
    val_mse = []
    test_mse = []

    min_loss = np.PINF

    print(f'Begin to train ICT models with seed {model.seed} for Obj. {model.which_obj}')

    for e in range(n_epochs):
        # adjust lr
        adjust_learning_rate(optimizer, lr, e, n_epochs)

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
                .format(e+1, n_epochs, train_loss.item()))
            
            train_mse.append(train_loss.item())

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


def tri_mentoring_train_models(
        args,
        models, x, y, x_test, y_test, device,
        retrain = False,
        lr = 0.1,
        weight_decay = 0.0,
        n_epochs = 3,
        batch_size = 128,
        split_ratio = 0.9
):
    assert len(models) % 2 == 1
    for model in models:
        set_seed(model.seed)
        tri_mentoring_train_one_model(
            model, x, y, x_test, y_test, device,
            retrain,
            lr,
            weight_decay,
            n_epochs,
            batch_size,
            split_ratio
        )

    for model in models:
        checkpoint = torch.load(model.save_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mse_loss = checkpoint['mse_loss']

    tkwargs = {
        'device': torch.device('cuda'),
        'dtype': torch.float32
    }

    tri_mentoring_kwargs = {
        'soft_label': 1,
        'majority_voting': 1,
        'epochs': 200,
        'Tmax': 200,
        'ft_lr': 1e-3,
        'topk': 128,
        'interval': 200,
        'K': 10,
        'method': 'triteach',
        'seed1': 2024,
        'seed2': 2025,
        'seed3': 2026
    }

    args.__dict__.update(tri_mentoring_kwargs)

    proxy1, proxy2, proxy3 = models[0], models[1], models[2]

    y = y.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x).to(**tkwargs)
    if isinstance(y, np.ndarray):
        y = torch.Tensor(y).to(**tkwargs)
    if isinstance(x_test, np.ndarray):
        x_test = torch.Tensor(x_test).to(**tkwargs)
    if isinstance(y_test, np.ndarray):
        y_test = torch.Tensor(y_test).to(**tkwargs)

    indexs = torch.argsort(y.squeeze())
    index = indexs[-1:]
    x_init = deepcopy(x[index])

    for x_i in range(x_init.shape[0]):
        # if args.method == 'simple' :
        #     proxy = SimpleMLP(task_x.shape[1]).to(device)
        #     proxy.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed) + ".pt", map_location=device))
        # else:
        #     proxy1 = SimpleMLP(task_x.shape[1]).to(device)
        #     proxy1.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed1) + ".pt", map_location=device))
        #     proxy2 = SimpleMLP(task_x.shape[1]).to(device)
        #     proxy2.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed2) + ".pt", map_location=device))
        #     proxy3 = SimpleMLP(task_x.shape[1]).to(device)
        #     proxy3.load_state_dict(torch.load("model/" + args.task + "_proxy_" + str(args.seed3) + ".pt", map_location=device))
        # define distill data
        candidate = x_init[x_i:x_i+1]
        #unmask2
        # score_before, _ = evaluate_sample(task, candidate, args.task, shape0)
        candidate.requires_grad = True
        candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
        for i in range(1, args.Tmax + 1):
            # if args.method == 'simple':
            #     loss = -proxy(candidate)
            if args.method == 'ensemble':
                loss = -1.0/3.0*(proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            elif args.method == 'triteach':
                adjust_proxy(proxy1, proxy2, proxy3, candidate.data, x0=x, y0=y, \
                K=args.K, majority_voting = args.majority_voting, soft_label=args.soft_label)
                loss = -1.0/3.0*(proxy1(candidate) + proxy2(candidate) + proxy3(candidate))
            candidate_opt.zero_grad()
            loss.backward()
            candidate_opt.step()
            # if i % args.Tmax == 0:
            #     score_after, _ = evaluate_sample(task, candidate.data, args.task, shape0)
            #     print("candidate {} score before {} score now {}".format(x_i, score_before.squeeze(), score_after.squeeze()))
        x_init[x_i] = candidate.data

    proxy1.to('cpu')
    proxy1_checkpoint = {
        'model_state_dict': proxy1.state_dict(),
        'mse_loss': proxy1.mse_loss
    }
    torch.save(proxy1_checkpoint, proxy1.save_path)

    proxy2.to('cpu')
    proxy2_checkpoint = {
        'model_state_dict': proxy2.state_dict(),
        'mse_loss': proxy2.mse_loss
    }
    torch.save(proxy2_checkpoint, proxy2.save_path)

    proxy3.to('cpu')
    proxy3_checkpoint = {
        'model_state_dict': proxy3.state_dict(),
        'mse_loss': proxy3.mse_loss
    }
    torch.save(proxy3_checkpoint, proxy3.save_path)