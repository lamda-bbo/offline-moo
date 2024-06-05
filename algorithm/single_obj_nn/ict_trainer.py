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
from algorithm.single_obj_nn.ict_utils import * 

def ict_grad_train_one_model(
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


    #     # random shuffle
    #     indexs = torch.randperm(train_L)
    #     train_logits = train_logits0[indexs]
    #     train_labels = train_labels0[indexs]
    #     tmp_loss = 0
    #     for t in range(T):
    #         x_batch = train_logits[t * args.bs:(t + 1) * args.bs, :]
    #         y_batch = train_labels[t * args.bs:(t + 1) * args.bs]
    #         pred = model(x_batch)
    #         loss = torch.mean(torch.pow(pred - y_batch, 2))
    #         tmp_loss = tmp_loss + loss.data
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     with torch.no_grad():
    #         valid_preds = model(valid_logits)
    #     pcc = compute_pcc(valid_preds.squeeze(), valid_labels.squeeze())
    #     # valid_loss = torch.mean(torch.pow(valid_preds.squeeze() - valid_labels.squeeze(),2))
    #     # print("epoch {} training loss {} loss {} best loss {}".format(e, tmp_loss/T, valid_loss, best_val))
    #     print("epoch {} training loss {} pcc {} best pcc {}".format(e, tmp_loss / T, pcc, best_pcc))
    #     if pcc > best_pcc:
    #         best_pcc = pcc
    #         print("epoch {} has the best loss {}".format(e, best_pcc))
    #         torch.save(model.state_dict(), args.store_path + args.task + "_proxy_" + str(args.seed) + ".pt")
    #         print('pred', valid_preds[0:20])
    # print('SEED', str(args.seed), 'has best pcc', str(best_pcc))
            
# Unpacked Co-teaching Loss function
def loss_coteaching(y_1, y_2, t, num_remember):
    # ind, noise_or_not
    loss_1 = F.mse_loss(y_1, t, reduction='none').view(128)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.mse_loss(y_2, t, reduction='none').view(128)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.mse_loss(y_1[ind_2_update], t[ind_2_update], reduction='none')
    loss_2_update = F.mse_loss(y_2[ind_1_update], t[ind_1_update], reduction='none')

    return loss_1_update, loss_2_update

def ict_train_models(
        args, reweight_mode,
        models, x, y, x_test, y_test, device,
        retrain = False,
        lr = 0.1,
        weight_decay = 0.0,
        n_epochs = 3,
        batch_size = 128,
        split_ratio = 0.9
):
    assert len(models) == 3
    n_epochs = 300
    for model in models:
        set_seed(model.seed)
        ict_grad_train_one_model(
           model, x, y, x_test, y_test, device,
            retrain,
            lr,
            weight_decay,
            n_epochs,
            batch_size,
            split_ratio
        )
    set_seed(args.train_model_seed)
    
    for model in models:
        checkpoint = torch.load(model.save_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mse_loss = checkpoint['mse_loss']

    ict_kwargs = dict(
        ft_lr=1e-1,
        alpha=1e-3,
        beta=3e-1,
        if_coteach=True,
        if_reweight=True,
        num_coteaching=8,
        Tmax=100,
        topk=128,
        K=128,
        wd=0.0,
        interval=100,
        mu=0,
        std=1,
        seed1=1,
        seed2=10,
        seed3=100,
        noise_coefficient=0.1,
        clamp_norm=1,
        clamp_min=-0.2,
        clamp_max=0.2
    )

    args.__dict__.update(ict_kwargs)
    
    f1, f2, f3 = models[0], models[1], models[2]

    tkwargs = {
        'device': torch.device('cuda'),
        'dtype': torch.float32
    }

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

    # get top k candidates
    if reweight_mode == "top128":
        index_val = indexs[-args.topk:]
    elif reweight_mode == "half":
        index_val = indexs[-(len(indexs) // 2):]
    else:
        index_val = indexs

    x_val = copy.deepcopy(x[index_val])
    label_val = copy.deepcopy(y[index_val])

    candidate = x_init[0]  # i.e., x_0
    candidate.requires_grad = True
    candidate_opt = optim.Adam([candidate], lr=args.ft_lr)
    optimizer1 = torch.optim.Adam(f1.parameters(), lr=args.alpha, weight_decay=args.wd)
    optimizer2 = torch.optim.Adam(f2.parameters(), lr=args.alpha, weight_decay=args.wd)
    optimizer3 = torch.optim.Adam(f3.parameters(), lr=args.alpha, weight_decay=args.wd)
    for i in range(1, args.Tmax + 1):
        loss = -1.0 / 3.0 * (f1(candidate) + f2(candidate) + f3(candidate))
        candidate_opt.zero_grad()
        loss.backward()
        candidate_opt.step()
        x_train = []
        y1_label = []
        y2_label = []
        y3_label = []
        # sample K points around current candidate
        for k in range(args.K):
            temp_x = candidate.data + args.noise_coefficient * np.random.normal(args.mu,
                                                                                args.std)  # add gaussian noise
            x_train.append(temp_x)
            temp_y1 = f1(temp_x)
            y1_label.append(temp_y1)

            temp_y2 = f2(temp_x)
            y2_label.append(temp_y2)

            temp_y3 = f3(temp_x)
            y3_label.append(temp_y3)

        x_train = torch.stack(x_train)
        y1_label = torch.Tensor(y1_label).to(device)
        y1_label = torch.reshape(y1_label, (args.K, 1))
        y2_label = torch.Tensor(y2_label).to(device)
        y2_label = torch.reshape(y2_label, (args.K, 1))
        y3_label = torch.Tensor(y3_label).to(device)
        y3_label = torch.reshape(y3_label, (args.K, 1))

        if args.if_reweight and args.if_coteach:
            # Round 1, use f3 to update f1 and f2
            weight_1 = torch.ones(args.num_coteaching).to(device)
            weight_1.requires_grad = True
            weight_2 = torch.ones(args.num_coteaching).to(device)
            weight_2.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                    l1, l2 = loss_coteaching(model1(x_train), model2(x_train), y3_label, args.num_coteaching)

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.num_coteaching
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm, dim=0)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.num_coteaching
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm, dim=0)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y3_label, reduction='none')
            loss1 = torch.sum(loss1) / args.num_coteaching
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = weight_2 * F.mse_loss(f2(x_train), y3_label, reduction='none')
            loss2 = torch.sum(loss2) / args.num_coteaching
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Round 2, use f2 to update f1 and f3
            weight_1 = torch.ones(args.num_coteaching).to(device)
            weight_1.requires_grad = True
            weight_3 = torch.ones(args.num_coteaching).to(device)
            weight_3.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l1, l3 = loss_coteaching(model1(x_train), model3(x_train), y2_label, args.num_coteaching)

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.num_coteaching
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm, dim=0)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.num_coteaching
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm, dim=0)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y2_label, reduction='none')
            loss1 = torch.sum(loss1) / args.num_coteaching
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y2_label, reduction='none')
            loss3 = torch.sum(loss3) / args.num_coteaching
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            # Round 3, use f1 to update f2 and f3
            weight_2 = torch.ones(args.num_coteaching).to(device)
            weight_2.requires_grad = True
            weight_3 = torch.ones(args.num_coteaching).to(device)
            weight_3.requires_grad = True
            with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l2, l3 = loss_coteaching(model2(x_train), model3(x_train), y1_label, args.num_coteaching)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.num_coteaching
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm, dim=0)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.num_coteaching
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm, dim=0)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss2 = weight_2 * F.mse_loss(f2(x_train), y1_label, reduction='none')
            loss2 = torch.sum(loss2) / args.num_coteaching
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y1_label, reduction='none')
            loss3 = torch.sum(loss3) / args.num_coteaching
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

        elif args.if_reweight and not args.if_coteach:

            # Round 1, use f3 to update f1 and f2
            weight_1 = torch.ones(args.K).to(device)
            weight_1.requires_grad = True
            weight_2 = torch.ones(args.K).to(device)
            weight_2.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                    l1 = F.mse_loss(model1(x_train), y3_label, reduction='none')
                    l2 = F.mse_loss(model2(x_train), y3_label, reduction='none')

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.K
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.K
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y3_label, reduction='none')
            loss1 = torch.sum(loss1) / args.K
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss2 = weight_2 * F.mse_loss(f2(x_train), y3_label, reduction='none')
            loss2 = torch.sum(loss2) / args.K
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            # Round 2, use f2 to update f1 and f3
            weight_1 = torch.ones(args.K).to(device)
            weight_1.requires_grad = True
            weight_3 = torch.ones(args.K).to(device)
            weight_3.requires_grad = True

            with higher.innerloop_ctx(f1, optimizer1) as (model1, opt1):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l1 = F.mse_loss(model1(x_train), y2_label, reduction='none')
                    l3 = F.mse_loss(model3(x_train), y2_label, reduction='none')

                    l1_t = weight_1 * l1
                    l1_t = torch.sum(l1_t) / args.K
                    opt1.step(l1_t)

                    logit1 = model1(x_val)
                    loss1_v = F.mse_loss(logit1, label_val)
                    g1 = torch.autograd.grad(loss1_v, weight_1)[0].data
                    g1 = F.normalize(g1, p=args.clamp_norm)
                    g1 = torch.clamp(g1, min=args.clamp_min, max=args.clamp_max)
                    weight_1 = weight_1 - args.beta * g1
                    weight_1 = torch.clamp(weight_1, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.K
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss1 = weight_1 * F.mse_loss(f1(x_train), y2_label, reduction='none')
            loss1 = torch.sum(loss1) / args.K
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y2_label, reduction='none')
            loss3 = torch.sum(loss3) / args.K
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

            # Round 3, use f1 to update f2 and f3
            weight_2 = torch.ones(args.K).to(device)
            weight_2.requires_grad = True
            weight_3 = torch.ones(args.K).to(device)
            weight_3.requires_grad = True
            with higher.innerloop_ctx(f2, optimizer2) as (model2, opt2):
                with higher.innerloop_ctx(f3, optimizer3) as (model3, opt3):
                    l2 = F.mse_loss(model2(x_train), y1_label, reduction='none')
                    l3 = F.mse_loss(model3(x_train), y1_label, reduction='none')

                    l2_t = weight_2 * l2
                    l2_t = torch.sum(l2_t) / args.K
                    opt2.step(l2_t)

                    logit2 = model2(x_val)
                    loss2_v = F.mse_loss(logit2, label_val)
                    g2 = torch.autograd.grad(loss2_v, weight_2)[0].data
                    g2 = F.normalize(g2, p=args.clamp_norm)
                    g2 = torch.clamp(g2, min=args.clamp_min, max=args.clamp_max)
                    weight_2 = weight_2 - args.beta * g2
                    weight_2 = torch.clamp(weight_2, min=0, max=2)

                    l3_t = weight_3 * l3
                    l3_t = torch.sum(l3_t) / args.K
                    opt3.step(l3_t)

                    logit3 = model3(x_val)
                    loss3_v = F.mse_loss(logit3, label_val)
                    g3 = torch.autograd.grad(loss3_v, weight_3)[0].data
                    g3 = F.normalize(g3, p=args.clamp_norm)
                    g3 = torch.clamp(g3, min=args.clamp_min, max=args.clamp_max)
                    weight_3 = weight_3 - args.beta * g3
                    weight_3 = torch.clamp(weight_3, min=0, max=2)

            loss2 = weight_2 * F.mse_loss(f2(x_train), y1_label, reduction='none')
            loss2 = torch.sum(loss2) / args.K
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            loss3 = weight_3 * F.mse_loss(f3(x_train), y1_label, reduction='none')
            loss3 = torch.sum(loss3) / args.K
            optimizer3.zero_grad()
            loss3.backward()
            optimizer3.step()

        elif not args.if_reweight and args.if_coteach:
            # f1 label, f2 and f3 coteaching
            loss_2, loss_3 = loss_coteaching(f2(x_train), f3(x_train), y1_label, args)
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            optimizer3.zero_grad()
            loss_3.backward()
            optimizer3.step()

            # f2 label, f1 and f3 coteaching
            loss_1, loss_33 = loss_coteaching(f1(x_train), f3(x_train), y2_label, args)
            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer3.zero_grad()
            loss_33.backward()
            optimizer3.step()

            # f3 label, f1 and f2 coteaching
            loss_11, loss_22 = loss_coteaching(f1(x_train), f2(x_train), y3_label, args)
            optimizer1.zero_grad()
            loss_11.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_22.backward()
            optimizer2.step()

            # optimizer1.zero_grad()
            # ((loss_1 + loss_11) / 2).backward()
            # optimizer1.step()
            # optimizer2.zero_grad()
            # ((loss_2 + loss_22) / 2).backward()
            # optimizer2.step()
            # optimizer3.zero_grad()
            # ((loss_3 + loss_33) / 2).backward()
            # optimizer3.step()

        elif not args.if_reweight and not args.if_coteach:
            pass
    
    f1.to('cpu')
    f1_checkpoint = {
        'model_state_dict': f1.state_dict(),
        'mse_loss': f1.mse_loss
    }
    torch.save(f1_checkpoint, f1.save_path)

    f2.to('cpu')
    f2_checkpoint = {
        'model_state_dict': f2.state_dict(),
        'mse_loss': f2.mse_loss
    }
    torch.save(f2_checkpoint, f2.save_path)

    f3.to('cpu')
    f3_checkpoint = {
        'model_state_dict': f3.state_dict(),
        'mse_loss': f3.mse_loss
    }
    torch.save(f3_checkpoint, f3.save_path)