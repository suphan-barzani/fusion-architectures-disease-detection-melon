import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.amp import autocast

import numpy as np
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold



from lib.dataset import BotryDataset
from lib.nn_models import (SpectralMLP, SpectralCNN, FluorMLP, FluorCNN) # Base
from lib.nn_models import (FusionNN_line, FusionNN_img) # Fusion

EPOCHS = 2
FORCE_CPU = False
USE_MULTIPROCESSING = False

def move_to(obj, device):
    if isinstance(obj, list):
        return [move_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, set):
        return set(move_to(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[move_to(key, device)] = move_to(value, device)
        return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj

def run_epoch(model, optimizer, data_loader, loss_func, device, results,
              score_funcs, prefix="", desc=None, tqdm_disabled=True):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    iterator = tqdm(data_loader, desc=desc, leave=False, disable=tqdm_disabled)
    for inputs, labels in iterator:
        inputs = move_to(inputs, device)
        labels = move_to(labels, device)

        with autocast(device_type=device):
            y_hat = model(inputs)
            loss = loss_func(y_hat, labels.unsqueeze(1))

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
            y_hat_np = y_hat.detach().cpu().float().numpy()
            y_true.extend(labels_np.tolist())
            y_pred.extend(y_hat_np.tolist())

    end = time.time()

    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    results[prefix + "_loss"].append(np.mean(running_loss))
    for name, score_func in score_funcs.items():
        try:
            pred = np.round(y_pred.squeeze())
            results[prefix + "_" + name].append(score_func(y_true, pred))
        except:
            results[prefix + "_" + name].append(float("NaN"))
    return end - start

def train_simple_network(model, loss_func, train_loader, epochs, optimizer,
                         device="cpu", test_loader=None, score_funcs=None,
                         checkpoint_file=None, tqdm_disabled=True):
    to_track = ["epoch", "total_time", "train_loss"]
    if test_loader is not None:
        to_track.append("test_loss")
    for eval_score in score_funcs:
        to_track.append("train_" + eval_score)
        if test_loader is not None:
            to_track.append("test_" + eval_score)

    total_train_time = 0
    results = {item: [] for item in to_track}

    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch", disable=tqdm_disabled):
        model.train()
        time = run_epoch(
            model, optimizer, train_loader, loss_func, device, results,
            score_funcs, prefix="train", desc="Training"
        )
        total_train_time += time
        results["total_time"].append(total_train_time)
        results["epoch"].append(epoch)
        if test_loader is not None:
            model.eval()
            with torch.inference_mode():
                run_epoch(
                    model, optimizer, test_loader, loss_func, device, results,
                    score_funcs, prefix="test", desc="Testing"
                )
        print(
            f'Epoch {epoch}\t'
            f'Time: {time:.2f}\t'
            f'Tra loss: {results["train_loss"][-1]:.2f}\t'
            f'Val loss: {results["test_loss"][-1]:.2f}\t'
            f'Tra acc: {results["train_Acc"][-1]:.2f}\t'
            f'Val acc: {results["test_Acc"][-1]:.2f}'
        )

    if checkpoint_file is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results
        }, checkpoint_file)

    return pd.DataFrame.from_dict(results)

def get_loaders_for_cv(train_df, val_df, class_, batch_size):
    train_dataset = BotryDataset(train_df, class_)
    val_dataset = BotryDataset(val_df, class_)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    in_size = train_dataset[0][0][0].shape[0]

    return train_loader, val_loader, in_size

def get_device():
    if torch.cuda.is_available() and not FORCE_CPU:
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device

def plot_results(all_results, prefix, models):
    colors = [
        ['#bd2929', '#ff6969'],  # SpectralMLP (Red)
        ['#295fbd', '#69a2ff'],  # SpectralCNN (Blue)
        ['#24bd4c', '#94ffb2'],  # FluorMLP (line) (Green)
        ['#ccbd24', '#f4e669'],  # BotryNN (line) (Yellow)
        ['#24bd94', '#94ffe0'],  # FluorCNN (square) (Cyan)
        ['#bd9424', '#ffdb94'],  # BotryNN (square) (Orange)
    ]
    figsize = (10, 6.18)
    plt.figure(figsize=figsize)
    for i, (model_name, _) in enumerate(models):
        data = all_results.loc[all_results['model_name'] == model_name]
        sns.lineplot(
            x='epoch',
            y='train_loss',
            data=data,
            label=f'Train {model_name}',
            color=colors[i][0]
        )
        sns.lineplot(
            x='epoch',
            y='test_loss',
            data=data,
            label=f'Validation {model_name}',
            color=colors[i][1]
        )
        plt.title('Loss')
    plt.savefig(f'results/{prefix}_loss', dpi=600)
    plt.close()

    plt.figure(figsize=figsize)
    for i, (model_name, _) in enumerate(models):
        data = all_results.loc[all_results['model_name'] == model_name]
        sns.lineplot(
            x='epoch',
            y='train_Acc',
            data=data,
            label=f'Train {model_name}',
            color=colors[i][0]
        )
        sns.lineplot(
            x='epoch',
            y='test_Acc',
            data=data,
            label=f'Validation {model_name}',
            color=colors[i][1],
            linestyle='--'
        )
        plt.title('Accuracy')
    plt.savefig(f'results/{prefix}_Acc', dpi=600)
    plt.close()

    with open(f'results/{prefix}.pkl', "wb") as file:
        pickle.dump(all_results, file)

def train_botry_nns(prefix, train_loader, val_loader, in_size, params):
    device = get_device()

    spectral_mlp = SpectralMLP(alone=True).to(device)
    spectral_cnn = SpectralCNN(alone=True).to(device)
    fluor_mlp = FluorMLP(alone=True).to(device)
    fluor_cnn = FluorCNN(alone=True).to(device)
    fusion_line = FusionNN_line().to(device)
    fusion_img = FusionNN_img().to(device)

    models = [
        ('SpectralMLP', spectral_mlp),
        ('SpectralCNN', spectral_cnn),
        ('FluorMLP', fluor_mlp),
        ('FluorCNN', fluor_cnn),
        ('FusionNN_line', fusion_line),
        ('FusionNN_img', fusion_img),
    ]

    criterion = torch.nn.BCELoss()
    all_results = None

    for model_name, model in models:
        optimizer = optim.Adam(model.parameters(),
                               lr=params['lr'],
                               betas=params['betas'],
                               eps=params['eps'],
                               weight_decay=params['wd'])

        results = train_simple_network(
            model,
            criterion,
            train_loader,
            optimizer=optimizer,
            epochs=EPOCHS,
            test_loader=val_loader,
            score_funcs={
                'Acc': accuracy_score,
                'F1': f1_score
            }
        )

        results = pd.concat((
            pd.DataFrame([model_name]*results.shape[0], columns=['model_name']),
            results,
        ), axis=1)

        if all_results is None:
            all_results = results
        else:
            all_results = pd.concat((all_results, results), ignore_index=True)

        torch.cuda.empty_cache()

    plot_results(all_results, prefix, models)

    return all_results

def run_cv_fold(data):
    df, params, round_seed, round_idx, fold_idx, train_index, val_index = data

    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

    train_loader, val_loader, in_size = get_loaders_for_cv(
        train_df, val_df,
        class_='group',  # adjust if needed
        batch_size=params['batch_size']
    )

    prefix = f'CvI_param_{params["id"]}_round_{round_idx}_fold_{fold_idx}'
    print(f"Running {prefix}")

    cv_results = train_botry_nns(prefix, train_loader, val_loader, in_size, params)

    return cv_results

def control_vs_infected_cv(df, params, n_splits=5, n_rounds=3):
    tasks = []
    base_seed = 0

    for round_idx in range(n_rounds):
        round_seed = base_seed + round_idx
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=round_seed)

        for fold_idx, (train_index, val_index) in enumerate(kf.split(df)):
            tasks.append((df, params, round_seed, round_idx, fold_idx, train_index, val_index))

    if USE_MULTIPROCESSING:
        with Pool(processes=21) as pool:
            results_list = pool.map(run_cv_fold, tasks)
    else:
        results_list = list(map(run_cv_fold, tasks))

    return results_list

def grid_search_adam():
    df = pd.read_excel('data/spectral_data.xlsx')
    df = df[(df['class'] == 'control') | (df['class'] == 'botrytis')]
    
    batch_sizes = [2**i for i in range(5, 7)]
    learning_rates = [1e-5, 1e-4, 1e-3]
    beta_pairs = [(0.9, 0.999), (0.95, 0.999)]
    epsilons = [1e-8, 1e-7]
    weight_decays = [0, 1e-5]

    results_file = "results/adam_grid_search_results.txt"
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("ID,Batch_Size,Learning_Rate,Beta1,Beta2,Epsilon,Weight_Decay\n")

    PARAMETER_SETS = []
    id_counter = 0
    for batch_size in batch_sizes:
        for lr in learning_rates:
            for betas in beta_pairs:
                for eps in epsilons:
                    for wd in weight_decays:
                        PARAMETER_SETS.append({
                            'id': id_counter,
                            'batch_size': batch_size,
                            'lr': lr,
                            'betas': betas,
                            'eps': eps,
                            'wd': wd
                        })
                        id_counter += 1

    for params in PARAMETER_SETS:
        if params['id'] >= 32:
            print(f"Processing parameter set ID: {params['id']}")
            
            cv_results = control_vs_infected_cv(df, params, n_splits=5, n_rounds=3)
            
            with open(results_file, "a") as f:
                f.write(f"{params['id']},{params['batch_size']},{params['lr']},"
                        f"{params['betas'][0]},{params['betas'][1]},"
                        f"{params['eps']},{params['wd']}\n")

if __name__ == '__main__':
    grid_search_adam()
