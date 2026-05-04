from tqdm import tqdm
import numpy as np
import torch
import copy
from os import path, listdir, makedirs

from model.eval import evaluate

def train_model(model, train_loader, val_loader, n_epochs, device, optimizer, criterion, checkpoint_path):

    train_losses = []
    val_losses = []

    best_val_loss = np.inf
    best_model = None
    patience = 5
    counter = 0

    for epoch in range(n_epochs):

        model.train()

        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)

        model.eval()

        val_loss = evaluate(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            counter = 0

            makedirs(checkpoint_path, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'mae': best_val_loss
            }, path.join(checkpoint_path, f"checkpoint{epoch}.pth"))

        else:
            counter += 1

        if counter == patience:
            print("Early stopping")
            break
        
    if best_model is not None:
        for param_model, param_best in zip(model.parameters(), best_model.parameters()):
            param_model.data = param_best.data
    
    return train_losses, val_losses


def train_epoch(model, train_loader, device, optimizer, criterion):

    total_loss = 0
    total_samples = 0

    for imgs, genders, ages in tqdm(train_loader):

        imgs = imgs.to(device)
        genders = genders.to(device)
        ages = ages.to(device)

        output = model(imgs, genders)

        optimizer.zero_grad()

        loss = criterion(output, ages) # we're iterating over batches of data, not full mae here

        loss.backward()

        optimizer.step()

        batch_size = ages.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mae = total_loss / total_samples
    
    return mae
    