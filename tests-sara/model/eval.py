from tqdm import tqdm
import torch


def evaluate(model, val_loader, device, criterion):

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for imgs, genders, ages, _ in tqdm(val_loader):

            imgs = imgs.to(device)
            genders = genders.to(device)
            ages = ages.to(device)

            output = model(imgs, genders)

            loss = criterion(output, ages) # we're iterating over batches of data, not full mae here

            batch_size = ages.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    mae = total_loss / total_samples

    return mae