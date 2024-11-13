import torch
# from typing import List, Optional, Any
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from model import TransformerForStoryGeneration
import wandb
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', DEVICE)

enable_logging = True

    
class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup = warmup_steps
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + math.cos(math.pi * epoch / self.max_steps))
        lr_factor *= min(epoch / self.warmup, 1.0)
        return lr_factor    
    

def plot_losses(train_losses, val_losses, model, optimizer):
    """
    Plot loss and perplexity of train and validation samples
    :param train_losses: list of train losses at each epoch
    :param val_losses: list of validation losses at each epoch
    """
    train_perplexities = list(torch.exp(torch.tensor(train_losses)))
    val_perplexities = list(torch.exp(torch.tensor(val_losses)))

    wandb.log({"train loss" : train_losses[-1], "train perplexity" : train_perplexities[-1], "val loss" : val_losses[-1], "val perplexity" : val_perplexities[-1], "lr" : optimizer.param_groups[0]['lr']})
    wandb.watch(model)


def training_epoch(model: TransformerForStoryGeneration, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str, scheduler = None, scaler=None):
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        optimizer.zero_grad()
        tokens = indices[:, :lengths.max()].to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          logits = model(tokens[:, :-1], lengths - 1)
          loss = criterion(logits.transpose(1, 2), tokens[:, 1:])
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        train_loss += loss.item() * tokens.shape[0]
        scaler.update()

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: TransformerForStoryGeneration, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str):
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    model.eval()
    for indices, lengths in tqdm(loader, desc=tqdm_desc):
        tokens = indices[:, :lengths.max()].to(device)
        logits = model(tokens[:, :-1], lengths - 1)
        loss = criterion(logits.transpose(1, 2), tokens[:, 1:])
        val_loss += loss.item() * tokens.shape[0]

    val_loss /= len(loader.dataset)
    return val_loss


def train(model: TransformerForStoryGeneration, optimizer: torch.optim.Optimizer, scheduler,
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, model_name: str, save_every=1, num_examples=5):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    :param num_examples: number of generation examples to print after each epoch
    """
    print("Start training")
    if enable_logging:
        wandb.finish()
        wandb.login(key="")
        wandb.init(project="bhw1-tiny-stories")
    
    train_losses, val_losses = [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.pad_id)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, num_epochs + 1):
    # for epoch in range(11, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}',
            scheduler=scheduler, scaler=scaler
        )
        val_loss = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]

        if enable_logging:
            plot_losses(train_losses, val_losses, model, optimizer)

        print('Generation examples (epoch ' + str(epoch) + '):')
        for _ in range(num_examples):
            print(model.inference())
            print()
            
        if epoch % save_every == 0:
            print("saving")
            torch.save(model.state_dict(), model_name+"_"+str(epoch)+"_model.pth")
            torch.save(optimizer.state_dict(), model_name+"_"+str(epoch)+"_optimizer.pth")
            torch.save(scheduler.state_dict(), model_name+"_"+str(epoch)+"_scheduler.pth")
        