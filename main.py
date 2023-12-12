import numpy as np
from pathlib import Path

import os
from subprocess import Popen, run
import platform

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def get_device():
    device = "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else device
    print(f"Using device: {device}")
    return device

# NEED TO USE AN EMBEDDING LAYER
# LOSS CROSS ENTROPY
# SOFTMAX
# return set of probabilities for each character for each character
# bag of words model for next week

device = get_device()

DEBUG = False

### Constant ###
BATCH_SIZE = 8
CHANNEL_SIZE = 16
CELL_SURVIVAL_RATE = 0.5
POOL_SIZE = 500
LEARNING_RATE = 0.0001
EPOCH_NUM = 1000
input_path = Path("./data/input03.txt")

file_path = str(input_path)
with open(file_path, "r", encoding="utf-8") as input_text:
    text = input_text.read()

TEXT_LEN= len(text)

chars = sorted(list(set(text)))
CHAR_SIZE = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

###### Model Construction #####
class NCA_LLM(nn.Module):
    def __init__(self, channel_num, cell_survival_rate, device=device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(CHAR_SIZE, CHAR_SIZE)
        self.channel_num = channel_num
        self.cell_survival_rate = cell_survival_rate
        self.device = device

        self.seq = nn.Sequential(
            nn.Conv1d(
                in_channels=TEXT_LEN,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=TEXT_LEN,
                kernel_size=3,
                padding=1,
                bias=False
            ),
        )

        # initialize weights to zero to prevent random noise
        with torch.no_grad():
            self.seq[2].weight.zero_()

    # XXX: Live cell mask deleted, 
    # stochastic update disabled
    def stochastic_update(self, X):
        mask = (torch.rand(X[:, :1, :].shape) <= self.cell_survival_rate).to(
            self.device, torch.float32
        )
        return X * mask
    
    def forward(self, X):
        emb_x = self.token_embedding_table(X) #(B, T, C)
        logits = self.seq(emb_x) #(B, T, C)
        probs = F.softmax(logits, dim=-1) # (B, C)
        return logits, probs.argmax(dim=-1)



def train_step(model, optimizer, pool_grid, targets, text_length, writer, epoch):
    batch_ids = torch.multinomial(
        torch.ones(POOL_SIZE), BATCH_SIZE, replacement=False
    ).to(device)
    batch_sample = pool_grid[batch_ids]
    loss_rank = get_loss(batch_sample, targets).argsort().flip(dims=(0,))

    batch_sample = batch_sample[loss_rank]
    batch_ids = batch_ids[loss_rank]
    batch_sample[0] = init_text(text_size=text_length)

    for _ in range(np.random.randint(80, 96)):
        batch_sample = model(batch_sample)

    loss = get_loss(batch_sample, targets).mean()

    pool_grid[batch_ids] = batch_sample.detach()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("train/loss", loss, epoch)
    return batch_sample, pool_grid, loss

def get_loss(logits, targets):
    B, T, C = logits.shape
    logits_flat =  logits.reshape(B * T, C) # Now shape is (8*13, 10)
    targets_flat = targets.reshape(B * T)  # Now shape is (8*13)
    # Calculate the loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none') 
    # Reshape the loss back to the batch shape (8, 13)
    loss = loss.reshape(B, T)
    loss = loss.mean(dim=-1)
    return loss

def main():
    # Construct target
    targets = torch.tensor(encode(text), dtype=torch.long)[None, ...].to(device)
    targets = targets.repeat(BATCH_SIZE, 1)


    # construct pool sample with poolsize of 1024
    #init_x = torch.zeros_like(targets).to(device)
    init_x = torch.zeros((1, TEXT_LEN), dtype=torch.long).to(device)
    pool_grid = init_x.repeat(POOL_SIZE,  1)
    assert pool_grid.shape[-1] == targets.shape[-1]

    model = NCA_LLM(channel_num=CHANNEL_SIZE, cell_survival_rate=CELL_SURVIVAL_RATE).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LOGGING FILES for tensorboard
    log_path = Path("logs")
    pwd = Path().resolve()

    macos_tb = None
    if platform.system() == "Darwin":
        run(["rm", "-r", "logs/"])
        macos_tb = Popen(
            [
                "/usr/bin/osascript",
                "-e",
                f'tell app "Terminal" to do script "cd {pwd} &&  python3 -m tensorboard.main --logdir=logs"',
            ]
        )

    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)
    try:
        for epoch in range(EPOCH_NUM):
            batch_ids = torch.multinomial(
                torch.ones(POOL_SIZE), BATCH_SIZE, replacement=False
            ).to(device)
            batch_x = pool_grid[batch_ids]
            
            logit = None
            for _ in range(np.random.randint(80, 96)):
                logit, batch_x = model(batch_x)

            loss = get_loss(logit, targets)
            loss_rank = loss.argsort().flip(dims=(0,))
            avg_loss = loss.mean()

            batch_x = batch_x[loss_rank]
            batch_ids = batch_ids[loss_rank]
            batch_x[0] = init_x
            pool_grid[batch_ids] = batch_x.detach()

            print(f"epoch: {epoch}, loss: {avg_loss.item()}")
            #print(decode(batch_x[0].cpu().numpy()))


            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", avg_loss, epoch)

            if epoch == 200 and platform.system() == "Darwin":
                run(
                    [
                        "open",
                        "-a",
                        "Safari",
                        "http://localhost:6006/?darkMode=true#timeseries",
                    ]
                )
    except KeyboardInterrupt:
        pass
    finally:
        model.eval()
        for _ in range(1000):
            logit, init_x = model(init_x)
        output = init_x
        print("Final Output: ", decode(output[0].cpu().numpy()))
        print("# of chars: ",TEXT_LEN, "\n# of unique chars: ", CHAR_SIZE)
        weight_path = Path(f"data/weights/new_{input_path.stem}.pt")
        torch.save(model.state_dict(), weight_path)
        print("\nSaved model to\n\n", weight_path)

        if macos_tb is not None:
            terminate = input("Terminate Tensorboard? (y/n): ")
            if terminate.lower() == "y":
                Popen(
                    [
                        "/usr/bin/osascript",
                        "-e",
                        'tell app "Terminal" to quit'
                    ]
                )

if __name__ == "__main__":
    main()
