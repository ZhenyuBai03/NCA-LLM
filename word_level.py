import os
import numpy as np
from pathlib import Path

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

device = get_device()

DEBUG = False

### Constant ###
BATCH_SIZE = 8
CHANNEL_SIZE = 16
CELL_SURVIVAL_RATE = 0.5
POOL_SIZE = 500
LEARNING_RATE = 0.001
EPOCH_NUM = 1000
EMBD_SIZE = 128

input_path = Path("./data/input02.txt")

file_path = str(input_path)
with open(file_path, "r", encoding="utf-8") as input_text:
    text = input_text.read()

TEXT_LEN= len(text)

chars = sorted(list(set(text)))
CHAR_SIZE = len(chars)

vocab_text = text.split(" ")
VOCAB_LEN = len(vocab_text)
vocab = set(vocab_text)
VOCAB_SIZE = len(vocab)
print(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
encode_word = lambda s: [word_to_ix[c] for c in s] # encoder: take a string, output a list of integers
decode_word = lambda l: ' '.join([ix_to_word[i] for i in l]) # decoder: take a list of integers, output a string


###### Model Construction #####
class NCA_LLM(nn.Module):
    def __init__(self, device=device):
        super().__init__()
        self.device = device

        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)

        self.filter = nn.Conv1d(in_channels = VOCAB_LEN,
                                out_channels = VOCAB_LEN * 3,
                                kernel_size = 3,
                                padding=1,
                                groups=VOCAB_LEN,)

        self.seq = nn.Sequential(
            nn.Conv1d(
                in_channels=VOCAB_LEN * 3,
                out_channels= 128,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=VOCAB_LEN,
                kernel_size=1,
                bias=False
            ),
        )

        # initialize weights to zero to prevent random noise
        with torch.no_grad():
            self.seq[2].weight.zero_()

    def forward(self, X):
        emb_x = self.token_embedding_table(X) #(B, T, C)
        filtered_emb = self.filter(emb_x)
        logits = self.seq(filtered_emb) #(B, T, C)

        probs = F.softmax(logits, dim=-1) # (B, T, C)

        all_probs = probs.reshape(-1, VOCAB_SIZE) # (B*T, C)
        output = torch.multinomial(all_probs, 1) # (B*T, 1)
        output = output.reshape(-1, VOCAB_LEN)

        live_mask = torch.max(probs, dim=-1).values > 0.9
        assert live_mask.shape == output.shape
        output = output * live_mask

        return logits, output

def get_loss(logits, targets):
    B, T, C = logits.shape
    logits_flat =  logits.reshape(B * T, C)
    targets_flat = targets.reshape(B * T)  
    # Calculate the loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none') 

    loss = loss.reshape(B, T)
    loss = loss.mean(dim=-1)
    return loss

def main():
    # Construct target
    targets = torch.tensor(encode_word(vocab_text), dtype=torch.long)[None, ...].to(device)
    targets = targets.repeat(BATCH_SIZE, 1)

    print("Target: ", targets.shape)
    print("Types of vocab: ", VOCAB_SIZE)
    print("Total word count: ", VOCAB_LEN)
    input("press enter to continue: ")

    # construct pool sample with poolsize of 1024
    init_x = torch.zeros((1, VOCAB_LEN), dtype=torch.long).to(device)
    pool_grid = init_x.repeat(POOL_SIZE,  1)
    assert pool_grid.shape[-1] == targets.shape[-1]

    model = NCA_LLM().to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # LOGGING FILES for tensorboard
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    try:
        for epoch in range(EPOCH_NUM):
            batch_ids = torch.multinomial(
                torch.ones(POOL_SIZE), BATCH_SIZE, replacement=False
            ).to(device)
            batch_x = pool_grid[batch_ids]
            
            logit = None
            for _ in range(np.random.randint(10, 26)):
                logit, batch_x = model(batch_x)

            loss = get_loss(logit, targets)
            loss_rank = loss.argsort().flip(dims=(0,))
            avg_loss = loss.mean()

            batch_x = batch_x[loss_rank]
            batch_ids = batch_ids[loss_rank]
            batch_x[0] = init_x
            pool_grid[batch_ids] = batch_x.detach()

            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"epoch: {epoch}, loss: {avg_loss.item()}")
            print(decode_word(batch_x[-1].cpu().numpy()), "\n")

            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", avg_loss, epoch)

    except KeyboardInterrupt:
        pass

    finally:
        model.eval()
        print("\n=====Test=====\n")
        print("Initial Text:\n", decode_word(init_x[0].cpu().numpy()))
        for _ in range(30):
            logit, init_x = model(init_x)
            print(decode_word(init_x[0].cpu().numpy()), "\n")
        output = init_x
        print("\n=====Final Result=====\n", decode_word(output[0].cpu().numpy()))
        print("# of chars: ",TEXT_LEN, "\n# of unique chars: ", CHAR_SIZE)

        weight_dir = Path("data/weights")
        weight_dir.mkdir(parents=True, exist_ok=True)

        weight_path = weight_dir / f"{input_path.stem}.pt"

        torch.save(model.state_dict(), weight_path)
        print("\nSaved model to\n\n", weight_path)

if __name__ == "__main__":
    main()
