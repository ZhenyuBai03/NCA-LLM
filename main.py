import numpy as np
from pathlib import Path
import os
from time import sleep

import torch
from torch import nn
import torch.nn.functional as F


def get_device():
    device = "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else device
    print(f"Using device: {device}")
    return device
device = get_device()

### Constant ###
BATCH_SIZE = 8
CHANNEL_SIZE = 16
CELL_SURVIVAL_RATE = 0.1
POOL_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCH_NUM = 500

###### Utility Functions ######
def load_text(file_path):
    file_path = str(file_path)
    with open(file_path, "r", encoding='utf-8') as input_text:
        text = input_text.read()
    text_length = len(text)
    print("the length of the text is:", text_length)
    return text, text_length

def create_charmap(text):
    """
    Currently, we only want to make sure the first character is "<space>" and the last one is "."

    return: 
    ston: map, String to nums
    ntos: map, nums to String
    """
    chars = sorted(list(set(text)))
    chars.append(chars.pop(chars.index(".")))
    char_size = len(chars)

    #NOTE: might be used in the future
    #for i in range(chars.index(".")):
    #    chars.append(chars.pop(0))
    #chars = chars[::-1] 

    print(f'there are {char_size} characters')
    print(''.join(chars))

    # create mapping for "<space> \n zyxwvutsrqponmlkjihgfedcba." to integers
    ston = { ch:i for i,ch in enumerate(chars) }
    ntos = { i:ch for i,ch in enumerate(chars) }
    return ston, ntos

def init_text(text_size, channel_size=CHANNEL_SIZE):
    """
    The first character and its hidden states іs assigned as 1, which corresponds to "."
    The rest of characters are all zeros.
    
    return: 
        init_ntext: pytorch.tensor with shape(1, channel_size, text_size)
    """
    init_ntext = torch.zeros((1, channel_size, text_size))
    init_ntext[:, :, 0] = 1
    init_ntext = F.pad(init_ntext, (1,1), "constant", 0)
    return init_ntext

def get_loss(X, target_ntext):
    return ((target_ntext - X[:, :2, ...]) ** 2).mean(dim=[1, 2])

###### Model Construction #####
#FIXME: the model structure need to be checked
class CANN(nn.Module):
    def __init__(self, channel_num, cell_survival_rate, device=device):
        super().__init__()
        self.channel_num = channel_num
        self.cell_survival_rate = cell_survival_rate
        self.device = device

        self.seq = nn.Sequential(
            nn.Conv1d(in_channels=channel_num, 
                    out_channels=128, 
                    kernel_size=3, 
                    padding=1, 
                    groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, 
                    out_channels=channel_num, 
                    kernel_size=3, 
                    padding=1, 
                    groups=1,
                    bias=False),
        )

        # initialize weights to zero to prevent random noise
        with torch.no_grad():
            self.seq[2].weight.zero_()

    def stochastic_update(self, X):
        mask = (torch.rand(X[:, :1, :].shape) <= self.cell_survival_rate).to(self.device, torch.float32)
        return X * mask

    def live_cell_mask(self, X, alpha_threshold=0.1):
        live_mask = X[..., 1:2] > alpha_threshold
        return (live_mask)

    def neighbor_vector(self, X, neighbor_len=1):
        """
        Currently, the percieved vector is only defined as [-1, 0, 1] to
        apply convolution on the neighbor 
        Args:
            X, torch.tensor with shape of (BATCH_SIZE, CHANNEL_SIZE, CHAR_NUM)
            neighbor_len: in case we want to increase the size of neighbor, the argument is added
                          if neighbor_len=2, then the filter tensor would be [-2, -1, 0, 1, 2]

        return: 
            perceived: torch.tensor with the shape same as input 
        """
        filter = torch.arange(-neighbor_len, neighbor_len+1)[None, ...] / (neighbor_len*2)
        filter = filter.repeat((self.channel_num, 1)).unsqueeze(1).to(device)
        perceived = F.conv1d(X, filter, padding=neighbor_len, groups=self.channel_num)
        return perceived

    def forward(self, X):
        pre_mask = self.live_cell_mask(X)
        y = self.neighbor_vector(X)
        dx = self.seq(y)
        dx = self.stochastic_update(dx)

        X = X + dx

        post_mask = self.live_cell_mask(X)
        live_mask = (pre_mask & post_mask).to(torch.float32)

        return X * live_mask


def train_step(model, optimizer, pool_grid, target_ntext, text_length):
    batch_ids = torch.multinomial(torch.ones(POOL_SIZE), BATCH_SIZE, replacement=False).to(device)
    batch_sample = pool_grid[batch_ids]
    loss_rank = get_loss(batch_sample, target_ntext).argsort().flip(dims=(0,))

    batch_sample = batch_sample[loss_rank]
    batch_ids = batch_ids[loss_rank]
    batch_sample[0] = init_text(text_size=text_length)

    for _ in range(np.random.randint(64, 96)):
        batch_sample = model(batch_sample)

    loss = get_loss(batch_sample, target_ntext).mean()

    pool_grid[batch_ids] = batch_sample.detach()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return batch_sample, pool_grid, loss

def main():
    # loading input data and construct maps
    input_path = Path("./data/input01.txt")
    text, text_length = load_text(input_path)
    ston, ntos = create_charmap(text)
    encode = lambda s: [float(ston[c])/(len(ston) - 1) for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([ntos[round(i * (len(ntos) - 1))] for i in l]) # decoder: take a list of integers, output a string

    # Construct target 
    target_ntext = torch.tensor(encode(text))[None, ...]
    target_ntext = F.pad(target_ntext, (1,1), "constant", 0)
    target_ntext = torch.concat((target_ntext, torch.ones_like(target_ntext)), dim=0).to(device)

    # to show that decode function works
    target_stext = decode(target_ntext[0,:].squeeze().tolist())

    # construct pool sample with poolsize of 1024
    init_ntext = init_text(text_size=len(text)).to(device)
    pool_grid = init_ntext.clone().repeat(POOL_SIZE, 1, 1)

    # visualize the inital state, ensuring that the string starts with "<space>."
    init_stext = decode(init_ntext[0, 0, :].squeeze().tolist())

    model = CANN(channel_num=CHANNEL_SIZE, cell_survival_rate=CELL_SURVIVAL_RATE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log_file_path = Path(f"./data/log_{input_path.stem}.txt")
    log_file = open(log_file_path, "w")

    print("\n ################\n please check the parameters above...")
    sleep(4)

    try: 
        for epoch in range(EPOCH_NUM):
            batch_sample, pool_grid, loss = train_step(model,
                            optimizer,
                            pool_grid,
                            target_ntext,
                            text_length)

            epoch_sign = f"=====EPOCH {epoch}====="
            result = ""
            for sample in batch_sample[:, 0, :]:
                result_text = decode(torch.clamp(sample, 0, 1).squeeze().tolist())
                result += (result_text+"\n")

            log_file.write(epoch_sign+"\n")
            log_file.write(result)

            if epoch % 10 == 0:
                os.system('clear')
                print("############## The Original Text is #################")
                print(target_stext)
                print("################ The Inital Text is #################")
                print(init_stext)
                print(epoch_sign)
                print("loss: ", loss.item())
                print(result)

    except KeyboardInterrupt:
        pass

    finally: 
        log_file.close()

        print(f"\nlog file saved to {log_file_path}...")
        weight_path = Path(f"data/weights/{input_path.stem}.pt")
        torch.save(model.state_dict(), weight_path)
        print("\nSaved model to\n\n", weight_path)

if __name__ == "__main__":
    main()
