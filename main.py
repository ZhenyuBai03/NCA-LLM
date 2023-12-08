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
EPOCH_NUM = 8000
input_path = Path("./data/input02.txt")

###### Utility Functions ######
def load_text(file_path):
    file_path = str(file_path)
    with open(file_path, "r", encoding="utf-8") as input_text:
        text = input_text.read()
    text_length = len(text)
    print("the length of the text is:", text_length)
    return text, text_length


# FIXME: remove
def create_charmap(text):
    """
    Currently, we only want to make sure the first character is "<space>" and the last one is "."

    return:
    ston: map, String to nums
    ntos: map, nums to String
    """
    chars = sorted(list(set(text)))
    char_size = len(chars)

    print(f"characters[{char_size}]: [{''.join(chars)}]")

    ston = {ch: i for i, ch in enumerate(chars)}
    ntos = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [
        float(ston[c]) / (len(ston) - 1) for c in s
    ]  # encoder: take a string, output a list of integers
    decode = lambda l: "".join(
        [ntos[round(i * (len(ntos) - 1))] for i in l]
    )  # decoder: take a list of integers, output a string
    return ston, ntos, encode, decode


def init_text(text_size, channel_size=CHANNEL_SIZE):
    """
    The first character and its hidden states іs assigned as 1, which corresponds to "."
    The rest of characters are all zeros.

    return:
        init_ntext: pytorch.tensor with shape(1, channel_size, text_size)
    """
    init_ntext = torch.zeros((1, channel_size, text_size))
    init_ntext[:, 1:, 0] = 1
    return init_ntext


def get_loss(X, target_ntext):
    loss = ((target_ntext - X[:, :2, ...]) ** 2).mean(dim=[1, 2])
    return loss


###### Model Construction #####
class NCA_LLM(nn.Module):
    def __init__(self, channel_num, cell_survival_rate, device=device):
        super().__init__()
        self.channel_num = channel_num
        self.cell_survival_rate = cell_survival_rate
        self.device = device

        self.seq = nn.Sequential(
            nn.Conv1d(
                in_channels=channel_num * 2,
                out_channels=128,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=128, out_channels=channel_num, kernel_size=1, bias=False
            ),
        )

        # initialize weights to zero to prevent random noise
        with torch.no_grad():
            self.seq[2].weight.zero_()

    def stochastic_update(self, X):
        mask = (torch.rand(X[:, :1, :].shape) <= self.cell_survival_rate).to(
            self.device, torch.float32
        )
        return X * mask

    def live_cell_mask(self, X, alive_threshold=0.1):
        val = (
            F.max_pool1d(X[:, 1:2, ...], kernel_size=3, stride=1, padding=1)
            > alive_threshold
        )
        return val

    # TODO: Use convolution in sequence
    # (will use transformer later)
    def neighbor_vector(self, X):
        """
        return:
            perceived: torch.tensor with the shape same as input
        """
        scalar = 2
        # NOTE: learn filter
        surrounding = torch.tensor([-1, 0, 1]) / scalar
        identity = torch.tensor([0, 1, 0]) / scalar

        filter = torch.stack([surrounding, identity])
        filter = filter.repeat((self.channel_num, 1)).unsqueeze(1).to(device)

        perceived = F.conv1d(X, filter, padding=1, groups=self.channel_num)
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


def train_step(model, optimizer, pool_grid, target_ntext, text_length, writer, epoch):
    batch_ids = torch.multinomial(
        torch.ones(POOL_SIZE), BATCH_SIZE, replacement=False
    ).to(device)
    batch_sample = pool_grid[batch_ids]
    loss_rank = get_loss(batch_sample, target_ntext).argsort().flip(dims=(0,))

    batch_sample = batch_sample[loss_rank]
    batch_ids = batch_ids[loss_rank]
    batch_sample[0] = init_text(text_size=text_length)

    for _ in range(np.random.randint(80, 96)):
        batch_sample = model(batch_sample)

    loss = get_loss(batch_sample, target_ntext).mean()

    pool_grid[batch_ids] = batch_sample.detach()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("train/loss", loss, epoch)
    return batch_sample, pool_grid, loss


def main():
    # loading input data and construct maps
    input_path = Path("./data/input02.txt")
    text, text_length = load_text(input_path)
    ston, _, encode, decode = create_charmap(text)

    if DEBUG:
        print("the charmap is:\n", ston)

    # Construct target
    target_ntext = torch.tensor(encode(text))[None, ...]
    target_ntext = torch.concat(
        (target_ntext, torch.ones_like(target_ntext)), dim=0
    ).to(device)

    # to show that decode function works
    target_stext = decode(target_ntext[0, :].squeeze().tolist())
    print("############## The Original Text is #################")
    print(target_stext)

    # construct pool sample with poolsize of 1024
    init_ntext = init_text(text_size=len(text)).to(device)
    pool_grid = init_ntext.clone().repeat(POOL_SIZE, 1, 1)

    # visualize the inital state, ensuring that the string starts with "<space>."
    init_stext = decode(init_ntext[0, 0, :].squeeze().tolist())
    print("############## The Inital Text is #################")
    print(init_stext)
    if DEBUG:
        print(init_ntext)

    model = NCA_LLM(channel_num=CHANNEL_SIZE, cell_survival_rate=CELL_SURVIVAL_RATE).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log_file_path = Path(f"./data/log_{input_path.stem}.txt")
    log_file = open(log_file_path, "w")

    # LOGGING FILES for tensorboard
    log_path = Path("logs")
    pwd = Path().resolve()
    if platform.system() == "Darwin":
        run(["rm", "-r", "logs/"])
        Popen(
            [
                "/usr/bin/osascript",
                "-e",
                f'tell app "Terminal" to do script "cd {pwd} &&  python3 -m tensorboard.main --logdir=logs"',
            ]
        )

    log_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_path)

    print("\n####################################################")
    input("please check the parameters above and press enter...\n")

    try:
        for epoch in range(EPOCH_NUM):
            batch_sample, pool_grid, loss = train_step(
                model, optimizer, pool_grid, target_ntext, text_length, writer, epoch
            )

            epoch_sign = f"===== EPOCH {epoch} ====="
            result = ""
            for sample in batch_sample[:, 0, :]:
                result_text = decode(torch.clamp(sample, 0, 1).squeeze().tolist())
                result += result_text + "\n"

            log_file.write(epoch_sign + "\n")
            log_file.write(result)

            if epoch % 1 == 0:
                os.system("clear")
                if DEBUG:
                    print(ston)
                print("############## The Original Text is #################")
                print(target_stext)
                print("##############  The Inital Text is  #################")
                print(init_stext)
                print(epoch_sign)
                print("loss: ", loss.item())
                print(result)

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
        log_file.close()
        writer.close()

        print(f"\nlog file saved to {log_file_path}...")
        weight_path = Path(f"data/weights/{input_path.stem}.pt")
        torch.save(model.state_dict(), weight_path)
        print("\nSaved model to\n\n", weight_path)


if __name__ == "__main__":
    main()
