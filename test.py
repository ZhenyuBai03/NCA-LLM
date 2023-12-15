import torch
import argparse

from pathlib import Path
import os

import main as ca


### Constant ###
input_path = Path("./data/input02.txt")

BATCH_SIZE = 8
CHANNEL_SIZE = 16
CELL_SURVIVAL_RATE = 0.5
POOL_SIZE = 500
LEARNING_RATE = 0.0001
EPOCH_NUM = 8000
input_path = Path("./data/input02.txt")
weight_path = Path(f'./data/weights/{input_path.stem}.pt')

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

def main():
    device = ca.get_device()
    print("Device: ", device)
    init_x = torch.zeros((1, TEXT_LEN), dtype=torch.long).to(device)

    model = ca.NCA_LLM().to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    with torch.no_grad():
        for _ in range(8):
            logit, init_x = model(init_x)
            print(decode(init_x[0].cpu().numpy()), "\n")
        output = init_x

    print("====Final Output====: \n", decode(output[0].cpu().numpy()))

    


if __name__ == "__main__":
    main()
