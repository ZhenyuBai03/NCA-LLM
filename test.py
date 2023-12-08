import torch
import argparse

from pathlib import Path
import os

import main as ca


### Constant ###
BATCH_SIZE = 8
CHANNEL_SIZE = 16
CELL_SURVIVAL_RATE = 0.5
POOL_SIZE = 100
LEARNING_RATE = 0.001
EPOCH_NUM = 5000

def main():
    parser = argparse.ArgumentParser(
            description="Show the result of text generation"
    )
    parser.add_argument(
            "-p",
            "--text_file",
            type=str,
            default=False,
            help="the file path of the text",
    )
    device = ca.get_device()
    args = parser.parse_args()
    print(vars(args))

    text_path = Path(args.text_file)
    weight_path = Path(f'./data/weights/{text_path.stem}.pt')

    # load model
    model = ca.NCA_LLM(channel_num=CHANNEL_SIZE, cell_survival_rate=CELL_SURVIVAL_RATE).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()


    text, text_len = ca.load_text(text_path)
    _, _, _, decode = ca.create_charmap(text)


    with torch.no_grad():
        input_ntext = ca.init_text(text_len).to(device)
        init_stext = decode(input_ntext[0, 0, :].squeeze().tolist())
        print(init_stext)
        input()
        result = init_stext
        for epoch in range(EPOCH_NUM):
            os.system('clear')
            print("Testing...")
            print("target text is: \n" + text + "\n")
            print(f"Epoch {epoch:10}/{EPOCH_NUM}")
            input_ntext = model(input_ntext)

            result = input_ntext[0, 0, :]
            result = decode(torch.clamp(result, 0, 1).squeeze().tolist())
            print(result)

        print("\n\n")
        print("Final Result is: \n", result)


if __name__ == "__main__":
    main()
