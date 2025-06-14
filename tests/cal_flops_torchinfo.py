# Calculates the flops and params of pre-trained models WITHOUT DeepSpeed.
# python cal_flops.py --model_name_or_path path_to_model --batch_size 1 --seq_length 512

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

import fire
import torch
from torchinfo import summary
from typing import Optional

from llmtuner import ChatModel

def calculate(
    model_name_or_path: str,
    batch_size: Optional[int] = 1,
    seq_length: Optional[int] = 256,
    flash_attn: Optional[bool] = False
):
    # Load model
    chat_model = ChatModel(dict(
        model_name_or_path=model_name_or_path,
        template="vanilla",
        flash_attn=flash_attn
    ))
    model = chat_model.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Prepare fake input
    fake_input = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
    input_dict = {"input_ids": fake_input}

    # Print model summary (params, estimated MACs/FLOPs)
    print("Model Summary:")
    print(summary(
        model,
        input_data=input_dict,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=2,
        device=device,
        verbose=1
    ))

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # Estimate FLOPs for one forward pass (approximate)
    print("\nNote: FLOPs estimation is approximate and based on MACs (multiply-adds).")
    print("For more accurate profiling, consider using DeepSpeed or fvcore if available.")

if __name__ == "__main__":
    fire.Fire(calculate)