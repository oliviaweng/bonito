"""
Bonito model viewer - display a model architecture for a given config.
"""

import toml
import argparse
from bonito.util import load_symbol
from torchinfo import summary

def print_model(model, input_size):
    summary(model, 
        input_size=input_size, 
        depth=9,
        col_names=['input_size', 'output_size', 'mult_adds'], 
        row_settings=['var_names']
    )

def main(args):
    config = toml.load(args.config)
    Model = load_symbol(config, "Model")
    model = Model(config)
    print(model)
    summary(model, 
        input_size=args.input_size, 
        depth=9,
        col_names=['input_size', 'output_size', 'mult_adds'], 
        row_settings=['var_names']
    )
    print("Total parameters in model", sum(p.numel() for p in model.parameters()))


def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--config", help="Model config that defines the model architecture")
    parser.add_argument("--input_size", nargs="+", type=int, help="batch_size x y")
    return parser
