import torch

def square_error(x, y):
    return (x-y).pow(2).sum()

def count_fails(x, y):
    return (x!=y).sum()
