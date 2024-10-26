import torch
from mlp import MLP
from dataset import KnotCNData
from evaluation import evaluate_accuracy

model_name = 'MLP_0'
MODEL_PATH = './models/' + model_name + '.pth'
DATA_PATH = './data/braid_data.csv'

dataset = KnotCNData(data_file=DATA_PATH)
model = MLP(input_size=dataset.max_length)
model.load_state_dict(torch.load(MODEL_PATH))

accuracy = evaluate_accuracy(model, dataset)

print(model_name, " accuracy: ", accuracy)