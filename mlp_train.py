import torch
import torch.nn as nn
from dataset import KnotCNData
from torch.utils.data import DataLoader
from mlp import MLP
from tqdm import tqdm

MODEL_NAME = 'MLP_0.pth'
DATA_PATH = './data/braid_data.csv'
NUM_EPOCHS = 1
LR = 0.01
BATCH_SIZE = 32


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print("Device: ", device)


def train(model, dataset, num_epochs, lr, batch_size, model_name):

    dataloader = DataLoader(knot_cn_data, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    cel = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train().to(device)
    
    for epoch in range(num_epochs):
        num_correct = 0
        num_total = 0 
        
        for x, y in tqdm(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = cel(outputs, y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            num_correct += torch.count_nonzero(preds == y)
            num_total += preds.size(0)
            print(f"Epoch: {epoch+1}, loss: {loss.item()}, acc: {num_correct / num_total}")
    
    torch.save(model.state_dict(), './models/' + model_name)


knot_cn_data = KnotCNData(data_file=DATA_PATH)
model = MLP(input_size=knot_cn_data.max_length)
train(model, knot_cn_data, NUM_EPOCHS, LR, BATCH_SIZE, MODEL_NAME)