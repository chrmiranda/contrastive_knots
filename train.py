import torch
import torch.nn as nn
import knpy
from dataset import PairKnotData, KnotCNData
from mlp import MLP

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(device)

num_epochs = 100
lr = 0.01


pair_knot_data = PairKnotData()
pair_knot_loader = torch.utils.data.DataLoader(pair_knot_data, batch_size=32, shuffle=True)

knot_cn_data = KnotCNData()
knot_cn_loader = torch.utils.data.DataLoader(knot_cn_data, batch_size=32, shuffle=True)

for dataset, dataloader in [(pair_knot_data, pair_knot_loader), (knot_cn_data, knot_cn_loader)]:
    model = MLP()
    cel = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    model.train().to(device)

    for epoch in range(num_epochs):
        num_correct = 0
        num_total = 0 
        
        for x, y in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss = cel(preds, y)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            num_correct += torch.count_nonzero(preds == y)
            num_total += preds.size(0)
            print(f"Epoch: {epoch+1}, loss: {loss.item()}, acc: {num_correct / num_total}")
            
    model.eval()

    num_correct = 0
    num_total = 0 

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)
        num_correct += torch.count_nonzero(preds == y)
        num_total += preds.size(0)
    
    print(f"Epoch: {epoch+1}, loss: {loss.item()}, acc: {num_correct / num_total}")