import torch
from torch.utils.data import DataLoader

def evaluate_accuracy(model, dataset):
    '''
    Evaluates accuracy on the specified dataset. Returns a scalar tensor.
    Loading the state dictionary before use is expected.
    '''

    model.eval()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
    num_correct = 0
    num_total = 0
    for x, y in dataloader:
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            num_correct += torch.count_nonzero(preds == y)
            num_total += preds.size(0)
    accuracy = num_correct / num_total
    return accuracy