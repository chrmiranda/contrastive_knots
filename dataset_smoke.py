from dataset import KnotData, collate_fn
from torch.utils.data import DataLoader


ds = KnotData('./data/braid_notation.csv')

loader = DataLoader(
    ds,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

batch = next(iter(loader))
print(batch[0].shape)
print(batch[1].shape)
print(batch[2].shape)