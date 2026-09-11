import sys

sys.path.append(r"D:\sih2026\Data Engineer\ml")

from dataset import FakeRealDataset


dataset = FakeRealDataset("training")

mel, label = dataset[0]

print("Mel shape:", mel.shape)
print("Label:", label.item())