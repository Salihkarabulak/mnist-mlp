import time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Tekrarlanabilirlik
SEED = 42
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", device)

# MNIST'in standart ortalama/std değerleriyle normalizasyon
MEAN, STD = 0.1307, 0.3081
transform = transforms.Compose([
    transforms.ToTensor(),                 # [0,255] -> [0,1], şekil (1,28,28)
    transforms.Normalize((MEAN,), (STD,))
])

# Ham veri: 60.000 eğitim, 10.000 test
full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set   = datasets.MNIST("data", train=False, download=True, transform=transform)

# Train / Validation ayrımı: 50.000 / 10.000
train_set, val_set = random_split(
    full_train, [50_000, 10_000],
    generator=torch.Generator().manual_seed(SEED)
)

BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1000)
test_loader  = DataLoader(test_set,  batch_size=1000)

print(f"Train: {len(train_set)} | Validation: {len(val_set)} | Test: {len(test_set)}")

class MLP(nn.Module):
    def __init__(self, hidden_sizes=(256, 128)):
        super().__init__()
        layers = [nn.Flatten()]
        in_dim = 784
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def evaluate(model, loader, criterion):
    """Verilen veri kümesinde ortalama loss ve accuracy hesaplar."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def train_model(model, epochs=10, lr=1e-3, optimizer_name="adam"):
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Epoch sonunda train ve validation metriklerini ölç
        tr_loss, tr_acc = evaluate(model, train_loader, criterion)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    history["time"] = time.time() - start
    print(f"Eğitim süresi: {history['time']:.1f} sn")
    return history

torch.manual_seed(SEED)
final_model = MLP(hidden_sizes=(256, 128)).to(device)
h = train_model(final_model, epochs=10, lr=1e-2, optimizer_name="sgd")
torch.save(final_model.state_dict(), "mlp_final.pt")

test_loss, test_acc = evaluate(final_model, test_loader, nn.CrossEntropyLoss())
print(f"FINAL MODEL TEST loss: {test_loss:.4f} | TEST accuracy: {test_acc:.4f}")