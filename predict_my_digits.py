import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt

STUDENT_NO = "23120205050"   # <-- kendi 11 haneli numaranı yaz
MEAN, STD = 0.1307, 0.3081  # eğitimdeki normalizasyonla aynı

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

def prepare_digit(path):
    """Fotoğrafı MNIST formatına çevirir: gri, ters çevrilmiş, 20x20'ye sığdırılmış, 28x28 ortalanmış."""
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if arr.mean() > 127:          # açık arka plan + koyu rakam -> MNIST gibi ters çevir
        arr = 255 - arr
    arr -= np.median(arr)         # zemin seviyesini (kağıt) çıkar
    arr[arr < 0] = 0.0            # zeminden koyu kenar çizgileri/gölgeler sıfırlansın
    arr /= (arr.max() + 1e-8)
    arr[arr < 0.35] = 0.0         # kağıt gürültüsünü temizle

    ys, xs = np.where(arr > 0)    # rakamın sınırlayıcı kutusu
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = arr.shape
    scale = 20.0 / max(h, w)      # en uzun kenar 20 piksel (MNIST gibi)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = Image.fromarray((arr * 255).astype(np.uint8)).resize((new_w, new_h), Image.LANCZOS)
    small = np.array(small, dtype=np.float32) / 255.0

    # 28x28 tuvale, kütle merkezi ortada olacak şekilde yerleştir
    yy, xx = np.indices(small.shape)
    cy = (yy * small).sum() / small.sum()
    cx = (xx * small).sum() / small.sum()
    top = min(max(int(round(14 - cy)), 0), 28 - new_h)
    left = min(max(int(round(14 - cx)), 0), 28 - new_w)
    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = small
    return canvas                 # 0..1 aralığında

model = MLP()
model.load_state_dict(torch.load("mlp_final.pt", map_location="cpu"))
model.eval()

fig, axes = plt.subplots(1, len(STUDENT_NO), figsize=(1.6 * len(STUDENT_NO), 2.4))
rows = []
correct = 0
for i, true_digit in enumerate(STUDENT_NO, start=1):
    img = prepare_digit(f"my_digits/d{i}.png")
    x = torch.tensor((img - MEAN) / STD).unsqueeze(0).unsqueeze(0)   # normalizasyon, (1,1,28,28)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    pred = probs.argmax().item()
    conf = probs[pred].item()
    correct += int(pred == int(true_digit))
    print(f"d{i}: gerçek {true_digit} | tahmin {pred} | güven {conf:.2%}")
    rows.append((i, true_digit, pred, conf))

    ax = axes[i - 1] if len(STUDENT_NO) > 1 else axes
    ax.imshow(img, cmap="gray")
    ax.set_title(f"gerçek {true_digit}\ntahmin {pred}", color="green" if pred == int(true_digit) else "red")
    ax.axis("off")

print(f"\nDoğru: {correct}/{len(STUDENT_NO)}")
with open("my_digits_sonuc.md", "w", encoding="utf-8") as f:
    f.write("| Hane | Gerçek | Tahmin | Güven | Sonuç |\n|---|---|---|---|---|\n")
    for i, t, p, c in rows:
        f.write(f"| d{i} | {t} | {p} | {c:.1%} | {'doğru' if p == int(t) else 'yanlış'} |\n")
fig.tight_layout()
fig.savefig("my_digits_sonuc.png", dpi=150)
plt.show()