"""
Self-test for stickers.py - no dataset or checkpoint needed.
Run from your project folder (the one containing attacks.py):   python check_stickers.py
Every line should print PASS. A FAIL tells you exactly which property is broken.
"""
import torch
import torch.nn as nn
from stickers import UniversalSticker

results = []


def check(cond, msg):
    results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + msg)


torch.manual_seed(0)
mean, std = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
mean_t = torch.tensor(mean).view(1, 3, 1, 1)
std_t = torch.tensor(std).view(1, 3, 1, 1)

# a toy network; the labels are its own predictions, so it is 100% "correct" before any attack
net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 32 * 32, 5)).eval()
X = torch.randn(200, 3, 32, 32)
with torch.no_grad():
    y = net(X).argmax(1)
dl = [(X[i:i + 50], y[i:i + 50]) for i in range(0, 200, 50)]

# ---- 1. two stickers (black then white) ----
hp = {"colors": ["black", "white"], "sizes": [(8, 8)], "region": (6, 6, 26, 26), "stride": 2}
attack = UniversalSticker(net, None, hp, mean=mean, std=std)
placements = attack.fit(dl, max_images=200)
check(len(placements) == 2, "fit() returns one placement per sticker")

top_bounds, left_bounds, bottom_bound, right_bound = hp["region"][0], hp["region"][1], hp["region"][2], hp["region"][3]
check(all(top_bounds <= t and left_bounds <= l and t + h <= bottom_bound and l + w <= right_bound
          for _, t, l, h, w in placements), "stickers stay inside the allowed region")

X_before = X.clone()
adv = attack.perturb(X, y)
check(torch.equal(X, X_before), "perturb() does not modify the input tensor")
check(adv.shape == X.shape, "adversarial images keep the input shape")

mask = torch.zeros(32, 32, dtype=torch.bool)
for _, t, l, h, w in placements:
    mask[t:t + h, l:l + w] = True
check(torch.equal(adv[:, :, ~mask], X[:, :, ~mask]), "pixels outside the stickers are untouched")
check(bool((adv[:, :, mask] == adv[0:1, :, mask]).all()), "the sticker is IDENTICAL on every image (universal)")

color, t, l, h, w = placements[-1]  # last sticker is never overwritten by another one
pixels = adv[:, :, t:t + h, l:l + w] * std_t + mean_t  # back to the [0,1] pixel range
expected = 0.0 if color == "black" else 1.0
check(torch.allclose(pixels, torch.full_like(pixels, expected), atol=1e-5),
      "'{}' sticker really is {} in pixel space (normalization handled)".format(color, expected))

# ---- 2. the search must beat random placement ----
one = UniversalSticker(net, None, {"colors": ["black"], "sizes": [(8, 8)], "region": (6, 6, 26, 26), "stride": 2},
                       mean=mean, std=std)
one.fit(dl, max_images=200)
with torch.no_grad():
    searched_rate = (net(one.perturb(X, y)).argmax(1) != y).float().mean().item()
    random_rates = []
    for _ in range(30):
        t, l = int(torch.randint(6, 19, (1,))), int(torch.randint(6, 19, (1,)))
        random_rates.append((net(one.apply(X, [("black", t, l, 8, 8)])).argmax(1) != y).float().mean().item())
mean_random = sum(random_rates) / len(random_rates)
check(searched_rate >= mean_random, "searched sticker fools at least as many images as random placement "
                                    "({:.2f} vs {:.2f})".format(searched_rate, mean_random))

# ---- 3. helpers ----
check(isinstance(attack.wrong_label_counts(dl), list), "wrong_label_counts() works")
try:
    UniversalSticker(net, None, hp, mean=mean, std=std).perturb(X, y)
    check(False, "perturb() before fit() should raise an error")
except AssertionError:
    check(True, "perturb() before fit() raises a clear error")

print("\n{} / {} checks passed".format(sum(results), len(results)))
