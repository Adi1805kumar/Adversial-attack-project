"""
Universal sticker attack (after Etim & Szefer, "Adversarial Universal Stickers", arXiv:2502.18724).

One fixed set of black / white rectangles, placed at the SAME pixel location on every image, is searched
so that as many images as possible get misclassified. It is a black-box attack: the search only uses the
network's output probabilities, never gradients.

It subclasses attacks.Attack, so test_attack(), helper.measure_resistance_on_test() etc. work unchanged.
Usage:
    sticker = UniversalSticker(net, loss_fn, hp, mean=(0.3403, 0.3121, 0.3214), std=(0.2724, 0.2608, 0.2669))
    sticker.fit(val_dl, device=device)                       # exhaustive location / size search
    acc = sticker.test_attack(test_dl, plot_results=False, save_results_figs=True,
                              fig_path="sticker.png", device=device)
"""
import torch
from collections import Counter

from attacks import Attack


class UniversalSticker(Attack):
    def __init__(self, net, loss_fn, hp, mean=(0.0,), std=(1.0,)):
        """
        :param hp: dict with
            colors: list like ["black"], ["white"], or ["black", "white"] - one sticker per entry (default ["black"])
            sizes:  list of (height, width) in pixels to try (default a few squares)
            region: (top, left, bottom, right) area in which stickers may be placed - the "merged mask"
                    (default: central part of the image, which lies on the sign for GTSRB crops)
            stride: step in pixels of the location search (default 2)
        :param mean, std: the normalization used by the data transform, so that pixel values 0 (black) and
                          1 (white) are converted to the network's input space. MNIST: leave the defaults.
        """
        super().__init__(net, loss_fn)
        self.name = "universal_sticker"
        self.colors = hp.get("colors", ["black"])
        self.sizes = hp.get("sizes", [(6, 6), (8, 8), (10, 10), (12, 12)])
        self.region = hp.get("region", (6, 6, 26, 26))
        self.stride = hp.get("stride", 2)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.placements = []  # list of (color, top, left, h, w) - filled by fit()

    # ---------- applying stickers ----------
    def _color_value(self, color, device):
        pixel = 0.0 if color == "black" else 1.0
        return ((pixel - self.mean) / self.std).to(device)  # shape (C, 1, 1)

    def apply(self, X, placements):
        adv = X.clone()
        for color, top, left, h, w in placements:
            adv[:, :, top:top + h, left:left + w] = self._color_value(color, X.device)
        return adv

    def perturb(self, X, y, device=None):
        """ the same stickers on every image (y is not used - the attack is input-agnostic) """
        assert len(self.placements) > 0, "call fit() first to search for the sticker location"
        return self.apply(X.detach(), self.placements)

    # ---------- search ----------
    @torch.no_grad()
    def _score(self, xs, ys, placements):
        """ (fooling rate, mean confidence of the wrong prediction on fooled images) - higher is better """
        probs = torch.softmax(self.net(self.apply(xs, placements)), dim=1)
        conf, preds = probs.max(dim=1)
        fooled = preds != ys
        if fooled.sum() == 0:
            return 0.0, 0.0
        return fooled.float().mean().item(), conf[fooled].mean().item()

    @torch.no_grad()
    def fit(self, dataloader, max_images=500, device=None):
        """
        Exhaustive search of size and location. Only images the net classifies correctly are used, so a
        'successful attack' always means the sticker changed a correct prediction.
        With 2 stickers the search is greedy (best first sticker, then best second given the first) - the
        paper searches both jointly, which costs (locations)^2 forward passes.
        """
        self.net = self.net.to(device)
        was_training = self.net.training
        self.net.eval()

        xs_list, ys_list, n = [], [], 0
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)
            correct = self.net(xs).argmax(dim=1) == ys
            xs_list.append(xs[correct])
            ys_list.append(ys[correct])
            n += int(correct.sum())
            if n >= max_images:
                break
        xs, ys = torch.cat(xs_list)[:max_images], torch.cat(ys_list)[:max_images]

        top0, left0, bottom, right = self.region
        self.placements = []
        for color in self.colors:
            best, best_score = None, (-1.0, -1.0)
            for h, w in self.sizes:
                for top in range(top0, bottom - h + 1, self.stride):
                    for left in range(left0, right - w + 1, self.stride):
                        cand = (color, top, left, h, w)
                        score = self._score(xs, ys, self.placements + [cand])
                        if score > best_score:
                            best, best_score = cand, score
            self.placements.append(best)
            print("sticker {}: {} -> fooling rate {:.3f} on {} search images".format(
                len(self.placements), best, best_score[0], len(xs)))

        self.net.train(was_training)
        return self.placements

    @torch.no_grad()
    def wrong_label_counts(self, dataloader, device=None):
        """ which classes do fooled images end up in? (the paper found one class dominating) """
        self.net.eval()
        counts = Counter()
        for xs, ys in dataloader:
            xs, ys = xs.to(device), ys.to(device)
            clean = self.net(xs).argmax(dim=1)
            adv = self.net(self.perturb(xs, ys)).argmax(dim=1)
            for p in adv[(clean == ys) & (adv != ys)].tolist():
                counts[p] += 1
        return counts.most_common()
