"""Online learner for image classification: the forecasting Learner with a ViT backbone.

Same buffer, metered updates, probes, snapshots and prefix caching; what differs is the
forward pass (images in, logits out), what is stored (labels) and the loss (cross-entropy).
The newest batch's per-channel mean and spread are kept for the input-shift gate.
"""
import numpy as np
import torch
import torch.nn.functional as F

from engine.learner import Learner


class VisionLearner(Learner):
    def _forecast(self, images):
        with torch.no_grad():
            x = torch.from_numpy(images).to(self.dev)
            self.last_mu = x.mean((0, 2, 3)).reshape(-1, 1)
            self.last_sd = x.std((0, 2, 3)).reshape(-1, 1)
            h = self.model.run_blocks(self.model.embed(x), 0, self.first)
            self._last_prefix = h.clone()
            logits = self.model.head_out(self.model.run_blocks(h, self.first))
            return logits.cpu().numpy(), x

    def push(self, images, labels):
        """Store the newest labelled batch (or its cached prefix). Returns the energy charged."""
        y = torch.from_numpy(labels).to(self.dev)
        item = (self._last_prefix if self.cache_prefix else self._last_xn, y)
        nbytes = item[0].numel() * 4 + y.numel() * 8
        self.buffer.append(item)
        nbytes += self._remember(item, nbytes)
        _, joules, _, _ = self.meter.measure(lambda: self.meter.store(nbytes), "store")
        return joules

    def _loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def _base_labels(self):
        return self._base()["labels"]

    def _base_items(self, idx):
        """Held-out base-domain images (uint8) as buffer items: (prefix or normalised images, labels)."""
        mean, std = self._base_args
        d = self._base()
        x = torch.from_numpy(d["images"][idx]).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1))
             / torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)).to(self.dev)
        y = torch.from_numpy(d["labels"][idx]).long().to(self.dev)
        return (self.model.run_blocks(self.model.embed(x), 0, self.first) if self.cache_prefix else x), y

    def retention(self, path, mean, std, batch=256):
        """Accuracy on a stored set of uint8 images (n, H, W, 3) and labels."""
        d = np.load(path)
        return self.accuracy(d["images"], d["labels"], mean, std, batch)

    def accuracy(self, images, labels, mean, std, batch=256):
        """Accuracy on uint8 images (n, H, W, 3); not metered, and leaves the learner's state alone."""
        m = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1)
        correct = 0
        with torch.no_grad():
            for i in range(0, len(labels), batch):
                x = torch.from_numpy(images[i:i + batch]).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - m) / s).to(self.dev)
                pred = self.model(x).argmax(1).cpu().numpy()
                correct += int((pred == labels[i:i + batch]).sum())
        return correct / len(labels)
