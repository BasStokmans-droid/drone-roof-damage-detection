"""
SQRT-balanced weighted sampling for YOLOv8 training (the version described in thesis chapter 5).

Copied verbatim from the Colab notebook used for the bachelor thesis
(code/notebooks/03_train_yolov8_sqrt_weighted_sampling.ipynb).
It subclasses Ultralytics' YOLODataset and is activated by monkey-patching
``ultralytics.data.build.YOLODataset`` before training (see the notebook).
Based on the weighted-dataloader approach described by Yasin (2024), cited in the thesis.
Tested only in the 2025 Colab environment; newer Ultralytics versions may differ.
"""

import numpy as np
from ultralytics.data.dataset import YOLODataset

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, balance_method="sqrt", debug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.balance_method = balance_method
        self.debug = debug

        # Count instances and calculate SQRT weights
        self.count_instances()
        self.class_weights = self.calculate_sqrt_weights()
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """Count instances per class"""
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1
        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

        if self.debug:
            print(f"\n🔍 Dataset Analysis:")
            print(f"Total images: {len(self.labels):,}")
            print(f"Total annotations: {np.sum(self.counts):,}")
            print(f"\n📊 Annotations per class:")
            for i, count in enumerate(self.counts):
                class_name = self.data["names"][i]
                percentage = (count / np.sum(self.counts)) * 100
                print(f"  {class_name:<18}: {count:6,} ({percentage:5.1f}%)")

    def calculate_sqrt_weights(self):
        """Calculate SQRT balancing weights - moderate improvement"""
        total_samples = np.sum(self.counts)

        # SQRT balancing: less aggressive than inverse
        weights = np.sqrt(total_samples / self.counts)

        # Normalize weights to have mean = 1
        weights = weights / np.mean(weights)

        if self.debug:
            print(f"\n⚖️ SQRT Balancing Weights:")
            print(f"Balance method: {self.balance_method.upper()}")
            for i, (count, weight) in enumerate(zip(self.counts, weights)):
                class_name = self.data["names"][i]
                print(f"  {class_name:<18}: weight={weight:.2f}x (count={count:,})")

            print(f"\nWeight statistics:")
            print(f"  Min weight: {np.min(weights):.2f}x")
            print(f"  Max weight: {np.max(weights):.2f}x")
            print(f"  Mean weight: {np.mean(weights):.2f}x")

        return weights

    def calculate_weights(self):
        """Calculate weight per image based on classes in that image"""
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue
            # Use mean weight of all classes in this image
            weight = np.mean(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """Calculate sampling probabilities"""
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]

        if self.debug:
            print(f"\n📈 Sampling Probabilities:")
            print(f"  Average probability: {np.mean(probabilities):.6f}")
            print(f"  Min probability: {np.min(probabilities):.6f}")
            print(f"  Max probability: {np.max(probabilities):.6f}")
            print(f"  Probability ratio (max/min): {np.max(probabilities)/np.min(probabilities):.1f}x")

        return probabilities

    def __getitem__(self, index):
        """Get item with weighted sampling during training"""
        if not self.train_mode:
            # During validation/test: use normal indexing
            return self.transforms(self.get_image_and_label(index))
        else:
            # During training: use weighted sampling
            weighted_index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(weighted_index))

print("✅ YOLOWeightedDataset with SQRT balancing loaded!")
