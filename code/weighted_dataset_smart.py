"""
'Smart' variant: SQRT base weights plus per-class caps and floors (damage protected, Puncture/Hail Impact capped, Big-Damage boosted).

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
    def __init__(self, *args, balance_method="smart_custom", debug=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.balance_method = balance_method
        self.debug = debug

        if self.train_mode:  # Only apply weighting during training
            self.count_instances()
            self.class_weights = self.calculate_smart_weights()
            self.weights = self.calculate_image_weights()
            self.probabilities = self.calculate_probabilities()

            if self.debug:
                print(f"🎯 Smart Weighted Dataset Active: {self.balance_method}")
        else:
            if self.debug:
                print("ℹ️ Validation mode - no weighting applied")

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

    def calculate_smart_weights(self):
        """Calculate SMART custom weights with protection and caps"""

        # Get class indices for special handling
        damage_idx = None
        puncture_idx = None
        hail_idx = None
        big_damage_idx = None

        for i, class_name in enumerate(self.data["names"]):
            if class_name.lower() == 'damage':
                damage_idx = i
            elif class_name.lower() == 'puncture':
                puncture_idx = i
            elif 'hail' in class_name.lower():
                hail_idx = i
            elif 'big-damage' in class_name.lower():
                big_damage_idx = i

        if self.debug:
            print(f"\n🔍 Special class indices:")
            print(f"  damage={damage_idx}, puncture={puncture_idx}")
            print(f"  hail={hail_idx}, big-damage={big_damage_idx}")

        # Calculate base weights based on method
        total_samples = np.sum(self.counts)

        if self.balance_method == "smart_custom":
            # Use SQRT as base, then apply custom adjustments
            base_weights = np.sqrt(total_samples / self.counts)
        elif self.balance_method == "sum":
            base_weights = total_samples / self.counts
        elif self.balance_method == "sqrt":
            base_weights = np.sqrt(total_samples / self.counts)
        else:  # normal
            base_weights = np.ones_like(self.counts).astype(float)

        # Apply smart custom adjustments
        smart_weights = base_weights.copy()

        if self.debug:
            print(f"\n🎯 Applying smart weight adjustments:")

        for i, class_name in enumerate(self.data["names"]):
            original_weight = base_weights[i]

            if i == damage_idx:
                # 🛡️ Protect damage - max 20% reduction
                smart_weights[i] = max(0.8, original_weight)
                if self.debug:
                    print(f"🛡️ {class_name}: {original_weight:.2f}x → {smart_weights[i]:.2f}x (PROTECTED)")

            elif i == puncture_idx:
                # 🔧 Cap puncture - max 1.3x boost
                smart_weights[i] = min(1.3, original_weight)
                if self.debug:
                    print(f"🔧 {class_name}: {original_weight:.2f}x → {smart_weights[i]:.2f}x (CAPPED)")

            elif i == hail_idx:
                # 🔧 Cap hail impact - max 1.2x boost (already frequent)
                smart_weights[i] = min(1.2, original_weight)
                if self.debug:
                    print(f"🔧 {class_name}: {original_weight:.2f}x → {smart_weights[i]:.2f}x (CAPPED)")

            elif i == big_damage_idx:
                # 🚀 Ensure big-damage gets good boost - min 3x
                smart_weights[i] = max(3.0, original_weight)
                if self.debug:
                    print(f"🚀 {class_name}: {original_weight:.2f}x → {smart_weights[i]:.2f}x (BOOSTED)")

            else:
                # 📊 Other rare classes - cap at 2.5x
                smart_weights[i] = min(2.5, original_weight)
                if self.debug and original_weight != smart_weights[i]:
                    print(f"📊 {class_name}: {original_weight:.2f}x → {smart_weights[i]:.2f}x (CAPPED)")
                elif self.debug:
                    print(f"✅ {class_name}: {smart_weights[i]:.2f}x (UNCHANGED)")

        # Normalize weights
        smart_weights = smart_weights / np.mean(smart_weights)

        if self.debug:
            print(f"\nFinal smart weights:")
            print(f"  Min weight: {np.min(smart_weights):.2f}x")
            print(f"  Max weight: {np.max(smart_weights):.2f}x")
            print(f"  Mean weight: {np.mean(smart_weights):.2f}x")

        return smart_weights

    def calculate_image_weights(self):
        """Calculate weight per image with smart logic"""
        weights = []

        # Get damage index for special handling
        damage_idx = None
        for i, class_name in enumerate(self.data["names"]):
            if class_name.lower() == 'damage':
                damage_idx = i
                break

        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue

            # Smart weighting logic
            class_weights_in_image = [self.class_weights[cls_id] for cls_id in cls]

            if damage_idx is not None and damage_idx in cls:
                # If image contains damage, use average of damage weight and max weight
                base_weight = self.class_weights[damage_idx]
                max_weight = max(class_weights_in_image)
                final_weight = (base_weight + max_weight) / 2
            else:
                # For non-damage images, use max weight
                final_weight = max(class_weights_in_image)

            weights.append(final_weight)

        return weights

    def calculate_probabilities(self):
        """Calculate sampling probabilities"""
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]

        if self.debug:
            print(f"\n📈 Smart Sampling Probabilities:")
            print(f"  Average probability: {np.mean(probabilities):.6f}")
            print(f"  Min probability: {np.min(probabilities):.6f}")
            print(f"  Max probability: {np.max(probabilities):.6f}")
            print(f"  Probability ratio (max/min): {np.max(probabilities)/np.min(probabilities):.1f}x")

        return probabilities

    def __getitem__(self, index):
        """Get item with smart weighted sampling during training"""
        if not self.train_mode:
            # During validation/test: use normal indexing
            return self.transforms(self.get_image_and_label(index))
        else:
            # During training: use smart weighted sampling
            weighted_index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(weighted_index))

print("✅ Smart YOLOWeightedDataset with custom protection and caps loaded!")
print("🎯 Features:")
print("  🛡️ Damage class protected (max 20% reduction)")
print("  🔧 Puncture capped (max 1.3x boost)")
print("  🚀 Big-Damage boosted (min 3x)")
print("  📊 Other classes balanced (max 2.5x)")
print("  🔧 Hail Impact capped (max 1.2x)")
