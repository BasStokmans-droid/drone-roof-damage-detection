# Thesis listing 5.4 (Codefragment): YOLOWeightedDataset class initialisation.
# Excerpt only; the full class is in ../weighted_dataset_sqrt.py
class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, balance_method="sqrt", debug=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.balance_method = balance_method
        self.debug = debug

        self.count_instances()
        self.class_weights = self.calculate_sqrt_weights()
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
