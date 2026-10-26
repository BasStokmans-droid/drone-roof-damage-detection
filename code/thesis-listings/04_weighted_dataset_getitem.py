# Thesis listing 5.5 (Codefragment): weighted sampling in YOLOWeightedDataset.
# During training an image index is drawn according to its sampling probability
# instead of iterating sequentially. Validation uses normal indexing.
def __getitem__(self, index):
    if not self.train_mode:
        return self.transforms(self.get_image_and_label(index))
    else:
        weighted_index = np.random.choice(len(self.labels),
                                          p=self.probabilities)
        return self.transforms(self.get_image_and_label(weighted_index))
