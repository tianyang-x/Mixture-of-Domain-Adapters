from datasets import Dataset


class CombinedDataset(Dataset):
    """
    Dataset that combines data from two datasets together.
    """
    def __init__(
        self, 
        dataset_1: Dataset, 
        dataset_2: Dataset,
        collate_1=None, collate_2=None
    ):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.collate_1 = collate_1
        self.collate_2 = collate_2

    def __len__(self):
        print(len(self.dataset_1))
        print(len(self.dataset_2))
        return min(len(self.dataset_1), len(self.dataset_2))

    def __getitem__(self, item):
        return self.dataset_1[item], self.dataset_2[item]

    def collate_fn(self, batch, proportion_2=0.06):
        from_1 = [x[0] for x in batch]
        from_2 = [x[1] for x in batch]
        # from_2 = from_2[:int(len(from_2) * proportion_2)]
        c1 = self.collate_1 if self.collate_1 is not None else self.dataset_1.collate_fn
        c2 = self.collate_2 if self.collate_2 is not None else self.dataset_2.collate_fn
        return c1(from_1), c2(from_2)
