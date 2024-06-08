from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, X, Y, M):
        self.__X = X
        self.__Y = Y
        self.__M = M

    def __len__(self):
        return self.__X.shape[0]

    def __getitem__(self, idx):
        return self.__X[idx, :].astype('float32'), self.__Y[idx, :].astype('float32'), self.__M[idx, :]


class MLDataset(Dataset):
    def __init__(self, X):
        self.__X = X

    def __len__(self):
        return self.__X.shape[0]

    def __getitem__(self, idx):
        return self.__X[idx, :].astype('float32')
