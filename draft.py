import numpy as np
import pandas as pd
import torch

def test_p_series():
    a = np.array([1212, 12143, 13123, 12312])
    b = np.array([23, 27, 39, 54])
    c = pd.Series(data=b, index=a)
    print(c)


def test_pandas_nunique():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 1, 1]})
    print(df.B)
    print(df)
    print(df.nunique())
    print(type(df['A'].nunique()))

def test_nonzero():
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = torch.tensor([[1, 2, 6], [5, 7, 9], [1, 1, 9]])
    print(a)
    print(b)
    print(a == b)
    c = (a == b).nonzero()
    n_hits = c[:, :-1].size(0)
    print(c[:, :-1])
    print(c)

if __name__ == '__main__':
    # test_p_series()
    # test_pandas_nunique()
    test_nonzero()
