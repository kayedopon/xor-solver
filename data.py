import numpy as np
import pandas as pd

np.random.seed(42)

# input
X = np.random.rand(75, 2)
X[X > 0.5] = 1
X[X < 0.5] = 0
X = X.astype(int)

# target
y = ((X[:, 0] == 1) != (X[:, 1] == 1)).astype(int).reshape(-1, 1)

data = {
    "input1": X[:, 0],
    "input2": X[:, 1],
    "Target": y.squeeze()
}

df = pd.DataFrame(data)
df.to_csv("data.csv", index=False)
