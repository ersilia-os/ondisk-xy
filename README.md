# On-disk X-Y
Manage X and Y matrices on disk

## Installation

```bash
git clone https://github.com/ersilia-os/ondisk-xy.git
cd ondisk-xy
pip install -e .
```

## Usage

Write a matrix on disk.

```python
import numpy as np
from ondiskxy import MatrixWriter

# create 10 random X matrices of shape (10000, 100) and stack them on disk
dir_path = "./X"
writer = MatrixWriter(dir_path=dir_path, max_file_size_mb=25)
for _ in range(10):
    X = np.random.sample((1000,100))
    writer.append(X)
```

Matrices can be read easily.

```python
from ondiskxy import MatrixReader

dir_path = "./X"
reader = MatrixReader(dir_path=dir_path)

# read all at once
X = reader.read()

# iterate by row
for row in reader.iterrows():
    row
```

## Learn more

The [Ersilia Open Source Initiative](https://ersilia.io) is on a mission to strenghten research capacity in low income countries. Please reach out to us if you want to contribute: [hello@ersilia.io]()