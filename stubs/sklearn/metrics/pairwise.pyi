from typing import Optional

import numpy as np
from numpy.typing import NDArray

def cosine_similarity(
    X: NDArray[np.float32],
    Y: Optional[NDArray[np.float32]] = None,
    dense_output: bool = True,
) -> NDArray[np.float32]: ...
def euclidean_distances(
    X: NDArray[np.float32],
    Y: Optional[NDArray[np.float32]] = None,
    squared: bool = False,
) -> NDArray[np.float32]: ...
