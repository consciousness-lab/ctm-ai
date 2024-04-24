from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

def load(
    audio_path: str, sr: Optional[int] = None
) -> Tuple[NDArray[np.float32], int]: ...
