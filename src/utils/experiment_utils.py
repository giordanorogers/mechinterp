import logging
import random
import torch
import numpy as np

logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """ Globally set random seed. """
    #logger.info("setting all seeds to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)