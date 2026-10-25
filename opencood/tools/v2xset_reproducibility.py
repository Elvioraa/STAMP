"""Opt-in seeding; configurations without v2xset_reproducibility are untouched."""
import random


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure(hypes):
    if not hypes.get("v2xset_reproducibility", False):
        return
    if hypes["fusion"]["dataset"].lower() != "v2xset":
        raise ValueError("v2xset_reproducibility is restricted to V2XSet")
    import numpy as np
    import torch
    seed = int(hypes["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader_kwargs(hypes, validation=False):
    if not hypes.get("v2xset_reproducibility", False):
        return {}
    import torch
    generator = torch.Generator()
    generator.manual_seed(int(hypes["seed"]) + int(validation))
    return {"generator": generator, "worker_init_fn": seed_worker}
