MAX_FLOAT32 = 2 ** 32 - 1


def sign(k_id):
    # mirrored sampling
    return -1. if k_id % 2 == 0 else 1.
