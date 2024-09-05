from time import time


def timer(f):
    def new_f(*args, **kwargs):
        t = time()
        rez = f(*args, **kwargs)
        t = time() - t
        return *rez, t

    return new_f