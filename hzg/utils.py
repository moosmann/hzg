import numpy as np


def indexing(array: np.ndarray, *ind):
    """Fancy indexing with relative, absolute and negative indices.

    Returns a copy of selected entries of the input array."""

    arg_len = np.size(ind)
    size = np.size(array)
    start = 0
    stop = size
    step = 1
    if arg_len is 0:
        pass
    elif arg_len is 1:
        step = ind[0]
        if not float(step).is_integer():
            raise TypeError(f'index arguments ({step}) of non-integer type')
    elif arg_len is 2:
        start = ind[0]
        stop = ind[1]
    elif arg_len is 3:
        start = ind[0]
        stop = ind[1]
        step = ind[2]
    elif arg_len > 3:
        if not np.iterable(ind):
            raise IndexError(f'argument not iterable')
        return array[ind]

    if step < 1 or step > size:
        raise IndexError(f'Step size {step} outside allowed range [0,{size}].')

    # Modify start index
    if start >= size:
        raise IndexError(f'Start index number {start} must be lower than the total number of elements {size}')
    elif 0 <= start < 1:
        start = np.floor(start * size)
    elif start < 0:
        raise IndexError(f'Start index number {start} must be positive')
    # modify stop index
    if stop > size:
        raise IndexError(f'Stop index number {stop} must not exceeds the total number of elements {size}.')
    elif 1 >= stop > 0:
        stop = np.ceil(stop * size)
    elif stop == 0:
        stop = 1
    elif 0 > stop > -1:
        stop = np.ceil((1 + stop) * size)
    elif -1 >= stop >= -size:
        stop = size + stop
    elif -size > stop:
        raise IndexError(f'Index number {stop} exceeds total number of elements {size}.')

    # create index range
    start = int(start)
    stop = int(stop)
    step = int(step)
    index_range = np.arange(start, stop, step, dtype=np.int)

    return array[index_range]


if __name__ == '__main__':
    indexing(np.arange(10), np.arange(1, 9, 2))
