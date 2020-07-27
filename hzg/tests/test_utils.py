from utils import indexing
import numpy as np
import pytest


a = np.arange(10)
N = a.size
r0 = 0.2
r1 = 0.8


class TestIndexing:
    def test_indexing_integer_arg1(self):
        assert np.array_equal(indexing(a, 1), a)
        assert indexing(a, 1) is not a
        assert np.array_equal(indexing(a, 2), a[::2])

    def test_indexing_integer_arg2(self):
        assert np.array_equal(indexing(a, 0, 0), a[0:1])
        assert np.array_equal(indexing(a, 0, 1), a)
        assert np.array_equal(indexing(a, 0, N), a)

    def test_indexing_integer_arg3(self):
        assert np.array_equal(indexing(a, 0, N, 1), a)
        assert np.array_equal(indexing(a, 0, N, 2), a[::2])

    def test_indexing_relative(self):
        i0 = int(np.floor(r0 * N))
        i1 = int(np.ceil(r1 * N))
        # two arguments i.e. without step size parameter
        assert np.array_equal(indexing(a, 0, r1), a[0:i1])
        assert np.array_equal(indexing(a, r0, 1), a[i0:])
        assert np.array_equal(indexing(a, r0, r1), a[i0:i1])
        # three argument step size 1
        assert np.array_equal(indexing(a, 0, r1, 1), a[0:i1])
        assert np.array_equal(indexing(a, r0, 1, 1), a[i0:])
        assert np.array_equal(indexing(a, r0, r1, 1), a[i0:i1])
        # three argument step size 2
        assert np.array_equal(indexing(a, 0, r1, 2), a[0:i1:2])
        assert np.array_equal(indexing(a, r0, 1, 2), a[i0::2])
        assert np.array_equal(indexing(a, r0, r1, 2), a[i0:i1:2])

    def test_negative_integer_indexing(self):
        assert np.array_equal(indexing(a, 0, -1), a[0:-1:1])
        assert np.array_equal(indexing(a, 0, -1, 1), a[0:-1:1])
        assert np.array_equal(indexing(a, 0, -1, 2), a[0:-1:2])
        assert np.array_equal(indexing(a, 2, -2, 3), a[2:-2:3])

    def test_negative_and_relative_indexing(self):
        i0 = int(np.floor(r0 * N))
        i1 = int(np.ceil((1 - r0) * N))
        # 1st index relative, 2nd negative
        assert np.array_equal(indexing(a, r0, -1), a[i0:-1])
        assert np.array_equal(indexing(a, r0, -1, 1), a[i0:-1])
        assert np.array_equal(indexing(a, r0, -1, 2), a[i0:-1:2])
        # 1st index relative, 2nd negative and relative
        assert np.array_equal(indexing(a, r0, -r0, 2), a[i0:i1:2])

    def test_indexing_range(self):
        assert np.array_equal(indexing(a, np.arange(1, 9, 2)), a[1:9:2])
        assert np.array_equal(indexing(a, [1, 3, 5, 7]), a[1:9:2])
        assert np.array_equal(indexing(a, (1, 3, 5, 7)), a[1:9:2])

    def test_indexing_exceptions(self):
        with pytest.raises(TypeError):
            indexing(a, 1.5)
        with pytest.raises(IndexError):
            indexing(a, N + 1)
        with pytest.raises(IndexError):
            indexing(a, -1)
        with pytest.raises(IndexError):
            indexing(a, -1, 1)
        with pytest.raises(IndexError):
            indexing(a, N + 1, 1)
        with pytest.raises(IndexError):
            indexing(a, 0, N + 1)
        with pytest.raises(IndexError):
            indexing(a, 0, -N - 1)
        with pytest.raises(IndexError):
            indexing(a, 0, N, N + 1)

