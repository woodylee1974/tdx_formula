import pandas as pd
import numpy as np
from tests.test_util import testfunc, testcase


@testcase
def test_REMEMB():
    text = """
       REMEMB(B,X);
    """

    param = {
        'X': pd.Series([10.2, 30.9, 30.48, 39.34, 43.3, 45.9, 30.48, 39.34, 45.9, 30.48, 39.34]),
        'B': pd.Series([False, False, True, True, True, False, False, True, False, False, False]),
        'RESULT': pd.Series([10.2, 30.9, 30.48, 39.34, 43.3, 43.3, 43.3, 39.34, 39.34, 39.34, 39.34])
    }

    return text, param, False, False


@testcase
def test_CROSS():
    text = """
        CROSS(UP_CLOSE, CLOSE);
    """

    param1 = {
        'CLOSE': 10,
        'UP_CLOSE': pd.Series([6, 9, 11, 12, 13]),
        'RESULT': pd.Series([False, False, True, False, False])
    }

    param2 = {
        'CLOSE': pd.Series([12, 11, 10, 9, 8]),
        'UP_CLOSE': 10,
        'RESULT': pd.Series([False, False, False, True, False])
    }

    param3 = {
        'CLOSE': 10,
        'UP_CLOSE': 12,
        'RESULT': False
    }

    param4 = {
        'CLOSE': pd.Series([12, 11, 10, 9, 8]),
        'UP_CLOSE': pd.Series([6, 9, 11, 12, 13]),
        'RESULT': pd.Series([False, False, True, False, False])
    }

    params = [param1, param2, param3, param4]

    return text, params


@testcase
def test_NOT():
    text = """
         NOT(X);
    """

    param1 = {
        'X': 10,
        'RESULT': False
    }

    param2 = {
        'X': pd.Series([10, 0, 0, 20]),
        'RESULT': pd.Series([False, True, True, False])
    }

    param3 = {
        'X': pd.Series([True, False, False, True]),
        'RESULT': pd.Series([False, True, True, False])
    }

    param4 = {
        'X': pd.Series([10.0909, 0.000000001, -0.000000000394, 20]),
        'RESULT': pd.Series([False, True, True, False])
    }

    params = [param1, param2, param3, param4]

    return text, params


@testcase
def test_IF():
    text = """
        IF(X, A, B);
    """

    param1 = {
        'X': True,
        'A': 10,
        'B': 0,
        'RESULT': 10
    }

    param2 = {
        'X': pd.Series([10, 0, 0, 20]),
        'A': pd.Series([-10, 10, 10, -10]),
        'B': pd.Series([10, -10, -10, 10]),
        'RESULT': pd.Series([-10, -10, -10, -10])
    }

    param3 = {
        'X': pd.Series([True, False, False, True]),
        'A': pd.Series([-10, 10, 10, -10]),
        'B': pd.Series([10, -10, -10, 10]),
        'RESULT': pd.Series([-10, -10, -10, -10])
    }

    param4 = {
        'X': pd.Series([10.0, 0.000000001, 0.0000000000343, 20.1]),
        'A': pd.Series([-10, 10, 10, -10]),
        'B': pd.Series([10, -10, -10, 10]),
        'RESULT': pd.Series([-10, -10, -10, -10])
    }

    params = [param1, param2, param3, param4]
    return text, params


@testcase
def test_EVERY():
    text = """
        EVERY(X, N);
    """

    param1 = {
        'X': pd.Series([False, True, True, True, False, False, True, True, True]),
        'N': 3,
        'RESULT': pd.Series([False, False, False, True, False, False, False, False, True])
    }

    params = [param1]
    return text, params


@testcase
def test_EXIST():
    text = """
        EXIST(X, N);
    """

    param1 = {
        'X': pd.Series([False, True, True, False, False, False, True, True, True]),
        'N': 3,
        'RESULT': pd.Series([False, False, True, True, True, False, True, True, True])
    }

    params = [param1]
    return text, params


@testcase
def test_STD():
    text = """
        STD(X, N);
    """

    param1 = {
        'X': pd.Series([12, 11.8, 11.31, 11.90, 12, 12.2, 12.5, 12.20, 12.90]),
        'N': 3,
        'RESULT': pd.Series([np.NaN, np.NaN, 0.355012, 0.315753, 0.372872, 0.152753, 0.251661, 0.173205, 0.351188])
    }

    params = [param1]
    return text, params, True


def test_VAR():
    text = """
        VAR(X, N);
    """

    param1 = {
        'X': pd.Series([12, 11.8, 11.31, 11.90, 12, 12.2, 12.5, 12.20, 12.90]),
        'N': 3,
        'RESULT': pd.Series([np.NaN, np.NaN, 0.126033, 0.099700, 0.139033, 0.023333, 0.063333, 0.030000, 0.123333])
    }

    params = [param1]
    testfunc(text, params, True)


def test_STDP():
    text = """
        STDP(X, N);
    """

    param1 = {
        'X': pd.Series([12, 11.8, 11.31, 11.90, 12, 12.2, 12.5, 12.20, 12.90]),
        'N': 3,
        'RESULT': pd.Series([np.NaN, np.NaN, 0.289866, 0.257811, 0.304448, 0.124722, 0.205480, 0.141421, 0.286744])
    }

    params = [param1]
    testfunc(text, params, True)


def test_VARP():
    text = """
        VARP(X, N);
    """

    param1 = {
        'X': pd.Series([12, 11.8, 11.31, 11.90, 12, 12.2, 12.5, 12.20, 12.90]),
        'N': 3,
        'RESULT': pd.Series([np.NaN, np.NaN, 0.084022, 0.066467, 0.092689, 0.015556, 0.042222, 0.020000, 0.082222])
    }

    params = [param1]
    testfunc(text, params, True)


def test_MAX():
    text = """
        MAX(A, B);
    """

    param1 = {
        'A': pd.Series([10, 11, 10, 9, 7]),
        'B': pd.Series([11, 10, 11, 11, 11]),
        'RESULT': pd.Series([11, 11, 11, 11, 11])
    }

    params = [param1]
    testfunc(text, params)


def test_MIN():
    text = """
        MIN(A, B);
    """

    param1 = {
        'A': pd.Series([19, 11, 11, 11.9, 19.1]),
        'B': pd.Series([11, 20, 11.5, 11, 11]),
        'RESULT': pd.Series([11, 11, 11, 11, 11])
    }

    params = [param1]
    testfunc(text, params)


def test_COUNT():
    text = """
        COUNT(X, N);
    """
    param1 = {
        'X': pd.Series([True, False, False, True, False, False, False, False, True]),
        'N': 3,
        'RESULT': pd.Series([np.NaN, np.NaN, 1, 1, 1, 1, 0, 0, 1])
    }

    params = [param1]
    testfunc(text, params)


def test_ABS():
    text = """
        ABS(X);
    """
    param1 = {
        'X': pd.Series([-2, -1, -0.5, 9.8]),
        'RESULT': pd.Series([2, 1, 0.5, 9.8])
    }

    param2 = {
        'X': pd.Series([-2, -1, 0, 9]),
        'RESULT': pd.Series([2, 1, 0, 9])
    }

    params = [param1, param2]
    testfunc(text, params)


def test_SQRT():
    text = """
        SQRT(X);
    """
    param1 = {
        'X': pd.Series([16.0, 25.0, 36.0, 49.0]),
        'RESULT': pd.Series([4.0, 5.0, 6.0, 7.0])
    }

    params = [param1]
    testfunc(text, params, True)


def test_POW():
    text = """
        POW(X, M);
    """
    param1 = {
        'X': pd.Series([2.0, 1.0, 3.0]),
        'M': 3,
        'RESULT': pd.Series([8.0, 1.0, 27.0])
    }

    params = [param1]
    testfunc(text, params)


def test_CONST():
    text = """
        CONST(X);
    """
    param1 = {
        'X': pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'RESULT': pd.Series([10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    }

    params = [param1]
    testfunc(text, params)


def test_INSIST():
    """
       LAST is the alias name of INSIST
    """
    text = """
       INSIST(X, A, B);
    """
    param1 = {
        'X': pd.Series(
            [False, True, False, True, True, True, False, False, True, False, False, True, True, True, True]),
        'A': (3, '3', 0),
        'B': (1, '1', 0),
        'RESULT': pd.Series(
            [False, False, False, False, False, True, True, False, False, False, False, False, False, True, True])
    }

    param2 = {
        'X': pd.Series(
            [False, True, False, True, True, True, False, False, True, False, False, True, True, True, True]),
        'A': (1, '1', 0),
        'B': (3, '3', 0),
        'RESULT': pd.Series(
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    }

    params = [param1, param2]
    testfunc(text, params)


def test_FILTER():
    text = """
       FILTER(X, N);
    """

    param1 = {
        'X': pd.Series([False, True, False, True, True, True, True, False, True, True, True, True, True, True, True]),
        'N': (2, '2', 0),
        'RESULT': pd.Series(
            [False, True, False, False, True, False, False, False, True, False, False, True, False, False, True])
    }

    params = [param1]
    testfunc(text, params, False, False)


def test_BARSLAST():
    text = """
       BARSLAST(X);
    """

    param1 = {
        'X': pd.Series([False, False, True, False, False, False, True, False, False, False, True, True, False, False]),
        'RESULT': pd.Series([0, 0, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 2])
    }

    params = [param1]
    testfunc(text, params, False, False)


def test_AVEDEV():
    text = """
       AVEDEV(X, N);
    """

    param1 = {
        'X': pd.Series([10, 20, 30, 20, 30, 40, 30, 40]),
        'N': 3,
        'RESULT': pd.Series([np.NaN, np.NaN, 6.666667, 4.444444, 4.444444, 6.666667, 4.444444, 4.444444])
    }

    params = [param1]
    testfunc(text, params, True, True)


def test_EXPEMA():
    text = """
       EMA(X, N);
    """

    param1 = {
        'X': pd.Series([10.2, 30.9, 30.48, 39.34, 43.3, 45.9, 30.48, 39.34, 45.9, 30.48, 39.34]),
        'N': 3,
        'RESULT': pd.Series(
            [10.2, 24, 27.702857, 33.909333, 38.756129, 42.384762, 36.385512, 37.868549, 41.892133, 36.180489,
             37.761016])
    }

    params = [param1]
    testfunc(text, params, True, True)


def test_MEMA():
    text = """
       MEMA(X, N);
    """

    param1 = {
        'X': pd.Series([10.2, 30.9, 30.48, 39.34, 43.3, 45.9, 30.48, 39.34, 45.9, 30.48, 39.34]),
        'N': 3,
        'RESULT': pd.Series(
            [10.2, 17.10, 21.56, 27.486667, 32.757778, 37.138519, 34.919012, 36.392675, 39.561783, 36.534522, 37.46968])
    }

    params = [param1]
    testfunc(text, params, True, True)


def test_DMA():
    text = """
       DMA(X, A);
    """

    param1 = {
        'X': pd.Series([10.2, 30.9, 30.48, 39.34, 43.3, 45.9, 30.48, 39.34, 45.9, 30.48, 39.34]),
        'A': pd.Series([0.8, 0.1, 0.5, 0.28, 0.19, 0.22, 0.43, 0.9, 0.8, 0.43, 0.28]),
        'RESULT': pd.Series(
            [10.2, 12.2700, 20.3400, 18.3592, 16.4890, 18.0540, 18.9204, 36.4260, 38.7600, 18.9204, 18.3592])
    }

    params = [param1]
    testfunc(text, params, True, True)


def test_SMA():
    text = """
       SMA(X, M, N);
    """

    param1 = {
        'X': pd.Series([10.2, 30.9, 30.48, 39.34, 43.3, 45.9, 30.48, 39.34, 45.9, 30.48, 39.34]),
        'M': 5,
        'N': 3,
        'RESULT': pd.Series(
            [10.2, 24.985714, 28.507692, 35.177833, 40.101552, 43.594930, 35.713058, 37.890650, 42.697520, 35.366239,
             37.750596])
    }

    params = [param1]
    testfunc(text, params, True, True)

if __name__ == '__main__':
    test_CROSS()
    test_NOT()
    test_IF()
    test_EVERY()
    test_EXIST()
    test_STD()
    test_VAR()
    test_STDP()
    test_VARP()
    test_MAX()
    test_MIN()
    test_COUNT()
    test_ABS()
    test_SQRT()
    test_POW()
    test_CONST()
    test_INSIST()
    test_FILTER()
    test_BARSLAST()
    test_AVEDEV()
    test_EXPEMA()
    test_MEMA()
    test_DMA() #Failed.
    test_SMA()

