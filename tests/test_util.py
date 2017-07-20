import pandas as pd
import numpy as np
from tdx import Formula

def verify_result(result, expected, float_cmp, debug):
    if not isinstance(result, pd.core.series.Series):
        if result == expected:
            print('TEST OK')
            return
        else:
            print('TEST Failed.')
            return
    result = result.dropna()
    expected = expected.dropna()
    if debug:
        print('RESULT:')
        print(result)
        print('EXPECTED:')
        print(expected)
    if float_cmp:
        cmp = (np.abs(result - expected) < 2.631048e-06)
    else:
        cmp = (result == expected)
    if len(cmp[cmp == False]) > 0 :
        print('TEST Failed.')
        return
    print('TEST OK.')
    return


def testfunc(text, param, float_comp = False, debug = False):
    formula = Formula(text, "")
    if isinstance(param, dict):
        params = [param]
    elif isinstance(param, list):
        params = param
    else:
        raise ValueError("param should be dict or list.")

    for pa in params:
        #print(formula.annotate(pa))
        result = formula.evaluate(pa)
        if result is None:
            print('TEST Failed in evaluate strategy.')
            return
        verify_result(result, pa['RESULT'], float_comp, debug)


def testcase(func):
    def test_case_impl():
        print('Perform %s...' % func.__name__)
        ret = func()
        testfunc(*ret)
    return test_case_impl
