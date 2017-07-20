from .tdx_parser import TDXParser
import pandas as pd
import numpy as np
import json
from collections import deque

class Formula(object):
    buy_kw = [r'买入', r'买', 'BUY', 'BUYIN', 'ENTERLONG']
    sell_kw = [r'卖出', r'卖', 'SELL', 'SELLOUT', 'EXITLONG']

    FIGURE_DATA_LEN = 200    
    
    def __init__(self, text, param_desc='', **kwargs):
        self.figure_data_len = Formula.FIGURE_DATA_LEN
        if 'json_len' in kwargs:
            self.figure_data_len = kwargs['json_len']
        self.text = text
        self.param_desc = param_desc
        parser = TDXParser()
        self.ast = parser.parse(text)
        self.const = {
            'PERCENT_HLINE' : [0, 20, 50, 80, 100],
        }
        self._df = None
        self.local = {}
        self.buy = None
        self.sell = None
        self.builtin = {
            'HHV': (self.HHV, r'%s的%s日内的最高价', 2),
            'LLV': (self.LLV, r'%s的%s日内的最低价', 2),
            'SUM': (self.SUM, r'%s的%s日数值之和', 2),
            'REF': (self.REF, r'%s的%s日前的参考值', 2),
            'CROSS': (self.CROSS, r'在%s上穿%s时触发', 2),
            'NOT': (self.NOT, r'对%s取反逻辑', 1),
            'IF' : (self.IF, r'IF(%s, %s, %s)', 3),
            'IFF' : (self.IF, r'IFF(%s, %s, %s)', 3),
            'EVERY' : (self.EVERY, r'%s周期内均满足%s', 2),
            'EXIST' : (self.EXIST, r'EXIST(%s, %s)', 2),
            'STD': (self.STD, r'%s周期%s的样本标准差', 2),
            'VAR': (self.VAR, r'%s周期%s的样本方差', 2),
            'STDP': (self.STDP, r'%s周期%s的总体标准差', 2),
            'VARP': (self.VARP, r'%s周期%s的总体方差', 2),
            'MAX': (self.MAX, r'取最大值(%s, %s)', 2),
            'MIN': (self.MIN, r'取最小值(%s, %s)', 2),
            'COUNT': (self.COUNT, r'满足条件%s在统计周期%s内的个数', 2),
            'ABS': (self.ABS, r'%s的绝对值', 1),
            'SQRT': (self.SQRT, r'%s的平方根', 1),
            'POW': (self.POW, r'%s的%s次方', 2),
            'LOG': (self.LOG, r'%s的对数', 1),
            'CONST': (self.CONST, r'%s的最后值为常量', 1),
            'INSIST': (self.INSIST, r'%s在周期%s到周期%s全为真', 3),
            'LAST': (self.INSIST, r'%s在周期%s到周期%s全为真', 3),
            'FILTER': (self.FILTER, r'过滤%s连续出现的%s个信号', 2),
            'BARSLAST': (self.BARSLAST, r'满足条件%s到当前的周期数', 1),
            'AVEDEV' : (self.AVEDEV, r'%s的周期%s的平均绝对偏差', 2),
            'MA': (self.MA, r'%s的%s日简单移动平均', 2),
            'EMA': (self.EMA, r'%s的%s日指数移动平均', 2),
            'EXPEMA': (self.EMA, r'%s的%s日指数移动平均', 2),
            'MEMA' : (self.MEMA, r'%s的%s周期平滑指数移动平均', 2),
            'EXPMEMA' : (self.MEMA, r'%s的%s周期平滑指数移动平均', 2),
            'DMA' :  (self.DMA, r'%s的%s周期动态平均', 2),
            'SMA' :  (self.SMA, r'%s的%s周期(权重:%s)动态平均', 3),
            'CONV':  (self.CONV, r'%s与%s的%s周期卷积', 3),
            'SAR' :  (self.SAR, r'周期为%s步长为%s极值为%s的抛物转向', 3),
            'SLOPE': (self.SLOPE, r'%s的周期为%s的线性回归斜率', 2),
            'CLAMP': (self.CLAMP, r'限定%s的输出在(%s, %s)之间', 3),
            'FORCAST': (self.FORCAST, r'%s的周期为%s的线性预测', 2),
            'DRAWFUNC': (self.DRAWFUNC, r'DRAWFUNC(%s, %s, %s)', 3),
            'DRAWICON': (self.DRAWICON, r'DRAWICON(%s, %s, %s)', 3),
            'DRAWICONF': (self.DRAWICONF, r'DRAWICONF(%s, %s, %s)', 3),
            'STICKLINE': (self.STICKLINE, r'STICKLINE(%s,%s,%s,%s,%s)', 5),
            'DRAWKLINE' : (self.DRAWKLINE, r'DRAWKLINE(%s, %s, %s, %s)', 4),
            'DRAWSKLINE' : (self.DRAWSKLINE, r'DRAWSKLINE(%s, %s, %s, %s)', 4),
            'DRAWPKLINE' : (self.DRAWPKLINE, r'DRAWPKLINE(%s, %s, %s, %s, %s)', 5),
            'DRAWNUMBER' : (self.DRAWNUMBER, r'DRAWNUMBER(%s, %s, %s)', 3),
            'DRAWTEXT' : (self.DRAWTEXT, r'DRAWTEXT(%s, %s, %s)', 3),
            'DRAWNULL' : (np.NaN, r'DRAWNULL', 0),
            'DRAWGRID' : (self.DRAWGRID, r'DRAWGRID(%s)', 1),
            'DRAWVOL' : (self.DRAWVOL, r'DRAWVOL(%s, %s)', 2),
            'SIGNAL': (self.SIGNAL, r'从仓位数据%s导出指定的买入%s卖出%s信号指示',3),
            'BASE': (self.BASE, r'创建%s的基准', 1),
            'ASSET': (self.ASSET, r'根据价格%s,仓位%s和建仓比例%s创建资产', 3),
            'SUGAR': (self.SUGAR, r'根据盈利情况进行调仓操作(仓位:%s, 价格:%s, 资产:%s)', 3),
            'REMEMB': (self.REMEMB, r'记录价格(条件:%s, 值:%s)', 2),
            'STOP':(self.STOP, r'止盈止损点(条件:%s, 价格:%s, 比例1:%s, 比例2:%s)', 4),
            'HORIZON':(self.HORIZON, r'%s的周期为%s的地平线', 2)
        }


    def __mergeParam(self, param):
        self.local = {}
        #for k, v in self.const.items():
        #    self.local[k] = v
        self.local.update(self.const)
        if param is None:
            return
        #for k, v in param.items():
        #    self.local[k] = v
        self.local.update(param)
    
    
    def annotate(self, param = None):
        """
            取注解, 返回文符串
        """
        if self.ast is None:
            print('This formula failed to be parsed.')
            return None
        
        self.__mergeParam(param)

        return self.ast.annotate(self)

    def paramDesc(self):
        """
            取参数描述，返回参数名与描述
        """
        if self.param_desc is None:
            return ''
        if self.param_desc == '':
            return ''
        return json.dumps(self.param_desc)

    def get_figure(self):
        """
           取绘图数据
           在调用这个函数前，需要调用求值函数
           调用求值函数，也可用于图表数据的刷新
        """
        if self.ast is None:
            print('This formula failed to be parsed.')
            return None

        return self.ast.figure(self)
    

    def evaluate(self, param, fields = None, is_last_value = False):
        """
            求值，设置股票代码，参数
            param: 参数以dict形式给出，键名为参数的变量名
        """
        if self.ast is None:
            print('This formula failed to be parsed.')
            raise ValueError('The formula failed to be parsed')
        if isinstance(param, dict):
            self.__mergeParam(param)
        if isinstance(param, pd.DataFrame):
            self._df = param

        default_retval = self.ast.evaluate(self)
        if default_retval is None:
            raise ValueError('Failed to evaluate. The formula failed to be parsed.')

        if fields is None:
            return default_retval

        retv = {}
        for col in fields:
            retv[col] = self.resolveSymbol(col, 0)

        if not is_last_value:
            return retv

        last_values = np.array([v.iloc[-1] for k, v in retv.items()])
        return last_values


    def asDataFrame(self, columns, data_index = None):
        if data_index is None:
            close = self.resolveSymbol('CLOSE', 0)
            if close is None or len(close) == 0:
                return None
            df = pd.DataFrame(index = close.index)
        else:
            df = pd.DataFrame(index = data_index)
        for col in columns:
            s = self.resolveSymbol(col, 0)
            if s is not None:
                df[col] = s
        return df
    
    def setSymbol(self, symbol, func_def):
        old_def = None
        if symbol in self.builtin:
            old_def = self.builtin[symbol]
        self.builtin[symbol] = func_def
        return old_def            
            
        
    def resolveSymbol(self, symbol, n):
        if n == 0:
            if self._df is not None:
                if symbol in self._df.columns:
                    return self._df[symbol]
            if symbol in self.local:
                variable = self.local[symbol]
                if type(variable) == tuple:
                    variable = variable[0]
                if variable is not None:
                    if hasattr(variable, '__call__'):
                        return variable()
                    return variable

        symdef = None
        if symbol in self.builtin:
            symdef = self.builtin[symbol]
            if n == symdef[2]:
                return symdef[0]
        if symdef is not None:
            print('function: %s is resolved, expect %d parameters, but %d is given.' % (symbol, symdef[2], n))
        
        if symbol in self.local:
            funcdef = self.local[symbol]
            if type(funcdef) == tuple:
                if n == funcdef[2]:
                    func = funcdef[0]
                else:
                    print('function: %s is resolved, expect %d parameters, but %d is given.' % (symbol, funcdef[2], n))
            else:
                func = funcdef
            return func
        return None
    
    def resolveAnnotate(self, symbol):
        if symbol in self.local:
            variable = self.local[symbol]
            if type(variable) != tuple:
                return '[%s]' % symbol
            return variable[1]
        if symbol in self.builtin:
            symdef = self.builtin[symbol]
            return symdef[1]
        return None
    
    def registerFunctor(self, name, func, n):
        self.builtin[name] = (func, 'DefineFunction', n)

#############  内部函数  #################    
    def addSymbol(self, symbol, value):
        self.local[symbol] = value
        if symbol in Formula.buy_kw:
            if isinstance(value, pd.core.series.Series):
                if value.dtype == bool:
                    self.buy = value
        
        if symbol in Formula.sell_kw:
            if isinstance(value, pd.core.series.Series):
                if value.dtype == bool:
                    self.sell = value

### Formula Function Implementation ###    

#引用函数


    def HHV(self, param):
        if param[1] == 0:
            return pd.expanding_max(param[0])
        return pd.rolling_max(param[0], param[1])
    
    def LLV(self, param):
        if param[1] == 0:
            return pd.expanding_min(param[0])
        return pd.rolling_min(param[0], param[1])
        
    def REF(self, param):
        return param[0].shift(param[1])
        

    def EMA(self, param):
        return pd.ewma(param[0], span=param[1], adjust = True)

    def MA(self, param):
        if param[1] == 0:
            return pd.rolling_mean(param[0])
        return pd.rolling_mean(param[0], param[1])
        
    def SUM(self, param):
        if param[1] == 0:
            return pd.expanding_sum(param[0])
        return pd.rolling_sum(param[0], param[1])
    
    def CONST(self, param):
        ret = pd.Series(index = param[0].index)
        ret[:] = param[0][-1:].values[0]
        return ret
    
    def MEMA(self, param):
        return pd.ewma(param[0], span=param[1] * 2 - 1, adjust = False)
    
    def DMA(self, param):
        df = pd.DataFrame(index = param[0].index)
        df['X'] = param[0]
        df['A'] = param[1]
        class Averager:
            def __init__(self):
                self.Y = 0
                self.start = True

            def handleInput(self, row):
                if self.start:
                    self.start = False
                    self.Y = row['X']
                    return self.Y
                X = row['X']
                A = row['A']
                if A > 1:
                    A = 1
                if A < 0:
                    A = 0
                self.Y = A * X + (1 - A) * self.Y
                return self.Y
        avger = Averager()
        result = df.apply(avger.handleInput, axis = 1, reduce = True)
        return result
        
    def SMA(self, param):
        M = param[2]
        if param[2] == 0:
            M = 1
        return pd.ewma(param[0], span = 2 * param[1] / M - 1)
        
    def CONV(self, param):
        df = pd.DataFrame(index = param[0].index)
        df['X'] = param[0]
        df['W'] = param[1]
        class Convolution:
            def __init__(self, N):
                self.N = N
                self.q = deque([], self.N)
                self.tq = deque([], self.N)
                self.s = 0
                self.t = 0
            
            def handleInput(self, row):
                if len(self.q) < self.N:
                    if pd.isnull(row['W']) or pd.isnull(row['X']):
                        return np.NaN
                    self.q.append(row['W'] * row['X'])
                    self.tq.append(row['W'])
                    self.s += row['W'] * row['X']
                    self.t += row['W']
                    return np.NaN
                ret = self.s / self.t
                self.s -= self.q[0]
                self.t -= self.tq[0]
                delta_s = row['W'] * row['X']
                delta_t = row['W']
                self.s += delta_s
                self.t += delta_t
                self.q.append(delta_s)
                self.tq.append(delta_t)
                return ret
        conv = Convolution(param[2])
        result = df.apply(conv.handleInput, axis = 1, reduce = True)
        return result
                
    
#算术逻辑函数
    EPSLON = 0.0000001
    
    def CROSS(self, param):
        if not isinstance(param[0], pd.core.series.Series) and not isinstance(param[1], pd.core.series.Series):
            print('Invalid data type is detected.')
            return False

        if not isinstance(param[0], pd.core.series.Series):
            x1 = param[0]
            x2 = param[0]
            y1 = param[1].shift(1)
            y2 = param[1]
        
        if not isinstance(param[1], pd.core.series.Series):
            x1 = param[0].shift(1)
            x2 = param[0]
            y1 = param[1]
            y2 = param[1]
        
        if isinstance(param[0], pd.core.series.Series) and isinstance(param[1], pd.core.series.Series):
            x1 = param[0].shift(1)
            x2 = param[0]
            y1 = param[1].shift(1)
            y2 = param[1]
        
        return (x1 <= y1) & (x2 > y2)

    def NOT(self, param):
        if not isinstance(param[0], pd.core.series.Series):
            if type(param[0]) != bool:
                return (param[0] == 0)
            else:
                return not param[0]

        if param[0].dtype == bool:
            return (param[0] == False)

        if param[0].dtype == float:
            return (param[0] > -Formula.EPSLON) & (param[0] < Formula.EPSLON)

        return (param[0] == 0)


    def IF(self, param):
        EPSLON = 0.0000001
        if not isinstance(param[0], pd.core.series.Series):
            if type(param[0]) == bool:
                if param[0]:
                    return param[1]
                else:
                    return param[2]
            elif type(param[0]) == int:
                if param[0] != 0:
                    return param[1]
                else:
                    return param[2]
            elif type(param[0]) == float:
                if (param[0] < -Formula.EPSLON) or (param[0] > Formula.EPSLON):
                    return param[1]
                else:
                    return param[2]
        df = pd.DataFrame(index = param[0].index)
        if param[0].dtype == bool:
            df['X'] = param[0]
        elif param[0].dtype == float:
            df['X'] = ~ ((param[0] > -EPSLON) & (param[0] < EPSLON))
        else:
            df['X'] = (param[0] != 0)
            
        df['A'] = param[1]
        df['B'] = param[2]
        
        def callback(row):
            if row['X']:
                return row['A']
            else:
                return row['B']
        
        result = df.apply(callback, axis=1, reduce = True)
        return result
        
    def EVERY(self, param):
        norm = self.IF([param[0], 1, 0])
        result = pd.rolling_sum(norm, param[1])
        return result == param[1]
        
    def EXIST(self, param):
        norm = self.IF([param[0], 1, 0])
        result = pd.rolling_sum(norm, param[1])
        return result > 0

    def COUNT(self, param):
        norm = self.IF([param[0], 1, 0])
        result = pd.rolling_sum(norm, param[1])
        return result
    
    def INSIST(self, param):
        norm = self.IF([param[0], 1, 0])
        x1 = pd.rolling_sum(norm, param[1])
        x2 = pd.rolling_sum(norm, param[2])
        ret =((x1 - x2) == np.abs(param[1] - param[2]))
        return ret

    def FILTER(self, param):
        class Counter:
            def __init__(self, N):
                self.state = 0
                self.count = 0
                self.num = N
            def handleInput(self, value):
                if self.state == 0:
                    if value:
                        self.state = 1
                        self.count = 0
                        return True
                    else:
                        return False
                else:
                    self.count += 1
                    if self.count >= self.num:
                        self.state = 0
                    return False
        counter = Counter(param[1])
        ret = param[0].apply(counter.handleInput)
        return ret

    def BARSLAST(self, param):
        class Counter:
            def __init__(self):
                self.count = -1
            def handleInput(self, value):
                if value:
                    self.count = 0
                    return self.count
                elif self.count != -1:
                    self.count += 1
                    return self.count
                else:
                    return 0
        counter = Counter()
        ret = param[0].apply(counter.handleInput)
        return ret
        
#统计函数
    def STD(self, param):
        return pd.rolling_std(param[0], param[1])

    def VAR(self, param):
        return pd.rolling_var(param[0], param[1])

    def STDP(self, param):
        return pd.rolling_std(param[0], param[1], ddof = 0)

    def VARP(self, param):
        return pd.rolling_var(param[0], param[1], ddof = 0)
    
    def MAX(self, param):
        if isinstance(param[0], pd.core.series.Series):
            df = pd.DataFrame(index = param[0].index)
        elif isinstance(param[1], pd.core.series.Series):
            df = pd.DataFrame(index = param[1].index)
        else:
            df = None
        
        if df is None:
            return np.max(param)

        df['A'] = param[0]
        df['B'] = param[1]
        def callback(row):
            if row['A'] >= row['B']:
                return row['A']
            else:
                return row['B']
        result = df.apply(callback, axis = 1, reduce = True)
        return result

    def MIN(self, param):
        if isinstance(param[0], pd.core.series.Series):
            df = pd.DataFrame(index = param[0].index)
        elif isinstance(param[1], pd.core.series.Series):
            df = pd.DataFrame(index = param[1].index)
        else:
            df = None
        
        if df is None:
            return np.max(param)
        df['A'] = param[0]
        df['B'] = param[1]
        def callback(row):
            if row['A'] <= row['B']:
                return row['A']
            else:
                return row['B']
        result = df.apply(callback, axis = 1, reduce = True)
        return result
        
    def AVEDEV(self, param):
        return pd.rolling_apply(param[0], param[1], lambda x: pd.DataFrame(x).mad())
        
    def SLOPE(self, param):
        class Context:
            def __init__(self, N):
                self.N = N
                self.q = deque([], self.N)
                self.x = [i for i in range(self.N)]
            
            def handleInput(self, value):
                if len(self.q) < self.N:
                    self.q.append(value)
                    return 0
                self.q.append(value)
                z1 = np.polyfit(self.x, self.q, 1)
                return z1[0]
        ctx = Context(param[1])
        result = param[0].apply(ctx.handleInput)
        return result
        
        
    def FORCAST(self, param):
        class Context:
            def __init__(self, N):
                self.N = N
                self.q = deque([], self.N)
                self.x = [i for i in range(self.N)]
            
            def handleInput(self, value):
                if len(self.q) < self.N:
                    self.q.append(value)
                    return np.NaN
                z1 = np.polyfit(self.x, self.q, 1)
                fn = np.poly1d(z1)
                y = fn(self.N + 1)
                self.q.append(value)
                return y
            
        ctx = Context(param[1])
        result = param[0].apply(ctx.handleInput)
        return result
        
#数学函数
    def ABS(self, param):
        return np.abs(param[0])

    def SQRT(self, param):
        return np.sqrt(param[0])

    def POW(self, param):
        return np.power(param[0], param[1])

    def LOG(self, param):
        return param[0].apply(np.log)


#指标函数
    def SAR(self, param):
        N = param[0]
        iaf = param[1] / 100
        maxaf = param[2] / 100
        
        close = self.resolveSymbol('CLOSE', 0)
        high = self.resolveSymbol('HIGH', 0)
        low = self.resolveSymbol('LOW', 0)
        
        df = pd.DataFrame(index = close.index)
        df['CLOSE'] = close
        df['HIGH'] = high
        df['LOW'] = low        
        
        class Context:
            def __init__(self, N, iaf, maxaf):
                self.N = N
                self.iaf = iaf
                self.maxaf = maxaf
                self.q_low = deque([], self.N)
                self.q_high = deque([], self.N)
                self.psar = 0
                self.bull = True
                self.af = iaf
                self.hp = 0
                self.lp = 0
                
            
            def handleInput(self, row):
                if len(self.q_low) < self.N:
                    self.psar = row['CLOSE']
                    self.q_low.append(row['LOW'])
                    self.q_high.append(row['HIGH'])
                    self.hp = row['HIGH']
                    self.lp = row['LOW']
                    return self.psar
                
                low = row['LOW']
                high = row['HIGH']
                if self.bull:
                    psar = self.psar + self.af * (self.hp - self.psar)
                else:
                    psar = self.psar + self.af * (self.lp - self.psar)
                
                reverse = False
                if self.bull:
                    if low < psar:
                        self.bull = False
                        reverse = True
                        psar = self.hp
                        self.lp = low
                        self.af = self.iaf
                else:
                    if high > psar:
                        self.bull = True
                        reverse = True
                        psar = self.lp
                        self.hp = high
                        self.af = self.iaf
                if not reverse:
                    if self.bull:
                        if high > self.hp:
                            self.hp = high
                            self.af = min(self.af + self.iaf, self.maxaf)
                        lowest = np.min(self.q_low)
                        psar = min(lowest, psar)
                    else:
                        if low < self.lp:
                            self.lp = low
                            self.af = min(self.af + self.iaf, self.maxaf)
                        highest = np.max(self.q_high)
                        psar = max(highest, psar)
                self.psar = psar
                self.q_low.append(low)
                self.q_high.append(high)
                return psar

        ctx = Context(N, iaf, maxaf)
        result = df.apply(ctx.handleInput, axis = 1, reduce = True)
        return result



#绘图函数
    def DRAWFUNC(self, param):
        '''
          这个函数包裹了默认数据的图形表示
          param[0] == data
          param[1] == attr
          param[2] == color
          param[3] == function name
        '''
        result = param[0]
        if result is None:
            return None
        if type(result) == bool:
            return None
        if isinstance(result, dict):
            if not param[1] is None:
                result['attr'] = param[1]
            if not param[2] is None:
                result['color'] = param[2]
            return result
        
        # NODRAW if indicator == 'BUY' or 'SELL'
        if param[3] in Formula.buy_kw:
            return None
        if param[3] in Formula.sell_kw:
            return None
        if param[1] == 'NODRAW':
            return None
        if param[1] == 'DRAWNULL':
            return None
        figure = {'figure': result}
        if not param[1] is None:
            figure['attr'] = param[1]
        if not param[2] is None:
            figure['color'] = param[2]
        if type(result) == float or type(result) == int:
            figure['function'] = 'DRAWHLINE'
            return figure
        if result.dtype == bool:
            figure['function'] = 'DRAWBOOL'
            return figure
        figure['function'] = 'DRAWDATA'
        return figure

    def STICKLINE(self, param):
        df = pd.DataFrame(index = param[0].index)
        df['COND'] = param[0]
        df['P1'] = param[1]
        df['P2'] = param[2]
        df = df.drop(df[df['COND'] == False].index)
        del df['COND']
        return {'function': 'STICKLINE', 'figure' : df, 'width': param[3], 'fill': param[4]}
        
    def DRAWICONF(self, param):
        figure = pd.DataFrame(index = param[1].index)
        price = pd.Series(param[1], name='PRICE')
        figure = figure.join(price)
        figure['COND'] = param[0]
        figure['ICON'] = param[2]
        figure.loc[figure['COND'] == False, 'ICON'] = 0
        if len(figure) == 0:
            return {'function':'DRAWICON','figure': None}
        return {'function':'DRAWICON', 'figure':figure}

    def DRAWICON(self, param):
        figure = pd.DataFrame(index = param[1].index)
        price = pd.Series(param[1], name='PRICE')
        figure = figure.join(price)
        figure['COND'] = param[0]
        figure['ICON'] = param[2]
        figure = figure.drop(figure[figure['COND'] == False].index)
        del figure['COND']
        if len(figure) == 0:
            return {'function':'DRAWICON','figure': None}
        
        return {'function':'DRAWICON', 'figure':figure}
        
    def DRAWKLINE(self, param):
        df = pd.DataFrame(index = param[0].index)
        df['HIGH'] = param[0]
        df['OPEN'] = param[1]
        df['LOW'] = param[2]
        df['CLOSE'] = param[3]
        return {'function': 'DRAWKLINE', 'figure': df}

    def DRAWSKLINE(self, param):
        df = pd.DataFrame(index = param[0].index)
        df['HIGH'] = param[0]
        df['OPEN'] = param[1]
        df['LOW'] = param[2]
        df['CLOSE'] = param[3]
        return {'function': 'DRAWKLINE', 'figure': df, 'attr':'SIMPLE'}

    def DRAWPKLINE(self, param):
        df = pd.DataFrame(index = param[0].index)
        df['HIGH'] = param[0]
        df['OPEN'] = param[1]
        df['LOW'] = param[2]
        df['CLOSE'] = param[3]
        df['POSITION'] = param[4]
        return {'function': 'DRAWPKLINE', 'figure': df}

    def DRAWNUMBER(self, param):
        figure = pd.DataFrame(index = param[1].index)
        figure['COND'] = param[0]
        figure['PRICE'] = param[1]
        figure['NUMBER'] = param[2]
        figure = figure.drop(figure[figure['COND'] == False].index)
        del figure['COND']
        return {'function':'DRAWNUMBER', 'figure':figure}

    def DRAWTEXT(self, param):
        figure = pd.DataFrame(index = param[1].index)
        figure['COND'] = param[0]
        figure['PRICE'] = param[1]
        figure['TEXT'] = param[2]
        figure = figure.drop(figure[figure['COND'] == False].index)
        del figure['COND']
        return {'function':'DRAWTEXT', 'figure':figure}
        
    def DRAWGRID(self, param):
        return {'function': 'DRAWGRID', 'figure': param[0], 'attr': 'DOTLINE'}

    def DRAWVOL(self, param):
        figure = pd.DataFrame(index = param[0].index)
        figure['VOL'] = param[0]
        figure['FILL'] = param[1]
        return {'function': 'DRAWVOL', 'figure': figure}
        
#交易分析的相关功能
    #根据仓位状态生成买卖信号
    def SIGNAL(self, param):
        if not isinstance(param[0], pd.core.series.Series):
            close = self.resolveSymbol('CLOSE', 0)
            src = pd.Series(index=close.index)
            src.fillna(param[0])
        else:
            src = param[0]
        class Context:
            def __init__(self, bs, ss):
                self.current = -1
                self.bs = bs
                self.ss = ss
            
            def handleInput(self, value):
                if self.current < 0:
                    self.current = value
                    return 0
                if value - self.current > 0:
                    self.current = value
                    return self.bs
                else:
                    if value - self.current < 0:
                        self.current = value
                        return self.ss
                return 0
            
        ctx = Context(param[1], param[2])
        result = src.apply(ctx.handleInput)
        return result

    def BASE(self, param):
        """
        将输入序列转化为以1为基准的基准序列
        """
        class Context:
            def __init__(self):
                self.origin = 100000
                self.stock = 0
            
            def handleInput(self, value):
                if self.stock == 0:
                    if value == 0:
                        return 1
                    self.stock = self.origin / value
                    return 1
                return self.stock * value / self.origin
            
        ctx = Context()
        result = param[0].apply(ctx.handleInput)
        return result
        
    def ASSET(self, param):
        """
        计算资产  p1: CLOSE, p2: POSITION, p3: TIMES
        """
        df = pd.DataFrame(index = param[0].index)
        df['CLOSE'] = param[0]
        df['POSITION'] = param[1]
        class Context:
            def __init__(self, times):
                self.origin = 100000
                self.cash = 100000
                self.position = 0
                self.times = times
                self.cash_brick = self.cash / self.times
                self.quant_brick = 0
                self.quant = 0
                self.price = 0
            
            def callback(self, row):
                target = row['POSITION']
                self.price = row['CLOSE']
                delta = target - self.position
                if delta > self.times:
                    delta = self.times
                if delta < -self.times:
                    delta = -self.times
                if delta > 0:
                    budget = self.cash_brick * delta
                    self.cash -= budget
                    self.quant += budget / self.price
                    self.quant_brick = self.quant / target
                if delta < 0:
                    delta = -delta
                    budget = self.quant_brick * delta
                    self.quant -= budget
                    self.cash += budget * self.price
                    self.cash_brick = self.cash / (self.times - target)
                self.position = target
                return (self.cash + self.quant * self.price) / self.origin
        context = Context(param[2])
        asset = df.apply(context.callback, axis=1)
        return asset
    
    def SUGAR(self, param):
        """
        糖化  p1: POSITION, p2: close, p3: asset
        """
        df = pd.DataFrame(index = param[0].index)
        df['POSITION'] = param[0]
        df['PRICE'] = param[1]
        df['ASSET'] = param[2]
        class Context:
            def __init__(self):
                self.short_idx = []
                self.long_idx = []
                self.erase = []
                self.index = 0
                self.position = -1
                self.long = False
                self.short = False
                self.start = 0
            
            def callback(self, row):
                if self.position < 0:
                    self.position = row['POSITION']
                    self.start = row['ASSET']
                    return self.position

                if self.position == 0 and row['POSITION'] > 0:
                    self.short = False
                    self.long = True
                    self.start = row['ASSET']
                    self.asset = row['ASSET']
                    self.short_idx = []
                    self.long_idx = []

                if self.position > 0 and row['POSITION'] == 0:
                    self.long = False
                    if row['ASSET'] < self.start:
                        #d = len(self.long_idx) - len(self.short_idx)
                        #if d > 3:
                        #    self.erase += self.short_idx
                        #else:
                        self.erase += self.long_idx
                        self.erase.append(self.index)
                    else:
                        x = row['ASSET'] - self.start

                if self.long:
                    self.long_idx.append(self.index)
                    if self.asset > row['ASSET']:
                        self.short = True
                    if self.short:
                        self.short_idx.append(self.index)
                self.index += 1
                self.asset = row['ASSET']
                self.position = row['POSITION']
                return row['POSITION']
                
        context = Context()
        position = df.apply(context.callback, axis=1)
        position.iloc[context.erase] = 0
        
        return position


    def REMEMB(self, param):
        """
        REMEMB(true/false, value)
        
        remember the value when the first param is true
        """
        values = param[1]
        class Context:
            def __init__(self, judgement):
                self.judgement = judgement.values
                self.index = 0
                self.current = 0
                
            def handleInput(self, value):
                if self.judgement[self.index]:
                    self.current = value
                    self.index += 1
                    return value
                self.index += 1
                if self.current == 0:
                    return value
                else:
                    return self.current
        ctx = Context(param[0])
        result = values.apply(ctx.handleInput)
        return result        
        
        
    def STOP(self, param):
        """
        STOP(true/false, value, ratio)
        
        calculate the stop point according the value true point, return bool
        
        Generated:
            TRADE: a series contains the profit for each trade
            TOTAL_TRADES: total count that stop occurs
            WIN_TRADES: win count
        """
        values = param[1]
        ratio2 = param[2]
        ratio1 = param[3]
        class Context:
            def __init__(self, judgement):
                self.judgement = judgement.values
                self.index = -1
                self.current = 0
                self.toggled = False
                self.last_trade = 0
                self.win_count = 0
                self.total_count = 0
                
            def appendState(self, param):
                trade.append(param[0])
                win_count.append(param[1])
                total_count.append(param[2])
                
            def handleInput(self, value):
                self.index += 1
                if not self.toggled:
                    self.appendState((self.last_trade,self.win_count,self.total_count))
                    if self.judgement[self.index]:
                        self.toggled = True
                        self.current = value
                    return False
                else:
                    td = value / self.current - 1
                    if ratio1 != 100:
                        if td >= ratio1:
                            self.last_trade = td
                            self.win_count += 1
                            self.total_count += 1
                            self.appendState((td,self.win_count,self.total_count))
                            self.toggled = False
                            return True
                    if ratio2 != -100:
                        if td <= ratio2:
                            self.last_trade = td
                            self.total_count += 1
                            self.toggled = False
                            self.appendState((td,self.win_count,self.total_count))
                            return True
                    self.appendState((self.last_trade,self.win_count,self.total_count))
                    return False
        trade = []
        win_count = []
        total_count = []
        ctx = Context(param[0])
        result = values.apply(ctx.handleInput)
        self.local['TRADE'] = pd.Series(trade, index = values.index)
        self.local['TOTAL_TRADES']  = pd.Series(total_count, index = values.index)
        self.local['WIN_TRADES'] = pd.Series(win_count, index = values.index)
        return result



    def HORIZON(self, param):
        """
        HORIZON(values, N_DAY)
        
        Make a horizon from current to history, length is N_DAY.
        If nearest m days are lower than current, return -m
        If nearest m days are larger than current, return m
        """
        n_day = param[1]
        values = param[0]
        class Context:
            def __init__(self, n_day):
                self.q = deque([], n_day)

            def handleInput(self, value):
                if len(self.q) < n_day:
                    self.q.append(value)
                    return np.NaN

                m = 0
                for x in reversed(self.q):
                    if m == 0:
                        if x > value:
                            m = 1
                        else:
                            m = -1
                        continue
                    if m > 0:
                        if x > value:
                            m += 1
                        else:
                            break
                    if m < 0:
                        if x < value:
                            m -= 1
                        else:
                            break
                self.q.append(value)
                return m
        ctx = Context(n_day)
        result = values.apply(ctx.handleInput)
        return result

    def CLAMP(self, param):
        """
        CLAMP(value, min, max)

        make the value to be clamped into the range of [min, max]
        """
        values = param[0]
        min_ = param[1]
        max_ = param[2]

        class Context:
            def __init__(self, min_, max_):
                self.min_ = min_
                self.max_ = max_

            def handleInput(self, value):
                if value < self.min_:
                    return self.min_
                elif value > self.max_:
                    return self.max_
                return value

        ctx = Context(min_, max_)
        result = values.apply(ctx.handleInput)
        return result
