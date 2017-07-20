import sys

class Node(object):
    """ Abstract base class for AST nodes.
    """
    
    def isIndicator(self):
        return False

    def show(self, buf=sys.stdout, offset=0, attrnames=False, nodenames=False, showcoord=False, _my_node_name=None):
        lead = ' ' * offset
        if nodenames and _my_node_name is not None:
            buf.write(lead + self.__class__.__name__+ ' <' + _my_node_name + '>: ')
        else:
            buf.write(lead + self.__class__.__name__+ ': ')

        if self.attr_names:
            if attrnames:
                nvlist = [(n, getattr(self,n)) for n in self.attr_names]
                attrstr = ', '.join('%s=%s' % nv for nv in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ', '.join('%s' % v for v in vlist)
            buf.write(attrstr)

        if showcoord:
            buf.write(' (at %s)' % self.coord)
        
        self._show(buf)
        buf.write('\n')
    
    def _show(self, buf):
        pass
    
    def invoke(self, method_name, param = None):
        method = getattr(self, method_name, None)
        if hasattr(method, '__call__'):
            return method(param)
        return None

class TransUnit(Node):
    def __init__(self, name = 'TransUnit'):
        self.name = name
        self.funcdefs = []
        self.functors = []
        self.stmts = []
        self.result = None
        self.figures = []
        
    def addFuncDef(self, func_def):
        self.funcdefs.append(func_def)
        self.stmts.append(func_def)
    
    def addFunctor(self, functor):
        self.functors.append(functor)
        self.stmts.append(functor)

    def evaluate(self, ctx):
        result = None
        for node in self.funcdefs:
            result = node.evaluate(ctx)
            ctx.addSymbol(node.name, result)
            figure = node.figure(ctx, result)
            if not figure is None:
                self.figures.append(figure)

        for func in self.functors:
            result = func.evaluate(ctx)
            figure = func.figure(ctx, result, self.name)
            if not figure is None:
                self.figures.append(figure)
        
        self.result = result
        return result
    
    def figure(self, ctx):
        return self.figures
    
    def annotate(self, ctx):
        result = r''
        i = 0
        for stmt in self.stmts:
            if i > 0:
                result += '\n'
            i += 1
            result += stmt.annotate(ctx)
            result += ';'
        return result
    
    attr_names = ('name',)

class FuncDef(Node):
    def __init__(self, name, functor):
        self.name = name
        self.functor = functor
        self.is_indicator = False

    def setAsIndicator(self):
        self.is_indicator = True

    def figure(self, ctx, result):
        if not self.is_indicator:
            return None
        
        figure = self.functor.figure(ctx, result, self.name)
        if figure is None:
            return None
        figure['indicator'] = self.name
        return figure

    def setColor(self, color):
        self.functor.setColor(color)
    
    def setDrawAttr(self, attr):
        self.functor.setDrawAttr(attr)
    
    def evaluate(self, ctx):
        return self.functor.evaluate(ctx)
    
    def annotate(self, ctx):
        result = r'[%s]的定义: ' % self.name
        result += self.functor.annotate(ctx)
        return result

    attr_names = ('name',)


class UnaryExpr(Node):
    def __init__(self):
        self.neg = False
        self.attr = None
        self.color = None

    def negative(self):
        self.neg = True

    def evaluate(self, ctx):
        ret = self._evaluate(ctx)
        if ret is None:
            return None
        if self.neg:
            return -ret
        return ret

    def _figure(self, ctx, result, name):
        draw_default = ctx.resolveSymbol('DRAWFUNC', 3)
        if draw_default is None:
            return None
        else:
            result = draw_default([result, self.attr, self.color, name])
        return result
    
    def setColor(self, color):
        self.color = color
    
    def setDrawAttr(self, attr):
        self.attr = attr

class Functor(UnaryExpr):
    def __init__(self, name, param):
        UnaryExpr.__init__(self)
        self.name = name
        self.param = param
    
    def figure(self, ctx, result, name):
        return self._figure(ctx, result, name)
    
    def _show(self, buf):
        if self.attr:
            buf.write(' attr:%s' % self.attr)
        if self.color:
            buf.write(' color:%s' % self.color)

    def _evaluate(self, ctx):
        if self.param is None:
            param_num = 0
        else:
            param_num = len(self.param)
        func = ctx.resolveSymbol(self.name, param_num)
        if not func is None:
            if hasattr(func, '__call__'):
                if self.param is None:
                    return func()
                params = [v.evaluate(ctx) for v in self.param]
                if any([x is None for x in params]):
                    return None
                return func(params)
            else:
                return func
        print("Symbol is not resolved: %s" % self.name)
        return None

    def annotate(self, ctx):
        result = r''
        anno = ctx.resolveAnnotate(self.name)
        if not anno is None:
            if self.param is None:
                result = anno
            else:
                param = [v.annotate(ctx) for v in self.param]
                param = tuple(param)
                result = anno % param
        else:
            if self.param is None:
                result = self.name
            else:
                params = [v.annotate(ctx) for v in self.param]
                result = r'(' + params[0] 
                for param in params[1:]:
                    result += ',' + param
                result += r')'
        return result

    attr_names = ('name',)

    
class Parenthesis(UnaryExpr):
    def __init__(self, expr):
        UnaryExpr.__init__(self)
        self.name = '()'
        self.expr = expr
    
    def figure(self, ctx, result, name):
        return self._figure(ctx, result, name)

    def _evaluate(self, ctx):
        return self.expr.evaluate(ctx)
    
    def annotate(self, ctx):
        return self.expr.annotate(ctx)

    attr_names = ('name',)
    

def ValidDataType(a, b):
    if type(a)  != type(b):
        if type(a)  != float and type(a) != int:
            return False
        if type(b) != float and type(b) != int:
            return False
        return True
    return True

class Operator(UnaryExpr):
    def __init__(self, left, right, operator):
        UnaryExpr.__init__(self)
        self.name = operator
        self.left = left
        self.right = right
    
    def figure(self, ctx, result, name):
        return self._figure(ctx, result, name)
        
    def _show(self, buf):
        buf.write(' left:%s right:%s' % (self.left.name, self.right.name))

    def _evaluate(self, ctx):
        if self.left.evaluate(ctx) is None:
            return None
        if self.right.evaluate(ctx) is None:
            return None
        if self.name == '+':
            return self.left.evaluate(ctx) + self.right.evaluate(ctx)
        elif self.name == '-':
            return self.left.evaluate(ctx) - self.right.evaluate(ctx)
        elif self.name == '*':
            return self.left.evaluate(ctx) * self.right.evaluate(ctx)
        elif self.name == '/':
            return self.left.evaluate(ctx) / self.right.evaluate(ctx)
        elif self.name == 'AND' or self.name == '&&':
            return self.left.evaluate(ctx) & self.right.evaluate(ctx)
        elif self.name == 'OR' or self.name == '||':
            return self.left.evaluate(ctx) | self.right.evaluate(ctx)
        elif self.name == '=':
            return self.left.evaluate(ctx) == self.right.evaluate(ctx)
        elif self.name == '>':
            return self.left.evaluate(ctx) > self.right.evaluate(ctx)
        elif self.name == '>=':
            return self.left.evaluate(ctx) >= self.right.evaluate(ctx)
        elif self.name == '<':
            return self.left.evaluate(ctx) < self.right.evaluate(ctx)
        elif self.name == '<=':
            return self.left.evaluate(ctx) <= self.right.evaluate(ctx)
        elif self.name == '!=':
            return self.left.evaluate(ctx) != self.right.evaluate(ctx)
    
    def annotate(self, ctx):
        return self.left.annotate(ctx) + ' ' + self.name + ' ' + self.right.annotate(ctx)

    attr_names = ('name',)

class Integer(UnaryExpr):
    def __init__(self, name, value):
        UnaryExpr.__init__(self)
        self.name = name
        self.result = value
    
    def figure(self, ctx, result, name):
        return self._figure(ctx, result, name)

    def _evaluate(self, ctx):
        return self.result
    
    def annotate(self, ctx):
        return '%d' % self.result
    
    attr_names = ('name',)

class Float(UnaryExpr):
    def __init__(self, name, value):
        UnaryExpr.__init__(self)
        self.name = name
        self.result = value
    
    def figure(self, ctx, result, name):
        return self._figure(ctx, result, name)

    def _evaluate(self, ctx):
        return self.result
    
    def annotate(self, ctx):
        return '%0.3f' % self.result
    
    attr_names = ('name',)

class String(UnaryExpr):
    def __init__(self, name, value):
        UnaryExpr.__init__(self)
        self.name = name
        self.result = value

    def figure(self, ctx, result, name):
        return self._figure(ctx, result, name)

    def evaluate(self, ctx):
        return self.result
    
    def annotate(self, ctx):
        return self.result
    
    attr_names = ('name',)
