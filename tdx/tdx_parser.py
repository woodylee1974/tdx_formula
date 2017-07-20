from ply.yacc import yacc
from .tdx_lex import TDXLex
from .tdx_ast import *
import re

class TDXParser(object):
    precedence = (
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'EQ', 'NE'),
        ('left', 'GT', 'GE', 'LT', 'LE'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE')
    )
    
    def __init__(self):
        self.lex = TDXLex()
        self.lex.build()
        self.tokens = self.lex.tokens
        self.literals = self.lex.literals
        self.parser = yacc(module = self,
                                start = 'tu_e',
                                debug = False
                                )
        self.attr_re = re.compile(r'(LINETHICK[1-9])|STICK|VOLSTICK|LINESTICK|CROSSDOT|CIRCLEDOT|POINTDOT|DRAWNULL|DOTLINE|NODRAW|COLORSTICK')
        self.color_re = re.compile(r'COLOR[0-9A-Z]+')
    
    def parse(self, text):
        return self.parser.parse(input=text,
                                 lexer = self.lex,
                                 debug = False
                                 )

    def p_tu_e(self, p):
        """ tu_e   :    tu
                      | empty
        """
        p[0] = p[1]
    
    def p_tu_1(self, p):
        """ tu   :      stmt_d 
                   | tu stmt_d
        """
        n = None
        if (len(p) == 2):
            p[0] = TransUnit()
            n = p[1]
        else:
            p[0] = p[1]
            n = p[2]
        
        if n.__class__.__name__ == 'Functor' or \
           n.__class__.__name__ == 'Operator' or \
           n.__class__.__name__ == 'Parenthesis' :
            p[0].addFunctor(n)
        
        if n.__class__.__name__ == 'FuncDef':
            p[0].addFuncDef(n)
    
    
    def p_stmt_d(self, p):
        """ stmt_d    :    stmt ';' 
                         | stmt_attr ';'
        """
        p[0] = p[1]
    
    def p_stmt_attr(self, p):
        """ stmt_attr :  stmt      ',' FUNCNAME
                       | stmt_attr ',' FUNCNAME
        """
        p[0] = p[1]
        if self.attr_re.match(p[3]):
            p[0].setDrawAttr(p[3])
        else:
            if self.color_re.match(p[3]):
                p[0].setColor(p[3])
    
    def p_stmt_1(self, p):
        """ stmt  : FUNCNAME ':' expr
        """
        p[0] = FuncDef(p[1], p[3])
        p[0].setAsIndicator()
    
    def p_stmt_2(self, p):
        """ stmt  : FUNCNAME ASSIGN expr
        """
        p[0] = FuncDef(p[1], p[3])

    def p_stmt_3(self, p):
        """ stmt  : expr
        """
        p[0] = p[1]
    

    def p_expr_0(self, p):
        """ expr :  unary_expr
        """
        p[0] = p[1]
    
    def p_expr_1(self, p):
        """ expr :  MINUS unary_expr
        """
        p[0] = p[2]
        p[0].negative()
    
    def p_expr_2(self, p):
        """ expr  :  LPAREN expr RPAREN
        """
        p[0] = Parenthesis(p[2])
    
    def p_expr_3(self, p):
        """ expr   :   expr TIMES expr
                     | expr DIVIDE expr
                     | expr PLUS  expr
                     | expr MINUS expr
        """
        p[0] = Operator(p[1], p[3], p[2])
    
    def p_expr_4(self, p):
        """ expr :     expr LT expr
                     | expr LE expr
                     | expr GE expr
                     | expr GT expr
                     | expr EQ expr
                     | expr NE expr
        """
        p[0] = Operator(p[1], p[3], p[2])
    
    def p_expr_5(self, p):
        """ expr :     expr AND expr
                     | expr OR  expr
        """
        p[0] = Operator(p[1], p[3], p[2])

    def p_unary_expr_1(self, p):
        """ unary_expr   :     FUNCNAME 
                             | FUNCNAME LPAREN param_list RPAREN
        """
        if (len(p) == 2): 
            p[0] = Functor(p[1], None)
        else:
            p[0] = Functor(p[1], p[3])
    
    def p_unary_expr_2(self, p):
        """ unary_expr   :   INTEGER
        """
        p[0] = Integer(p[1], int(p[1]))

    def p_unary_expr_3(self, p):
        """ unary_expr   :   FLOAT
        """
        p[0] = Float(p[1], float(p[1]))
    
    def p_unary_expr_4(self, p):
        """ unary_expr   :   STRING
        """
        p[0] = String('text', p[1])
        
    def p_unary_expr_5(self, p):
        """ unary_expr   :  LPAREN MINUS INTEGER RPAREN
        """
        p[0] = Integer(p[3], int(p[3]))
        p[0].negative()
    
    def p_unary_expr_6(self, p):
        """ unary_expr  :  LPAREN MINUS FLOAT RPAREN
        """
        p[0] = Float(p[3], float(p[3]))
        p[0].negative()
    
    def p_param_list(self, p):
        """ param_list  :    expr
                          |  param_list ',' expr
        """
        if (len(p) == 2):
            p[0] = [p[1]]
        else:
            p[1].append(p[3])
            p[0] = p[1]

        

    def p_empty(self, p):
        'empty : '
        pass

    def p_error(self, p):
        if p:
            print('Error in line %d, pos: %d before "%s"' % (p.lineno, p.lexpos, p.value))
        else:
            print('Error at the end of input.')




#=================TEST =========================================================
if __name__ == '__main__':
    import numpy as np
    
    text1 = """
        EMA(9) + 10; 
        """
    result1 = 19
    
    text2 = """
        EMA(9) - 10;
    """
    result2 = -1
    
    text3 = """
        EMA(9) - (EMA(EMA(8)) + 0.2);
    """
    result3 = 0.8


    text4 = """
        EMA(9) + EMA(10) - EMA(10) * EMA(4) / EMA(2);
    """
    result4 = -1
    
    text5 = """
        EMA(9) >= EMA(EMA(9));
    """
    result5 = True

    text6 = """
        EMA(9) > EMA(EMA(7));
    """
    result6 = True

    text7 = """
        EMA(9) < EMA(EMA(7));
    """
    result7 = False

    text8 = """
        EMA(9) <= EMA(EMA(9));
    """
    result8 = True

    text9 = """
        EMA(9) = EMA(EMA(9));
    """
    result9 = True

    text10 = """
        EMA(9) = EMA(EMA(7));
    """
    result10 = False

    text11 = """
      KKK := EMA(10) + EMA(20);
        EMA(KKK);
    """
    result11 = 30
    
    text12 = """
      K : EMA(10);
      EMA(20);
      KKK := EMA(10) + EMA(20);
      EMA(KKK);
    """
    result12 = 30
    
    text13="""
      J:(-0.3)*K-2*D;
    """
    result13 = -27
    
    text14="""
      J: EMA(-D);
    """
    result14 = -15
    
    #test_cases = (text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12)
    #results = (result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12)
    test_cases = (text14,)
    results = (result14,)
    
    parser = TDXParser()
    i = 1
    
    def TestEvaluate(ast, env):
        if ast:
            result = ast.evaluate(env)
            if result is None:
                print('Error in evaluation.')
            elif type(result) == bool or type(result) == int:
                if (result == results[i-1]):
                    print('case %d is [OK], result = %r' % (i, result))
                else:
                    print('case %d is [Failed], result = %r' % (i, result))
            elif type(result) == float:
                if np.isclose(result, results[i-1], rtol=1e-05, atol=1e-08, equal_nan=False):
                    print('case %d is [OK], result = %f' % (i, result))
                else:
                    print('case %d is [Failed], result = %f' % (i, result))
            
    
    class ExecEnv(object):
        def __init__(self):
            self.local = {
               'K': 10,
               'D': 15
            }
            self.figure = {}
            self.builtin = {
                'EMA': self.EMA
            }
            
        def resolveSymbol(self, symbol, num):
            if symbol in self.local:
                return self.local[symbol]
            if symbol in self.builtin:
                return self.builtin[symbol]
            return None
        
        def addSymbol(self, name, value):
            self.local[name] = value
        
        def addFigure(self, name, figure):
            self.figure[name] = figure
    
        def EMA(self, param):
            if (len(param) > 0):
                return param[0]
            return 0
        
        def dump(self):
            print('=== start dump ===')
            print('funcdef:')
            for k, v in self.local.items():
                print(k, v)
            print('figures:')
            for k, v in self.figure.items():
                print(k, v)
            print('=== end dump ===')

    for case in test_cases:
        ast = parser.parse(case)
        env = ExecEnv()
        TestEvaluate(ast, env)
        env.dump()
        #ShowTree(ast)
        i = i + 1
        
