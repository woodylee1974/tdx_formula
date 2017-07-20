from ply import lex

class TDXLex(object):
    def __init__(self):
        pass
    
    def build(self, **kwargs):
        self.lexer = lex.lex(object = self, debug = False)
    
    def token(self):
        self.last_token = self.lexer.token()
        return self.last_token
    
    def input(self, text):
        self.lexer.input(text)
        
    def findTok(self, token):
        last_cr = self.lexer.lexdata.rfind('\n', 0, token.lexpos)
        return token.lexpos - last_cr        

    literals = [';' , ',', ':']
        
    tokens = (
       'LPAREN',
       'RPAREN',
       'TIMES',
       'DIVIDE',
       'PLUS',
       'MINUS',
       'ASSIGN',
       'LT',
       'LE',
       'GE',
       'GT',
       'EQ',
       'NE',
       'AND',
       'OR',
       'FLOAT',
       'INTEGER',
       'STRING',
       'FUNCNAME'
    )
    
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_ASSIGN = r':='
    t_LT = r'<'
    t_LE = r'<='
    t_GE = r'>='
    t_GT = r'>'
    t_EQ = r'='
    t_NE = r'!='
    
    t_ignore  = ' \t'
    
    def t_AND(self, t):
        r'AND|\&\&'
        return t
    
    def t_OR(self, t):
        r'OR|\|\|'
        return t

    def t_FUNCNAME(self, t):
        '[_a-zA-Z\u4e00-\u9fa5a][0-9a-zA-Z\u4e00-\u9fa5a_]*'
        return t

    
    def t_FLOAT(self, t):
        r'([0-9]*\.[0-9]+)|([0-9]+\.)'
        return t
    
    def t_INTEGER(self, t):
        r'(0)|([1-9][0-9]*)'
        return t
    
    def t_STRING(self, t):
        r'(\'.*\')|(\".*\")'
        return t
        
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

#=================================Test ===========================================================    
def _test_case(xlex, text):
    xlex.input(text)
    while True:
        tok = xlex.lexer.token()
        if not tok: 
            break
        print(tok)    

if __name__ == '__main__':
    mylex = TDXLex()
    mylex.build()
    _test_case(mylex, '-0.9099 AND -.4 COLOR00FFFFFF 得胜归来 >= < - + := :=> () LINESTICK LINETHICK9 LINETHIC9 COLORRED')
    _test_case(mylex, '3')
    
    
    
    