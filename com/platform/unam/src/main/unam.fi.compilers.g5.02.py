'''
Interpreter 
With Lexical, Syntactic, and Semantic Analysis using PLY

Compilers 
Team: 02
Group: 05
Students:
320206102
316255819
423031180
320117174
320340312

INSTALLATION:
pip install ply
'''

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import ply.lex as lex
import ply.yacc as yacc
from typing import Any, Dict, List, Optional, Set
from enum import Enum

# ============================================================================
# AST NODE DEFINITIONS (Abstract Syntax Tree)
# ============================================================================

class NodeType(Enum):
    """Enumeration for different AST Node Types."""
    PROGRAM = "PROGRAM"
    FUNCTION = "FUNCTION"
    DECLARATION = "DECLARATION"
    ARRAY_DECLARATION = "ARRAY_DECLARATION"
    ASSIGNMENT = "ASSIGNMENT"
    ARRAY_ACCESS = "ARRAY_ACCESS"
    ARRAY_ASSIGNMENT = "ARRAY_ASSIGNMENT"
    IF_STMT = "IF_STMT"
    WHILE_STMT = "WHILE_STMT"
    FOR_STMT = "FOR_STMT"
    SWITCH_STMT = "SWITCH_STMT"
    CASE_STMT = "CASE_STMT"
    DEFAULT_STMT = "DEFAULT_STMT"
    BREAK_STMT = "BREAK_STMT"
    CONTINUE_STMT = "CONTINUE_STMT"
    RETURN_STMT = "RETURN_STMT"
    EXPRESSION = "EXPRESSION"
    BINARY_OP = "BINARY_OP"
    UNARY_OP = "UNARY_OP"
    FUNCTION_CALL = "FUNCTION_CALL"
    NUMBER = "NUMBER"
    STRING = "STRING"
    CHAR = "CHAR"
    IDENTIFIER = "IDENTIFIER"
    BLOCK = "BLOCK"
    INCREMENT = "INCREMENT"
    DECREMENT = "DECREMENT"

class ASTNode:
    """Base class for all AST nodes."""
    def __init__(self, node_type, line=0):
        self.node_type = node_type
        self.line = line

class ProgramNode(ASTNode):
    """Represents the entire program (root node)."""
    def __init__(self, declarations, line=0):
        super().__init__(NodeType.PROGRAM, line)
        self.declarations = declarations

class FunctionNode(ASTNode):
    """Represents a function definition."""
    def __init__(self, return_type, name, params, body, line=0):
        super().__init__(NodeType.FUNCTION, line)
        self.return_type = return_type
        self.name = name
        self.params = params
        self.body = body

class DeclarationNode(ASTNode):
    """Represents a variable declaration."""
    def __init__(self, var_type, name, value=None, line=0):
        super().__init__(NodeType.DECLARATION, line)
        self.var_type = var_type
        self.name = name
        self.value = value

class ArrayDeclarationNode(ASTNode):
    """Represents an array declaration."""
    def __init__(self, var_type, name, size, initial_values=None, line=0):
        super().__init__(NodeType.ARRAY_DECLARATION, line)
        self.var_type = var_type
        self.name = name
        self.size = size
        self.initial_values = initial_values

class ArrayAccessNode(ASTNode):
    """Represents accessing an array element (e.g., arr[i])."""
    def __init__(self, name, index, line=0):
        super().__init__(NodeType.ARRAY_ACCESS, line)
        self.name = name
        self.index = index

class ArrayAssignmentNode(ASTNode):
    """Represents assigning a value to an array element."""
    def __init__(self, name, index, value, line=0):
        super().__init__(NodeType.ARRAY_ASSIGNMENT, line)
        self.name = name
        self.index = index
        self.value = value

class AssignmentNode(ASTNode):
    """Represents a standard variable assignment."""
    def __init__(self, name, value, line=0):
        super().__init__(NodeType.ASSIGNMENT, line)
        self.name = name
        self.value = value

class IfNode(ASTNode):
    """Represents an if-else control structure."""
    def __init__(self, condition, if_body, else_body=None, line=0):
        super().__init__(NodeType.IF_STMT, line)
        self.condition = condition
        self.if_body = if_body
        self.else_body = else_body

class WhileNode(ASTNode):
    """Represents a while loop."""
    def __init__(self, condition, body, line=0):
        super().__init__(NodeType.WHILE_STMT, line)
        self.condition = condition
        self.body = body

class ForNode(ASTNode):
    """Represents a for loop."""
    def __init__(self, init, condition, update, body, line=0):
        super().__init__(NodeType.FOR_STMT, line)
        self.init = init        # Can be a declaration (int i=0) or assignment
        self.condition = condition
        self.update = update
        self.body = body

class SwitchNode(ASTNode):
    """Represents a switch-case structure."""
    def __init__(self, expression, cases, line=0):
        super().__init__(NodeType.SWITCH_STMT, line)
        self.expression = expression
        self.cases = cases

class CaseNode(ASTNode):
    """Represents a single case within a switch."""
    def __init__(self, value, body, line=0):
        super().__init__(NodeType.CASE_STMT, line)
        self.value = value
        self.body = body

class DefaultNode(ASTNode):
    """Represents the default case within a switch."""
    def __init__(self, body, line=0):
        super().__init__(NodeType.DEFAULT_STMT, line)
        self.body = body

class BreakNode(ASTNode):
    """Represents a break statement."""
    def __init__(self, line=0):
        super().__init__(NodeType.BREAK_STMT, line)

class ContinueNode(ASTNode):
    """Represents a continue statement."""
    def __init__(self, line=0):
        super().__init__(NodeType.CONTINUE_STMT, line)

class ReturnNode(ASTNode):
    """Represents a return statement."""
    def __init__(self, value, line=0):
        super().__init__(NodeType.RETURN_STMT, line)
        self.value = value

class BinaryOpNode(ASTNode):
    """Represents a binary operation (e.g., +, -, *, /)."""
    def __init__(self, operator, left, right, line=0):
        super().__init__(NodeType.BINARY_OP, line)
        self.operator = operator
        self.left = left
        self.right = right

class UnaryOpNode(ASTNode):
    """Represents a unary operation (e.g., -x, !x)."""
    def __init__(self, operator, operand, line=0):
        super().__init__(NodeType.UNARY_OP, line)
        self.operator = operator
        self.operand = operand

class FunctionCallNode(ASTNode):
    """Represents a function call."""
    def __init__(self, name, arguments, line=0):
        super().__init__(NodeType.FUNCTION_CALL, line)
        self.name = name
        self.arguments = arguments

class NumberNode(ASTNode):
    """Represents a numeric literal (int or float)."""
    def __init__(self, value, line=0):
        super().__init__(NodeType.NUMBER, line)
        self.value = value

class StringNode(ASTNode):
    """Represents a string literal."""
    def __init__(self, value, line=0):
        super().__init__(NodeType.STRING, line)
        self.value = value

class CharNode(ASTNode):
    """Represents a character literal."""
    def __init__(self, value, line=0):
        super().__init__(NodeType.CHAR, line)
        self.value = value

class IdentifierNode(ASTNode):
    """Represents an identifier (variable name)."""
    def __init__(self, name, line=0):
        super().__init__(NodeType.IDENTIFIER, line)
        self.name = name

class BlockNode(ASTNode):
    """Represents a block of code (enclosed in braces)."""
    def __init__(self, statements, line=0):
        super().__init__(NodeType.BLOCK, line)
        self.statements = statements

class IncrementNode(ASTNode):
    """Represents an increment operation (e.g., i++)."""
    def __init__(self, name, line=0):
        super().__init__(NodeType.INCREMENT, line)
        self.name = name

class DecrementNode(ASTNode):
    """Represents a decrement operation (e.g., i--)."""
    def __init__(self, name, line=0):
        super().__init__(NodeType.DECREMENT, line)
        self.name = name

# ============================================================================
# LEXICAL ANALYZER using PLY
# ============================================================================

class CLexer:
    """The lexical analyzer (Lexer) for C language, utilizing PLY."""
    
    # Reserved words mapping
    reserved = {
        'int': 'INT', 'float': 'FLOAT', 'char': 'CHAR',
        'bool': 'BOOL', 'void': 'VOID', 'if': 'IF', 'else': 'ELSE',
        'while': 'WHILE', 'for': 'FOR', 'return': 'RETURN', 'true': 'TRUE',
        'false': 'FALSE', 'include': 'INCLUDE', 'switch': 'SWITCH',
        'case': 'CASE', 'default': 'DEFAULT', 'break': 'BREAK',
        'continue': 'CONTINUE', 'define': 'DEFINE', 'NULL': 'NULL',
        'stdbool': 'STDBOOL',
    }
    
    # List of token names
    tokens = [
        'ID',
        'NUMBER',
        'FLOAT_NUM',
        'STRING_LITERAL',
        'CHAR_LITERAL',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO',
        'ASSIGN',
        'EQ', 'NE', 'LT', 'GT', 'LE', 'GE',
        'AND', 'OR', 'NOT',
        'LPAREN', 'RPAREN',
        'LBRACE', 'RBRACE',
        'LBRACKET', 'RBRACKET',
        'SEMICOLON', 'COMMA',
        'AMPERSAND',
        'INCREMENT', 'DECREMENT',
        'HASH',
        'DOT',
        'COLON',
    ] + list(reserved.values())
    
    # Regular expressions for simple tokens
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_DIVIDE = r'/'
    t_MODULO = r'%'
    t_ASSIGN = r'='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_SEMICOLON = r';'
    t_COMMA = r','
    t_AMPERSAND = r'&'
    t_HASH = r'\#'
    t_DOT = r'\.'
    t_COLON = r':'
    
    # Ignore spaces and tabs
    t_ignore = ' \t'
    
    # Handling single-line comments
    def t_COMMENT_SINGLE(self, t):
        r'//.*'
        pass  # Ignore comments
    
    # Handling multi-line comments
    def t_COMMENT_MULTI(self, t):
        r'/\*(.|\n)*?\*/'
        t.lexer.lineno += t.value.count('\n')
        pass  # Ignore comments
    
    # Compound operators
    def t_INCREMENT(self, t):
        r'\+\+'
        return t
    
    def t_DECREMENT(self, t):
        r'--'
        return t
    
    def t_EQ(self, t):
        r'=='
        return t
    
    def t_NE(self, t):
        r'!='
        return t
    
    def t_LE(self, t):
        r'<='
        return t
    
    def t_GE(self, t):
        r'>='
        return t
    
    def t_LT(self, t):
        r'<'
        return t
    
    def t_GT(self, t):
        r'>'
        return t
    
    def t_AND(self, t):
        r'&&'
        return t
    
    def t_OR(self, t):
        r'\|\|'
        return t
    
    def t_NOT(self, t):
        r'!'
        return t
    
    # Numeric literals
    def t_FLOAT_NUM(self, t):
        r'\d+\.\d+'
        t.value = float(t.value)
        return t
    
    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t
    
    # String and Char literals
    def t_STRING_LITERAL(self, t):
        r'"([^"\\]|\\.)*"'
        t.value = t.value[1:-1]  # Remove quotes
        # Process escape sequences
        t.value = t.value.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
        return t
    
    def t_CHAR_LITERAL(self, t):
        r"'([^'\\]|\\.)'"
        t.value = t.value[1:-1]  # Remove single quotes
        return t
    
    # Identifiers and Keywords
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        t.type = self.reserved.get(t.value, 'ID')
        return t
    
    # Track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    # Error handling
    def t_error(self, t):
        error_msg = f"Lexical Error: Illegal character '{t.value[0]}' at line {t.lineno}"
        raise Exception(error_msg)
    
    def __init__(self):
        self.lexer = lex.lex(module=self)
        self.tokens_list = []
    
    def tokenize(self, text):
        """Tokenize input string and return a list of tokens."""
        # Reset lexer to avoid state accumulation
        self.lexer.lineno = 1
        self.lexer.input(text)
        self.tokens_list = []
        
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            self.tokens_list.append(tok)
        
        return self.tokens_list

# ============================================================================
# SYNTACTIC ANALYZER (PARSER) using PLY
# ============================================================================

class CParser:
    """The syntactic analyzer (Parser) for C language, building the AST."""
    
    tokens = CLexer.tokens
    
    # Operator precedence rules
    precedence = (
        ('left', 'OR'),
        ('left', 'AND'),
        ('left', 'EQ', 'NE'),
        ('left', 'LT', 'GT', 'LE', 'GE'),
        ('left', 'PLUS', 'MINUS'),
        ('left', 'TIMES', 'DIVIDE', 'MODULO'),
        ('right', 'UMINUS', 'UNOT'),
    )
    
    def p_program(self, p):
        '''program : external_declarations'''
        p[0] = ProgramNode(p[1])
    
    def p_external_declarations(self, p):
        '''external_declarations : external_declaration
                                 | external_declarations external_declaration'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]
    
    def p_external_declaration(self, p):
        '''external_declaration : function_definition
                                | declaration
                                | preprocessor_directive'''
        p[0] = p[1]
    
    def p_preprocessor_directive(self, p):
        '''preprocessor_directive : HASH INCLUDE LT ID DOT ID GT
                                  | HASH INCLUDE LT STDBOOL DOT ID GT
                                  | HASH INCLUDE STRING_LITERAL'''
        p[0] = None  # Ignore preprocessor directives
    
    def p_function_definition(self, p):
        '''function_definition : type ID LPAREN parameter_list RPAREN compound_statement
                               | type ID LPAREN RPAREN compound_statement'''
        params = p[4] if len(p) == 7 else []
        body = p[6] if len(p) == 7 else p[5]
        p[0] = FunctionNode(p[1], p[2], params, body, p.lineno(2))
    
    def p_parameter_list(self, p):
        '''parameter_list : parameter
                          | parameter_list COMMA parameter'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]
    
    def p_parameter(self, p):
        '''parameter : type ID'''
        p[0] = (p[1], p[2])
    
    def p_type(self, p):
        '''type : INT
                | FLOAT
                | CHAR
                | BOOL
                | VOID'''
        p[0] = p[1]
    
    def p_compound_statement(self, p):
        '''compound_statement : LBRACE statement_list RBRACE
                              | LBRACE RBRACE'''
        statements = p[2] if len(p) == 4 else []
        p[0] = BlockNode(statements)
    
    def p_statement_list(self, p):
        '''statement_list : statement
                          | statement_list statement'''
        if len(p) == 2:
            p[0] = [p[1]] if p[1] is not None else []
        else:
            if p[2] is not None:
                p[0] = p[1] + [p[2]]
            else:
                p[0] = p[1]
    
    def p_statement(self, p):
        '''statement : declaration
                     | assignment
                     | if_statement
                     | while_statement
                     | for_statement
                     | switch_statement
                     | break_statement
                     | continue_statement
                     | return_statement
                     | expression_statement
                     | compound_statement
                     | increment_statement
                     | decrement_statement'''
        p[0] = p[1]
    
    def p_declaration(self, p):
        '''declaration : type declarator_list SEMICOLON
                       | type ID LBRACKET NUMBER RBRACKET SEMICOLON
                       | type ID LBRACKET NUMBER RBRACKET ASSIGN LBRACE argument_list RBRACE SEMICOLON
                       | type ID LBRACKET RBRACKET ASSIGN LBRACE argument_list RBRACE SEMICOLON
                       | type ID LBRACKET RBRACKET ASSIGN STRING_LITERAL SEMICOLON'''
        
        # Array with specific size
        if len(p) == 7 and p[3] == '[':
            p[0] = ArrayDeclarationNode(p[1], p[2], p[4], None, p.lineno(2))
        # Array with size and brace initialization
        elif len(p) == 11 and p[3] == '[' and p[4] != ']':
            p[0] = ArrayDeclarationNode(p[1], p[2], p[4], p[8], p.lineno(2))
        # Array without size and brace initialization
        elif len(p) == 10 and p[3] == '[':
            size = len(p[7]) if p[7] else 0
            p[0] = ArrayDeclarationNode(p[1], p[2], size, p[7], p.lineno(2))
        # Array char[] = "string" - convert string to char list
        elif len(p) == 8 and p[3] == '[' and isinstance(p[6], str):
            # Convert string to list of chars
            chars = list(p[6])
            size = len(chars)
            char_nodes = [StringNode(c, p.lineno(2)) for c in chars]
            p[0] = ArrayDeclarationNode(p[1], p[2], size, char_nodes, p.lineno(2))
        # Normal variable declaration
        else:
            var_type = p[1]
            declarations = []
            for decl_info in p[2]:
                if isinstance(decl_info, tuple):
                    name, value = decl_info
                    declarations.append(DeclarationNode(var_type, name, value, p.lineno(1)))
                else:
                    declarations.append(DeclarationNode(var_type, decl_info, None, p.lineno(1)))
            
            if len(declarations) == 1:
                p[0] = declarations[0]
            else:
                p[0] = declarations
    
    def p_declarator_list(self, p):
        '''declarator_list : declarator
                           | declarator_list COMMA declarator'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]

    def p_for_init(self, p):
        '''for_init : type declarator_list SEMICOLON
                    | assignment_no_semi SEMICOLON
                    | empty SEMICOLON'''
        if p[1] is None or p[1] == 'empty': # empty SEMICOLON
            p[0] = None
        elif p[1] in self.reserved: # Declaration: type declarator_list SEMICOLON
            # Recreate declaration node without final semicolon
            var_type = p[1]
            declarations = []
            for decl_info in p[2]:
                if isinstance(decl_info, tuple):
                    name, value = decl_info
                    declarations.append(DeclarationNode(var_type, name, value, p.lineno(1)))
                else:
                    declarations.append(DeclarationNode(var_type, decl_info, None, p.lineno(1)))
            
            if len(declarations) == 1:
                p[0] = declarations[0]
            else:
                p[0] = declarations
        else: # Assignment: ID ASSIGN expression SEMICOLON
            p[0] = p[1] # Already an AssignmentNode (from p_assignment_no_semi)

    def p_assignment_no_semi(self, p):
        '''assignment_no_semi : ID ASSIGN expression'''
        p[0] = AssignmentNode(p[1], p[3], p.lineno(1))

    def p_for_update(self, p):
        '''for_update : ID INCREMENT
                      | ID DECREMENT
                      | ID ASSIGN expression
                      | empty'''
        if len(p) == 2: # empty
            p[0] = None
        elif len(p) == 3: # Increment or decrement
            if p[2] == '++':
                p[0] = IncrementNode(p[1], p.lineno(1))
            else:
                p[0] = DecrementNode(p[1], p.lineno(1))
        else: # Normal assignment
            p[0] = AssignmentNode(p[1], p[3], p.lineno(1))

    def p_empty(self, p):
        'empty :'
        p[0] = 'empty'
        pass
    
    def p_declarator(self, p):
        '''declarator : ID
                      | ID ASSIGN expression'''
        if len(p) == 2:
            p[0] = p[1]  # Just the name
        else:
            p[0] = (p[1], p[3])  # Name and initial value
    
    def p_assignment(self, p):
        '''assignment : ID ASSIGN expression SEMICOLON
                      | ID LBRACKET expression RBRACKET ASSIGN expression SEMICOLON'''
        if len(p) == 5:
            # Normal assignment
            p[0] = AssignmentNode(p[1], p[3], p.lineno(1))
        else:
            # Array element assignment
            p[0] = ArrayAssignmentNode(p[1], p[3], p[6], p.lineno(1))
    
    def p_increment_statement(self, p):
        '''increment_statement : ID INCREMENT SEMICOLON'''
        p[0] = IncrementNode(p[1], p.lineno(1))
    
    def p_decrement_statement(self, p):
        '''decrement_statement : ID DECREMENT SEMICOLON'''
        p[0] = DecrementNode(p[1], p.lineno(1))
    
    def p_if_statement(self, p):
        '''if_statement : IF LPAREN expression RPAREN compound_statement
                        | IF LPAREN expression RPAREN compound_statement ELSE compound_statement
                        | IF LPAREN expression RPAREN compound_statement ELSE if_statement'''
        else_body = p[7] if len(p) == 8 else None
        p[0] = IfNode(p[3], p[5], else_body, p.lineno(1))
    
    def p_while_statement(self, p):
        '''while_statement : WHILE LPAREN expression RPAREN compound_statement'''
        p[0] = WhileNode(p[3], p[5], p.lineno(1))
    
    def p_switch_statement(self, p):
        '''switch_statement : SWITCH LPAREN expression RPAREN LBRACE case_list RBRACE'''
        p[0] = SwitchNode(p[3], p[6], p.lineno(1))
    
    def p_case_list(self, p):
        '''case_list : case_statement
                     | case_list case_statement'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]
    
    def p_case_statement(self, p):
        '''case_statement : CASE expression COLON statement_list
                          | CASE expression COLON
                          | DEFAULT COLON statement_list
                          | DEFAULT COLON'''
        if p[1] == 'case':
            statements = p[4] if len(p) == 5 else []
            p[0] = CaseNode(p[2], statements, p.lineno(1))
        else:  # default
            statements = p[3] if len(p) == 4 else []
            p[0] = DefaultNode(statements, p.lineno(1))
    
    def p_break_statement(self, p):
        '''break_statement : BREAK SEMICOLON'''
        p[0] = BreakNode(p.lineno(1))
    
    def p_continue_statement(self, p):
        '''continue_statement : CONTINUE SEMICOLON'''
        p[0] = ContinueNode(p.lineno(1))
    
    def p_for_statement(self, p):
        '''for_statement : FOR LPAREN for_init expression SEMICOLON for_update RPAREN compound_statement'''
        # p[3] is init, p[4] is condition, p[6] is update
        p[0] = ForNode(p[3], p[4], p[6], p[8], p.lineno(1))

    def p_for_init(self, p):
        '''for_init : declaration
                    | assignment
                    | SEMICOLON'''
        # declaration and assignment already include SEMICOLON in their base rules
        if p[1] == ';':
            p[0] = None
        else:
            p[0] = p[1]

    def p_for_update(self, p):
        '''for_update : ID INCREMENT
                      | ID DECREMENT
                      | ID ASSIGN expression
                      | '''
        # Update in a for loop does NOT have a semicolon (e.g. i++)
        if len(p) == 1: # Empty
            p[0] = None
        elif len(p) == 3: # i++ or i--
            if p[2] == '++':
                p[0] = IncrementNode(p[1], p.lineno(1))
            else:
                p[0] = DecrementNode(p[1], p.lineno(1))
        else: # i = i + 1
            p[0] = AssignmentNode(p[1], p[3], p.lineno(1))
    
    def p_return_statement(self, p):
        '''return_statement : RETURN expression SEMICOLON
                            | RETURN SEMICOLON'''
        value = p[2] if len(p) == 4 else None
        p[0] = ReturnNode(value, p.lineno(1))
    
    def p_expression_statement(self, p):
        '''expression_statement : expression SEMICOLON'''
        p[0] = p[1]
    
    def p_expression_binop(self, p):
        '''expression : expression PLUS expression
                      | expression MINUS expression
                      | expression TIMES expression
                      | expression DIVIDE expression
                      | expression MODULO expression
                      | expression EQ expression
                      | expression NE expression
                      | expression LT expression
                      | expression GT expression
                      | expression LE expression
                      | expression GE expression
                      | expression AND expression
                      | expression OR expression'''
        p[0] = BinaryOpNode(p[2], p[1], p[3], p.lineno(2))
    
    def p_expression_unary(self, p):
        '''expression : MINUS expression %prec UMINUS
                      | NOT expression %prec UNOT
                      | AMPERSAND ID'''
        if p[1] == '&':
            # Variable reference (for printf with &i)
            p[0] = UnaryOpNode('&', IdentifierNode(p[2]), p.lineno(1))
        else:
            p[0] = UnaryOpNode(p[1], p[2], p.lineno(1))
    
    def p_expression_group(self, p):
        '''expression : LPAREN expression RPAREN'''
        p[0] = p[2]
    
    def p_expression_number(self, p):
        '''expression : NUMBER
                      | FLOAT_NUM'''
        p[0] = NumberNode(p[1], p.lineno(1))
    
    def p_expression_string(self, p):
        '''expression : STRING_LITERAL'''
        p[0] = StringNode(p[1], p.lineno(1))
    
    def p_expression_char(self, p):
        '''expression : CHAR_LITERAL'''
        p[0] = CharNode(p[1], p.lineno(1))
    
    def p_expression_bool(self, p):
        '''expression : TRUE
                      | FALSE'''
        p[0] = NumberNode(True if p[1] == 'true' else False, p.lineno(1))
    
    def p_expression_null(self, p):
        '''expression : NULL'''
        p[0] = NumberNode(None, p.lineno(1))
    
    def p_expression_id(self, p):
        '''expression : ID
                      | ID LBRACKET expression RBRACKET'''
        if len(p) == 2:
            p[0] = IdentifierNode(p[1], p.lineno(1))
        else:
            # Array element access (e.g., A[i])
            p[0] = ArrayAccessNode(p[1], p[3], p.lineno(1))
    
    def p_expression_function_call(self, p):
        '''expression : ID LPAREN argument_list RPAREN
                      | ID LPAREN RPAREN'''
        args = p[3] if len(p) == 5 else []
        p[0] = FunctionCallNode(p[1], args, p.lineno(1))
    
    def p_argument_list(self, p):
        '''argument_list : expression
                         | argument_list COMMA expression'''
        if len(p) == 2:
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[3]]
    
    def p_error(self, p):
        if p:
            # Get the real code line
            raise Exception(f"Syntax error at '{p.value}' (line {p.lineno})")
        else:
            raise Exception("Syntax error: unexpected end of file")
    
    def __init__(self):
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)
        self.current_lexer = None
    
    def parse(self, text, lexer):
        """Analyze input and return AST."""
        # Reset parser for each analysis
        self.current_lexer = lexer
        # Reset lexer line counter
        lexer.lexer.lineno = 1
        return self.parser.parse(text, lexer=lexer.lexer, tracking=True)

# ============================================================================
# INTERPRETER (Executes AST)
# ============================================================================

class CInterpreter:
    """The executor that traverses the AST and runs the logic."""
    
    def __init__(self):
        # Memory storage for variables (Symbol Table for execution)
        self.variables = {}
        # Registry for defined functions (built-ins + user defined)
        self.functions = {
            'printf': self._builtin_printf,
            'scanf': self._builtin_scanf,
            'print': self._builtin_print,
            # Lambda functions for string manipulation helper
            'strlen': lambda s: len(str(s)),
            'strcmp': lambda s1, s2: 0 if str(s1) == str(s2) else (1 if str(s1) > str(s2) else -1),
            'strcat': lambda s1, s2: str(s1) + str(s2),
        }
        # Buffer to capture standard output for the GUI
        self.output = []
        # Register for function return values
        self.return_value = None
        # Control flow flags
        self.break_flag = False
        self.continue_flag = False
    
    def _builtin_printf(self, *args):
        """Simplified printf function implementation simulating C behavior."""
        if args:
            msg = str(args[0])
            arg_index = 1
            
            # Replace format specifiers in order using Regex
            import re
            # Finds %d, %f, %s, %i, %c
            for match in re.finditer(r'%[dfsic]', msg):
                if arg_index < len(args):
                    value = args[arg_index]
                    specifier = match.group()
                    
                    # Format value based on specifier type
                    if specifier == '%d':
                        replacement = str(int(value)) if not isinstance(value, bool) else str(int(value))
                    elif specifier == '%f':
                        replacement = str(float(value))
                    elif specifier == '%s':
                        replacement = str(value)
                    elif specifier == '%c':
                        replacement = str(value)[0] if str(value) else ''
                    elif specifier == '%i':
                        replacement = str(int(value))
                    else:
                        replacement = str(value)
                    
                    # Replace only the first occurrence for this iteration
                    msg = msg.replace(specifier, replacement, 1)
                    arg_index += 1
            
            self.output.append(f"printf: {msg}")
        return 0
    
    def _builtin_scanf(self, *args):
        """Simplified scanf function stub (input not implemented in GUI)."""
        self.output.append(f"scanf called")
        return 0
    
    def _builtin_print(self, *args):
        """Standard print function for testing/debugging."""
        self.output.append(f"print: {' '.join(map(str, args))}")
        return 0
    
    def log(self, message):
        """Append message to the output log for GUI display."""
        self.output.append(message)
    
    def execute(self, ast):
        """Main method to execute the AST. Resets state before running."""
        self.output.clear()
        self.return_value = None
        self.break_flag = False
        self.continue_flag = False
        return self._execute_node(ast)
    
    def _execute_node(self, node):
        """
        Recursive method to execute individual nodes.
        Dispatches logic based on NodeType.
        """
        if node is None:
            return None
        
        # If the node is a list of declarations/statements, execute them sequentially
        if isinstance(node, list):
            result = None
            for n in node:
                result = self._execute_node(n)
            return result
        
        # --- Program Entry Point ---
        if node.node_type == NodeType.PROGRAM:
            result = None
            # First pass: Register all functions (hoisting concept)
            for decl in node.declarations:
                if decl is not None and decl.node_type == NodeType.FUNCTION:
                    if decl.name != 'main':
                        self.functions[decl.name] = decl
                        self.log(f"Function registered: {decl.return_type} {decl.name}(...)")
            
            # Second pass: Execute 'main' function if it exists
            # We look for the function named 'main' in the declarations
            if 'main' in self.functions or any(d and d.node_type == NodeType.FUNCTION and d.name == 'main' for d in node.declarations if d):
                main_func = None
                for decl in node.declarations:
                    if decl is not None and decl.node_type == NodeType.FUNCTION and decl.name == 'main':
                        main_func = decl
                        break
                
                if main_func:
                    self.log(f"\nExecuting main() function")
                    # Start execution from main's body
                    result = self._execute_node(main_func.body)
            
            return result
        
        # --- Function Definition ---
        elif node.node_type == NodeType.FUNCTION:
            # Register function in the lookup table (do not execute body yet)
            self.functions[node.name] = node
            self.log(f"Function registered: {node.return_type} {node.name}(...)")
            return None
        
        # --- Code Block execution ---
        elif node.node_type == NodeType.BLOCK:
            result = None
            for stmt in node.statements:
                if stmt is not None:
                    result = self._execute_node(stmt)
                    # Flow control checks:
                    # If a return statement was hit, bubble it up
                    if self.return_value is not None:
                        return result
                    # Check for break/continue to stop block execution
                    if self.break_flag or self.continue_flag:
                        return result
            return result
        
        # --- Variable Declaration ---
        elif node.node_type == NodeType.DECLARATION:
            if node.value:
                # Calculate initialization value
                value = self._evaluate_expression(node.value)
            else:
                # Assign default values based on C-like defaults
                type_defaults = {
                    'int': 0, 'float': 0.0, 'char': '', 
                    'string': '', 'bool': False
                }
                value = type_defaults.get(node.var_type, 0)
            
            # Store in current memory scope
            self.variables[node.name] = value
            self.log(f"Declaration: {node.var_type} {node.name} = {value}")
            return None
        
        # --- Array Declaration ---
        elif node.node_type == NodeType.ARRAY_DECLARATION:
            # Determine size dynamically or statically
            size = node.size if isinstance(node.size, int) else self._evaluate_expression(node.size)
            
            if node.initial_values:
                # Array with explicit initialization list
                array = [self._evaluate_expression(val) for val in node.initial_values]
                
                # Special handling for char[] initialized with chars
                if node.var_type == 'char' and all(isinstance(v, str) and len(v) <= 1 for v in array):
                    string_value = ''.join(array)
                    self.variables[node.name] = string_value
                    self.log(f"Array declaration: char {node.name}[] = \"{string_value}\"")
                    return None
                
                # Padding: If initialization list is shorter than size, fill with zeros
                while len(array) < size:
                    array.append(0)
                self.variables[node.name] = array
                self.log(f"Array declaration: {node.var_type} {node.name}[{size}] = {array}")
            else:
                # Uninitialized array, fill entire size with default values
                type_defaults = {'int': 0, 'float': 0.0, 'char': '', 'bool': False}
                default = type_defaults.get(node.var_type, 0)
                array = [default] * size
                self.variables[node.name] = array
                self.log(f"Array declaration: {node.var_type} {node.name}[{size}] = {array}")
            
            return None
        
        # --- Array Access (Read) ---
        elif node.node_type == NodeType.ARRAY_ACCESS:
            # Ensure array exists
            if node.name not in self.variables:
                raise Exception(f"Array '{node.name}' not defined")
            
            value = self.variables[node.name]
            
            # If it's a string (char[]), access character by index
            if isinstance(value, str):
                index = int(self._evaluate_expression(node.index))
                if index < 0 or index >= len(value):
                    raise Exception(f"Index {index} out of range for array '{node.name}'")
                return value[index]
            
            # If it's a standard list/array
            if not isinstance(value, list):
                raise Exception(f"'{node.name}' is not an array")
            
            index = int(self._evaluate_expression(node.index))
            if index < 0 or index >= len(value):
                raise Exception(f"Index {index} out of range for array '{node.name}'")
            
            return value[index]
        
        # --- Array Assignment (Write) ---
        elif node.node_type == NodeType.ARRAY_ASSIGNMENT:
            if node.name not in self.variables:
                raise Exception(f"Array '{node.name}' not defined")
            
            value_data = self.variables[node.name]
            index = int(self._evaluate_expression(node.index))
            new_value = self._evaluate_expression(node.value)
            
            # String handling: strings in Python are immutable, so we rebuild it
            if isinstance(value_data, str):
                if index < 0 or index >= len(value_data):
                    raise Exception(f"Index {index} out of range for array '{node.name}'")
                char_list = list(value_data)
                char_list[index] = str(new_value)[0] if str(new_value) else ''
                self.variables[node.name] = ''.join(char_list)
                self.log(f"Array assignment: {node.name}[{index}] = {new_value}")
                return new_value
            
            # Standard list handling
            if not isinstance(value_data, list):
                raise Exception(f"'{node.name}' is not an array")
            
            if index < 0 or index >= len(value_data):
                raise Exception(f"Index {index} out of range for array '{node.name}'")
            
            value_data[index] = new_value
            self.log(f"Array assignment: {node.name}[{index}] = {new_value}")
            return new_value
        
        # --- Variable Assignment ---
        elif node.node_type == NodeType.ASSIGNMENT:
            value = self._evaluate_expression(node.value)
            # Update variable in memory
            self.variables[node.name] = value
            self.log(f"Assignment: {node.name} = {value}")
            return value
        
        # --- Increment/Decrement ---
        elif node.node_type == NodeType.INCREMENT:
            if node.name in self.variables:
                self.variables[node.name] += 1
                self.log(f"Increment: {node.name}++ = {self.variables[node.name]}")
            return None
        
        elif node.node_type == NodeType.DECREMENT:
            if node.name in self.variables:
                self.variables[node.name] -= 1
                self.log(f"Decrement: {node.name}-- = {self.variables[node.name]}")
            return None
        
        # --- Control Flow: IF ---
        elif node.node_type == NodeType.IF_STMT:
            condition = self._evaluate_expression(node.condition)
            if condition:
                self.log(f"IF: True condition, executing if block")
                return self._execute_node(node.if_body)
            elif node.else_body:
                self.log(f"IF: False condition, executing else block")
                return self._execute_node(node.else_body)
            return None
        
        # --- Control Flow: WHILE ---
        elif node.node_type == NodeType.WHILE_STMT:
            iterations = 0
            while self._evaluate_expression(node.condition):
                iterations += 1
                if iterations > 1000:  # Safety mechanism against infinite loops
                    self.log(f"WHILE: Stopped after 1000 iterations")
                    break
                
                self._execute_node(node.body)
                
                # Check control flags
                if self.return_value is not None:
                    break
                if self.break_flag:
                    self.break_flag = False # Reset flag and exit loop
                    break
                if self.continue_flag:
                    self.continue_flag = False # Reset flag and continue next iteration
                    continue
            self.log(f"WHILE: Executed {iterations} iterations")
            return None
        
        # --- Control Flow: FOR ---
        elif node.node_type == NodeType.FOR_STMT:
            # 1. Execute initialization (e.g., int i = 0;)
            # Note: This creates the variable in the current scope.
            self._execute_node(node.init)
            
            iterations = 0
            # 2. Loop Execution
            while True:
                # Evaluate condition (if None, it's an infinite loop 'for(;;)')
                cond = True
                if node.condition is not None:
                    cond = self._evaluate_expression(node.condition)
                    # In C, 0 is false.
                    if cond == 0 or cond is False:
                        break
                
                iterations += 1
                if iterations > 5000: # Protection against infinite loops
                    self.log("FOR: Stopped for safety (5000 iters)")
                    break

                # 3. Execute body
                self._execute_node(node.body)

                # Handle control flow breaks
                if self.return_value is not None: break
                if self.break_flag:
                    self.break_flag = False
                    break
                if self.continue_flag:
                    self.continue_flag = False
                
                # 4. Execute update (e.g., i++)
                self._execute_node(node.update)

            return None
        
        # --- Control Flow: SWITCH ---
        elif node.node_type == NodeType.SWITCH_STMT:
            switch_value = self._evaluate_expression(node.expression)
            self.log(f"SWITCH: Evaluating expression = {switch_value}")
            
            matched = False
            fall_through = False
            
            for case in node.cases:
                # Logic for Fall-through execution (executing subsequent cases until break)
                if fall_through or matched:
                    if case.node_type == NodeType.CASE_STMT:
                        self.log(f"  CASE (fall-through): {self._evaluate_expression(case.value)}")
                    else:
                        self.log(f"  DEFAULT (fall-through)")
                    
                    for stmt in case.body:
                        self._execute_node(stmt)
                        if self.break_flag:
                            self.break_flag = False
                            self.log(f"  BREAK found")
                            return None
                    fall_through = True # Continue falling through
                    continue
                
                # Check for initial match
                if case.node_type == NodeType.CASE_STMT:
                    case_value = self._evaluate_expression(case.value)
                    if case_value == switch_value:
                        matched = True
                        self.log(f"  CASE match: {case_value}")
                        for stmt in case.body:
                            self._execute_node(stmt)
                            if self.break_flag:
                                self.break_flag = False
                                self.log(f"  BREAK found")
                                return None
                        fall_through = True
                # Check default case
                elif case.node_type == NodeType.DEFAULT_STMT:
                    if not matched:
                        matched = True
                        self.log(f"  DEFAULT executed")
                        for stmt in case.body:
                            self._execute_node(stmt)
                            if self.break_flag:
                                self.break_flag = False
                                self.log(f"  BREAK found")
                                return None
            return None
        
        # --- Control Flow Keywords ---
        elif node.node_type == NodeType.BREAK_STMT:
            self.break_flag = True
            return None
        
        elif node.node_type == NodeType.CONTINUE_STMT:
            self.continue_flag = True
            return None
        
        elif node.node_type == NodeType.RETURN_STMT:
            if node.value:
                self.return_value = self._evaluate_expression(node.value)
                self.log(f"Return: {self.return_value}")
            else:
                self.return_value = 0
                self.log(f"Return: 0")
            return self.return_value
        
        # --- Function Call Wrapper ---
        elif node.node_type == NodeType.FUNCTION_CALL:
            return self._evaluate_expression(node)
        
        else:
            return self._evaluate_expression(node)
    
    def _evaluate_expression(self, node):
        """Evaluate an expression node (math, logic, values) and return result."""
        if node is None:
            return None
        
        # --- Literals ---
        if node.node_type == NodeType.NUMBER:
            return node.value
        
        elif node.node_type == NodeType.STRING:
            return str(node.value)
        
        elif node.node_type == NodeType.CHAR:
            return str(node.value)
        
        # --- Array Element Value ---
        elif node.node_type == NodeType.ARRAY_ACCESS:
            # Logic similar to execution but returns the value directly
            name = node.name
            index = self._evaluate_expression(node.index)
            
            if name not in self.variables:
                raise Exception(f"Array '{name}' not declared")
            
            value_data = self.variables[name]
            
            if isinstance(value_data, str):
                if index < 0 or index >= len(value_data):
                    raise Exception(f"Index {index} out of range for string '{name}'")
                return value_data[index] 
            
            elif isinstance(value_data, list):
                if index < 0 or index >= len(value_data):
                    raise Exception(f"Index {index} out of range for array '{name}'")
                return value_data[index]
            
            else:
                raise Exception(f"'{name}' is not an array")
        
        # --- Variable Lookup ---
        elif node.node_type == NodeType.IDENTIFIER:
            if node.name in self.variables:
                value = self.variables[node.name]
                return value
            # Reserved boolean keywords
            elif node.name == 'true':
                return True
            elif node.name == 'false':
                return False
            elif node.name == 'NULL':
                return None
            else:
                raise Exception(f"Variable '{node.name}' not initialized")
        
        # --- Binary Operations ---
        elif node.node_type == NodeType.BINARY_OP:
            left = self._evaluate_expression(node.left)
            right = self._evaluate_expression(node.right)
            
            op = node.operator
            # Arithmetic
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                if right == 0:
                    raise Exception("Division by zero at runtime")
                return left / right
            elif op == '%':
                return left % right
            # Comparison (Returns 1 for True, 0 for False)
            elif op == '==':
                return 1 if left == right else 0
            elif op == '!=':
                return 1 if left != right else 0
            elif op == '<':
                return 1 if left < right else 0
            elif op == '>':
                return 1 if left > right else 0
            elif op == '<=':
                return 1 if left <= right else 0
            elif op == '>=':
                return 1 if left >= right else 0
            # Logical
            elif op == '&&':
                return 1 if left and right else 0
            elif op == '||':
                return 1 if left or right else 0
        
        # --- Unary Operations ---
        elif node.node_type == NodeType.UNARY_OP:
            operand = self._evaluate_expression(node.operand)
            ops = {
                '-': lambda: -operand,
                '+': lambda: operand,
                '!': lambda: not operand,
                '&': lambda: operand  # Reference (mock)
            }
            return ops[node.operator]()
        
        # --- Function Calls ---
        elif node.node_type == NodeType.FUNCTION_CALL:
            # 1. Evaluate arguments before calling
            args = [self._evaluate_expression(arg) for arg in node.arguments]
            
            if node.name in self.functions:
                func = self.functions[node.name]
                if callable(func):
                    # Direct call for Built-in functions (printf, etc.)
                    return func(*args)
                else:
                    # === User-defined functions Execution ===
                    
                    # 1. Context Switch: Save current state (variables and return)
                    # This simulates pushing the stack frame
                    previous_variables = self.variables.copy()
                    previous_return = self.return_value
                    
                    # 2. Create new local context (scope)
                    local_variables = {}
                    
                    # Validate argument count
                    if len(args) != len(func.params):
                        raise Exception(f"Function '{node.name}' expects {len(func.params)} arguments, but received {len(args)}")
                    
                    # 3. Bind arguments to parameters in local scope
                    for (param_type, param_name), arg_value in zip(func.params, args):
                        local_variables[param_name] = arg_value
                    
                    # Switch global pointer to local context
                    self.variables = local_variables
                    self.return_value = None  # Reset return for the new function
                    
                    # 4. Execute function body
                    result = 0 # Default return
                    try:
                        self._execute_node(func.body)
                        result = self.return_value # Capture return value
                    except Exception as e:
                        # If error occurs, restore context before crashing
                        self.variables = previous_variables
                        self.return_value = previous_return
                        raise e
                    
                    # 5. Restore previous context (Popping the stack frame)
                    self.variables = previous_variables
                    self.return_value = previous_return
                    
                    return result if result is not None else 0
            else:
                raise Exception(f"Function '{node.name}' not defined")
        
        return None

# ============================================================================
# SEMANTIC ANALYZER (Check Types & Scopes)
# ============================================================================

class SemanticAnalyzer:
    """Semantic Analyzer for C, handling type checking and scope verification."""
    
    def __init__(self):
        # Symbol table for Global variables
        self.symbol_table = {}  
        # Table for Function signatures
        self.function_table = {
            'printf': {'return_type': 'void', 'params': []},  # Variable args support
            'scanf': {'return_type': 'int', 'params': []},
            'print': {'return_type': 'void', 'params': []},
        }
        # Symbol table for Local variables (current scope)
        self.current_scope = {}  
        self.errors = []
        self.warnings = []
        # Mapping C types to Python types for validation
        self.type_map = {
            'int': int,
            'float': float,
            'char': str,
            'string': str,
            'bool': bool,
        }
    
    def analyze(self, ast):
        """Perform semantic analysis on the AST. Returns True if no errors."""
        self.errors = []
        self.warnings = []
        try:
            if ast:
                self._analyze_node(ast)
            return len(self.errors) == 0
        except Exception as e:
            self.errors.append(str(e))
            return False
    
    def _analyze_node(self, node):
        """Recursive method to analyze semantic validity of nodes."""
        if node is None:
            return
        
        # If node is a list, iterate through it
        if isinstance(node, list):
            for n in node:
                self._analyze_node(n)
            return
        
        # --- Program Root ---
        if node.node_type == NodeType.PROGRAM:
            for decl in node.declarations:
                if decl is not None:
                    self._analyze_node(decl)
        
        # --- Function Definition ---
        elif node.node_type == NodeType.FUNCTION:
            # Register function signature
            self.function_table[node.name] = {
                'return_type': node.return_type,
                'params': node.params
            }
            # Scope Management: Enter new scope
            old_scope = self.current_scope
            self.current_scope = {}
            
            # Add parameters to local scope
            for param_type, param_name in node.params:
                self.current_scope[param_name] = param_type
            
            # Analyze body inside the new scope
            self._analyze_node(node.body)
            
            # Scope Management: Exit scope (restore old)
            self.current_scope = old_scope
        
        # --- Variable Declaration ---
        elif node.node_type == NodeType.DECLARATION:
            # Check for valid type
            if node.var_type not in self.type_map and node.var_type != 'void':
                self.errors.append(f"Semantic error line {node.line}: Type '{node.var_type}' not recognized")
            
            # Check for Shadowing/Redeclaration in current scope
            if node.name in self.current_scope:
                self.warnings.append(f"Warning line {node.line}: Variable '{node.name}' already declared")
            
            # Add to current scope
            self.current_scope[node.name] = node.var_type
            
            # Analyze initialization expression
            if node.value:
                self._analyze_node(node.value)
        
        # --- Array Declaration ---
        elif node.node_type == NodeType.ARRAY_DECLARATION:
            if node.var_type not in self.type_map and node.var_type != 'void':
                self.errors.append(f"Semantic error line {node.line}: Type '{node.var_type}' not recognized")
            if node.name in self.current_scope:
                self.warnings.append(f"Warning line {node.line}: Array '{node.name}' already declared")
            
            # Register array type notation
            self.current_scope[node.name] = f"{node.var_type}[]"
            
            if node.initial_values:
                for val in node.initial_values:
                    self._analyze_node(val)
        
        # --- Assignment ---
        elif node.node_type == NodeType.ASSIGNMENT:
            # Check if variable exists in any scope
            if node.name not in self.current_scope and node.name not in self.symbol_table:
                self.errors.append(f"Semantic error line {node.line}: Variable '{node.name}' not declared")
            self._analyze_node(node.value)
        
        # --- Array Usage ---
        elif node.node_type == NodeType.ARRAY_ACCESS:
            if node.name not in self.current_scope and node.name not in self.symbol_table:
                self.errors.append(f"Semantic error line {node.line}: Array '{node.name}' not declared")
            self._analyze_node(node.index)
        
        elif node.node_type == NodeType.ARRAY_ASSIGNMENT:
            if node.name not in self.current_scope and node.name not in self.symbol_table:
                self.errors.append(f"Semantic error line {node.line}: Array '{node.name}' not declared")
            self._analyze_node(node.index)
            self._analyze_node(node.value)
        
        # --- Control Structures ---
        elif node.node_type == NodeType.IF_STMT:
            self._analyze_node(node.condition)
            self._analyze_node(node.if_body)
            if node.else_body:
                self._analyze_node(node.else_body)
        
        elif node.node_type == NodeType.WHILE_STMT:
            self._analyze_node(node.condition)
            self._analyze_node(node.body)

        elif node.node_type == NodeType.FOR_STMT:
            # Validate all parts of the loop
            self._analyze_node(node.init)
            self._analyze_node(node.condition)
            self._analyze_node(node.update)
            self._analyze_node(node.body)
        
        elif node.node_type == NodeType.SWITCH_STMT:
            self._analyze_node(node.expression)
            for case in node.cases:
                self._analyze_node(case)
        
        elif node.node_type == NodeType.CASE_STMT:
            if node.value:
                self._analyze_node(node.value)
            for stmt in node.body:
                self._analyze_node(stmt)
        
        elif node.node_type == NodeType.DEFAULT_STMT:
            for stmt in node.body:
                self._analyze_node(stmt)
        
        elif node.node_type == NodeType.BREAK_STMT:
            pass  # Logic check for break placement could be added here
        
        elif node.node_type == NodeType.CONTINUE_STMT:
            pass  # Logic check for continue placement could be added here
        
        elif node.node_type == NodeType.RETURN_STMT:
            if node.value:
                self._analyze_node(node.value)
        
        # --- Binary Operations ---
        elif node.node_type == NodeType.BINARY_OP:
            self._analyze_node(node.left)
            self._analyze_node(node.right)
            # Static check for division by zero (only works for literals)
            if node.operator == '/' and isinstance(node.right, NumberNode):
                if node.right.value == 0:
                    self.errors.append(
                        f"Semantic error line {node.line}: "
                        f"Division by zero"
                    )
        
        # --- Unary Operations ---
        elif node.node_type == NodeType.UNARY_OP:
            self._analyze_node(node.operand)
        
        # --- Function Call ---
        elif node.node_type == NodeType.FUNCTION_CALL:
            # Verify that function exists in the table
            if node.name not in self.function_table:
                self.errors.append(
                    f"Semantic error line {node.line}: "
                    f"Function '{node.name}' not declared"
                )
            # Analyze arguments
            for arg in node.arguments:
                self._analyze_node(arg)
        
        # --- Variable Usage (Identifier) ---
        elif node.node_type == NodeType.IDENTIFIER:
            # Verify that variable exists in current or global scope
            if node.name not in self.current_scope and node.name not in self.symbol_table:
                self.errors.append(
                    f"Semantic error line {node.line}: "
                    f"Variable '{node.name}' not declared"
                )
        
        # --- Block ---
        elif node.node_type == NodeType.BLOCK:
            for stmt in node.statements:
                if stmt is not None:
                    self._analyze_node(stmt)
        
        # --- Increment/Decrement ---
        elif node.node_type == NodeType.INCREMENT:
            if node.name not in self.current_scope and node.name not in self.symbol_table:
                self.errors.append(
                    f"Semantic error line {node.line}: "
                    f"Variable '{node.name}' not declared"
                )
        
        elif node.node_type == NodeType.DECREMENT:
            if node.name not in self.current_scope and node.name not in self.symbol_table:
                self.errors.append(
                    f"Semantic error line {node.line}: "
                    f"Variable '{node.name}' not declared"
                )

# ============================================================================
# INTERPRETER COORDINATOR
# ============================================================================

class CInterpreterEngine:
    """Coordinates all interpretation phases"""
    
    def __init__(self):
        self.lexer = None
        self.parser = None
        self.semantic_analyzer = None
        self.interpreter = None
        self.output = []
        self.token_stats = {}
        self.reset()
    
    def reset(self):
        """Reset all components"""
        self.lexer = CLexer()
        self.parser = CParser()
        self.semantic_analyzer = SemanticAnalyzer()
        self.interpreter = CInterpreter()
        self.output = []
        self.token_stats = {}
    
    def interpret(self, text):
        """Execute all interpretation phases"""
        # Reset everything before each interpretation
        self.reset()
        
        try:
            # Phase 1: Lexical Analysis
            self.output.append("=== PHASE 1: LEXICAL ANALYSIS ===")
            tokens = self.lexer.tokenize(text)
            
            # Token statistics
            token_details = {}
            for tok in tokens:
                if tok.type not in token_details:
                    token_details[tok.type] = {'count': 0, 'values': []}
                token_details[tok.type]['count'] += 1
                token_details[tok.type]['values'].append(str(tok.value))
            
            self.token_stats = token_details
            
            self.output.append(f"  Total tokens: {len(tokens)}")
            self.output.append("  Token types found:")
            for tok_type, info in sorted(token_details.items()):
                self.output.append(f"    - {tok_type}: {info['count']} times")
            self.output.append("")
            
            # Phase 2: Syntactic Analysis
            self.output.append("=== PHASE 2: SYNTACTIC ANALYSIS ===")
            ast = self.parser.parse(text, self.lexer)
            
            if ast:
                self.output.append(f"  ✓ Syntax tree built successfully")
                if hasattr(ast, 'declarations'):
                    func_count = sum(1 for d in ast.declarations 
                                   if d and d.node_type == NodeType.FUNCTION)
                    self.output.append(f"  Functions found: {func_count}")
            else:
                self.output.append(f"  ✓ Analysis completed")
            self.output.append("")
            
            # Phase 3: Semantic Analysis
            self.output.append("=== PHASE 3: SEMANTIC ANALYSIS ===")
            if self.semantic_analyzer.analyze(ast):
                self.output.append("  ✓ All semantic checks passed")
                
                # Show symbol table
                if self.semantic_analyzer.current_scope or self.semantic_analyzer.symbol_table:
                    self.output.append("\n  Symbol Table:")
                    all_vars = {**self.semantic_analyzer.symbol_table, 
                              **self.semantic_analyzer.current_scope}
                    for var, var_type in all_vars.items():
                        self.output.append(f"    - {var}: {var_type}")
                
                # Show functions
                if self.semantic_analyzer.function_table:
                    self.output.append("\n  Declared functions:")
                    for func, info in self.semantic_analyzer.function_table.items():
                        if func not in ['printf', 'scanf', 'print']:
                            params = ', '.join(f"{t} {n}" for t, n in info['params'])
                            self.output.append(
                                f"    - {info['return_type']} {func}({params})"
                            )
                
                # Show warnings
                if self.semantic_analyzer.warnings:
                    self.output.append("\n  Warnings:")
                    for warning in self.semantic_analyzer.warnings:
                        self.output.append(f"    ⚠ {warning}")
            else:
                self.output.append("  ✗ Semantic errors found:")
                for error in self.semantic_analyzer.errors:
                    self.output.append(f"    ✗ {error}")
                raise Exception("Semantic analysis failed")
            
            self.output.append("")
            
            # Phase 4: Execution
            self.output.append("=== PHASE 4: EXECUTION ===")
            result = self.interpreter.execute(ast)
            
            if self.interpreter.output:
                self.output.append("  Execution Output:")
                for msg in self.interpreter.output:
                    self.output.append(f"    {msg}")
            
            if result is not None:
                self.output.append(f"\n  Return Value: {result}")
            
            self.output.append("")
            self.output.append("=== ANALYSIS AND EXECUTION COMPLETED ===")
            self.output.append("✓ Lexical: OK")
            self.output.append("✓ Syntactic: OK")
            self.output.append("✓ Semantic: OK")
            self.output.append("✓ Execution: OK")
            
            return True
            
        except Exception as e:
            self.output.append(f"\n✗ ERROR: {e}")
            return False

# ============================================================================
# GRAPHICAL INTERFACE
# ============================================================================

class CInterpreterGUI:
    """GUI for C Interpreter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("C Language Interpreter - Lexical, Syntactic, and Semantic Analysis")
        self.root.geometry("1400x900")
        
        self.engine = CInterpreterEngine()
        
        # C code examples
        self.examples = {
            "Hello World": """#include <stdio.h>

int main() {
    printf("Hello World");
    return 0;
}""",
            "Multiple Variables": """#include <stdio.h>

int main() {
    int x, y, z;
    float a, b;
    
    x = 10;
    y = 20;
    z = x + y;
    
    a = 3.14;
    b = 2.71;
    
    return 0;
}""",
            "Variables and Operations": """#include <stdio.h>

int main() {
    int x;
    int y;
    int z;
    
    x = 10;
    y = 20;
    z = x + y * 2;
    
    return 0;
}""",
            "Control Structures": """#include <stdio.h>

int main() {
    int x;
    x = 10;
    
    if (x > 0) {
        printf("x is positive");
    } else {
        printf("x is negative or zero");
    }
    
    return 0;
}""",
            "While Loop": """#include <stdio.h>

int main() {
    int counter;
    counter = 0;
    
    while (counter < 5) {
        printf("Counter: %d", counter);
        counter++;
    }
    
    return 0;
}""",
            "Switch-Case": """#include <stdio.h>

int main() {
    int day;
    day = 3;
    
    switch (day) {
        case 1:
            printf("Monday");
            break;
        case 2:
            printf("Tuesday");
            break;
        case 3:
            printf("Wednesday");
            break;
        case 4:
            printf("Thursday");
            break;
        case 5:
            printf("Friday");
            break;
        default:
            printf("Weekend");
            break;
    }
    
    return 0;
}""",
            "Custom Functions": """#include <stdio.h>

int sum(int a, int b) {
    int result;
    result = a + b;
    return result;
}

int multiply(int x, int y) {
    return x * y;
}

int main() {
    int a, b, c;
    a = 5;
    b = 3;
    
    c = sum(a, b);
    printf("Sum: %d", c);
    
    c = multiply(a, b);
    printf("Product: %d", c);
    
    return 0;
}""",
            "Strings and Booleans": """#include <stdio.h>
#include <stdbool.h>

int main() {
    char name[] = "John";
    char surname[] = "Doe";
    bool isAdult;
    int age;
    
    age = 25;
    
    if (age >= 18) {
        isAdult = true;
    } else {
        isAdult = false;
    }
    
    printf("Name: %s", name);
    printf("Surname: %s", surname);
    printf("Age: %d", age);
    
    if (isAdult) {
        printf("Is an adult");
    }
    
    return 0;
}""",
            "Basic Arrays": """#include <stdio.h>

int main() {
    int numbers[5];
    int i;
    
    numbers[0] = 10;
    numbers[1] = 20;
    numbers[2] = 30;
    numbers[3] = 40;
    numbers[4] = 50;
    
    printf("First element: %d", numbers[0]);
    printf("Third element: %d", numbers[2]);
    printf("Last element: %d", numbers[4]);
    
    return 0;
}""",
            "Arrays with Initialization": """#include <stdio.h>

int main() {
    int grades[5] = {85, 90, 78, 92, 88};
    float averages[3] = {8.5, 9.2, 7.8};
    char names[4] = {'A', 'n', 'a', 'a'};
    
    printf("First grade: %d", grades[0]);
    printf("First average: %f", averages[0]);
    printf("First letter: %c", names[0]);
    
    grades[0] = 95;
    printf("Modified grade: %d", grades[0]);
    
    return 0;
}""",
            "Arrays in Loops": """#include <stdio.h>

int main() {
    int numbers[5] = {2, 4, 6, 8, 10};
    int sum;
    int i;
    
    sum = 0;
    i = 0;
    
    while (i < 5) {
        sum = sum + numbers[i];
        printf("numbers[%d] = %d", i, numbers[i]);
        i++;
    }
    
    printf("Total sum: %d", sum);
    
    return 0;
}""",
            "Full Code": """#include <stdio.h>
#include <stdbool.h>

int main() {
    // Variable declarations
    int x, y, z;
    float pi;
    char letter;
    char message[] = "Hello World";
    bool flag;
    
    // Initialize variables
    pi = 3.14159;
    letter = 'A';
    flag = true;
    
    // Arithmetic operations
    x = 10;
    y = 20;
    z = x + y * 2;
    
    // Control structures
    if (x > 0) {
        printf("x is positive: %d", x);
    } else if (x < 0) {
        printf("x is negative");
    } else {
        printf("x is zero");
    }
    
    // While loop
    int counter;
    counter = 0;
    while (counter < 5) {
        printf("Counter: %d", counter);
        counter++;
    }
    
    // Results
    printf("Result z: %d", z);
    printf("Message: %s", message);
    printf("Pi: %f", pi);
    printf("Letter: %c", letter);
    
    if (flag) {
        printf("Flag is true");
    }
    
    return 0;
}""",
            "Error: Variable not declared": """#include <stdio.h>

int main() {
    x = 10;
    y = x + 5;
    return 0;
}""",
            "Error: Division by zero": """#include <stdio.h>

int main() {
    int x;
    x = 10 / 0;
    return 0;
}""",
            "Error: Function not declared": """#include <stdio.h>

int main() {
    int x;
    x = sum(5, 3);
    return 0;
}""",
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create graphical interface"""
        # Header
        header = tk.Frame(self.root, bg='#2c3e50', pady=20)
        header.pack(fill='x')
        
        tk.Label(header, text="🔧 C Language Interpreter", 
                font=('Arial', 24, 'bold'),
                bg='#2c3e50', fg='white').pack()
        tk.Label(header, text="Lexical → Syntactic → Semantic Analysis", 
                font=('Arial', 12),
                bg='#2c3e50', fg='#ecf0f1').pack()
        tk.Label(header, text="Team: 02 | Group: 05", 
                font=('Arial', 9),
                bg='#2c3e50', fg='#95a5a6').pack()
        
        # Main area
        main = tk.PanedWindow(self.root, orient='horizontal', sashwidth=5)
        main.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel
        left = tk.Frame(main, bg='#ecf0f1')
        main.add(left, width=600)
        
        # Editor Title
        editor_header = tk.Frame(left, bg='#34495e', pady=8)
        editor_header.pack(fill='x')
        tk.Label(editor_header, text="📝 C Code Editor", 
                font=('Arial', 13, 'bold'),
                bg='#34495e', fg='white').pack()
        
        # Code editor with line numbers
        editor_frame = tk.Frame(left)
        editor_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Frame for line numbers
        line_numbers_frame = tk.Frame(editor_frame, bg='#3a3a3a', width=50)
        line_numbers_frame.pack(side='left', fill='y')
        
        self.line_numbers = tk.Text(
            line_numbers_frame,
            width=4,
            padx=5,
            takefocus=0,
            border=0,
            background='#3a3a3a',
            foreground='#858585',
            state='disabled',
            font=('Consolas', 11),
            wrap='none'
        )
        self.line_numbers.pack(fill='y', expand=True)
        
        # Shared scrollbar
        scrollbar = tk.Scrollbar(editor_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Text editor
        self.input_text = tk.Text(
            editor_frame, 
            height=20, 
            font=('Consolas', 11),
            bg='#2b2b2b',
            fg='#f8f8f2',
            insertbackground='white',
            selectbackground='#44475a',
            wrap='none',
            yscrollcommand=scrollbar.set
        )
        self.input_text.pack(side='left', fill='both', expand=True)
        self.input_text.insert('1.0', self.examples["Hello World"])
        
        # Configure scrollbar
        scrollbar.config(command=self.on_scrollbar)
        
        # Tag to highlight error line
        self.input_text.tag_config('error_line', background='#ff4444', foreground='white')
        
        # Bind events to update line numbers
        self.input_text.bind('<KeyRelease>', self.update_line_numbers)
        self.input_text.bind('<MouseWheel>', lambda e: self.update_line_numbers(e))
        self.input_text.bind('<Button-1>', self.update_line_numbers)
        
        # Update line numbers initially
        self.update_line_numbers()
        
        # Button frame
        btn_frame = tk.Frame(left, bg='#ecf0f1')
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(btn_frame, text="📂 Load File", 
                 command=self.load_file,
                 bg='#27ae60', fg='white', 
                 font=('Arial', 10, 'bold'),
                 padx=15, pady=10, 
                 cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="▶ Analyze Code", 
                 command=self.run_interpreter,
                 bg='#3498db', fg='white', 
                 font=('Arial', 12, 'bold'),
                 padx=25, pady=10,
                 cursor='hand2').pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="🗑 Clear", 
                 command=self.clear_all,
                 bg='#e74c3c', fg='white', 
                 font=('Arial', 10, 'bold'),
                 padx=15, pady=10,
                 cursor='hand2').pack(side='left', padx=5)
        
        # Example selector
        example_frame = tk.Frame(left, bg='#ecf0f1')
        example_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(example_frame, text="📚 Examples:", 
                font=('Arial', 10, 'bold'),
                bg='#ecf0f1').pack(side='left', padx=5)
        
        self.example_var = tk.StringVar()
        example_menu = ttk.Combobox(
            example_frame, 
            textvariable=self.example_var,
            values=list(self.examples.keys()),
            state='readonly', 
            width=40,
            font=('Arial', 10)
        )
        example_menu.pack(side='left', padx=5, fill='x', expand=True)
        example_menu.bind('<<ComboboxSelected>>', self.load_example)
        
        # Right panel
        right = tk.Frame(main, bg='#ecf0f1')
        main.add(right, width=700)
        
        # Analysis status
        status_frame = tk.Frame(right, bg='#34495e', pady=10)
        status_frame.pack(fill='x')
        
        tk.Label(status_frame, text="📊 Analysis Status", 
                font=('Arial', 13, 'bold'),
                bg='#34495e', fg='white').pack()
        
        # Status indicators
        indicators_frame = tk.Frame(right, bg='#ecf0f1', pady=10)
        indicators_frame.pack(fill='x', padx=10)
        
        self.status_lexico = tk.Label(
            indicators_frame, 
            text="⚪ Lexical", 
            font=('Arial', 11),
            bg='#ecf0f1', 
            anchor='w',
            width=15
        )
        self.status_lexico.pack(side='left', padx=5)
        
        self.status_sintactico = tk.Label(
            indicators_frame, 
            text="⚪ Syntactic", 
            font=('Arial', 11),
            bg='#ecf0f1', 
            anchor='w',
            width=15
        )
        self.status_sintactico.pack(side='left', padx=5)
        
        self.status_semantico = tk.Label(
            indicators_frame, 
            text="⚪ Semantic", 
            font=('Arial', 11),
            bg='#ecf0f1', 
            anchor='w',
            width=15
        )
        self.status_semantico.pack(side='left', padx=5)
        
        # Output area
        output_notebook = ttk.Notebook(right)
        output_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Results
        results_tab = tk.Frame(output_notebook)
        output_notebook.add(results_tab, text="📋 Results")
        
        self.output_text = scrolledtext.ScrolledText(
            results_tab, 
            height=35, 
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#2c3e50',
            wrap=tk.WORD
        )
        self.output_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 2: Tokens table
        tokens_tab = tk.Frame(output_notebook)
        output_notebook.add(tokens_tab, text="🔤 Tokens")
        
        # Create Treeview for tokens
        tokens_frame = tk.Frame(tokens_tab)
        tokens_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbar
        tokens_scroll = tk.Scrollbar(tokens_frame)
        tokens_scroll.pack(side='right', fill='y')
        
        self.tokens_tree = ttk.Treeview(
            tokens_frame,
            columns=('Token', 'Type', 'Frequency', 'Examples'),
            show='headings',
            yscrollcommand=tokens_scroll.set,
            height=20
        )
        tokens_scroll.config(command=self.tokens_tree.yview)
        
        # Configure columns
        self.tokens_tree.heading('Token', text='#')
        self.tokens_tree.heading('Type', text='Token Type')
        self.tokens_tree.heading('Frequency', text='Frequency')
        self.tokens_tree.heading('Examples', text='Example Values')
        
        self.tokens_tree.column('Token', width=50, anchor='center')
        self.tokens_tree.column('Type', width=150, anchor='w')
        self.tokens_tree.column('Frequency', width=100, anchor='center')
        self.tokens_tree.column('Examples', width=400, anchor='w')
        
        self.tokens_tree.pack(fill='both', expand=True)
        
        # Style for Treeview
        style = ttk.Style()
        style.configure("Treeview", rowheight=25, font=('Consolas', 9))
        style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))
        
        # Tab 3: Variables
        vars_tab = tk.Frame(output_notebook)
        output_notebook.add(vars_tab, text="📊 Variables")
        
        self.var_text = scrolledtext.ScrolledText(
            vars_tab, 
            height=20, 
            font=('Consolas', 11),
            bg='#f8f9fa',
            fg='#2c3e50'
        )
        self.var_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tags for variables
        self.var_text.tag_config('header', foreground='#2980b9', font=('Consolas', 11, 'bold'))
        self.var_text.tag_config('var_name', foreground='#8e44ad', font=('Consolas', 11, 'bold'))
        self.var_text.tag_config('var_type', foreground='#16a085')
        self.var_text.tag_config('var_value', foreground='#27ae60')
        
        # Configure tags for colors
        self.output_text.tag_config('header', foreground='#2980b9', font=('Consolas', 10, 'bold'))
        self.output_text.tag_config('success', foreground='#27ae60', font=('Consolas', 10, 'bold'))
        self.output_text.tag_config('error', foreground='#e74c3c', font=('Consolas', 10, 'bold'))
        self.output_text.tag_config('warning', foreground='#f39c12', font=('Consolas', 10, 'bold'))
        
        # Footer
        footer = tk.Frame(self.root, bg='#34495e', pady=8)
        footer.pack(fill='x')
        tk.Label(footer, 
                text="💡 This interpreter analyzes C code using PLY (Python Lex-Yacc)", 
                font=('Arial', 9),
                bg='#34495e', 
                fg='#bdc3c7').pack()
    
    def highlight_error_line(self, line_number):
        """Highlight line with error"""
        # Remove previous highlights
        self.input_text.tag_remove('error_line', '1.0', 'end')
        
        if line_number and line_number > 0:
            # Highlight the error line
            start = f"{line_number}.0"
            end = f"{line_number}.end"
            self.input_text.tag_add('error_line', start, end)
            
            # Scroll to line
            self.input_text.see(start)
    
    def update_line_numbers(self, event=None):
        """Update line numbers"""
        # Get editor content
        line_count = self.input_text.get('1.0', 'end-1c').count('\n') + 1
        
        # Generate line numbers
        line_numbers_string = '\n'.join(str(i) for i in range(1, line_count + 1))
        
        # Update line numbers widget
        self.line_numbers.config(state='normal')
        self.line_numbers.delete('1.0', 'end')
        self.line_numbers.insert('1.0', line_numbers_string)
        self.line_numbers.config(state='disabled')
        
        # Sync scroll
        self.line_numbers.yview_moveto(self.input_text.yview()[0])
    
    def on_scrollbar(self, *args):
        """Handle scrollbar scroll"""
        self.input_text.yview(*args)
        self.line_numbers.yview(*args)
    
    def load_example(self, event=None):
        """Load selected example"""
        name = self.example_var.get()
        if name in self.examples:
            self.input_text.delete('1.0', 'end')
            self.input_text.insert('1.0', self.examples[name])
            self.clear_output()
            self.update_line_numbers()
    
    def load_file(self):
        """Load code from file"""
        filename = filedialog.askopenfilename(
            title="Select C Code File",
            filetypes=[
                ("C Files", "*.c"),
                ("Text Files", "*.txt"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.input_text.delete('1.0', 'end')
                    self.input_text.insert('1.0', content)
                    self.clear_output()
                    self.update_line_numbers()
                messagebox.showinfo("Success", f"File loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error loading file", str(e))
    
    def clear_all(self):
        """Clear all"""
        self.input_text.delete('1.0', 'end')
        self.clear_output()
    
    def clear_output(self):
        """Clear output"""
        self.output_text.delete('1.0', 'end')
        self.var_text.delete('1.0', 'end')
        # Clear tokens table
        for item in self.tokens_tree.get_children():
            self.tokens_tree.delete(item)
        self.status_lexico.config(text="⚪ Lexical", fg='#2c3e50')
        self.status_sintactico.config(text="⚪ Syntactic", fg='#2c3e50')
        self.status_semantico.config(text="⚪ Semantic", fg='#2c3e50')
        # Remove error highlight
        self.input_text.tag_remove('error_line', '1.0', 'end')
    
    def run_interpreter(self):
        """Run interpreter"""
        code = self.input_text.get('1.0', 'end-1c').strip()
        
        if not code:
            messagebox.showwarning("Warning", "Enter code to analyze")
            return
        
        self.clear_output()
        
        # Reset indicators
        self.status_lexico.config(text="⏳ Lexical...", fg='#f39c12')
        self.status_sintactico.config(text="⚪ Syntactic", fg='#2c3e50')
        self.status_semantico.config(text="⚪ Semantic", fg='#2c3e50')
        self.root.update()
        
        try:
            success = self.engine.interpret(code)
            
            # Update indicators
            self.status_lexico.config(text="✓ Lexical", fg='#27ae60')
            self.status_sintactico.config(text="✓ Syntactic", fg='#27ae60')
            
            if success:
                self.status_semantico.config(text="✓ Semantic", fg='#27ae60')
            else:
                self.status_semantico.config(text="✗ Semantic", fg='#e74c3c')
            
            # Fill tokens table
            for i, (tok_type, info) in enumerate(sorted(self.engine.token_stats.items()), 1):
                # Take only first 5 examples to avoid clutter
                examples = ', '.join(info['values'][:5])
                if len(info['values']) > 5:
                    examples += '...'
                
                self.tokens_tree.insert('', 'end', values=(
                    i,
                    tok_type,
                    info['count'],
                    examples
                ))
            
            # Show formatted output
            for line in self.engine.output:
                if line.startswith("==="):
                    self.output_text.insert('end', line + '\n', 'header')
                elif "✓" in line and ("OK" in line or "Success" in line or "passed" in line):
                    self.output_text.insert('end', line + '\n', 'success')
                elif "✗" in line or "ERROR" in line or "Error" in line:
                    self.output_text.insert('end', line + '\n', 'error')
                elif "⚠" in line or "Warning" in line:
                    self.output_text.insert('end', line + '\n', 'warning')
                else:
                    self.output_text.insert('end', line + '\n')
            
            # Show variables
            if self.engine.interpreter.variables:
                self.var_text.insert('end', "Variables in memory:\n\n", 'header')
                for name, value in sorted(self.engine.interpreter.variables.items()):
                    # Determine type
                    if isinstance(value, list):
                        tipo = "array"
                        # Show array compactly
                        if len(value) <= 10:
                            display_val = f"[{', '.join(map(str, value))}]"
                        else:
                            display_val = f"[{', '.join(map(str, value[:5]))}, ... (total: {len(value)})]"
                    elif isinstance(value, bool):
                        tipo = "bool"
                        display_val = "true" if value else "false"
                    elif isinstance(value, str):
                        # Check if it's char[] (string) vs single char
                        if len(value) > 1 or (len(value) == 1 and name.endswith('name') or name.endswith('surname') or 'text' in name.lower() or 'message' in name.lower()):
                            tipo = "char[]"
                            display_val = f'"{value}"'
                        else:
                            tipo = "char"
                            display_val = f"'{value}'" if value else "''"
                    elif isinstance(value, float):
                        tipo = "float"
                        display_val = str(value)
                    elif isinstance(value, int):
                        tipo = "int"
                        display_val = str(value)
                    else:
                        tipo = "unknown"
                        display_val = str(value)
                    
                    self.var_text.insert('end', f"{name} ", 'var_name')
                    self.var_text.insert('end', f"({tipo}) ", 'var_type')
                    self.var_text.insert('end', "= ")
                    self.var_text.insert('end', f"{display_val}\n", 'var_value')
            else:
                self.var_text.insert('end', "No variables in memory\n")
            
            if success:
                messagebox.showinfo("Success", "✓ Analysis and execution completed successfully\n\nThe code has no errors.")
            
        except Exception as e:
            error_message = str(e)
            
            # Extract line number from error message
            import re
            line_match = re.search(r'line (\d+)', error_message)
            if line_match:
                error_line = int(line_match.group(1))
                self.highlight_error_line(error_line)
            
            # Determine which phase failed
            if "Lexical" in error_message or "lexical" in error_message.lower():
                self.status_lexico.config(text="✗ Lexical", fg='#e74c3c')
            elif "Syntax" in error_message or "syntax" in error_message.lower():
                self.status_lexico.config(text="✓ Lexical", fg='#27ae60')
                self.status_sintactico.config(text="✗ Syntactic", fg='#e74c3c')
            else:
                self.status_lexico.config(text="✓ Lexical", fg='#27ae60')
                self.status_sintactico.config(text="✓ Syntactic", fg='#27ae60')
                self.status_semantico.config(text="✗ Semantic", fg='#e74c3c')
            
            # Fill tokens table if available
            if self.engine.token_stats:
                for i, (tok_type, info) in enumerate(sorted(self.engine.token_stats.items()), 1):
                    examples = ', '.join(info['values'][:5])
                    if len(info['values']) > 5:
                        examples += '...'
                    
                    self.tokens_tree.insert('', 'end', values=(
                        i,
                        tok_type,
                        info['count'],
                        examples
                    ))
            
            for line in self.engine.output:
                if line.startswith("==="):
                    self.output_text.insert('end', line + '\n', 'header')
                elif "✓" in line:
                    self.output_text.insert('end', line + '\n', 'success')
                elif "✗" in line or "ERROR" in line:
                    self.output_text.insert('end', line + '\n', 'error')
                elif "⚠" in line:
                    self.output_text.insert('end', line + '\n', 'warning')
                else:
                    self.output_text.insert('end', line + '\n')
            
            # Show highlighted error
            error_display = f"\n⚠️ ERROR FOUND:\n{error_message}\n"
            if line_match:
                error_display += f"\n📍 Line {error_line} has been highlighted in the editor."
            
            messagebox.showerror("Analysis Error", error_display)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function to create and run the Tkinter application."""
    root = tk.Tk()
    app = CInterpreterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
