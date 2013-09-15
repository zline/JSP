
import sys
from abc import ABCMeta, abstractmethod

from jsp import tokenize


class ParseError(ValueError):
    pass

def init_parse():
    global _PARSER
    
    AssignmentExpression = ForwardDeclaredParser()
    
    ArrayLiteral = ArrayLiteralParser(AssignmentExpression)
    ObjectLiteral = ObjectLiteralParser(AssignmentExpression)
    
    Identifier = TokenParser(tokenize.Identifier)
    Expression = ForwardDeclaredParser()
    PrimaryExpression = AltParser((
        TokenParser(tokenize.Keyword, u"this"),
        Identifier,
        TokenParser(tokenize.Literal),
        ArrayLiteral,
        ObjectLiteral,
        ParenthesedExpressionParser(Expression)
    ))
    FunctionExpression = ForwardDeclaredParser()
    FunctionDeclaration = ForwardDeclaredParser()
    MemberExpression = MemberExpressionParser(PrimaryExpression, FunctionExpression, AssignmentExpression, Expression)
    NewExpression = NewExpressionParser(MemberExpression)
    CallExpression = CallExpressionParser(MemberExpression, AssignmentExpression, Expression)
    LeftHandSideExpression = AltParser((
        # Call & new expressions are reordered by purpose - otherwise new expression could eat
        # object of call expression (and we dont have backtracking), so call expression has a chance first.
        # Looks like it doesnt affect resulting syntactic & semantic behavior.
        CallExpression,
        NewExpression
    ))
    
    PostfixExpression = PostfixExpressionParser(LeftHandSideExpression)
    UnaryExpression = UnaryExpressionParser(PostfixExpression)
    MultiplicativeExpression = BinOpParser(UnaryExpression, TokenParser(tokenize.Punctuator, (u"*", u"/", u"%")))
    AdditiveExpression = BinOpParser(MultiplicativeExpression, TokenParser(tokenize.Punctuator, (u"+", u"-")))
    ShiftExpression = BinOpParser(AdditiveExpression, TokenParser(tokenize.Punctuator, (u"<<", u">>", u">>>")))
    
    RelationalExpression = BinOpParser(ShiftExpression, AltParser((
        TokenParser(tokenize.Punctuator, (u"<", u">", u"<=", u">=")),
        TokenParser(tokenize.Keyword, (u"instanceof", u"in")),
    )))
    RelationalExpressionNoIn = BinOpParser(ShiftExpression, AltParser((
        TokenParser(tokenize.Punctuator, (u"<", u">", u"<=", u">=")),
        TokenParser(tokenize.Keyword, u"instanceof"),
    )))
    EqualityExpression = BinOpParser(RelationalExpression, TokenParser(tokenize.Punctuator, (u"==", u"!=", u"===", u"!==")))
    EqualityExpressionNoIn = BinOpParser(RelationalExpressionNoIn, TokenParser(tokenize.Punctuator, (u"==", u"!=", u"===", u"!==")))
    
    BitwiseANDExpression = BinOpParser(EqualityExpression, TokenParser(tokenize.Punctuator, u"&"))
    BitwiseXORExpression = BinOpParser(BitwiseANDExpression, TokenParser(tokenize.Punctuator, u"^"))
    BitwiseORExpression = BinOpParser(BitwiseXORExpression, TokenParser(tokenize.Punctuator, u"|"))
    BitwiseANDExpressionNoIn = BinOpParser(EqualityExpressionNoIn, TokenParser(tokenize.Punctuator, u"&"))
    BitwiseXORExpressionNoIn = BinOpParser(BitwiseANDExpressionNoIn, TokenParser(tokenize.Punctuator, u"^"))
    BitwiseORExpressionNoIn = BinOpParser(BitwiseXORExpressionNoIn, TokenParser(tokenize.Punctuator, u"|"))
    
    LogicalANDExpression = BinOpParser(BitwiseORExpression, TokenParser(tokenize.Punctuator, u"&&"))
    LogicalORExpression = BinOpParser(LogicalANDExpression, TokenParser(tokenize.Punctuator, u"||"))
    LogicalANDExpressionNoIn = BinOpParser(BitwiseORExpressionNoIn, TokenParser(tokenize.Punctuator, u"&&"))
    LogicalORExpressionNoIn = BinOpParser(LogicalANDExpressionNoIn, TokenParser(tokenize.Punctuator, u"||"))
    
    ConditionalExpression = AltParser((
        ConditionalExpressionParser(LogicalORExpression, AssignmentExpression, AssignmentExpression),
        LogicalORExpression
    ))
    AssignmentExpressionNoIn = ForwardDeclaredParser()
    ConditionalExpressionNoIn = AltParser((
        ConditionalExpressionParser(LogicalORExpressionNoIn, AssignmentExpression, AssignmentExpressionNoIn),
        LogicalORExpressionNoIn
    ))
    
    AssignmentExpression.target = AltParser((
        AssignmentExpressionParser(LeftHandSideExpression, AssignmentExpression),
        ConditionalExpression
    ))
    AssignmentExpressionNoIn.target = AltParser((
        AssignmentExpressionParser(LeftHandSideExpression, AssignmentExpressionNoIn),
        ConditionalExpressionNoIn
    ))
    
    Expression.target = BinOpParser(AssignmentExpression, TokenParser(tokenize.Punctuator, u","))
    ExpressionNoIn = BinOpParser(AssignmentExpressionNoIn, TokenParser(tokenize.Punctuator, u","))

    
    VariableDeclarationNoIn = VariableDeclarationParser(AssignmentExpressionNoIn)
    VariableDeclarationList = VariableDeclarationListParser(AssignmentExpression)
    VariableDeclarationListNoIn = VariableDeclarationListParser(AssignmentExpressionNoIn)
    
    Statement = ForwardDeclaredParser()
    Block = BlockParser(Statement)
    VariableStatement = VariableStatementParser(VariableDeclarationList)
    EmptyStatement = EmptyStatementParser()
    ExpressionStatement = ExpressionStatementParser(Expression)
    IfStatement = IfStatementParser(Expression, Statement)
    WhileStatement = WhileStatementParser(Expression, Statement)
    ForStatement = ForStatementParser(ExpressionNoIn, Expression, Statement, VariableDeclarationListNoIn, LeftHandSideExpression,
        VariableDeclarationNoIn)
    ContinueStatement = ControlStatementParser(u"continue", Identifier, PTreeContinueStatementNode)
    BreakStatement = ControlStatementParser(u"break", Identifier, PTreeBreakStatementNode)
    ReturnStatement = ControlStatementParser(u"return", Expression, PTreeReturnStatementNode)
    WithStatement = WithStatementParser(Expression, Statement)
    SwitchStatement = SwitchStatementParser(Expression, Statement)
    LabelledStatement = LabelledStatementParser(Statement)
    ThrowStatement = ControlStatementParser(u"throw", Expression, PTreeThrowStatementNode, opt_required=True)
    TryStatement = TryStatementParser(Block)
    # TODO fast, hash-based switching in AltParser below
    Statement.target = AltParser((
        EmptyStatement,         # ;
        Block,                  # {
        ReturnStatement,        # return
        IfStatement,            # if
        ForStatement,           # for
        WhileStatement,         # while, do
        VariableStatement,      # var
        ContinueStatement,      # continue
        BreakStatement,         # break
        WithStatement,          # with
        SwitchStatement,        # switch
        ThrowStatement,         # throw
        TryStatement,           # try
        ExpressionStatement,    # 
        LabelledStatement,      #
    ))
    
    SourceElements = SourceElementsParser(Statement, FunctionDeclaration)
    FunctionExpression.target = FunctionParser(OptParser(Identifier), SourceElements, PTreeFunctionExpressionNode)
    FunctionDeclaration.target = FunctionParser(Identifier, SourceElements, PTreeFunctionDeclarationNode)
    
    Program = SourceElements
    
    _PARSER = Program


def parse(toks):
    global _PARSER
    ret = _PARSER.parse(toks, 0)
    if ret is None:
        raise ParseError("cant parse anything")
    (ptree, end) = ret
    assert end <= len(toks)
    if end < len(toks) and filter(lambda tok: type(tok) != tokenize.LineTerminator, toks[end:]):
        raise ParseError("cant parse starting at token {}: {}...".format(
            end, tuple(toks[end : end + 10])))

    return ptree


_PARSER = None

class Parser(object):
    """
    Abstract syntactic parser
    """
    __metaclass__ = ABCMeta
    
    def parse(self, toks, start):
        start = self.pre_parse(toks, start) # FIXME remove, if no longer needed
        assert start <= len(toks)
        if start == len(toks):
            return None
        
        return self.do_parse(toks, start)
    
    def pre_parse(self, toks, start):
        """
        should return new starting position
        """
        return start
    
    @abstractmethod
    def do_parse(self, toks, start):
        """
        should return (parse tree object, position of last consumed token + 1) or None
        """
        raise NotImplementedError()


class TokenParser(Parser):
    """
    Matches specified token class(es) (or subclass(es)) and optionally specified token data
    """
    def __init__(self, tok_isinstance, tok_data=None, keep_line_terms=False):
        self.tok_isinstance = tok_isinstance
        self.tok_data = tok_data
        self.keep_line_terms = keep_line_terms
        self._multitoks = isinstance(self.tok_data, (list, tuple))

    def do_parse(self, toks, start):
        if not self.keep_line_terms:
            while start < len(toks) and type(toks[start]) == tokenize.LineTerminator:
                start += 1
        
        assert start <= len(toks)
        if start == len(toks):
            return None
        
        tok = toks[start]
        
        # type check
        if not isinstance(tok, self.tok_isinstance):
            return None
        
        # data check
        if self.tok_data is not None:
            if self._multitoks:
                if tok.data not in self.tok_data:
                    return None
            else:
                if tok.data != self.tok_data:
                    return None
        
        return (PTreeTokenNode(tok), start+1)

class ForwardDeclaredParser(Parser):
    def __init__(self):
        self.target = None
    
    def parse(self, toks, start):
        return self.target.parse(toks, start)
    
    def do_parse(self, toks, start):
        assert False, 'should not be reached'

class AltParser(Parser):
    """
    Matches first matched alternative subparser
    """
    def __init__(self, alternatives):
        self.alternatives = alternatives

    def do_parse(self, toks, start):
        for alt in self.alternatives:
            # TODO remove excess call to pre_parse
            ret = alt.parse(toks, start)
            if ret is not None:
                return (self.change_node(ret[0]), ret[1])
        
        return None
    
    def change_node(self, ptree):
        return ptree

class SeqParser(Parser):
    """
    Matches sequence of subparsers
    """
    def __init__(self, subparsers, node_class=None, pick_node=None, 
            optional_right_items=None, optional_left_items=None):
        self.subparsers = subparsers
        self.node_class = node_class
        self.pick_node = pick_node
        self.optional_right_items = optional_right_items
        self.optional_left_items = optional_left_items

    def do_parse(self, toks, start):
        subtree_list = list()
        for (idx, subp) in enumerate(self.subparsers):
            ret = subp.parse(toks, start)
            if ret is None: # not parsed
                if (self.optional_left_items is not None and idx < self.optional_left_items
                        or self.optional_right_items is not None and idx >= len(self.subparsers) - self.optional_right_items):
                    # item was marked as optional
                    subtree_list.append(None)
                else:
                    return None
            
            else:           # parsed
                (ptree, end) = ret
                subtree_list.append(ptree)
                start = end
        
        return (self.mknode(subtree_list), start)
    
    def mknode(self, subtree_list):
        if self.pick_node is not None:
            return subtree_list[self.pick_node]
        elif self.node_class is not None:
            return self.node_class(subtree_list)
        else:
            raise RuntimeError("dunno how to construct node")

class RepeatedParser(Parser):
    """
    Matches specified number of repetitions of subparser
    """
    def __init__(self, subparser, min_matches=1, node_class=None, separator=None,
            include_separator_tree=False):
        self.subparser = subparser
        self.min_matches = min_matches
        self.node_class = node_class
        self.separator = separator
        self.include_separator_tree = include_separator_tree
    
    def do_parse(self, toks, start):
        subtree_list = list()
        num = 0
        while True:
            sep_ptree = None
            pre_sep_start = start
            if num > 0 and self.separator is not None:
                ret = self.separator.parse(toks, start)
                if ret is None:
                    break
                (ptree, end) = ret
                if self.include_separator_tree:
                    sep_ptree = ptree
                start = end
            
            ret = self.subparser.parse(toks, start)
            if ret is None:
                start = pre_sep_start
                break
            num += 1
            (ptree, end) = ret
            if sep_ptree is not None:
                subtree_list.append(sep_ptree)
            subtree_list.append(ptree)
            start = end
        
        if num < self.min_matches:
            return None
        return (self.mknode(subtree_list), start)
    
    def mknode(self, subtree_list):
        # define node_class or override mknode
        return self.node_class(subtree_list)

class EmptyParser(Parser):
    """
    Always matches, consumes no tokens
    """
    def __init__(self, node_class):
        self.node_class = node_class
    
    def do_parse(self, toks, start):
        return (self.node_class(), start)

class OptParser(Parser):
    """
    Always matches, returns subparser results if it matches, (None, start) otherwise
    """
    def __init__(self, subparser):
        self.subparser = subparser
    
    def do_parse(self, toks, start):
        ret = self.subparser.parse(toks, start)
        return (None, start) if ret is None else ret

class LookAheadParser(Parser):
    """
    Non-capturing match
    """
    def __init__(self, subparser, inverse=False):
        self.subparser = subparser
        self.inverse = inverse
    
    def do_parse(self, toks, start):
        ret = self.subparser.parse(toks, start)
        success = (ret is None) if self.inverse else (ret is not None)
        return None if not success else (None, start)


class ArrayLiteralParser(SeqParser):
    def __init__(self, AssignmentExpression):
        ItemsParser = RepeatedParser(
            AltParser((AssignmentExpression, EmptyParser(PTreeEmptyArrayElementNode))),
            min_matches=0,
            node_class=PTreeArrayLiteralNode,
            separator=TokenParser(tokenize.Punctuator, u",")
        )
        super(ArrayLiteralParser, self).__init__((
            TokenParser(tokenize.Punctuator, u"["),
            ItemsParser,
            TokenParser(tokenize.Punctuator, u"]"),
        ))
    
    def mknode(self, subtree_list):
        array_node = subtree_list[1]
        if array_node.item_list and isinstance(array_node.item_list[-1], PTreeEmptyArrayElementNode):
            array_node.item_list.pop()
        return array_node

class ObjectLiteralParser(SeqParser):
    def __init__(self, AssignmentExpression):
        PropertyNameParser = TokenParser((tokenize.Identifier, tokenize.StringLiteral, tokenize.NumericLiteral))
        ItemsParser = RepeatedParser(
            SeqParser((PropertyNameParser, TokenParser(tokenize.Punctuator, u":"), AssignmentExpression), node_class=tuple),
            min_matches=0,
            node_class=list,    # hack, see mknode
            separator=TokenParser(tokenize.Punctuator, u",")
        )
        super(ObjectLiteralParser, self).__init__((
            TokenParser(tokenize.Punctuator, u"{"),
            ItemsParser,
            TokenParser(tokenize.Punctuator, u"}"),
        ))
    
    def mknode(self, subtree_list):
        filtered_list = list()
        for item in subtree_list[1]:
            filtered_list.append((item[0], item[2]))
        
        return PTreeObjectLiteralNode(filtered_list)

class ParenthesedExpressionParser(SeqParser):
    def __init__(self, Expression):
        super(ParenthesedExpressionParser, self).__init__((
            TokenParser(tokenize.Punctuator, u"("),
            Expression,
            TokenParser(tokenize.Punctuator, u")"),
        ))
    
    def mknode(self, subtree_list):
        return subtree_list[1]

class ArgumentsParser(SeqParser):
    def __init__(self, AssignmentExpression):
        ItemsParser = RepeatedParser(
            AssignmentExpression,
            min_matches=0,
            node_class=PTreeArgumentsNode,
            separator=TokenParser(tokenize.Punctuator, u",")
        )
        super(ArgumentsParser, self).__init__((
            TokenParser(tokenize.Punctuator, u"("),
            ItemsParser,
            TokenParser(tokenize.Punctuator, u")"),
        ))
    
    def mknode(self, subtree_list):
        return subtree_list[1]

class CtorParser(SeqParser):
    def __init__(self, MemberExpression, AssignmentExpression):
        super(CtorParser, self).__init__((
            TokenParser(tokenize.Keyword, u"new"),
            MemberExpression,
            ArgumentsParser(AssignmentExpression),
        ))
    
    def mknode(self, subtree_list):
        return PTreeCtorNode(subtree_list[1], subtree_list[2])

class MemberAccessByIdentifierParser(SeqParser):
    def __init__(self):
        super(MemberAccessByIdentifierParser, self).__init__((
            TokenParser(tokenize.Punctuator, u"."),
            TokenParser(tokenize.Identifier)))
    def mknode(self, subtree_list):
        return PTreeTokenNode(tokenize.StringLiteral(subtree_list[1].token.data))

class MemberAccessParser(AltParser):
    def __init__(self, Expression):
        super(MemberAccessParser, self).__init__((
            SeqParser((
                TokenParser(tokenize.Punctuator, u"["),
                Expression,
                TokenParser(tokenize.Punctuator, u"]")),
                pick_node=1),
            MemberAccessByIdentifierParser()
        ))

class MemberExpressionParser(SeqParser):
    def __init__(self, PrimaryExpression, FunctionExpression, AssignmentExpression, Expression):
        # PrimaryExpression, FunctionExpression - non-recursive, CtorParser - finite recursion
        basic = AltParser((PrimaryExpression, FunctionExpression, CtorParser(self, AssignmentExpression)))
        member_access_repeated = RepeatedParser(
            MemberAccessParser(Expression),
            node_class=list,    # hack, see mknode
        )
        super(MemberExpressionParser, self).__init__((basic, member_access_repeated), optional_right_items=1)
    
    def mknode(self, subtree_list):
        if subtree_list[1] is None:
            return subtree_list[0]
        
        return PTreeMemberSelectorsNode(subtree_list[0], subtree_list[1])

class CallExpressionParser(SeqParser):
    def __init__(self, MemberExpression, AssignmentExpression, Expression):
        arg_parser = ArgumentsParser(AssignmentExpression)
        basic = SeqParser((MemberExpression, arg_parser), node_class=list)
        member_access_or_call = AltParser((MemberAccessParser(Expression), arg_parser))
        member_access_or_call_repeated = RepeatedParser(
            member_access_or_call,
            node_class=list,    # hack, see mknode
        )
        super(CallExpressionParser, self).__init__((basic, member_access_or_call_repeated), optional_right_items=1)

    def mknode(self, subtree_list):
        obj = PTreeCallNode(subtree_list[0][0], subtree_list[0][1])
        if subtree_list[1] is None:
            return obj
        
        for action in subtree_list[1]:
            if isinstance(action, PTreeArgumentsNode):
                obj = PTreeCallNode(obj, action)
            else:
                obj = PTreeMemberSelectorsNode(obj, [action])   # consecutive selectors could be combined into one node..
        
        return obj

class NewExpressionParser(AltParser):
    def __init__(self, MemberExpression):
        super(NewExpressionParser, self).__init__((
            MemberExpression,
            SeqParser((TokenParser(tokenize.Keyword, u"new"), self), node_class=list)
        ))

    def change_node(self, ptree):
        if isinstance(ptree, list):
            assert ptree[0].token.data == u"new"
            return PTreeCtorNode(ptree[1], PTreeArgumentsNode(list()))
        
        return ptree

class PostfixExpressionParser(SeqParser):
    def __init__(self, LeftHandSideExpression):
        token_parser = TokenParser(tokenize.Punctuator, (u"++", u"--"), keep_line_terms=True)
        super(PostfixExpressionParser, self).__init__((
            LeftHandSideExpression,
            RepeatedParser(
                token_parser,
                min_matches=0,
                node_class=list
            )
        ))
    
    def mknode(self, subtree_list):
        node = subtree_list[0]
        for action in subtree_list[1]:
            node = PTreePostfixExpressionNode(node, action)
        return node

class UnaryExpressionParser(SeqParser):
    def __init__(self, PostfixExpression):
        token_parser = AltParser((
            TokenParser(tokenize.Punctuator, (u"++", u"--", u"-", u"+", u"~", u"!")),
            TokenParser(tokenize.Keyword, (u"delete", u"void", u"typeof")),
        ))
        super(UnaryExpressionParser, self).__init__((
            RepeatedParser(
                token_parser,
                min_matches=0,
                node_class=list
            ),
            PostfixExpression,
        ))
    
    def mknode(self, subtree_list):
        node = subtree_list[1]
        subtree_list[0].reverse()
        for action in subtree_list[0]:
            node = PTreeUnaryExpressionNode(node, action)
        return node

class BinOpParser(RepeatedParser):
    def __init__(self, op_parser, token_parser):
        super(BinOpParser, self).__init__(
            op_parser,
            separator=token_parser,
            include_separator_tree=True
        )
    
    def mknode(self, subtree_list):
        node = subtree_list[0]
        for tok_idx in xrange(1, len(subtree_list), 2):
            node = PTreeBinaryOpNode(subtree_list[tok_idx], (node, subtree_list[tok_idx + 1]))
        return node

class ConditionalExpressionParser(SeqParser):
    def __init__(self, cond_parser, iftrue_parser, iffalse_parser):
        super(ConditionalExpressionParser, self).__init__((
            cond_parser,
            TokenParser(tokenize.Punctuator, u"?"),
            iftrue_parser,
            TokenParser(tokenize.Punctuator, u":"),
            iffalse_parser
        ))
    
    def mknode(self, subtree_list):
        return PTreeConditionalExpressionNode(subtree_list[0], subtree_list[2], subtree_list[4])

class AssignmentExpressionParser(SeqParser):
    def __init__(self, target_parser, source_parser):
        super(AssignmentExpressionParser, self).__init__((
            target_parser,
            TokenParser(tokenize.Punctuator, 
                (u"=", u"*=", u"/=", u"%=", u"+=", u"-=", u"<<=", u">>=", u">>>=", u"&=", u"^=", u"|=")),
            source_parser,
        ))
    
    def mknode(self, subtree_list):
        return PTreeAssignmentExpressionNode(subtree_list[0], subtree_list[1], subtree_list[2])

class StatementListParser(RepeatedParser):
    def __init__(self, Statement):
        super(StatementListParser, self).__init__(Statement, min_matches=0, node_class=list)

class BlockParser(SeqParser):
    def __init__(self, Statement):
        super(BlockParser, self).__init__((
            TokenParser(tokenize.Punctuator, u"{"),
            StatementListParser(Statement),
            TokenParser(tokenize.Punctuator, u"}"),
        ))
    
    def mknode(self, subtree_list):
        return PTreeBlockNode(subtree_list[1])

class VariableDeclarationParser(SeqParser):
    def __init__(self, AssignmentExpression):
        super(VariableDeclarationParser, self).__init__((
            TokenParser(tokenize.Identifier),
            OptParser(
                SeqParser((
                    TokenParser(tokenize.Punctuator, u"="),
                    AssignmentExpression
                ), node_class=list)
            )
        ))
    
    def mknode(self, subtree_list):
        return (subtree_list[0], None if subtree_list[1] is None else subtree_list[1][1])

class VariableDeclarationListParser(RepeatedParser):
    def __init__(self, AssignmentExpression):
        super(VariableDeclarationListParser, self).__init__(
            VariableDeclarationParser(AssignmentExpression),
            separator=TokenParser(tokenize.Punctuator, u","),
            node_class=PTreeVariableDeclarationListNode
        )

class VariableStatementParser(SeqParser):
    def __init__(self, VariableDeclarationList):
        super(VariableStatementParser, self).__init__((
            TokenParser(tokenize.Keyword, u"var"),
            VariableDeclarationList,
            TokenParser(tokenize.Punctuator, u";")
        ))
    
    def mknode(self, subtree_list):
        return PTreeVariableStatementNode(declaration_it=subtree_list[1].declaration_it)

class EmptyStatementParser(TokenParser):
    def __init__(self):
        super(EmptyStatementParser, self).__init__(tokenize.Punctuator, u";")
    
    def do_parse(self, toks, start):
        ret = super(EmptyStatementParser, self).do_parse(toks, start)
        return None if ret is None else (PTreeEmptyStatementNode(), ret[1])

class ExpressionStatementParser(SeqParser):
    def __init__(self, Expression):
        super(ExpressionStatementParser, self).__init__((
            LookAheadParser(
                AltParser((
                    TokenParser(tokenize.Punctuator, u"{"),
                    TokenParser(tokenize.Keyword, u"function"),
                )),
                inverse=True
            ),
            Expression,
            TokenParser(tokenize.Punctuator, u";")
        ))
    
    def mknode(self, subtree_list):
        return PTreeExpressionStatementNode(expr=subtree_list[1])

class IfStatementParser(SeqParser):
    def __init__(self, Expression, Statement):
        super(IfStatementParser, self).__init__((
            TokenParser(tokenize.Keyword, u"if"),
            TokenParser(tokenize.Punctuator, u"("),
            Expression,
            TokenParser(tokenize.Punctuator, u")"),
            Statement,
            SeqParser((TokenParser(tokenize.Keyword, u"else"), Statement), pick_node=1)
        ),
        optional_right_items=1, node_class=list)
    
    def mknode(self, subtree_list):
        return PTreeIfStatementNode(
            condition=subtree_list[2],
            iftrue=subtree_list[4],
            iffalse=subtree_list[5],
        )

class WhileStatementParser(AltParser):
    def __init__(self, Expression, Statement):
        
        class PreWhileStatementParser(SeqParser):
            def __init__(self):
                super(PreWhileStatementParser, self).__init__((
                    TokenParser(tokenize.Keyword, u"while"),
                    TokenParser(tokenize.Punctuator, u"("),
                    Expression,
                    TokenParser(tokenize.Punctuator, u")"),
                    Statement
                ))
            def mknode(self, subtree_list):
                return PTreeWhileStatementNode(condition=subtree_list[2], statement=subtree_list[4], is_postcondition=False)
        
        class PostWhileStatementParser(SeqParser):
            def __init__(self):
                super(PostWhileStatementParser, self).__init__((
                    TokenParser(tokenize.Keyword, u"do"),
                    Statement,
                    TokenParser(tokenize.Keyword, u"while"),
                    TokenParser(tokenize.Punctuator, u"("),
                    Expression,
                    TokenParser(tokenize.Punctuator, u")"),
                    TokenParser(tokenize.Punctuator, u";"),
                ))
            def mknode(self, subtree_list):
                return PTreeWhileStatementNode(condition=subtree_list[4], statement=subtree_list[1], is_postcondition=True)
        
        super(WhileStatementParser, self).__init__((PreWhileStatementParser(), PostWhileStatementParser()))

class ForStatementParser(AltParser):
    def __init__(self, ExpressionNoIn, Expression, Statement, VariableDeclarationListNoIn, LeftHandSideExpression,
        VariableDeclarationNoIn):
        
        class ClassicForStatementParser(SeqParser):
            def __init__(self):
                super(ClassicForStatementParser, self).__init__((
                    TokenParser(tokenize.Keyword, u"for"),
                    TokenParser(tokenize.Punctuator, u"("),
                    AltParser((
                        SeqParser((TokenParser(tokenize.Keyword, u"var"), VariableDeclarationListNoIn), pick_node=1),
                        OptParser(ExpressionNoIn)
                    )),
                    TokenParser(tokenize.Punctuator, u";"),
                    OptParser(Expression),
                    TokenParser(tokenize.Punctuator, u";"),
                    OptParser(Expression),
                    TokenParser(tokenize.Punctuator, u")"),
                    Statement
                ))
            def mknode(self, subtree_list):
                return PTreeForStatementNode(expr_first=subtree_list[2], expr_second=subtree_list[4], 
                    expr_third=subtree_list[6], statement=subtree_list[8])
        
        class ForInStatementParser(SeqParser):
            def __init__(self):
                super(ForInStatementParser, self).__init__((
                    TokenParser(tokenize.Keyword, u"for"),
                    TokenParser(tokenize.Punctuator, u"("),
                    AltParser((
                        SeqParser((TokenParser(tokenize.Keyword, u"var"), VariableDeclarationNoIn), pick_node=1),
                        OptParser(LeftHandSideExpression)
                    )),
                    TokenParser(tokenize.Keyword, u"in"),
                    OptParser(Expression),
                    TokenParser(tokenize.Punctuator, u")"),
                    Statement
                ))
            def mknode(self, subtree_list):
                if type(subtree_list[2]) == tuple:
                    subtree_list[2] = PTreeVariableDeclarationListNode(declaration_it=(subtree_list[2], ))  # var
                return PTreeForInStatementNode(expr_first=subtree_list[2], expr_second=subtree_list[4], 
                    statement=subtree_list[6])
        
        super(ForStatementParser, self).__init__((ClassicForStatementParser(), ForInStatementParser()))

class ControlStatementParser(SeqParser):
    def __init__(self, ctrl_keyword, ctrl_option_parser, node_class, opt_required=False):
        super(ControlStatementParser, self).__init__((
            TokenParser(tokenize.Keyword, ctrl_keyword),
            LookAheadParser(TokenParser(tokenize.LineTerminator, keep_line_terms=True), inverse=True),
            ctrl_option_parser if opt_required else OptParser(ctrl_option_parser),
            TokenParser(tokenize.Punctuator, u";")
        ))
        self.node_class = node_class
    
    def mknode(self, subtree_list):
        return self.node_class(subtree_list[2])

class WithStatementParser(SeqParser):
    def __init__(self, Expression, Statement):
        super(WithStatementParser, self).__init__((
            TokenParser(tokenize.Keyword, u"with"),
            TokenParser(tokenize.Punctuator, u"("),
            Expression,
            TokenParser(tokenize.Punctuator, u")"),
            Statement
        ))
    
    def mknode(self, subtree_list):
        return PTreeWithStatementNode(expr=subtree_list[2], statement=subtree_list[4])

class SwitchStatementParser(SeqParser):
    def __init__(self, Expression, Statement):
        
        class CaseNDefaultClause(AltParser):
            def __init__(self):
                StatementList = StatementListParser(Statement)
                super(CaseNDefaultClause, self).__init__((
                    SeqParser((
                        TokenParser(tokenize.Keyword, u"case"),
                        Expression,
                        TokenParser(tokenize.Punctuator, u":"),
                        StatementList
                    ), node_class=list),
                    SeqParser((
                        TokenParser(tokenize.Keyword, u"default"),
                        TokenParser(tokenize.Punctuator, u":"),
                        StatementList
                    ), node_class=list)
                ))
            
            def change_node(self, ptree):
                if ptree[0].token.data == u"case":
                    return (ptree[1], ptree[3])
                else:
                    return (ptree[0], ptree[2])
        
        CaseClauses = RepeatedParser(CaseNDefaultClause(), min_matches=0, node_class=list)
        super(SwitchStatementParser, self).__init__((
            TokenParser(tokenize.Keyword, u"switch"),
            TokenParser(tokenize.Punctuator, u"("),
            Expression,
            TokenParser(tokenize.Punctuator, u")"),
            TokenParser(tokenize.Punctuator, u"{"),
            CaseClauses,
            TokenParser(tokenize.Punctuator, u"}"),
        ))

    def mknode(self, subtree_list):
        return PTreeSwitchStatementNode(expr=subtree_list[2], case_block=subtree_list[5])

class LabelledStatementParser(SeqParser):
    def __init__(self, Statement):
        super(LabelledStatementParser, self).__init__((
            TokenParser(tokenize.Identifier),
            TokenParser(tokenize.Punctuator, u":"),
            Statement))

    def mknode(self, subtree_list):
        subtree_list[2].labels.add(subtree_list[0].token.data)
        return subtree_list[2]

class TryStatementParser(SeqParser):
    def __init__(self, Block):
        
        class CatchParser(SeqParser):
            def __init__(self):
                super(CatchParser, self).__init__((
                    TokenParser(tokenize.Keyword, u"catch"),
                    TokenParser(tokenize.Punctuator, u"("),
                    TokenParser(tokenize.Identifier),
                    TokenParser(tokenize.Punctuator, u")"),
                    Block))

            def mknode(self, subtree_list):
                return (subtree_list[2], subtree_list[4])
        
        Catch = CatchParser()
        Finally = SeqParser((
            TokenParser(tokenize.Keyword, u"finally"),
            Block), pick_node=1)
        
        super(TryStatementParser, self).__init__((
            TokenParser(tokenize.Keyword, u"try"),
            Block,
            AltParser((
                SeqParser((Catch, Finally), node_class=list),   # list
                Catch,  # tuple
                Finally # node
            ))), node_class=list)

    def mknode(self, subtree_list):
        block = subtree_list[1]
        catch_finally = subtree_list[2]
        if isinstance(catch_finally, list):
            catch_identifier = catch_finally[0][0]
            catch_block = catch_finally[0][1]
            finally_block = catch_finally[1]
        elif isinstance(catch_finally, tuple):
            catch_identifier = catch_finally[0]
            catch_block = catch_finally[1]
            finally_block = None
        elif isinstance(catch_finally, PTreeNode):
            catch_identifier = None
            catch_block = None
            finally_block = catch_finally
        else:
            assert False
        
        return PTreeTryStatementNode(block=block, catch_identifier=catch_identifier,
            catch_block=catch_block, finally_block=finally_block)

class SourceElementsParser(RepeatedParser):
    def __init__(self, Statement, FunctionDeclaration):
        super(SourceElementsParser, self).__init__(
            AltParser((FunctionDeclaration, Statement)), node_class=PTreeSourceElementsNode)

class FunctionParser(SeqParser):
    def __init__(self, name_parser, FunctionBody, node_class):
        FormalParameterList = RepeatedParser(
            TokenParser(tokenize.Identifier),
            min_matches=0,
            node_class=list,
            separator=TokenParser(tokenize.Punctuator, u",")
        )
        
        super(FunctionParser, self).__init__((
            TokenParser(tokenize.Keyword, u"function"),
            name_parser,
            TokenParser(tokenize.Punctuator, u"("),
            FormalParameterList,
            TokenParser(tokenize.Punctuator, u")"),
            TokenParser(tokenize.Punctuator, u"{"),
            FunctionBody,
            TokenParser(tokenize.Punctuator, u"}")),
            node_class=node_class)

    def mknode(self, subtree_list):
        return self.node_class(name=subtree_list[1], formal_params=subtree_list[3], body=subtree_list[6])


class PTreeNode(object):
    """
    Base class of hierarchy of parse tree nodes.
    The most important goal of this hierarchy is to represent script code in a machine-friendly way in python code.
    *dump* methods have purely demonstrative purpose.
    """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def dump(self, level):
        raise NotImplementedError()
    
    # aux
    def _dump_subnodes(self, level, subnodes):
        for subnode in subnodes if isinstance(subnodes, (list, tuple)) else (subnodes, ):
            for (slevel, node, item) in subnode.dump(level):
                yield (slevel, node, item)
    
    @staticmethod
    def simple_dump(ptree, fh=None):
        if fh is None:
            fh = sys.stdout
        for (level, node, item) in ptree.dump(0):
            print >> fh, "{:32s} {}{}".format(node.__class__.__name__, '    '*level, unicode(item).encode('UTF-8'))


class PTreeTokenNode(PTreeNode):
    def __init__(self, token):
        self.token = token

    def dump(self, level):
        yield (level, self, self.token)

class PTreeEmptyArrayElementNode(PTreeNode):
    def dump(self, level):
        yield (level, self, u"<empty element>")

class PTreeListNode(PTreeNode):
    "aux class"
    def __init__(self, item_list, opening_bracket=None, closing_bracket=None, **kwargs):
        super(PTreeListNode, self).__init__(**kwargs)
        self.item_list = item_list
        self.opening_bracket = opening_bracket
        self.closing_bracket = closing_bracket

    def dump(self, level):
        if self.opening_bracket:
            yield (level, self, self.opening_bracket)
        for (slevel, node, item) in self._dump_subnodes(level + 1, self.item_list):
            yield (slevel, node, item)
        if self.closing_bracket:
            yield (level, self, self.closing_bracket)

class PTreeArrayLiteralNode(PTreeListNode):
    def __init__(self, init_list):
        super(PTreeArrayLiteralNode, self).__init__(item_list=init_list, opening_bracket=u'[', closing_bracket=u']')

class PTreeObjectLiteralNode(PTreeNode):
    def __init__(self, init_list):
        self.init_list = init_list

    def dump(self, level):
        yield (level, self, u'{')
        for init_item in self.init_list:
            for (slevel, node, item) in init_item[0].dump(level + 1):
                yield (slevel, node, item)
            for (slevel, node, item) in init_item[1].dump(level + 2):
                yield (slevel, node, item)
        yield (level, self, u'}')

class PTreeArgumentsNode(PTreeListNode):
    def __init__(self, arg_list):
        super(PTreeArgumentsNode, self).__init__(item_list=arg_list, opening_bracket=u'(', closing_bracket=u')')

class PTreeCtorNode(PTreeNode):
    def __init__(self, ctor_fn, arguments):
        self.ctor_fn = ctor_fn
        self.arguments = arguments

    def dump(self, level):
        yield (level, self, u'new')
        for (slevel, node, item) in self.ctor_fn.dump(level + 1):
            yield (slevel, node, item)
        for (slevel, node, item) in self.arguments.dump(level + 2):
            yield (slevel, node, item)

class PTreeMemberSelectorsNode(PTreeNode):
    def __init__(self, obj, selectors):
        self.obj = obj
        self.selectors = selectors
        assert selectors

    def dump(self, level):
        for (slevel, node, item) in self.obj.dump(level):
            yield (slevel, node, item)
        for selector in self.selectors:
            yield (level + 1, self, u'.')
            for (slevel, node, item) in selector.dump(level + 2):
                yield (slevel, node, item)

class PTreeCallNode(PTreeNode):
    def __init__(self, call_expr, arguments):
        self.call_expr = call_expr
        self.arguments = arguments

    def dump(self, level):
        yield (level, self, u'call')
        for (slevel, node, item) in self.call_expr.dump(level + 1):
            yield (slevel, node, item)
        for (slevel, node, item) in self.arguments.dump(level + 2):
            yield (slevel, node, item)

class PTreeXfixExpressionNode(PTreeNode):
    "aux class"
    def __init__(self, expr, op):
        self.expr = expr
        self.op = op

    def dump(self, level):
        yield (level, self, self.op.token.data)
        for (slevel, node, item) in self.expr.dump(level + 1):
            yield (slevel, node, item)

class PTreePostfixExpressionNode(PTreeXfixExpressionNode):
    pass

class PTreeUnaryExpressionNode(PTreeXfixExpressionNode):
    pass

class PTreeBinaryOpNode(PTreeNode):
    def __init__(self, op, arg_list):
        self.op = op
        self.arg_list = arg_list

    def dump(self, level):
        yield (level, self, self.op.token.data)
        for (slevel, node, item) in self._dump_subnodes(level + 1, self.arg_list):
            yield (slevel, node, item)

class PTreeConditionalExpressionNode(PTreeNode):
    def __init__(self, cond, iftrue, iffalse):
        self.cond = cond
        self.iftrue = iftrue
        self.iffalse = iffalse

    def dump(self, level):
        yield (level, self, u"?")
        for (slevel, node, item) in self.cond.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u"iftrue")
        for (slevel, node, item) in self.iftrue.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u"iffalse")
        for (slevel, node, item) in self.iffalse.dump(level + 1):
            yield (slevel, node, item)

class PTreeAssignmentExpressionNode(PTreeNode):
    def __init__(self, target, op, source):
        self.target = target
        self.op = op
        self.source = source

    def dump(self, level):
        yield (level, self, self.op.token.data)
        for (slevel, node, item) in self.target.dump(level + 1):
            yield (slevel, node, item)
        for (slevel, node, item) in self.source.dump(level + 1):
            yield (slevel, node, item)

class PTreeStatementNode(PTreeNode):
    def __init__(self, labels=None, **kwargs):
        super(PTreeStatementNode, self).__init__(**kwargs)
        self.labels = set() if labels is None else set(labels)

class PTreeBlockNode(PTreeListNode, PTreeStatementNode):
    def __init__(self, st_list):
        super(PTreeBlockNode, self).__init__(item_list=st_list, opening_bracket=u'{', closing_bracket=u'}')

class PTreeVariableDeclarationListNode(PTreeNode):
    def __init__(self, declaration_it, **kwargs):
        self.declaration_it = declaration_it    # list of (Identifier, Initializer) pairs

    def dump(self, level):
        yield (level, self, u"var")
        for decl in self.declaration_it:
            for (slevel, node, item) in decl[0].dump(level + 1):
                yield (slevel, node, item)
            if decl[1] is not None:
                for (slevel, node, item) in decl[1].dump(level + 2):
                    yield (slevel, node, item)

class PTreeVariableStatementNode(PTreeStatementNode, PTreeVariableDeclarationListNode):
    pass

class PTreeEmptyStatementNode(PTreeStatementNode):
    def dump(self, level):
        yield (level, self, u";")

class PTreeExpressionStatementNode(PTreeStatementNode):
    def __init__(self, expr, **kwargs):
        super(PTreeExpressionStatementNode, self).__init__(**kwargs)
        self.expr = expr

    def dump(self, level):
        yield (level, self, u";")
        for (slevel, node, item) in self.expr.dump(level + 1):
            yield (slevel, node, item)

class PTreeIfStatementNode(PTreeStatementNode):
    def __init__(self, condition, iftrue, iffalse=None, **kwargs):
        super(PTreeIfStatementNode, self).__init__(**kwargs)
        self.condition = condition
        self.iftrue = iftrue
        self.iffalse = iffalse

    def dump(self, level):
        yield (level, self, u"if")
        for (slevel, node, item) in self.condition.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u"then")
        for (slevel, node, item) in self.iftrue.dump(level + 1):
            yield (slevel, node, item)
        if self.iffalse is not None:
            yield (level, self, u"else")
            for (slevel, node, item) in self.iffalse.dump(level + 1):
                yield (slevel, node, item)

class PTreeWhileStatementNode(PTreeStatementNode):
    def __init__(self, condition, statement, is_postcondition, **kwargs):
        super(PTreeWhileStatementNode, self).__init__(labels=(u'empty', ), **kwargs)
        self.condition = condition
        self.statement = statement
        self.is_postcondition = is_postcondition

    def dump(self, level):
        if self.is_postcondition:
            (pre, first, post, second) = (u"do", self.statement, u"while", self.condition)
        else:
            (pre, first, post, second) = (u"while", self.condition, u"do", self.statement)
        for (codeword, subnode) in ((pre, first), (post, second)):
            yield (level, self, codeword)
            for (slevel, node, item) in subnode.dump(level + 1):
                yield (slevel, node, item)

class PTreeForStatementNode(PTreeStatementNode):
    def __init__(self, expr_first, expr_second, expr_third, statement, **kwargs):
        super(PTreeForStatementNode, self).__init__(labels=(u'empty', ), **kwargs)
        self.expr_first = expr_first
        self.expr_second = expr_second
        self.expr_third = expr_third
        self.statement = statement

    def dump(self, level):
        yield (level, self, u'for (')
        for (idx, expr) in enumerate((self.expr_first, self.expr_second, self.expr_third)):
            if idx:
                yield (level, self, u';')
            if expr:
                for (slevel, node, item) in expr.dump(level + 1):
                    yield (slevel, node, item)
        yield (level, self, u')')
        for (slevel, node, item) in self.statement.dump(level + 1):
            yield (slevel, node, item)

class PTreeForInStatementNode(PTreeStatementNode):
    def __init__(self, expr_first, expr_second, statement, **kwargs):
        super(PTreeForInStatementNode, self).__init__(labels=(u'empty', ), **kwargs)
        self.expr_first = expr_first
        self.expr_second = expr_second
        self.statement = statement

    def dump(self, level):
        yield (level, self, u'for (')
        for (slevel, node, item) in self.expr_first.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u'in')
        for (slevel, node, item) in self.expr_second.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u')')
        for (slevel, node, item) in self.statement.dump(level + 1):
            yield (slevel, node, item)

class PTreeControlStatementNode(PTreeStatementNode):
    def __init__(self, ctrl_type, ctrl_option=None, **kwargs):
        super(PTreeControlStatementNode, self).__init__(**kwargs)
        self.ctrl_type = ctrl_type
        self.ctrl_option = ctrl_option

    def dump(self, level):
        yield (level, self, self.ctrl_type)
        if self.ctrl_option is not None:
            for (slevel, node, item) in self.ctrl_option.dump(level + 1):
                yield (slevel, node, item)

class PTreeContinueStatementNode(PTreeControlStatementNode):
    def __init__(self, ctrl_option):
        super(PTreeContinueStatementNode, self).__init__(ctrl_type=u'continue', ctrl_option=ctrl_option)

class PTreeBreakStatementNode(PTreeControlStatementNode):
    def __init__(self, ctrl_option):
        super(PTreeBreakStatementNode, self).__init__(ctrl_type=u'break', ctrl_option=ctrl_option)

class PTreeReturnStatementNode(PTreeControlStatementNode):
    def __init__(self, ctrl_option):
        super(PTreeReturnStatementNode, self).__init__(ctrl_type=u'return', ctrl_option=ctrl_option)

class PTreeThrowStatementNode(PTreeControlStatementNode):
    def __init__(self, ctrl_option):
        assert ctrl_option is not None
        super(PTreeThrowStatementNode, self).__init__(ctrl_type=u'throw', ctrl_option=ctrl_option)

class PTreeWithStatementNode(PTreeStatementNode):
    def __init__(self, expr, statement, **kwargs):
        super(PTreeWithStatementNode, self).__init__(**kwargs)
        self.expr = expr
        self.statement = statement

    def dump(self, level):
        yield (level, self, u"with (")
        for (slevel, node, item) in self.expr.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u")")
        for (slevel, node, item) in self.statement.dump(level + 1):
            yield (slevel, node, item)

class PTreeSwitchStatementNode(PTreeStatementNode):
    def __init__(self, expr, case_block, **kwargs):
        super(PTreeSwitchStatementNode, self).__init__(**kwargs)
        self.expr = expr
        assert type(case_block) == list
        self.case_block = case_block    # list of (case expr, StatementList)

    def dump(self, level):
        yield (level, self, u"switch (")
        for (slevel, node, item) in self.expr.dump(level + 1):
            yield (slevel, node, item)
        yield (level, self, u")")
        
        for case in self.case_block:
            assert type(case) == tuple
            yield (level + 1, self, u"case")
            for (slevel, node, item) in case[0].dump(level + 2):
                yield (slevel, node, item)
            yield (level + 1, self, u":")
            assert type(case[1]) == list
            for (slevel, node, item) in self._dump_subnodes(level + 2, case[1]):
                yield (slevel, node, item)

class PTreeTryStatementNode(PTreeStatementNode):
    def __init__(self, block, catch_identifier, catch_block, finally_block, **kwargs):
        super(PTreeTryStatementNode, self).__init__(**kwargs)
        assert catch_block is not None or finally_block is not None
        assert (catch_identifier is None) == (catch_block is None)
        self.block = block
        self.catch_identifier = catch_identifier
        self.catch_block = catch_block
        self.finally_block = finally_block

    def dump(self, level):
        yield (level, self, u"try")
        for (slevel, node, item) in self.block.dump(level + 1):
            yield (slevel, node, item)
        if self.catch_block is not None:
            yield (level, self, u"catch")
            for (slevel, node, item) in self.catch_identifier.dump(level + 1):
                yield (slevel, node, item)
            for (slevel, node, item) in self.catch_block.dump(level + 1):
                yield (slevel, node, item)
        if self.finally_block is not None:
            yield (level, self, u"finally")
            for (slevel, node, item) in self.finally_block.dump(level + 1):
                yield (slevel, node, item)

class PTreeSourceElementsNode(PTreeListNode):
    def __init__(self, item_list):
        super(PTreeSourceElementsNode, self).__init__(item_list=item_list)

class PTreeFunctionBaseNode(PTreeStatementNode):
    def __init__(self, name, formal_params, body, **kwargs):
        super(PTreeFunctionBaseNode, self).__init__(**kwargs)
        self.name = name
        self.formal_params = formal_params
        self.body = body

    def dump(self, level):
        yield (level, self, u"function")
        if self.name is not None:
            for (slevel, node, item) in self.name.dump(level + 1):
                yield (slevel, node, item)
        yield (level, self, u"(")
        for (slevel, node, item) in self._dump_subnodes(level + 1, self.formal_params):
            yield (slevel, node, item)
        yield (level, self, u")")
        yield (level, self, u"{")
        for (slevel, node, item) in self.body.dump(level):
            yield (slevel, node, item)
        yield (level, self, u"}")

class PTreeFunctionDeclarationNode(PTreeFunctionBaseNode):
    def __init__(self, **kwargs):
        super(PTreeFunctionDeclarationNode, self).__init__(**kwargs)
        if self.name is None:
            raise ValueError('function declaration without a name')

class PTreeFunctionExpressionNode(PTreeFunctionBaseNode):
    pass

