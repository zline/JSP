#!/usr/bin/env python

import unittest
import cStringIO

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
from jsp import tokenize, parse


class TestBasics(unittest.TestCase):

    def setUp(self):
        tokenize.init_tokenize()
        parse.init_parse()

    
    def test_exprs(self):
        self.check_parser("i;", """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <Identifier: u'i'>""")

        self.check_parser('[];', """
PTreeExpressionStatementNode         ;
PTreeArrayLiteralNode                    [
PTreeArrayLiteralNode                    ]""")
        self.check_parser('[,];', """
PTreeExpressionStatementNode         ;
PTreeArrayLiteralNode                    [
PTreeEmptyArrayElementNode                   <empty element>
PTreeArrayLiteralNode                    ]""")
        self.check_parser('[,,];', """
PTreeExpressionStatementNode         ;
PTreeArrayLiteralNode                    [
PTreeEmptyArrayElementNode                   <empty element>
PTreeEmptyArrayElementNode                   <empty element>
PTreeArrayLiteralNode                    ]""")
        self.check_parser('[5, 7];', """
PTreeExpressionStatementNode         ;
PTreeArrayLiteralNode                    [
PTreeTokenNode                               <NumericLiteral: u'5'>
PTreeTokenNode                               <NumericLiteral: u'7'>
PTreeArrayLiteralNode                    ]""")
        self.check_parser('["foo", x, 0.0];', """
PTreeExpressionStatementNode         ;
PTreeArrayLiteralNode                    [
PTreeTokenNode                               <StringLiteral: u'"foo"'>
PTreeTokenNode                               <Identifier: u'x'>
PTreeTokenNode                               <NumericLiteral: u'0.0'>
PTreeArrayLiteralNode                    ]""")
        self.check_parser('["foo", x, []];', """
PTreeExpressionStatementNode         ;
PTreeArrayLiteralNode                    [
PTreeTokenNode                               <StringLiteral: u'"foo"'>
PTreeTokenNode                               <Identifier: u'x'>
PTreeArrayLiteralNode                        [
PTreeArrayLiteralNode                        ]
PTreeArrayLiteralNode                    ]""")

        self.check_parser('({"foo": "bar"});', """
PTreeExpressionStatementNode         ;
PTreeObjectLiteralNode                   {
PTreeTokenNode                               <StringLiteral: u'"foo"'>
PTreeTokenNode                                   <StringLiteral: u'"bar"'>
PTreeObjectLiteralNode                   }""")
        self.check_parser('({"foo": [0]});', """
PTreeExpressionStatementNode         ;
PTreeObjectLiteralNode                   {
PTreeTokenNode                               <StringLiteral: u'"foo"'>
PTreeArrayLiteralNode                            [
PTreeTokenNode                                       <NumericLiteral: u'0'>
PTreeArrayLiteralNode                            ]
PTreeObjectLiteralNode                   }""")
        self.check_parser('({prop: value});', """
PTreeExpressionStatementNode         ;
PTreeObjectLiteralNode                   {
PTreeTokenNode                               <Identifier: u'prop'>
PTreeTokenNode                                   <Identifier: u'value'>
PTreeObjectLiteralNode                   }""")
        self.check_parser('({"foo": "bar" , "bar": [0], prop: value});', """
PTreeExpressionStatementNode         ;
PTreeObjectLiteralNode                   {
PTreeTokenNode                               <StringLiteral: u'"foo"'>
PTreeTokenNode                                   <StringLiteral: u'"bar"'>
PTreeTokenNode                               <StringLiteral: u'"bar"'>
PTreeArrayLiteralNode                            [
PTreeTokenNode                                       <NumericLiteral: u'0'>
PTreeArrayLiteralNode                            ]
PTreeTokenNode                               <Identifier: u'prop'>
PTreeTokenNode                                   <Identifier: u'value'>
PTreeObjectLiteralNode                   }""")
        
        self.check_parser('this;', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <Keyword: u'this'>""")
        self.check_parser('5;', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <NumericLiteral: u'5'>""")
        self.check_parser('"f o o";', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <StringLiteral: u'"f o o"'>""")
        self.check_parser('null;', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <NullLiteral: u'null'>""")
        self.check_parser('true;', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <BooleanLiteral: u'true'>""")
        self.check_parser('false;', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <BooleanLiteral: u'false'>""")
        self.check_parser('(x);', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <Identifier: u'x'>""")
        self.check_parser(' ( ( y ) ) ;', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <Identifier: u'y'>""")
        self.check_parser('this.a.b[c]["d"].e[0];', """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <Keyword: u'this'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <StringLiteral: u'a'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <StringLiteral: u'b'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <Identifier: u'c'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <StringLiteral: u'"d"'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <StringLiteral: u'e'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <NumericLiteral: u'0'>""")
        self.check_parser('a.b(c)(d).e[0](f);', """
PTreeExpressionStatementNode         ;
PTreeCallNode                            call
PTreeCallNode                                call
PTreeCallNode                                    call
PTreeTokenNode                                       <Identifier: u'a'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <StringLiteral: u'b'>
PTreeArgumentsNode                                       (
PTreeTokenNode                                               <Identifier: u'c'>
PTreeArgumentsNode                                       )
PTreeArgumentsNode                                   (
PTreeTokenNode                                           <Identifier: u'd'>
PTreeArgumentsNode                                   )
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'e'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <NumericLiteral: u'0'>
PTreeArgumentsNode                               (
PTreeTokenNode                                       <Identifier: u'f'>
PTreeArgumentsNode                               )""")
        self.check_parser('new a.b[c];', """
PTreeExpressionStatementNode         ;
PTreeCtorNode                            new
PTreeTokenNode                               <Identifier: u'a'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'b'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <Identifier: u'c'>
PTreeArgumentsNode                               (
PTreeArgumentsNode                               )""")
        self.check_parser('new x(y)(z);', """
PTreeExpressionStatementNode         ;
PTreeCallNode                            call
PTreeCtorNode                                new
PTreeTokenNode                                   <Identifier: u'x'>
PTreeArgumentsNode                                   (
PTreeTokenNode                                           <Identifier: u'y'>
PTreeArgumentsNode                                   )
PTreeArgumentsNode                               (
PTreeTokenNode                                       <Identifier: u'z'>
PTreeArgumentsNode                               )""")
        
        self.check_parser('x++;', """
PTreeExpressionStatementNode         ;
PTreePostfixExpressionNode               ++
PTreeTokenNode                               <Identifier: u'x'>""")
        self.check_parser('y[0]--;', """
PTreeExpressionStatementNode         ;
PTreePostfixExpressionNode               --
PTreeTokenNode                               <Identifier: u'y'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <NumericLiteral: u'0'>""")
        self.check_parser('x++--;', """
PTreeExpressionStatementNode         ;
PTreePostfixExpressionNode               --
PTreePostfixExpressionNode                   ++
PTreeTokenNode                                   <Identifier: u'x'>""")
        self.check_parser('delete x;', """
PTreeExpressionStatementNode         ;
PTreeUnaryExpressionNode                 delete
PTreeTokenNode                               <Identifier: u'x'>""")
        self.check_parser('- x.y;', """
PTreeExpressionStatementNode         ;
PTreeUnaryExpressionNode                 -
PTreeTokenNode                               <Identifier: u'x'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'y'>""")
        self.check_parser('~--x;', """
PTreeExpressionStatementNode         ;
PTreeUnaryExpressionNode                 ~
PTreeUnaryExpressionNode                     --
PTreeTokenNode                                   <Identifier: u'x'>""")
        self.check_parser('y * z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        *
PTreeTokenNode                               <Identifier: u'y'>
PTreeTokenNode                               <Identifier: u'z'>""")
        self.check_parser('x + y * z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        +
PTreeTokenNode                               <Identifier: u'x'>
PTreeBinaryOpNode                            *
PTreeTokenNode                                   <Identifier: u'y'>
PTreeTokenNode                                   <Identifier: u'z'>""")
        self.check_parser('2 << x + y * z << 1;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        <<
PTreeBinaryOpNode                            <<
PTreeTokenNode                                   <NumericLiteral: u'2'>
PTreeBinaryOpNode                                +
PTreeTokenNode                                       <Identifier: u'x'>
PTreeBinaryOpNode                                    *
PTreeTokenNode                                           <Identifier: u'y'>
PTreeTokenNode                                           <Identifier: u'z'>
PTreeTokenNode                               <NumericLiteral: u'1'>""")
        self.check_parser('w < 2 << x + y * z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        <
PTreeTokenNode                               <Identifier: u'w'>
PTreeBinaryOpNode                            <<
PTreeTokenNode                                   <NumericLiteral: u'2'>
PTreeBinaryOpNode                                +
PTreeTokenNode                                       <Identifier: u'x'>
PTreeBinaryOpNode                                    *
PTreeTokenNode                                           <Identifier: u'y'>
PTreeTokenNode                                           <Identifier: u'z'>""")
        self.check_parser('w in 2 << x + y * z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        in
PTreeTokenNode                               <Identifier: u'w'>
PTreeBinaryOpNode                            <<
PTreeTokenNode                                   <NumericLiteral: u'2'>
PTreeBinaryOpNode                                +
PTreeTokenNode                                       <Identifier: u'x'>
PTreeBinaryOpNode                                    *
PTreeTokenNode                                           <Identifier: u'y'>
PTreeTokenNode                                           <Identifier: u'z'>""")
        self.check_parser('true == w < 2 << x + y * z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        ==
PTreeTokenNode                               <BooleanLiteral: u'true'>
PTreeBinaryOpNode                            <
PTreeTokenNode                                   <Identifier: u'w'>
PTreeBinaryOpNode                                <<
PTreeTokenNode                                       <NumericLiteral: u'2'>
PTreeBinaryOpNode                                    +
PTreeTokenNode                                           <Identifier: u'x'>
PTreeBinaryOpNode                                        *
PTreeTokenNode                                               <Identifier: u'y'>
PTreeTokenNode                                               <Identifier: u'z'>""")
        
        self.check_parser('y & z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        &
PTreeTokenNode                               <Identifier: u'y'>
PTreeTokenNode                               <Identifier: u'z'>""")
        self.check_parser('x ^ y & z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        ^
PTreeTokenNode                               <Identifier: u'x'>
PTreeBinaryOpNode                            &
PTreeTokenNode                                   <Identifier: u'y'>
PTreeTokenNode                                   <Identifier: u'z'>""")
        self.check_parser('w | x ^ y & z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        |
PTreeTokenNode                               <Identifier: u'w'>
PTreeBinaryOpNode                            ^
PTreeTokenNode                                   <Identifier: u'x'>
PTreeBinaryOpNode                                &
PTreeTokenNode                                       <Identifier: u'y'>
PTreeTokenNode                                       <Identifier: u'z'>""")
        self.check_parser('y && z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        &&
PTreeTokenNode                               <Identifier: u'y'>
PTreeTokenNode                               <Identifier: u'z'>""")
        self.check_parser('x || y && z;', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        ||
PTreeTokenNode                               <Identifier: u'x'>
PTreeBinaryOpNode                            &&
PTreeTokenNode                                   <Identifier: u'y'>
PTreeTokenNode                                   <Identifier: u'z'>""")
        self.check_parser('true? x || y && z : false;', """
PTreeExpressionStatementNode         ;
PTreeConditionalExpressionNode           ?
PTreeTokenNode                               <BooleanLiteral: u'true'>
PTreeConditionalExpressionNode           iftrue
PTreeBinaryOpNode                            ||
PTreeTokenNode                                   <Identifier: u'x'>
PTreeBinaryOpNode                                &&
PTreeTokenNode                                       <Identifier: u'y'>
PTreeTokenNode                                       <Identifier: u'z'>
PTreeConditionalExpressionNode           iffalse
PTreeTokenNode                               <BooleanLiteral: u'false'>""")
        
        self.check_parser('y = x < i;', """
PTreeExpressionStatementNode         ;
PTreeAssignmentExpressionNode            =
PTreeTokenNode                               <Identifier: u'y'>
PTreeBinaryOpNode                            <
PTreeTokenNode                                   <Identifier: u'x'>
PTreeTokenNode                                   <Identifier: u'i'>""")
        self.check_parser('y *= x < i ;', """
PTreeExpressionStatementNode         ;
PTreeAssignmentExpressionNode            *=
PTreeTokenNode                               <Identifier: u'y'>
PTreeBinaryOpNode                            <
PTreeTokenNode                                   <Identifier: u'x'>
PTreeTokenNode                                   <Identifier: u'i'>""")
        self.check_parser('y *= x < i , foo, "bar";', """
PTreeExpressionStatementNode         ;
PTreeBinaryOpNode                        ,
PTreeBinaryOpNode                            ,
PTreeAssignmentExpressionNode                    *=
PTreeTokenNode                                       <Identifier: u'y'>
PTreeBinaryOpNode                                    <
PTreeTokenNode                                           <Identifier: u'x'>
PTreeTokenNode                                           <Identifier: u'i'>
PTreeTokenNode                                   <Identifier: u'foo'>
PTreeTokenNode                               <StringLiteral: u'"bar"'>""")


    def test_regression(self):
        self.check_parser("/* double \n */ i; /* multi\nline */", """
PTreeExpressionStatementNode         ;
PTreeTokenNode                           <Identifier: u'i'>""")
    
    
    def check_parser(self, source, etalon):
        ptree = parse.parse(tokenize.tokenize(source))
        output = cStringIO.StringIO()
        parse.PTreeNode.simple_dump(ptree, fh=output)
        try:
            self.assertEqual(output.getvalue().strip(), etalon.strip())
        finally:
            output.close()


if __name__ == '__main__':
    unittest.main()
