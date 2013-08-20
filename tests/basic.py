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


    def test_statements(self):
        self.check_parser("""
{
    // Set our document
    document = doc;
    docElem = doc.documentElement;

    // Support tests
    documentIsHTML = !isXML( doc );
}""", """
PTreeBlockNode                       {
PTreeExpressionStatementNode             ;
PTreeAssignmentExpressionNode                =
PTreeTokenNode                                   <Identifier: u'document'>
PTreeTokenNode                                   <Identifier: u'doc'>
PTreeExpressionStatementNode             ;
PTreeAssignmentExpressionNode                =
PTreeTokenNode                                   <Identifier: u'docElem'>
PTreeTokenNode                                   <Identifier: u'doc'>
PTreeMemberSelectorsNode                             .
PTreeTokenNode                                           <StringLiteral: u'documentElement'>
PTreeExpressionStatementNode             ;
PTreeAssignmentExpressionNode                =
PTreeTokenNode                                   <Identifier: u'documentIsHTML'>
PTreeUnaryExpressionNode                         !
PTreeCallNode                                        call
PTreeTokenNode                                           <Identifier: u'isXML'>
PTreeArgumentsNode                                           (
PTreeTokenNode                                                   <Identifier: u'doc'>
PTreeArgumentsNode                                           )
PTreeBlockNode                       }""")

        self.check_parser("""
var doc = node ? node.ownerDocument || node : preferredDoc,
    parent = doc.defaultView;
var j,
    matchIndexes = fn( [], seed.length, argument ),
    i = matchIndexes.length;""", """
PTreeVariableStatementNode           var
PTreeTokenNode                           <Identifier: u'doc'>
PTreeConditionalExpressionNode               ?
PTreeTokenNode                                   <Identifier: u'node'>
PTreeConditionalExpressionNode               iftrue
PTreeBinaryOpNode                                ||
PTreeTokenNode                                       <Identifier: u'node'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <StringLiteral: u'ownerDocument'>
PTreeTokenNode                                       <Identifier: u'node'>
PTreeConditionalExpressionNode               iffalse
PTreeTokenNode                                   <Identifier: u'preferredDoc'>
PTreeTokenNode                           <Identifier: u'parent'>
PTreeTokenNode                               <Identifier: u'doc'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'defaultView'>
PTreeVariableStatementNode           var
PTreeTokenNode                           <Identifier: u'j'>
PTreeTokenNode                           <Identifier: u'matchIndexes'>
PTreeCallNode                                call
PTreeTokenNode                                   <Identifier: u'fn'>
PTreeArgumentsNode                                   (
PTreeArrayLiteralNode                                    [
PTreeArrayLiteralNode                                    ]
PTreeTokenNode                                           <Identifier: u'seed'>
PTreeMemberSelectorsNode                                     .
PTreeTokenNode                                                   <StringLiteral: u'length'>
PTreeTokenNode                                           <Identifier: u'argument'>
PTreeArgumentsNode                                   )
PTreeTokenNode                           <Identifier: u'i'>
PTreeTokenNode                               <Identifier: u'matchIndexes'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'length'>""")

        self.check_parser(";", """
PTreeEmptyStatementNode              ;""")

        self.check_parser("""
if ( postFinder ) {
    postFinder( null, results, matcherOut, xml );
} else {
    push.apply( results, matcherOut );
}
if ( postFinder )
        // Get the final matcherOut by condensing this intermediate into postFinder contexts
        temp = [];""", """
PTreeIfStatementNode                 if
PTreeTokenNode                           <Identifier: u'postFinder'>
PTreeIfStatementNode                 then
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeCallNode                                    call
PTreeTokenNode                                       <Identifier: u'postFinder'>
PTreeArgumentsNode                                       (
PTreeTokenNode                                               <NullLiteral: u'null'>
PTreeTokenNode                                               <Identifier: u'results'>
PTreeTokenNode                                               <Identifier: u'matcherOut'>
PTreeTokenNode                                               <Identifier: u'xml'>
PTreeArgumentsNode                                       )
PTreeBlockNode                           }
PTreeIfStatementNode                 else
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeCallNode                                    call
PTreeTokenNode                                       <Identifier: u'push'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <StringLiteral: u'apply'>
PTreeArgumentsNode                                       (
PTreeTokenNode                                               <Identifier: u'results'>
PTreeTokenNode                                               <Identifier: u'matcherOut'>
PTreeArgumentsNode                                       )
PTreeBlockNode                           }
PTreeIfStatementNode                 if
PTreeTokenNode                           <Identifier: u'postFinder'>
PTreeIfStatementNode                 then
PTreeExpressionStatementNode             ;
PTreeAssignmentExpressionNode                =
PTreeTokenNode                                   <Identifier: u'temp'>
PTreeArrayLiteralNode                            [
PTreeArrayLiteralNode                            ]""")

        self.check_parser("""
while ( i-- ) {
    if ( !matchers[i]( elem, context, xml ) ) {
        return false;
    }
}
do {
    cur = cur[ dir ];
} while ( cur && cur.nodeType !== 1 );""", """
PTreeWhileStatementNode              while
PTreePostfixExpressionNode               --
PTreeTokenNode                               <Identifier: u'i'>
PTreeWhileStatementNode              do
PTreeBlockNode                           {
PTreeIfStatementNode                         if
PTreeUnaryExpressionNode                         !
PTreeCallNode                                        call
PTreeTokenNode                                           <Identifier: u'matchers'>
PTreeMemberSelectorsNode                                     .
PTreeTokenNode                                                   <Identifier: u'i'>
PTreeArgumentsNode                                           (
PTreeTokenNode                                                   <Identifier: u'elem'>
PTreeTokenNode                                                   <Identifier: u'context'>
PTreeTokenNode                                                   <Identifier: u'xml'>
PTreeArgumentsNode                                           )
PTreeIfStatementNode                         then
PTreeBlockNode                                   {
PTreeReturnStatementNode                             return
PTreeTokenNode                                           <BooleanLiteral: u'false'>
PTreeBlockNode                                   }
PTreeBlockNode                           }
PTreeWhileStatementNode              do
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'cur'>
PTreeTokenNode                                       <Identifier: u'cur'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <Identifier: u'dir'>
PTreeBlockNode                           }
PTreeWhileStatementNode              while
PTreeBinaryOpNode                        &&
PTreeTokenNode                               <Identifier: u'cur'>
PTreeBinaryOpNode                            !==
PTreeTokenNode                                   <Identifier: u'cur'>
PTreeMemberSelectorsNode                             .
PTreeTokenNode                                           <StringLiteral: u'nodeType'>
PTreeTokenNode                                   <NumericLiteral: u'1'>""")

        self.check_parser("""
for ( ; n; n = n.nextSibling ) {
    if ( n.nodeType === 1 && n !== elem ) {
        r.push( n );
    }
}
for ( e in data.events ) {
    jQuery.removeEvent( dest, e, data.handle );
}""", """
PTreeForStatementNode                for (
PTreeForStatementNode                ;
PTreeTokenNode                           <Identifier: u'n'>
PTreeForStatementNode                ;
PTreeAssignmentExpressionNode            =
PTreeTokenNode                               <Identifier: u'n'>
PTreeTokenNode                               <Identifier: u'n'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'nextSibling'>
PTreeForStatementNode                )
PTreeBlockNode                           {
PTreeIfStatementNode                         if
PTreeBinaryOpNode                                &&
PTreeBinaryOpNode                                    ===
PTreeTokenNode                                           <Identifier: u'n'>
PTreeMemberSelectorsNode                                     .
PTreeTokenNode                                                   <StringLiteral: u'nodeType'>
PTreeTokenNode                                           <NumericLiteral: u'1'>
PTreeBinaryOpNode                                    !==
PTreeTokenNode                                           <Identifier: u'n'>
PTreeTokenNode                                           <Identifier: u'elem'>
PTreeIfStatementNode                         then
PTreeBlockNode                                   {
PTreeExpressionStatementNode                         ;
PTreeCallNode                                            call
PTreeTokenNode                                               <Identifier: u'r'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'push'>
PTreeArgumentsNode                                               (
PTreeTokenNode                                                       <Identifier: u'n'>
PTreeArgumentsNode                                               )
PTreeBlockNode                                   }
PTreeBlockNode                           }
PTreeForInStatementNode              for (
PTreeTokenNode                           <Identifier: u'e'>
PTreeForInStatementNode              in
PTreeTokenNode                           <Identifier: u'data'>
PTreeMemberSelectorsNode                     .
PTreeTokenNode                                   <StringLiteral: u'events'>
PTreeForInStatementNode              )
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeCallNode                                    call
PTreeTokenNode                                       <Identifier: u'jQuery'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <StringLiteral: u'removeEvent'>
PTreeArgumentsNode                                       (
PTreeTokenNode                                               <Identifier: u'dest'>
PTreeTokenNode                                               <Identifier: u'e'>
PTreeTokenNode                                               <Identifier: u'data'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'handle'>
PTreeArgumentsNode                                       )
PTreeBlockNode                           }""")

        self.check_parser("""
if ( selection )
    continue;
if ( value === false ) {
    break;
}
return text == null ?
    "" :
    core_trim.call( text );""", """
PTreeIfStatementNode                 if
PTreeTokenNode                           <Identifier: u'selection'>
PTreeIfStatementNode                 then
PTreeContinueStatementNode               continue
PTreeIfStatementNode                 if
PTreeBinaryOpNode                        ===
PTreeTokenNode                               <Identifier: u'value'>
PTreeTokenNode                               <BooleanLiteral: u'false'>
PTreeIfStatementNode                 then
PTreeBlockNode                           {
PTreeBreakStatementNode                      break
PTreeBlockNode                           }
PTreeReturnStatementNode             return
PTreeConditionalExpressionNode           ?
PTreeBinaryOpNode                            ==
PTreeTokenNode                                   <Identifier: u'text'>
PTreeTokenNode                                   <NullLiteral: u'null'>
PTreeConditionalExpressionNode           iftrue
PTreeTokenNode                               <StringLiteral: u'""'>
PTreeConditionalExpressionNode           iffalse
PTreeCallNode                                call
PTreeTokenNode                                   <Identifier: u'core_trim'>
PTreeMemberSelectorsNode                             .
PTreeTokenNode                                           <StringLiteral: u'call'>
PTreeArgumentsNode                                   (
PTreeTokenNode                                           <Identifier: u'text'>
PTreeArgumentsNode                                   )""")

        self.check_parser("""
with (Math) {
  a = PI * r * r;
  x = r * cos(PI);
  y = r * sin(PI * 0.5);
}""", """
PTreeWithStatementNode               with (
PTreeTokenNode                           <Identifier: u'Math'>
PTreeWithStatementNode               )
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'a'>
PTreeBinaryOpNode                                    *
PTreeBinaryOpNode                                        *
PTreeTokenNode                                               <Identifier: u'PI'>
PTreeTokenNode                                               <Identifier: u'r'>
PTreeTokenNode                                           <Identifier: u'r'>
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'x'>
PTreeBinaryOpNode                                    *
PTreeTokenNode                                           <Identifier: u'r'>
PTreeCallNode                                            call
PTreeTokenNode                                               <Identifier: u'cos'>
PTreeArgumentsNode                                               (
PTreeTokenNode                                                       <Identifier: u'PI'>
PTreeArgumentsNode                                               )
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'y'>
PTreeBinaryOpNode                                    *
PTreeTokenNode                                           <Identifier: u'r'>
PTreeCallNode                                            call
PTreeTokenNode                                               <Identifier: u'sin'>
PTreeArgumentsNode                                               (
PTreeBinaryOpNode                                                    *
PTreeTokenNode                                                           <Identifier: u'PI'>
PTreeTokenNode                                                           <NumericLiteral: u'0.5'>
PTreeArgumentsNode                                               )
PTreeBlockNode                           }""")

        self.check_parser("""
var day=new Date().getDay();
 switch (day)
 {
 case 6:
   x="Today it's Saturday";
   break;
 case 0:
   x="Today it's Sunday";
   break;
 default:
   x="Looking forward to the Weekend";
 }""", r"""
PTreeVariableStatementNode           var
PTreeTokenNode                           <Identifier: u'day'>
PTreeCallNode                                call
PTreeCtorNode                                    new
PTreeTokenNode                                       <Identifier: u'Date'>
PTreeArgumentsNode                                       (
PTreeArgumentsNode                                       )
PTreeMemberSelectorsNode                             .
PTreeTokenNode                                           <StringLiteral: u'getDay'>
PTreeArgumentsNode                                   (
PTreeArgumentsNode                                   )
PTreeSwitchStatementNode             switch (
PTreeTokenNode                           <Identifier: u'day'>
PTreeSwitchStatementNode             )
PTreeSwitchStatementNode                 case
PTreeTokenNode                               <NumericLiteral: u'6'>
PTreeSwitchStatementNode                 :
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'x'>
PTreeTokenNode                                       <StringLiteral: u'"Today it\'s Saturday"'>
PTreeBreakStatementNode                      break
PTreeSwitchStatementNode                 case
PTreeTokenNode                               <NumericLiteral: u'0'>
PTreeSwitchStatementNode                 :
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'x'>
PTreeTokenNode                                       <StringLiteral: u'"Today it\'s Sunday"'>
PTreeBreakStatementNode                      break
PTreeSwitchStatementNode                 case
PTreeTokenNode                               <Keyword: u'default'>
PTreeSwitchStatementNode                 :
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'x'>
PTreeTokenNode                                       <StringLiteral: u'"Looking forward to the Weekend"'>""")

        self.check_parser("""
top:
for (i = 0; i < items.length; i++){
  for (j = 0; j < tests.length; j++)
    if (!tests[j].pass(items[i]))
      continue top;
  itemsPassed++;
}""", """
PTreeForStatementNode                for (
PTreeAssignmentExpressionNode            =
PTreeTokenNode                               <Identifier: u'i'>
PTreeTokenNode                               <NumericLiteral: u'0'>
PTreeForStatementNode                ;
PTreeBinaryOpNode                        <
PTreeTokenNode                               <Identifier: u'i'>
PTreeTokenNode                               <Identifier: u'items'>
PTreeMemberSelectorsNode                         .
PTreeTokenNode                                       <StringLiteral: u'length'>
PTreeForStatementNode                ;
PTreePostfixExpressionNode               ++
PTreeTokenNode                               <Identifier: u'i'>
PTreeForStatementNode                )
PTreeBlockNode                           {
PTreeForStatementNode                        for (
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'j'>
PTreeTokenNode                                       <NumericLiteral: u'0'>
PTreeForStatementNode                        ;
PTreeBinaryOpNode                                <
PTreeTokenNode                                       <Identifier: u'j'>
PTreeTokenNode                                       <Identifier: u'tests'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <StringLiteral: u'length'>
PTreeForStatementNode                        ;
PTreePostfixExpressionNode                       ++
PTreeTokenNode                                       <Identifier: u'j'>
PTreeForStatementNode                        )
PTreeIfStatementNode                             if
PTreeUnaryExpressionNode                             !
PTreeCallNode                                            call
PTreeTokenNode                                               <Identifier: u'tests'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <Identifier: u'j'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'pass'>
PTreeArgumentsNode                                               (
PTreeTokenNode                                                       <Identifier: u'items'>
PTreeMemberSelectorsNode                                                 .
PTreeTokenNode                                                               <Identifier: u'i'>
PTreeArgumentsNode                                               )
PTreeIfStatementNode                             then
PTreeContinueStatementNode                           continue
PTreeTokenNode                                           <Identifier: u'top'>
PTreeExpressionStatementNode                 ;
PTreePostfixExpressionNode                       ++
PTreeTokenNode                                       <Identifier: u'itemsPassed'>
PTreeBlockNode                           }""")

        self.check_parser("throw e;", """
PTreeThrowStatementNode              throw
PTreeTokenNode                           <Identifier: u'e'>""")

        self.check_parser("""
try {
    response = conv( response );
} catch ( e ) {
    return { state: "parsererror", error: conv ? e : "No conversion from " + prev + " to " + current };
}
try {
    push.apply( results,
        newContext.querySelectorAll( newSelector )
    );
    return results;
} catch(qsaError) {
} finally {
    if ( !old ) {
        context.removeAttribute("id");
    }
}""", """
PTreeTryStatementNode                try
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeAssignmentExpressionNode                    =
PTreeTokenNode                                       <Identifier: u'response'>
PTreeCallNode                                        call
PTreeTokenNode                                           <Identifier: u'conv'>
PTreeArgumentsNode                                           (
PTreeTokenNode                                                   <Identifier: u'response'>
PTreeArgumentsNode                                           )
PTreeBlockNode                           }
PTreeTryStatementNode                catch
PTreeTokenNode                           <Identifier: u'e'>
PTreeBlockNode                           {
PTreeReturnStatementNode                     return
PTreeObjectLiteralNode                           {
PTreeTokenNode                                       <Identifier: u'state'>
PTreeTokenNode                                           <StringLiteral: u'"parsererror"'>
PTreeTokenNode                                       <Identifier: u'error'>
PTreeConditionalExpressionNode                           ?
PTreeTokenNode                                               <Identifier: u'conv'>
PTreeConditionalExpressionNode                           iftrue
PTreeTokenNode                                               <Identifier: u'e'>
PTreeConditionalExpressionNode                           iffalse
PTreeBinaryOpNode                                            +
PTreeBinaryOpNode                                                +
PTreeBinaryOpNode                                                    +
PTreeTokenNode                                                           <StringLiteral: u'"No conversion from "'>
PTreeTokenNode                                                           <Identifier: u'prev'>
PTreeTokenNode                                                       <StringLiteral: u'" to "'>
PTreeTokenNode                                                   <Identifier: u'current'>
PTreeObjectLiteralNode                           }
PTreeBlockNode                           }
PTreeTryStatementNode                try
PTreeBlockNode                           {
PTreeExpressionStatementNode                 ;
PTreeCallNode                                    call
PTreeTokenNode                                       <Identifier: u'push'>
PTreeMemberSelectorsNode                                 .
PTreeTokenNode                                               <StringLiteral: u'apply'>
PTreeArgumentsNode                                       (
PTreeTokenNode                                               <Identifier: u'results'>
PTreeCallNode                                                call
PTreeTokenNode                                                   <Identifier: u'newContext'>
PTreeMemberSelectorsNode                                             .
PTreeTokenNode                                                           <StringLiteral: u'querySelectorAll'>
PTreeArgumentsNode                                                   (
PTreeTokenNode                                                           <Identifier: u'newSelector'>
PTreeArgumentsNode                                                   )
PTreeArgumentsNode                                       )
PTreeReturnStatementNode                     return
PTreeTokenNode                                   <Identifier: u'results'>
PTreeBlockNode                           }
PTreeTryStatementNode                catch
PTreeTokenNode                           <Identifier: u'qsaError'>
PTreeBlockNode                           {
PTreeBlockNode                           }
PTreeTryStatementNode                finally
PTreeBlockNode                           {
PTreeIfStatementNode                         if
PTreeUnaryExpressionNode                         !
PTreeTokenNode                                       <Identifier: u'old'>
PTreeIfStatementNode                         then
PTreeBlockNode                                   {
PTreeExpressionStatementNode                         ;
PTreeCallNode                                            call
PTreeTokenNode                                               <Identifier: u'context'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'removeAttribute'>
PTreeArgumentsNode                                               (
PTreeTokenNode                                                       <StringLiteral: u'"id"'>
PTreeArgumentsNode                                               )
PTreeBlockNode                                   }
PTreeBlockNode                           }""")


    def test_functions(self):
        self.check_parser("""
function markFunction( fn ) {
    fn[ expando ] = true;
    return fn;
}

function createInputPseudo( type ) {
    return function( elem ) {
        var name = elem.nodeName.toLowerCase();
        return name === "input" && elem.type === type;
    };
}""", """
PTreeFunctionDeclarationNode         function
PTreeTokenNode                           <Identifier: u'markFunction'>
PTreeFunctionDeclarationNode         (
PTreeTokenNode                           <Identifier: u'fn'>
PTreeFunctionDeclarationNode         )
PTreeFunctionDeclarationNode         {
PTreeExpressionStatementNode             ;
PTreeAssignmentExpressionNode                =
PTreeTokenNode                                   <Identifier: u'fn'>
PTreeMemberSelectorsNode                             .
PTreeTokenNode                                           <Identifier: u'expando'>
PTreeTokenNode                                   <BooleanLiteral: u'true'>
PTreeReturnStatementNode                 return
PTreeTokenNode                               <Identifier: u'fn'>
PTreeFunctionDeclarationNode         }
PTreeFunctionDeclarationNode         function
PTreeTokenNode                           <Identifier: u'createInputPseudo'>
PTreeFunctionDeclarationNode         (
PTreeTokenNode                           <Identifier: u'type'>
PTreeFunctionDeclarationNode         )
PTreeFunctionDeclarationNode         {
PTreeReturnStatementNode                 return
PTreeFunctionExpressionNode                  function
PTreeFunctionExpressionNode                  (
PTreeTokenNode                                   <Identifier: u'elem'>
PTreeFunctionExpressionNode                  )
PTreeFunctionExpressionNode                  {
PTreeVariableStatementNode                       var
PTreeTokenNode                                       <Identifier: u'name'>
PTreeCallNode                                            call
PTreeTokenNode                                               <Identifier: u'elem'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'nodeName'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'toLowerCase'>
PTreeArgumentsNode                                               (
PTreeArgumentsNode                                               )
PTreeReturnStatementNode                         return
PTreeBinaryOpNode                                    &&
PTreeBinaryOpNode                                        ===
PTreeTokenNode                                               <Identifier: u'name'>
PTreeTokenNode                                               <StringLiteral: u'"input"'>
PTreeBinaryOpNode                                        ===
PTreeTokenNode                                               <Identifier: u'elem'>
PTreeMemberSelectorsNode                                         .
PTreeTokenNode                                                       <StringLiteral: u'type'>
PTreeTokenNode                                               <Identifier: u'type'>
PTreeFunctionExpressionNode                  }
PTreeFunctionDeclarationNode         }""")

    
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
