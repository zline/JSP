
import re
import operator
import unicodedata
from abc import ABCMeta, abstractmethod
from collections import defaultdict


def init_tokenize():
    global _TOKENIZERS
    global LineTerminator_tk

    WhiteSpace_tk = CharRangeTokenizer(WhiteSpace, 
            unichar_by_cat('Zs') + list(u'\u0009\u000B\u000C\u0020\u00A0'))
    LineTerminator_tk = CharRangeTokenizer(LineTerminator, u'\u000A\u000D\u2028\u2029')
    
    MultiLineComment_tk = RegexTokenizer(MultiLineComment,
        u"/\*.*?\*/", re.DOTALL)
    SingleLineComment_tk = SingleLineCommentTokenizer(LineTerminator_tk)
    
    Keyword_tk = KeywordTokenizer(Keyword,
    u"""break case catch continue default delete do else finally for 
        function if in instanceof new return switch this throw try
        typeof var void while with""")
    
    FutureReservedWord_tk = KeywordTokenizer(FutureReservedWord,
    u"""abstract boolean byte char class const debugger double enum
        export extends final float goto implements import int 
        interface long native package private protected public short
        static super synchronized throws transient volatile""")

    NullLiteral_tk = KeywordTokenizer(NullLiteral, u"null")
    BooleanLiteral_tk = KeywordTokenizer(BooleanLiteral, u"true false")
    
    Identifier_tk = IdentifierTokenizer()
    
    Punctuator_tk = KeywordTokenizer(Punctuator,
    u"""{ } ( ) [ ] <= . ; , < > >= == != === !== + - * % ++ -- << >>
        >>> & | ^ ! ~ && || ? : = += -= *= %= <<= >>= >>>= &= |= ^=""")
    DivPunctuator_tk = KeywordTokenizer(DivPunctuator, u"/ /=")

    NumericLiteral_tk = NumericLiteralTokenizer(Identifier_tk)
    StringLiteral_tk = StringLiteralTokenizer(LineTerminator_tk)
    RegularExpressionLiteral_tk = RegularExpressionLiteralTokenizer(LineTerminator_tk, Identifier_tk)
    

    _TOKENIZERS = [
        WhiteSpace_tk,
        LineTerminator_tk,
        
        MultiLineComment_tk,
        SingleLineComment_tk,
        
        # ReservedWord
        Keyword_tk,
        FutureReservedWord_tk,
        NullLiteral_tk,
        BooleanLiteral_tk,
        
        Identifier_tk,
        Punctuator_tk,
        NumericLiteral_tk,
        StringLiteral_tk,
    ]

def tokenize(data):
    global _TOKENIZERS
    
    if not isinstance(data, unicode):
        data = unicode(data, 'utf-8')
    
    formatControlCleaner = CharRangeTokenizer(None, unichar_by_cat('Cf')).re
    data = formatControlCleaner.sub("", data)
    
    start = 0
    end = 0 # points after the last char of the token
    data_len = len(data)
    toks = list()
    while start < data_len:
        best_tkzer = None
        for tkzer in _TOKENIZERS:
            tkzer.set_input(data, start)
            this_end = tkzer.get_next_tok_end()
            assert this_end is None or this_end > start
            # searching for match of the maximum length, as requested by standart
            # in case of equal lengths, the first tokenizer has the highest priority
            if this_end is not None and this_end > end:
                end = this_end
                best_tkzer = tkzer
        
        if best_tkzer is None:
            raise RuntimeError("cant parse starting at byte {}: {}...".format(
                start, data[start : start + 32]))
        
        tok = best_tkzer.get_next_tok()
        # token postprocessing
        if isinstance(tok, MultiLineComment):
            if LineTerminator_tk.re.search(tok.data):
                toks.append(LineTerminator(u"\n"))
        else:
            if not isinstance(tok, (WhiteSpace, SingleLineComment)):
                toks.append(tok)
        
        start = end
    
    assert start == data_len
    return toks


class Tokenizer(object):
    """
    Abstract token parser
    """
    __metaclass__ = ABCMeta
    
    def set_input(self, string, start):
        self.string = string
        self.start = start
        self.end = None
    
    @abstractmethod
    def get_next_tok_end(self):
        """
        Preceeded by set_input
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_next_tok(self):
        """
        Should be preceeded by get_next_tok_end
        """
        raise NotImplementedError()

class RegexTokenizer(Tokenizer):
    def __init__(self, tok_class, _re, re_flags=0):
        super(RegexTokenizer, self).__init__()
        self.tok_class = tok_class
        self.re = re.compile(_re, re_flags|re.U) if isinstance(_re, basestring) else _re

    def get_next_tok_end(self):
        match = self.re.match(self.string, self.start)
        if match is None:
            return None
        
        self.end = match.end()
        return self.end
    
    def get_next_tok(self):
        return self.tok_class(self.string[self.start : self.end])

class CharRangeTokenizer(RegexTokenizer):
    def __init__(self, tok_class, uchars_it):
        self.chars = set(uchars_it)
        if not self.chars:
            raise ValueError("empty uchars_it")
        re_txt = re.escape( u"".join(self.chars) )
        super(CharRangeTokenizer, self).__init__(tok_class, u"[{}]+".format(re_txt))

class KeywordTokenizer(RegexTokenizer):
    def __init__(self, tok_class, keywords_it):
        if isinstance(keywords_it, basestring):
            keywords_it = re.sub(r"\s+", ' ', keywords_it.strip()).split(' ')
        self.keywords = set(keywords_it)
        if not self.keywords:
            raise ValueError("empty keywords_it")
        re_txt = u"|".join( map(re.escape, sorted(self.keywords, key=len, reverse=True)) )
        super(KeywordTokenizer, self).__init__(tok_class, re_txt)


class SingleLineCommentTokenizer(RegexTokenizer):
    def __init__(self, line_term_tk):
        eol_re_txt = re.escape( u"".join(line_term_tk.chars) )
        super(SingleLineCommentTokenizer, self).__init__(
            SingleLineComment, u"//[^{}]*".format(eol_re_txt))

class IdentifierTokenizer(RegexTokenizer):
    def __init__(self):
        UnicodeLetter_chars = unichar_by_categories('Lu Ll Lt Lm Lo Nl'.split(' '))
        self.IdentifierStart_chars_re_txt = re.escape( u"".join(UnicodeLetter_chars+[u'$',u'_']) )
        IdentifierStart_re_txt = u"(?:[{}]|\\\\u[0-9a-fA-F]{{4}})".format(self.IdentifierStart_chars_re_txt)
        
        IdentifierPart_chars = UnicodeLetter_chars + unichar_by_categories('Mn Mc Nd Pc'.split(' ')) + [u'$',u'_']
        IdentifierPart_chars_re_txt = re.escape( u"".join(IdentifierPart_chars) )
        self.IdentifierPart_re_txt = u"(?:[{}]|\\\\u[0-9a-fA-F]{{4}})".format(IdentifierPart_chars_re_txt)
        super(IdentifierTokenizer, self).__init__(
            Identifier, u"{}{}*".format(IdentifierStart_re_txt, self.IdentifierPart_re_txt))

class NumericLiteralTokenizer(RegexTokenizer):
    def __init__(self, identifier_tk):
        DecimalIntegerLiteral_re_txt = u"(?:0|[1-9][0-9]*)"
        ExponentPart_re_txt = u"(?:[eE][\\+\\-]?[0-9]+)"
        DecimalLiteral_re_txt = u"(?:{0}\\.[0-9]*{1}?|\\.[0-9]+{1}?|{0}{1}?)".format(
            DecimalIntegerLiteral_re_txt, ExponentPart_re_txt)
        
        HexIntegerLiteral_re_txt = u"0[xX][0-9a-fA-F]+"
        super(NumericLiteralTokenizer, self).__init__(
            NumericLiteral, u"(?:{}|{})(?![{}0-9])".format(
                DecimalLiteral_re_txt, HexIntegerLiteral_re_txt, identifier_tk.IdentifierStart_chars_re_txt))

class StringLiteralTokenizer(RegexTokenizer):
    def __init__(self, line_term_tk):
        eol_re_txt = re.escape( u"".join(line_term_tk.chars) )
        CharacterEscapeSequence_re_txt = u"[^{}xu0-9]".format(eol_re_txt)
        EscapeSequence_re_txt = u"(?:{}|0(?![0-9])|x[0-9a-fA-F]{{2}}|u[0-9a-fA-F]{{4}})".format(
            CharacterEscapeSequence_re_txt)
        DoubleStringCharacter_re_txt = u"(?:[^\"\\\\{}]|\\\\{})".format(eol_re_txt, EscapeSequence_re_txt)
        SingleStringCharacter_re_txt = u"(?:[^\'\\\\{}]|\\\\{})".format(eol_re_txt, EscapeSequence_re_txt)
        StringLiteral_re_txt = u"(?:\"{}*\"|\'{}*\')".format(
            DoubleStringCharacter_re_txt, SingleStringCharacter_re_txt)
        super(StringLiteralTokenizer, self).__init__(StringLiteral, StringLiteral_re_txt)

class RegularExpressionLiteralTokenizer(RegexTokenizer):
    def __init__(self, line_term_tk, identifier_tk):
        eol_re_txt = re.escape( u"".join(line_term_tk.chars) )
        BackslashSequence = u"(?:\\\\[^{}])".format(eol_re_txt)
        RegularExpressionFirstChar_re_txt = u"(?:[^\\\\{}\\*\\/]|{})".format(
            eol_re_txt, BackslashSequence)
        RegularExpressionChars_re_txt = u"(?:[^\\\\{}\\/]|{})*".format(
            eol_re_txt, BackslashSequence)
        RegularExpressionBody_re_txt = RegularExpressionFirstChar_re_txt + RegularExpressionChars_re_txt
        
        RegularExpressionLiteral_re_txt = u"/{}/{}*".format(
            RegularExpressionBody_re_txt, identifier_tk.IdentifierPart_re_txt)
        super(RegularExpressionLiteralTokenizer, self).__init__(
            RegularExpressionLiteral, RegularExpressionLiteral_re_txt)


_TOKENIZERS = None
LineTerminator_tk = None


_UNICHAR_BY_CATEGORY = None
def unichar_by_cat(category):
    global _UNICHAR_BY_CATEGORY
    if _UNICHAR_BY_CATEGORY is None:
        _UNICHAR_BY_CATEGORY = defaultdict(list)
        for i in xrange(0, 0xFFFF + 1):
            unichar = unichr(i)
            _UNICHAR_BY_CATEGORY[ unicodedata.category(unichar) ].append(unichar)
    
    assert category in _UNICHAR_BY_CATEGORY
    return _UNICHAR_BY_CATEGORY[category]

def unichar_by_categories(cat_it):
    return reduce(operator.add, map(unichar_by_cat, cat_it))


class Lexem(object):
    def __init__(self, data):
        self.data = data
    
    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, repr(self.data))
    
    def __hash__(self):
        return hash(type(self)) ^ hash(self.data)
    
    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.data == other.data

    def __ne__(self, other):
        return not (self == other)
    

class WhiteSpace(Lexem):
    pass
    
class LineTerminator(Lexem):
    pass


class Comment(Lexem):
    pass

class SingleLineComment(Comment):
    pass

class MultiLineComment(Comment):
    pass


class Token(Lexem):
    pass

class Literal(Lexem):
    pass


class ReservedWord(Token):
    pass

class Keyword(ReservedWord):
    pass

class FutureReservedWord(ReservedWord):
    pass

class NullLiteral(ReservedWord, Literal):
    pass

class BooleanLiteral(ReservedWord, Literal):
    pass


class Identifier(Token):
    pass

class Punctuator(Token):
    pass

class NumericLiteral(Token, Literal):
    pass

class StringLiteral(Token, Literal):
    pass


class DivPunctuator(Lexem):
    pass

class RegularExpressionLiteral(Lexem):
    pass
    
