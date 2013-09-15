
class OP(object):
    """
    OP (grammar operator) is the root of hierarchy of fundamental grammar operators.
    Graph of instances (including descendants) of grammar ops forms grammatic structure of the language.
    Current grammar ops (see below) are derived from the first parsers implementation, so the are a bit 
    different from EBNF and other widely used notations.
    
    Implementation of grammar ops is ready for multiple inheritance and automatic processing.
    """
    def _use_props(self, prop_names, caller_kwargs, optional=None):
        """
        generic kwargs grabber
        """
        self.prop_names = prop_names
        for prop in prop_names:
            if optional is not None and prop in optional:
                if prop not in caller_kwargs:
                    caller_kwargs[prop] = None
            else:   # required prop
                if caller_kwargs.get(prop, None) is None:
                    raise TypeError("required argument '{}' is not found or is None".format(prop))
            
            setattr(self, prop, caller_kwargs[prop])
            del caller_kwargs[prop]
        
        return caller_kwargs
    
    def _check_operands(self, ops):
        erroneous = filter(lambda op: not isinstance(op, OP), ops)
        if erroneous:
            raise TypeError("got non-OPs in operands: {}".format(", ".join(map(str, erroneous))))

class TerminalOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('tok_isinstance', 'tok_data'), kwargs, ('tok_data', ))
        super(TerminalOP, self).__init__(**kwargs)

class ForwardDeclarationOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('target', ), kwargs, ('target', ))
        super(ForwardDeclarationOP, self).__init__(**kwargs)

class AlternativesOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('alternatives', ), kwargs)
        super(AlternativesOP, self).__init__(**kwargs)
        self._check_operands(self.alternatives)

class SequenceOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('subops', 'optional_right_items', 'optional_left_items'), kwargs,
            ('optional_right_items', 'optional_left_items'))
        super(SequenceOP, self).__init__(**kwargs)
        self._check_operands(self.subops)

class RepetitionOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('subop', 'min_matches', 'separator'), kwargs, ('separator', ))
        super(RepetitionOP, self).__init__(**kwargs)
        self._check_operands([self.subop] + ([self.separator] if self.separator is not None else []))

class EmptyOP(OP):
    def __init__(self, **kwargs):
        self._use_props((), kwargs)
        super(EmptyOP, self).__init__(**kwargs)

class OptionalOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('subop', ), kwargs)
        super(OptionalOP, self).__init__(**kwargs)
        self._check_operands([self.subop])

class LookAheadOP(OP):
    def __init__(self, **kwargs):
        self._use_props(('subop', 'inverse'), kwargs)
        super(LookAheadOP, self).__init__(**kwargs)
        self._check_operands([self.subop])



