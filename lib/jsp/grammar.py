
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


OP_TYPES = (TerminalOP, ForwardDeclarationOP, AlternativesOP, SequenceOP, RepetitionOP, EmptyOP, OptionalOP,
    LookAheadOP)

def find_op_type(obj):
    if not isinstance(obj, OP):
        raise TypeError("obj is a non-OP: {}".format(obj))
    for op_t in OP_TYPES:   # fixme is there a better way?
        if isinstance(obj, op_t):
            return op_t
    assert False


def convert_grammar(root, type_map, traversed_ids=None):
    """
    Converts grammar graph to a different type system.
    type_map has to hold corresponding type for each type of OPs,
    grammar op object will be passed as first parameter to corresponding type's constructor.
    """
    if traversed_ids is None:   # root of recursion
        traversed_ids = set()
        if not isinstance(type_map, dict):
            # one type for all ops
            type_map = dict((op, type_map) for op in OP_TYPES)

    if id(root) in traversed_ids:
        return
    
    converted = type_map[find_op_type(root)](root, level=len(traversed_ids))
    converted.prop_names = tuple(root.prop_names)
    traversed_ids = set(traversed_ids)  # copy
    traversed_ids.add(id(root))
    for prop_name in root.prop_names:
        prop = getattr(root, prop_name)
        if isinstance(prop, OP):
            setattr(converted, prop_name, convert_grammar(prop, type_map, traversed_ids))
        elif isinstance(prop, (list, tuple)) and filter(lambda so: isinstance(so, OP), prop):
            converted_subops = map(lambda so: convert_grammar(so, type_map, traversed_ids), prop)
            setattr(converted, prop_name, converted_subops if isinstance(prop, list) else tuple(converted_subops))
        else:
            setattr(converted, prop_name, prop)

    return converted

