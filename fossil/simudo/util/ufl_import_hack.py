import importlib
import re
import sys


def fix():
    try:
        import ufl_legacy
    except ImportError:
        return  # no need, we don't have to do the rename thing

    regex = re.compile("^ufl")

    for old in """
ufl.log
ufl.algebra
ufl.constantvalue
ufl.core.expr
ufl.core.ufl_type
ufl.equation
ufl.mathfunctions
    """.split():
        mod = ""
        for name in old.split("."):
            if not mod:
                mod = name
            else:
                mod = f"{mod}.{name}"

            new = regex.sub("ufl_legacy", mod)
            sys.modules[mod] = importlib.import_module(new)
