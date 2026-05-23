from __future__ import annotations

from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/torchao/utils.py")
text = path.read_text()
if "from enum import Enum" not in text:
    text = text.replace("import time\n", "import time\nfrom enum import Enum\n", 1)
old = '''def register_as_pytree_constant(cls):
    """Decorator to register a class as a pytree constant for dynamo non-strict trace mode."""
    torch.utils._pytree.register_constant(cls)
    return cls
'''
new = '''def register_as_pytree_constant(cls):
    """Decorator to register a class as a pytree constant for dynamo non-strict trace mode."""
    if isinstance(cls, type) and issubclass(cls, Enum):
        return cls
    torch.utils._pytree.register_constant(cls)
    return cls
'''
if old not in text:
    raise RuntimeError("TorchAO register_as_pytree_constant anchor not found")
path.write_text(text.replace(old, new, 1))
