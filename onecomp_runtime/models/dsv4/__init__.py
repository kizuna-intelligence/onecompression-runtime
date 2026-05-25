"""QEP-quantized DeepSeek-V4-Flash runtime (int4 dense / int2 routed experts).

:class:`Dsv4QuantizedModel` assembles the shipped reference arch block-by-block with
packed int4/int2 Linears and CPU-offloaded routed experts so the 284B MoE fits 24GB.
"""
from .loader import Dsv4QuantizedModel

__all__ = ["Dsv4QuantizedModel"]
