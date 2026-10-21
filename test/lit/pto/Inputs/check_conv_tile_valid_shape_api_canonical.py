"""The ConvTileBufType API canonicalizes valid == declared instead of failing.

Regression for the review finding: the textual parser normalizes an equal
valid_shape away, but the CAPI/Python getChecked path used to reject it, so
Python callers passing valid == shape silently got None. Both paths must
build the same canonical type.
"""
from ptoas.mlir.dialects import pto as pto_ir
from ptoas.mlir.ir import Attribute, Context, Type

with Context() as ctx:
    pto_ir.register_dialect(ctx)
    shape = [1, 1, 3, 4, 16]
    elem = Type.parse("f16", context=ctx)
    buffer = Attribute.parse("192 : i64", context=ctx)
    mem = Attribute.parse("#pto.address_space<mat>", context=ctx)
    layout = Attribute.parse("#pto.conv_layout<nc1hwc0>", context=ctx)

    plain = pto_ir.ConvTileBufType.get(shape, elem, buffer, mem, layout, context=ctx)
    assert plain is not None, "plain construction failed"

    equal_valid = pto_ir.ConvTileBufType.get(
        shape, elem, buffer, mem, layout, valid_shape=shape, context=ctx)
    assert equal_valid is not None, "valid == shape must not fail construction"
    assert equal_valid == plain, "valid == shape must build the canonical type"
    assert equal_valid.valid_shape is None, "canonical form stores no valid_shape"

    shrunk = pto_ir.ConvTileBufType.get(
        shape, elem, buffer, mem, layout, valid_shape=[1, 1, 2, 4, 16], context=ctx)
    assert shrunk is not None, "strictly smaller valid shape must construct"
    assert list(shrunk.valid_shape) == [1, 1, 2, 4, 16]

    # Pre-valid_shape callers bind the context positionally as the trailing
    # argument; that call shape must keep working.
    legacy = pto_ir.ConvTileBufType.get(shape, elem, buffer, mem, layout, ctx)
    assert legacy is not None, "positional context call must keep working"
    assert legacy == plain

    print("plain:", str(plain))
    print("equal:", str(equal_valid))
    print("shrunk:", str(shrunk))
    print("API-CANONICAL-OK")
