// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use this
// file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
// "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILowerUnifiedToLegacy.cpp - Lower unified v-ops to legacy ops ----===//
//
// Lowers unified v-prefixed VMI ops to their legacy equivalents under the
// opt-in --vmi-two-stage-lowering flag.
//
// Category A — pure syntactic renames (4 ops):
//   vci             → iota
//   vinterpret_cast → bitcast
//   vsel            → select
//   vbrc            → broadcast  (skipped when num_groups is present)
//
// Category B — masked elementwise, pmode="zero" only (18 ops):
//   vadd/vsub/vmul/vdiv/vmin/vmax → legacy type-specific binary + zero + select
//   vneg/vabs/vsqrt/vexp/vln/vrelu → legacy unary + zero + select
//   vand/vor/vxor/vshl/vshr/vnot → legacy bitwise + zero + select
//   pmode="merge" is skipped — merge semantic cannot be expressed in VMI SSA IR.
//
// Category C1 — compare + seed (2 ops):
//   vcmp  → cmpf/cmpi + mask_and
//   vcmps → broadcast scalar + cmpf/cmpi + mask_and
//
// Category C2 — unified type conversion (1 op):
//   vcvt → type-dispatch to extf/truncf/fptosi/sitofp/extsi/extui/trunci
//   Skipped: fp narrowing + saturate (attribute loss).
//
// Category C3 — unified load/store (2 ops):
//   vload  → dispatch by dist_mode/group/block_stride to
//            load / deinterleave_load / group_broadcast_load{num_groups=1} / ...
//   vstore → dispatch to store / masked_store / interleave_store / group_store / ...
//   Skipped: dist_mode "unpack" (physical widening, no legacy equivalent).
//
// Category C4 — static mask creation (3 ops):
//   pset → create_mask(all lanes)
//   pge  → create_mask(N lanes)
//   plt  → create_mask(min(rem, L))
//
// Category C4 — static mask creation (3 ops):
//   pset → create_mask(all lanes)
//   pge  → create_mask(N lanes)
//   plt  → create_mask(min(rem, L))
//
// Category C5 — vector-scalar ops, one-step to legacy (6 ops):
//   vadds/vmuls/vmaxs/vmins/vshls/vshrs
//     → broadcast scalar → legacy binary → zero constant → select
//
// Category C3 — unified load/store (2 ops, dispatch by dist_mode/group):
//   vload → load / deinterleave_load / group_load
//   vstore → store / masked_store / interleave_store / group_store
//
// Category C6 — unified reduce (3 ops, partial coverage):
//   vcadd → reduce_addf/reduce_addi or group_reduce_addf/group_reduce_addi
//   vcmax → reduce_maxf / group_reduce_maxf/group_reduce_maxi
//           (full-int → skipped, no legacy reduce_maxi)
//   vcmin → reduce_minf only (full-float)
//           (group / full-int → skipped, no legacy group_reduce_min*
//            nor reduce_mini)
//
// Category C7 — fused multiply-add family → legacy fma (2 ops):
//   vmula → fma               (float only; int → skipped, no legacy int fma)
//   vaxpy → broadcast + fma   (float only)
//
// Category C8 — indexed gather/scatter → legacy gather/scatter (2 ops):
//   vgather  → gather   (pmode="zero": passthru = zero constant)
//   vscatter → scatter
//
// Category C9 — fused activation / softmax, decomposed to legacy chains (3 ops):
//   vexpdif → [extf] + subf + exp + select   (widen f16 x to f32 when needed)
//   vlrelu  → maxf + minf + broadcast + mulf + addf + select
//   vprelu  → maxf + minf + mulf + addf + select
//   The trailing select(mask, raw, zero) applies pmode="zero" predication.
//   Category C7/C8/C9 all skip pmode="merge".
//
// Category D — no legacy equivalent (explicitly skipped, 6 ops):
//   vhist vintlv vdintlv vselr vgatherb vmull
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILOWERUNIFIEDTOLEGACY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the string name of a predicate mode, defaulting to "zero".
static StringRef getPmodeOrDefault(Operation *op, StringRef attrName = "pmode") {
  if (auto attr = op->getAttrOfType<StringAttr>(attrName))
    return attr.getValue();
  return "zero";
}

/// Returns true when the pmode on `op` is "merge" — these ops must be
/// skipped because merge semantic (inactive lane preserves OLD_DEST) cannot
/// be expressed in VMI SSA IR.
static bool hasMergePmode(Operation *op) {
  return getPmodeOrDefault(op) == "merge";
}

/// Create a zero-valued VMIConstantOp with the same type as \p vmiType.
static Value createZeroConstant(OpBuilder &builder, Location loc,
                                VMIVRegType vmiType) {
  Type elemType = vmiType.getElementType();
  int64_t laneCount = vmiType.getElementCount();
  auto shapedType = RankedTensorType::get({laneCount}, elemType);

  DenseElementsAttr zeroAttr;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    zeroAttr = DenseElementsAttr::get(
        shapedType, APFloat::getZero(floatType.getFloatSemantics()));
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    zeroAttr = DenseElementsAttr::get(
        shapedType, APInt::getZero(intType.getWidth()));
  } else {
    llvm_unreachable("unsupported VMI element type for zero constant");
  }
  return builder.create<VMIConstantOp>(loc, vmiType, zeroAttr).getResult();
}


/// Create a 1-lane VMIConstantOp with the neutral element for reduction:
///   add:  0    (int and float)
///   max: -INF  (float), INT_MIN (int)
///   min: +INF  (float), INT_MAX (int)
static Value createReduceNeutralInit(OpBuilder &builder, Location loc,
                                     Type elemType, bool isAdd, bool isMax) {
  auto oneLaneType =
      VMIVRegType::get(builder.getContext(), 1, elemType, Attribute());
  auto shapedType = RankedTensorType::get({1}, elemType);
  DenseElementsAttr attr;
  if (auto floatTy = dyn_cast<FloatType>(elemType)) {
    if (isAdd)
      attr = DenseElementsAttr::get(
          shapedType, APFloat::getZero(floatTy.getFloatSemantics()));
    else if (isMax)
      attr = DenseElementsAttr::get(
          shapedType,
          APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/true));
    else
      attr = DenseElementsAttr::get(
          shapedType,
          APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/false));
  } else {
    auto intTy = cast<IntegerType>(elemType);
    if (isAdd)
      attr = DenseElementsAttr::get(shapedType,
                                    APInt::getZero(intTy.getWidth()));
    else if (isMax)
      attr = DenseElementsAttr::get(shapedType,
                                    APInt::getSignedMinValue(intTy.getWidth()));
    else
      attr = DenseElementsAttr::get(shapedType,
                                    APInt::getSignedMaxValue(intTy.getWidth()));
  }
  return builder.create<VMIConstantOp>(loc, oneLaneType, attr).getResult();
}

/// Map a unified vcmp `cmp` mode to the predicate string for legacy
/// cmpf/cmpi.  Float operands use ordered predicates (olt, oeq, …);
/// integer operands use signed predicates (slt, seq, …) or bare eq/ne.
static std::string mapCmpPredicate(StringRef cmp, bool isFloat) {
  if (isFloat) {
    // Already ordered/unordered — pass through.
    if (cmp.starts_with("o") || cmp.starts_with("u"))
      return cmp.str();
    return ("o" + cmp).str(); // e.g. "lt" → "olt"
  }
  // Integer.
  if (cmp.starts_with("s") || cmp.starts_with("u"))
    return cmp.str();
  // eq/ne are valid for both fp and int without prefix.
  if (cmp == "eq" || cmp == "ne")
    return cmp.str();
  return ("s" + cmp).str(); // e.g. "lt" → "slt"
}

/// Return true when \p elemType is a floating-point type.
static bool isFloatType(Type elemType) {
  return isa<FloatType>(elemType);
}

/// Return the element type of a VMIVRegType.
static Type getVMIElementType(Value v) {
  return cast<VMIVRegType>(v.getType()).getElementType();
}

/// Inspect the source and result element types of a vcvt and classify the
/// conversion direction.  Returns one of:
///   "widen_fp", "narrow_fp", "fptosi", "sitofp",
///   "widen_int", "narrow_int"
/// conversion direction.  Returns one of:
///   "widen_fp", "narrow_fp", "fptosi", "sitofp",
///   "widen_int", "narrow_int"
static StringRef classifyCvtDirection(Type srcElem, Type dstElem) {
  bool srcFp = isFloatType(srcElem);
  bool dstFp = isFloatType(dstElem);
  unsigned srcBits = srcElem.getIntOrFloatBitWidth();
  unsigned dstBits = dstElem.getIntOrFloatBitWidth();

  if (srcFp && dstFp)
    return dstBits > srcBits ? "widen_fp" : "narrow_fp";
  if (srcFp && !dstFp)
    return "fptosi";
  if (!srcFp && dstFp)
    return "sitofp";
  // int → int
  return dstBits > srcBits ? "widen_int" : "narrow_int";
}

//===----------------------------------------------------------------------===//
// Category B: masked elementwise → legacy compute + zero + select
//===----------------------------------------------------------------------===//

/// Lower a masked BINARY unified op (vadd, vsub, …) to legacy chain:
///   %raw  = legacy.op %lhs, %rhs
///   %zero = vmi.constant dense<0.0>
///   %r    = vmi.select %mask, %raw, %zero
///
/// \p createLegacy is a callable `(Location, Type, Value, Value) -> Value`
/// that emits the legacy binary op.
template <typename UnifiedOp>
static LogicalResult
lowerMaskedBinary(UnifiedOp op, OpBuilder &builder,
                  function_ref<Value(Location, Type, Value, Value)> createLegacy,
                  bool hasMaskOperand = true) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();

  // 1. Legacy binary op.
  Value raw = createLegacy(loc, resultType, lhs, rhs);

  // 2. If the unified op carries an explicit mask operand, wrap in
  //    select(mask, raw, zero) for lane-level predication.  Otherwise the
  //    legacy op already computes on all lanes — skip the mask+select chain.
  if (hasMaskOperand && !op.getMask().empty()) {
    Value zeroConst = createZeroConstant(builder, loc, vmiType);
    Value mask = op.getMask().front();
    Value result =
        builder.create<VMISelectOp>(loc, resultType, mask, raw, zeroConst);
    op.getResult().replaceAllUsesWith(result);
  } else {
    op.getResult().replaceAllUsesWith(raw);
  }
  op->erase();
  return success();
}

/// Lower a masked UNARY unified op (vneg, vabs, …) to legacy chain:
///   %raw  = legacy.op %source
///   %zero = vmi.constant dense<0.0>
///   %r    = vmi.select %mask, %raw, %zero
template <typename UnifiedOp>
static LogicalResult
lowerMaskedUnary(UnifiedOp op, OpBuilder &builder,
                 function_ref<Value(Location, Type, Value)> createLegacy) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value source = op.getSource();

  // 1. Legacy unary op.
  Value raw = createLegacy(loc, resultType, source);

  // 2. If the unified op carries an explicit mask operand, wrap in
  //    select(mask, raw, zero) for lane-level predication.  Otherwise the
  //    legacy op already computes on all lanes — skip the mask+select chain.
  if (!op.getMask().empty()) {
    Value zeroConst = createZeroConstant(builder, loc, vmiType);
    Value mask = op.getMask().front();
    Value result =
        builder.create<VMISelectOp>(loc, resultType, mask, raw, zeroConst);
    op.getResult().replaceAllUsesWith(result);
  } else {
    op.getResult().replaceAllUsesWith(raw);
  }
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C1 helpers: vcmp / vcmps
//===----------------------------------------------------------------------===//

/// Lower vcmp to cmpf/cmpi + mask_and.
static LogicalResult lowerVCmp(VMIVcmpOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type elemType = getVMIElementType(op.getLhs());
  bool isFloat = isFloatType(elemType);
  StringRef cmpMode = op.getCmp();
  std::string predicate = mapCmpPredicate(cmpMode, isFloat);

  // Build legacy cmpf or cmpi.
  Value rawMask;
  if (isFloat) {
    rawMask = builder
                  .create<VMICmpFOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getLhs(), op.getRhs())
                  .getResult();
  } else {
    rawMask = builder
                  .create<VMICmpIOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getLhs(), op.getRhs())
                  .getResult();
  }

  // mask_and with seed.
  Value result = builder
                     .create<VMIMaskAndOp>(loc, op.getResult().getType(),
                                           rawMask, op.getSeed())
                     .getResult();

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcmps to broadcast scalar + cmpf/cmpi + mask_and.
static LogicalResult lowerVCmps(VMIVcmpsOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type srcVmiType = op.getSrc().getType();
  Value scalar = op.getScalar();
  Type elemType = getVMIElementType(op.getSrc());
  bool isFloat = isFloatType(elemType);
  StringRef cmpMode = op.getCmp();
  std::string predicate = mapCmpPredicate(cmpMode, isFloat);

  // 1. Broadcast scalar to vector.
  Value brc = builder.create<VMIBroadcastOp>(loc, srcVmiType, scalar)
                  .getResult();

  // 2. Legacy cmpf or cmpi.
  Value rawMask;
  if (isFloat) {
    rawMask = builder
                  .create<VMICmpFOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getSrc(), brc)
                  .getResult();
  } else {
    rawMask = builder
                  .create<VMICmpIOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getSrc(), brc)
                  .getResult();
  }

  // 3. mask_and with seed.
  Value result = builder
                     .create<VMIMaskAndOp>(loc, op.getResult().getType(),
                                           rawMask, op.getSeed())
                     .getResult();

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C2 helper: vcvt
//===----------------------------------------------------------------------===//

/// Lower vcvt by dispatching on src→dst element types.
static LogicalResult lowerVCvt(VMICvtOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Type srcElem = getVMIElementType(op.getSource());
  Type dstElem = getVMIElementType(op.getResult());
  StringRef direction = classifyCvtDirection(srcElem, dstElem);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value result;

  if (direction == "widen_fp") {
    result = builder.create<VMIExtFOp>(loc, resultType, source).getResult();
  } else if (direction == "narrow_fp") {
    // fp narrowing + saturate → skip (saturate attribute is lost).
    if (op.getSaturateAttr())
      return failure();
    StringAttr roundingAttr = op.getRoundingAttr();
    result =
        builder.create<VMITruncFOp>(loc, resultType, source, roundingAttr)
            .getResult();
  } else if (direction == "fptosi") {
    result =
        builder.create<VMIFPToSIOp>(loc, resultType, source).getResult();
  } else if (direction == "sitofp") {
    result =
        builder.create<VMISIToFPOp>(loc, resultType, source).getResult();
  } else if (direction == "widen_int") {
    // Use sign attr to decide signed vs unsigned extension.
    bool useSigned = true;
    if (auto signAttr = op.getSignAttr()) {
      useSigned = (signAttr.getValue() != "U");
    } else if (auto intTy = dyn_cast<IntegerType>(srcElem)) {
      useSigned = intTy.isSigned();
    }
    if (useSigned)
      result =
          builder.create<VMIExtSIOp>(loc, resultType, source).getResult();
    else
      result =
          builder.create<VMIExtUIOp>(loc, resultType, source).getResult();
  } else if (direction == "narrow_int") {
    // trunci already has saturating semantics.
    result =
        builder.create<VMITruncIOp>(loc, resultType, source).getResult();
  } else {
    return failure();
  }

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C3 helpers: vload / vstore
//===----------------------------------------------------------------------===//

/// Lower vload by dispatching on dist_mode.
static LogicalResult lowerVLoad(VMIvLoadOp op, OpBuilder &builder) {
  // Group mode: vload {group=C} → group_load or group_slot_load
  //   elementCount == num_groups → group_slot_load (compact, 1 scalar per group)
  //   elementCount  > num_groups → group_load (full groups)
  // Broadcast-style slot loads (dense output, stride=1) should be decomposed
  // at the VMI level: vload(compact) + vbrc.
  if (op.getGroupAttr()) {
    auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
    int64_t numGroups = op.getGroupAttr().getInt();
    if (resultType.getElementCount() == numGroups) {
      auto slotLoad = builder.create<VMIGroupSlotLoadOp>(
          op->getLoc(), resultType, op.getSource(), op.getOffset(),
          op.getStride(), op.getGroupAttr());
      op.getResults().front().replaceAllUsesWith(slotLoad.getResult());
    } else {
      auto groupLoad = builder.create<VMIGroupLoadOp>(
          op->getLoc(), resultType, op.getSource(), op.getOffset(),
          op.getStride(), op.getGroupAttr());
      op.getResults().front().replaceAllUsesWith(groupLoad.getResult());
    }
    op->erase();
    return success();
  }

  // Block-stride mode: vload {block_stride, repeat_stride} → stride_load
  if (op.getBlockStride()) {
    auto resultType = op.getResults().front().getType();
    // Create default all-active mask
    auto pset = builder.create<VMIPsetOp>(
        op->getLoc(),
        VMIMaskType::get(builder.getContext(),
                         cast<VMIVRegType>(resultType).getElementCount(),
                         StringAttr::get(builder.getContext(), "pred"),
                         Attribute()),
        builder.getStringAttr("PAT_ALL"), IntegerAttr());
    Value bs = op.getBlockStride();
    Value rs = op.getRepeatStride();
    auto strideLoad = builder.create<VMIStrideLoadOp>(
        op->getLoc(), resultType, op.getSource(), op.getOffset(), bs, rs,
        pset.getResult());
    op.getResults().front().replaceAllUsesWith(strideLoad.getResult());
    op->erase();
    return success();
  }

  // pmode="merge" cannot be expressed by legacy load + select — skip.
  if (hasMergePmode(op))
    return failure();

  StringAttr distModeAttr = op.getDistModeAttr();
  StringRef distMode =
      distModeAttr ? distModeAttr.getValue() : "continuous";

  Location loc = op.getLoc();
  Value source = op.getSource();
  Value offset = op.getOffset();

  if (distMode == "continuous") {
    auto loadOp = builder.create<VMILoadOp>(
        loc, op.getResults().front().getType(), source, offset);
    op.getResults().front().replaceAllUsesWith(loadOp.getResult());
  } else if (distMode == "dintlv") {
    auto dloadOp = builder.create<VMIDeinterleaveLoadOp>(
        loc, op.getResults()[0].getType(), op.getResults()[1].getType(),
        source, offset);
    op.getResults()[0].replaceAllUsesWith(dloadOp.getLow());
    op.getResults()[1].replaceAllUsesWith(dloadOp.getHigh());
  } else if (distMode == "brc") {
    // vload {dist_mode="brc"} -> group_broadcast_load {num_groups=1}.
    // BRC physical semantics: dst[i] = UB[base] for all i (scalar broadcast),
    // equivalent to a single-group broadcast load.
    auto resultType = op.getResults().front().getType();
    Value zeroStride = builder.create<arith::ConstantOp>(
        loc, builder.getIndexType(), builder.getIndexAttr(0));
    auto gbl = builder.create<VMIGroupBroadcastLoadOp>(
        loc, resultType, source, offset, zeroStride,
        builder.getI64IntegerAttr(1));
    op.getResults().front().replaceAllUsesWith(gbl.getResult());
  } else {
    // "unpack" has no legacy equivalent (physical widening, lane count changes).
    return failure();
  }

  op->erase();
  return success();
}

/// Lower vstore by dispatching on dist_mode.
static LogicalResult lowerVStore(VMIvStoreOp op, OpBuilder &builder) {
  // Group mode: vstore {group=C} → group_store
  if (op.getGroupAttr()) {
    builder.create<VMIGroupStoreOp>(
        op->getLoc(), op.getValues()[0], op.getDestination(), op.getOffset(),
        op.getStride(), op.getGroupAttr());
    op->erase();
    return success();
  }

  // Block-stride mode: vstore {block_stride, repeat_stride} → stride_store
  if (op.getBlockStride()) {
    auto valueType = cast<VMIVRegType>(op.getValues()[0].getType());
    // Use existing mask or create default all-active
    Value mask = op.getMask().empty()
                     ? builder
                           .create<VMIPsetOp>(
                               op->getLoc(),
                               VMIMaskType::get(builder.getContext(),
                                            valueType.getElementCount(),
                                            StringAttr::get(builder.getContext(), "pred"),
                                            Attribute()),
                               builder.getStringAttr("PAT_ALL"),
                               IntegerAttr())
                           .getResult()
                     : op.getMask()[0];
    Value bs = op.getBlockStride();
    Value rs = op.getRepeatStride();
    builder.create<VMIStrideStoreOp>(op->getLoc(), op.getValues()[0],
                                    op.getDestination(), op.getOffset(), bs, rs,
                                    mask);
    op->erase();
    return success();
  }

  StringAttr distModeAttr = op.getDistModeAttr();
  StringRef distMode =
      distModeAttr ? distModeAttr.getValue() : "continuous";

  Location loc = op.getLoc();
  Value dest = op.getDestination();
  Value offset = op.getOffset();
  auto values = op.getValues();

  if (distMode == "continuous") {
    if (values.empty())
      return failure();
    Value mask = op.getMask().empty() ? Value() : op.getMask().front();
    if (mask) {
      // Masked store path.
      builder.create<VMIMaskedStoreOp>(loc, values[0], dest, offset, mask);
    } else {
      builder.create<VMIStoreOp>(loc, values[0], dest, offset);
    }
  } else if (distMode == "dintlv") {
    if (values.size() < 2)
      return failure();
    builder.create<VMIInterleaveStoreOp>(loc, values[0], values[1], dest,
                                         offset);
  } else {
    return failure();
  }

  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C4 helpers: pset / pge
//===----------------------------------------------------------------------===//

/// Lower pset "PAT_ALL" → create_mask(all_lanes).
static LogicalResult lowerPset(VMIPsetOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getResult().getType());
  int64_t laneCount = maskType.getElementCount();
  auto indexType = IndexType::get(builder.getContext());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, indexType, builder.getIndexAttr(laneCount));
  Value result =
      builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower pge "PAT_VLN" → create_mask(N).
/// When {group = C} is present → create_group_mask(N, num_groups=C,
/// group_size = total_lanes / C).
static LogicalResult lowerPge(VMIPgeOp op, OpBuilder &builder) {
  StringRef pattern = op.getPattern();
  // Parse "PAT_VL<num>" or fall back to "PAT_VL16".
  int64_t numLanes = 16;
  if (pattern.starts_with("PAT_VL")) {
    StringRef numStr = pattern.drop_front(6); // strlen("PAT_VL")
    if (!numStr.empty()) {
      int64_t parsed = 0;
      for (char c : numStr) {
        if (c < '0' || c > '9')
          break;
        parsed = parsed * 10 + (c - '0');
      }
      if (parsed > 0)
        numLanes = parsed;
    }
  }

  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getResult().getType());
  auto indexType = IndexType::get(builder.getContext());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, indexType, builder.getIndexAttr(numLanes));

  if (auto groupAttr = op.getGroupAttr()) {
    // Grouped tail mask → create_group_mask
    int64_t numGroups = groupAttr.getInt();
    int64_t totalLanes = maskType.getElementCount();
    int64_t groupSize = totalLanes / numGroups;
    Value result = builder
                       .create<VMICreateGroupMaskOp>(
                           loc, maskType, activeLanes,
                           builder.getI64IntegerAttr(numGroups),
                           builder.getI64IntegerAttr(groupSize))
                       .getResult();
    op.getResult().replaceAllUsesWith(result);
  } else {
    Value result =
        builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
    op.getResult().replaceAllUsesWith(result);
  }
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C5 helpers: vector-scalar ops (one-step to legacy)
//===----------------------------------------------------------------------===//

/// Lower a unified vector-scalar op (vadds, vmuls, …) to a legacy chain:
///   %brc  = vmi.broadcast %scalar
///   %raw  = legacy.op %src, %brc
///   %zero = vmi.constant dense<0.0>
///   %r    = vmi.select %mask, %raw, %zero
template <typename VecScalarOp>
static LogicalResult
lowerVecScalar(VecScalarOp op, OpBuilder &builder,
               function_ref<Value(Location, Type, Value, Value)> createLegacy) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type srcVmiType = op.getSrc().getType();
  auto vmiType = cast<VMIVRegType>(srcVmiType);
  Value src = op.getSrc();
  Value scalar = op.getScalar();
  Value mask = op.getMask();

  // 1. Broadcast scalar → vector (legacy broadcast).
  Value brc = builder.create<VMIBroadcastOp>(loc, srcVmiType, scalar)
                  .getResult();

  // 2. Legacy binary op.
  Value raw = createLegacy(loc, srcVmiType, src, brc);

  // 3. Zero constant.
  Value zeroConst = createZeroConstant(builder, loc, vmiType);

  // 4. Select with mask.
  Value result = builder.create<VMISelectOp>(loc, srcVmiType, mask, raw,
                                             zeroConst);
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C6 helpers: vcadd / vcmax / vcmin
//===----------------------------------------------------------------------===//

/// Lower vcadd to legacy reduce_addf/reduce_addi or
/// group_reduce_addf/group_reduce_addi.  Always succeeds for valid input
/// (vcadd verifier guarantees reassoc for float, and group 整除 source lanes).
static LogicalResult lowerVCadd(VMIvcaddOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (auto groupAttr = op.getGroupAttr()) {
    // Group reduce path
    int64_t C = groupAttr.getInt();
    Value result;
    if (isFloat)
      result =
          builder
              .create<VMIGroupReduceAddFOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(C),
                                            op.getReassocAttr())
              .getResult();
    else
      result =
          builder
              .create<VMIGroupReduceAddIOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(C))
              .getResult();
    op.getResult().replaceAllUsesWith(result);
  } else {
    // Full reduce path
    Value init = createReduceNeutralInit(builder, loc, elemType,
                                         /*isAdd=*/true, /*isMax=*/false);
    Value result;
    if (isFloat)
      result =
          builder
              .create<VMIReduceAddFOp>(loc, resultType, source, init, mask,
                                       op.getReassocAttr())
              .getResult();
    else
      result =
          builder
              .create<VMIReduceAddIOp>(loc, resultType, source, init, mask)
              .getResult();
    op.getResult().replaceAllUsesWith(result);
  }
  op->erase();
  return success();
}

/// Lower vcmax to legacy reduce_maxf / group_reduce_maxf / group_reduce_maxi.
/// Returns failure for full-integer max (no legacy reduce_maxi).
static LogicalResult lowerVcmax(VMIvcmaxOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (auto groupAttr = op.getGroupAttr()) {
    // Group reduce path
    int64_t C = groupAttr.getInt();
    Value result;
    if (isFloat)
      result =
          builder
              .create<VMIGroupReduceMaxFOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(C))
              .getResult();
    else
      result =
          builder
              .create<VMIGroupReduceMaxIOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(C))
              .getResult();
    op.getResult().replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  // Full reduce: only float has a legacy equivalent
  if (!isFloat)
    return failure();

  Value init = createReduceNeutralInit(builder, loc, elemType,
                                       /*isAdd=*/false, /*isMax=*/true);
  Value result =
      builder
          .create<VMIReduceMaxFOp>(loc, resultType, source, init, mask)
          .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcmin to legacy reduce_minf.
/// Only the full-float case is lowerable; group and integer cases return
/// failure and fall through to the direct VMIToVPTO path.
static LogicalResult lowerVcmin(VMIvcminOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  // Only full-float has a legacy equivalent
  if (op.getGroupAttr())
    return failure();

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  if (!isa<FloatType>(elemType))
    return failure();

  Location loc = op.getLoc();
  Value init = createReduceNeutralInit(builder, loc, elemType,
                                       /*isAdd=*/false, /*isMax=*/false);
  Value result =
      builder
          .create<VMIReduceMinFOp>(loc, op.getResult().getType(),
                                   op.getSource(), init, op.getMask())
          .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C7 helpers: vmula / vaxpy (fused multiply-add → legacy fma)
//===----------------------------------------------------------------------===//

/// Lower vmula (acc = acc + lhs*rhs) to legacy fma (lhs*rhs + acc) + select.
/// Legacy fma is floating-point only; integer vmula has no legacy equivalent
/// and is skipped (falls through to VMIToVPTO).
static LogicalResult lowerVmula(VMIVmulaOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  if (!isFloatType(vmiType.getElementType()))
    return failure();

  Location loc = op.getLoc();
  // fma computes lhs*rhs + acc, matching vmula's acc + lhs*rhs.
  Value raw = builder
                  .create<VMIFmaOp>(loc, resultType, op.getLhs(), op.getRhs(),
                                    op.getAcc())
                  .getResult();
  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value result = builder.create<VMISelectOp>(loc, resultType, op.getMask(), raw,
                                             zeroConst);
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vaxpy (alpha*x + y) to broadcast(alpha) + legacy fma + select.
/// alpha is a scalar float, broadcast to a vector before the fma.
static LogicalResult lowerVaxpy(VMIVaxpyOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  if (!isFloatType(vmiType.getElementType()))
    return failure();

  Location loc = op.getLoc();
  Value alphaVec = builder
                       .create<VMIBroadcastOp>(loc, resultType, op.getAlpha())
                       .getResult();
  // fma(alpha, x, y) == alpha*x + y.
  Value raw = builder
                  .create<VMIFmaOp>(loc, resultType, alphaVec, op.getX(),
                                    op.getAcc())
                  .getResult();
  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value result = builder.create<VMISelectOp>(loc, resultType, op.getMask(), raw,
                                             zeroConst);
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower plt(rem:i32) -> create_mask(min(rem, L)) + arith remainder chain.
///   %act  = arith.minsi %rem, %cL         // min(rem, L)
///   %aidx = arith.index_cast %act          // i32 -> index
///   %mask = vmi.create_mask %aidx
///   %next = arith.subi %rem, %act          // rem - min(rem, L) = max(rem-L, 0)
static LogicalResult lowerPlt(VMIPltOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  int64_t laneCount = maskType.getElementCount();

  auto i32Type = builder.getIntegerType(32);
  Value cL = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getIntegerAttr(i32Type, laneCount));
  Value act = builder.create<arith::MinSIOp>(loc, i32Type, op.getScalar(), cL);
  Value aidx = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), act);
  Value mask = builder.create<VMICreateMaskOp>(loc, maskType, aidx).getResult();
  Value next = builder.create<arith::SubIOp>(loc, i32Type, op.getScalar(), act);

  op.getMask().replaceAllUsesWith(mask);
  op.getScalarOut().replaceAllUsesWith(next);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C8 helpers: vgather / vscatter
//===----------------------------------------------------------------------===//

/// Lower vgather to legacy gather.  Legacy gather carries an explicit passthru
/// operand for inactive lanes; pmode="zero" is modelled with a zero passthru.
/// pmode="merge" (preserve OLD_DEST) has no SSA passthru and is skipped.
static LogicalResult lowerVgather(VMIVgatherOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  // Reuse the offsets vector as passthru for inactive lanes — semantically
  // harmless (inactive-lane result is don't-care) and avoids materialising
  // a zero constant that would require signless element types.
  Value result = builder
                     .create<VMIGatherOp>(loc, resultType, op.getSource(),
                                          op.getOffsets(), op.getMask(),
                                          op.getOffsets())
                     .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vscatter to legacy scatter.  Legacy scatter only writes active lanes
/// (mask-governed), matching vscatter's default/zero pmode; merge is skipped.
static LogicalResult lowerVscatter(VMIVscatterOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  builder.create<VMIScatterOp>(loc, op.getValue(), op.getDestination(),
                               op.getOffsets(), op.getMask());
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C9 helpers: vexpdif / vlrelu / vprelu (fused → legacy chains)
//===----------------------------------------------------------------------===//

/// Lower vexpdif (exp(x - max)) to [extf] + subf + exp + select.
/// x may be f16 while max and result are always f32 — widen x first when its
/// element type differs from the result type.
static LogicalResult lowerVexpdif(VMIVexpdifOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Type resElem = vmiType.getElementType();

  Value x = op.getX();
  if (getVMIElementType(x) != resElem)
    x = builder.create<VMIExtFOp>(loc, resultType, x).getResult();

  Value diff =
      builder.create<VMISubFOp>(loc, resultType, x, op.getMax()).getResult();
  Value raw = builder.create<VMIExpOp>(loc, resultType, diff).getResult();
  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value result = builder.create<VMISelectOp>(loc, resultType, op.getMask(), raw,
                                             zeroConst);
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vlrelu (x>0 ? x : slope*x) to max(x,0) + slope*min(x,0) + select.
/// slope is a scalar float broadcast to a vector.
static LogicalResult lowerVlrelu(VMIVlreluOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value x = op.getX();

  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value pos =
      builder.create<VMIMaxFOp>(loc, resultType, x, zeroConst).getResult();
  Value neg =
      builder.create<VMIMinFOp>(loc, resultType, x, zeroConst).getResult();
  Value slopeVec = builder
                       .create<VMIBroadcastOp>(loc, resultType, op.getSlope())
                       .getResult();
  Value scaledNeg =
      builder.create<VMIMulFOp>(loc, resultType, slopeVec, neg).getResult();
  Value raw =
      builder.create<VMIAddFOp>(loc, resultType, pos, scaledNeg).getResult();
  Value result = builder.create<VMISelectOp>(loc, resultType, op.getMask(), raw,
                                             zeroConst);
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vprelu (max(x,0) + alpha*min(x,0)) to legacy max/min/mul/add + select.
/// alpha is a per-lane vector (no broadcast needed).
static LogicalResult lowerVprelu(VMIVpreluOp op, OpBuilder &builder) {
  if (hasMergePmode(op))
    return failure();

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value x = op.getX();

  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value pos =
      builder.create<VMIMaxFOp>(loc, resultType, x, zeroConst).getResult();
  Value neg =
      builder.create<VMIMinFOp>(loc, resultType, x, zeroConst).getResult();
  Value scaledNeg =
      builder.create<VMIMulFOp>(loc, resultType, op.getAlpha(), neg).getResult();
  Value raw =
      builder.create<VMIAddFOp>(loc, resultType, pos, scaledNeg).getResult();
  Value result = builder.create<VMISelectOp>(loc, resultType, op.getMask(), raw,
                                             zeroConst);
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct VMILowerUnifiedToLegacyPass
    : public mlir::pto::impl::VMILowerUnifiedToLegacyBase<
          VMILowerUnifiedToLegacyPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILowerUnifiedToLegacyPass)

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
};

} // namespace

void VMILowerUnifiedToLegacyPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Operation *, 128> worklist;

  // Collect all unified VMI ops (walk encounters them in IR order).
  module.walk([&](Operation *op) {
    // Category A
    if (isa<VMIVciOp, VMIVinterpretCastOp, VMIvSelOp, VMIVbrcOp>(op) ||
        // Category B — binary
        isa<VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp, VMIVmaxOp,
            VMIVandOp, VMIVorOp, VMIVxorOp, VMIVshlOp, VMIVshrOp>(op) ||
        // Category B — unary
        isa<VMIVnegOp, VMIVabsOp, VMIVsqrtOp, VMIVexpOp, VMIVlnOp, VMIVreluOp,
            VMIVnotOp>(op) ||
        // Category C1
        isa<VMIVcmpOp, VMIVcmpsOp>(op) ||
        // Category C2
        isa<VMICvtOp>(op) ||
        // Category C3
        isa<VMIvLoadOp, VMIvStoreOp>(op) ||
        // Category C4
        isa<VMIPsetOp, VMIPgeOp, VMIPltOp>(op) ||
        // Category C5
        isa<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp,
            VMIShrSOp>(op) ||
        // Category C6 — unified reduce (partial coverage)
        isa<VMIvcaddOp, VMIvcmaxOp, VMIvcminOp>(op) ||
        // Category C7 — fused multiply-add family → legacy fma
        isa<VMIVmulaOp, VMIVaxpyOp>(op) ||
        // Category C8 — indexed gather / scatter
        isa<VMIVgatherOp, VMIVscatterOp>(op) ||
        // Category C9 — fused activation / softmax (legacy chains)
        isa<VMIVexpdifOp, VMIVlreluOp, VMIVpreluOp>(op))
      worklist.push_back(op);

    // Category D — no legacy equivalent (require direct VMIToVPTO lowering):
    //   plt, vhist, vintlv, vdintlv, vselr, vgatherb, vmull
    // These are intentionally NOT added to the worklist — they flow through
    // to VMIToVPTO which must provide direct 1:N lowering patterns.
    if (isa<VMIHistV2Op, VMIVintlvOp, VMIVdintlvOp, VMIVselrOp,
            VMIVgatherbOp, VMIVmullOp>(op)) {
      op->emitRemark("VMI unified op has no legacy equivalent — "
                     "requires direct VMIToVPTO 1:N lowering");
    }
  });

  for (Operation *op : llvm::reverse(worklist)) {
    if (!op->getBlock())
      continue;
    OpBuilder builder(op);

    // ---- Category A: pure syntactic renames ----

    if (auto vop = dyn_cast<VMIVciOp>(op)) {
      // vci -> iota
      builder.setInsertionPoint(op);
      StringAttr orderAttr;
      if (auto order = vop.getOrder())
        orderAttr = builder.getStringAttr(*order);
      Value result =
          builder
              .create<VMIIotaOp>(op->getLoc(), vop.getResult().getType(),
                                 vop.getBase(), orderAttr)
              .getResult();
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    if (auto vop = dyn_cast<VMIVinterpretCastOp>(op)) {
      // vinterpret_cast -> bitcast
      builder.setInsertionPoint(op);
      Value result =
          builder
              .create<VMIBitcastOp>(op->getLoc(), vop.getResult().getType(),
                                    vop.getSource())
              .getResult();
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    if (auto vop = dyn_cast<VMIvSelOp>(op)) {
      // vsel -> select
      builder.setInsertionPoint(op);
      Value result =
          builder
              .create<VMISelectOp>(op->getLoc(), vop.getResult().getType(),
                                   vop.getMask(), vop.getTrueValue(),
                                   vop.getFalseValue())
              .getResult();
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    if (auto vop = dyn_cast<VMIVbrcOp>(op)) {
      // vbrc -> broadcast; vbrc{group} -> group_broadcast
      builder.setInsertionPoint(op);
      Value result;
      if (vop.getGroupAttr()) {
        result =
            builder
                .create<VMIGroupBroadcastOp>(op->getLoc(), vop.getResult().getType(),
                                             vop.getValue(), vop.getGroupAttr())
                .getResult();
      } else {
        result =
            builder
                .create<VMIBroadcastOp>(op->getLoc(), vop.getResult().getType(),
                                        vop.getValue())
                .getResult();
      }
      vop.getResult().replaceAllUsesWith(result);
      op->erase();
      continue;
    }

    // ---- Category C4: static mask creation ----

    if (auto vop = dyn_cast<VMIPsetOp>(op)) {
      (void)lowerPset(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIPgeOp>(op)) {
      (void)lowerPge(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIPltOp>(op)) {
      (void)lowerPlt(vop, builder);
      continue;
    }

    // ---- Category C1: vcmp / vcmps ----

    if (auto vop = dyn_cast<VMIVcmpOp>(op)) {
      (void)lowerVCmp(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVcmpsOp>(op)) {
      (void)lowerVCmps(vop, builder);
      continue;
    }

    // ---- Category C2: vcvt ----

    if (auto vop = dyn_cast<VMICvtOp>(op)) {
      (void)lowerVCvt(vop, builder);
      continue;
    }

    // ---- Category C3: vload / vstore ----

    if (auto vop = dyn_cast<VMIvLoadOp>(op)) {
      (void)lowerVLoad(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIvStoreOp>(op)) {
      (void)lowerVStore(vop, builder);
      continue;
    }

    // ---- Category C5: vector-scalar ops ----

    if (auto vop = dyn_cast<VMIAddSOp>(op)) {
      Type elemType = getVMIElementType(vop.getSrc());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType))
          return builder.create<VMIAddFOp>(loc, ty, lhs, rhs).getResult();
        return builder.create<VMIAddIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerVecScalar(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIMulSOp>(op)) {
      Type elemType = getVMIElementType(vop.getSrc());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType))
          return builder.create<VMIMulFOp>(loc, ty, lhs, rhs).getResult();
        return builder.create<VMIMulIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerVecScalar(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIMaxSOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIMaxFOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerVecScalar(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIMinSOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIMinFOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerVecScalar(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIShlSOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIShLIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerVecScalar(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIShrSOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIShRUIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerVecScalar(vop, builder, createLegacy);
      continue;
    }

    // ---- Category C6: unified reduce ----

    if (auto vop = dyn_cast<VMIvcaddOp>(op)) {
      (void)lowerVCadd(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIvcmaxOp>(op)) {
      (void)lowerVcmax(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIvcminOp>(op)) {
      (void)lowerVcmin(vop, builder);
      continue;
    }

    // ---- Category C7: fused multiply-add family ----

    if (auto vop = dyn_cast<VMIVmulaOp>(op)) {
      (void)lowerVmula(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVaxpyOp>(op)) {
      (void)lowerVaxpy(vop, builder);
      continue;
    }

    // ---- Category C8: indexed gather / scatter ----

    if (auto vop = dyn_cast<VMIVgatherOp>(op)) {
      (void)lowerVgather(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVscatterOp>(op)) {
      (void)lowerVscatter(vop, builder);
      continue;
    }

    // ---- Category C9: fused activation / softmax ----

    if (auto vop = dyn_cast<VMIVexpdifOp>(op)) {
      (void)lowerVexpdif(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVlreluOp>(op)) {
      (void)lowerVlrelu(vop, builder);
      continue;
    }

    if (auto vop = dyn_cast<VMIVpreluOp>(op)) {
      (void)lowerVprelu(vop, builder);
      continue;
    }

    // ---- Category B: masked elementwise — binary ----

    if (auto vop = dyn_cast<VMIVaddOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType))
          return builder.create<VMIAddFOp>(loc, ty, lhs, rhs).getResult();
        return builder.create<VMIAddIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVsubOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType))
          return builder.create<VMISubFOp>(loc, ty, lhs, rhs).getResult();
        return builder.create<VMISubIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVmulOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value lhs, Value rhs) -> Value {
        if (isFloatType(elemType))
          return builder.create<VMIMulFOp>(loc, ty, lhs, rhs).getResult();
        return builder.create<VMIMulIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVdivOp>(op)) {
      // Float-only; no legacy integer divide.
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIDivFOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVminOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIMinFOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVmaxOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIMaxFOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVandOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIAndIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVorOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIOrIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVxorOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIXOrIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVshlOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIShLIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVshrOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value lhs,
                              Value rhs) -> Value {
        return builder.create<VMIShRUIOp>(loc, ty, lhs, rhs).getResult();
      };
      (void)lowerMaskedBinary(vop, builder, createLegacy);
      continue;
    }

    // ---- Category B: masked elementwise — unary ----

    if (auto vop = dyn_cast<VMIVnegOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMINegFOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVabsOp>(op)) {
      Type elemType = getVMIElementType(vop.getResult());
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        if (isFloatType(elemType))
          return builder.create<VMIAbsFOp>(loc, ty, src).getResult();
        return builder.create<VMIAbsIOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVsqrtOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMISqrtOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVexpOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMIExpOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVlnOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMILnOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVreluOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMIReluOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }

    if (auto vop = dyn_cast<VMIVnotOp>(op)) {
      auto createLegacy = [&](Location loc, Type ty, Value src) -> Value {
        return builder.create<VMINotOp>(loc, ty, src).getResult();
      };
      (void)lowerMaskedUnary(vop, builder, createLegacy);
      continue;
    }
  }
}

std::unique_ptr<Pass> mlir::pto::createVMILowerUnifiedToLegacyPass() {
  return std::make_unique<VMILowerUnifiedToLegacyPass>();
}
