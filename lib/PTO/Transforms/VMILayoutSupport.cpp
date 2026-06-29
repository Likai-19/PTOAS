// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutSupport.cpp - VMI layout support queries --------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VMILayoutSupport.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/VMITargetCapabilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/Twine.h"

#include <cassert>

using namespace mlir;
using namespace mlir::pto;

namespace {

static LogicalResult failWithReason(const Twine &message, std::string *reason) {
  if (reason)
    *reason = message.str();
  return failure();
}

static LogicalResult checkFullDataPhysicalChunks(VMIVRegType type,
                                                 std::string *reason) {
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart))
    return failWithReason("requires known physical lanes per part", reason);

  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  if (failed(arity))
    return failWithReason("requires computable physical arity", reason);

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return failWithReason("requires assigned layout", reason);
  int64_t factor = layout.isDeinterleaved() ? layout.getFactor() : 1;
  if (factor <= 0 || *arity % factor != 0)
    return failWithReason("requires arity divisible by layout factor", reason);

  int64_t chunksPerPart = *arity / factor;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPaddingLane(type, part, chunk, lane);
        if (failed(padding))
          return failWithReason("failed to map physical padding lane", reason);
        if (*padding)
          return failWithReason("found padding lane in physical chunk", reason);
      }
    }
  }

  return success();
}

static bool hasX2MemoryDistToken(Type elementType) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  return elementBits == 8 || elementBits == 16 || elementBits == 32;
}

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>()) {
    if (constant.getType().isIndex())
      return constant.value();
  }
  return std::nullopt;
}

static int64_t ceilDivNonNegative(int64_t lhs, int64_t rhs) {
  assert(lhs >= 0 && rhs > 0);
  return (lhs + rhs - 1) / rhs;
}

static FailureOr<int64_t> getVMITypeElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return vregType.getElementCount();
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getElementCount();
  return failure();
}

static FailureOr<int64_t> getVMITypeLayoutFactor(Type type) {
  VMILayoutAttr layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layout = vregType.getLayoutAttr();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layout = maskType.getLayoutAttr();
  else
    return failure();
  if (!layout)
    return failure();
  return layout.isDeinterleaved() ? layout.getFactor() : 1;
}

static FailureOr<int64_t> getVMITypeLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return getDataLanesPerPart(vregType.getElementType());
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return getMaskLanesPerPart(maskType.getGranularity());
  return failure();
}

static FailureOr<int64_t> getVMITypeChunksInPart(Type type, int64_t part) {
  FailureOr<int64_t> elementCount = getVMITypeElementCount(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(lanesPerPart) ||
      part < 0 || part >= *factor)
    return failure();

  int64_t logicalLanesInPart = (*elementCount + *factor - 1 - part) / *factor;
  return ceilDivNonNegative(logicalLanesInPart, *lanesPerPart);
}

static LogicalResult checkFullVMIPhysicalChunks(Type type,
                                                std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  if (failed(factor) || failed(lanesPerPart))
    return fail("requires assigned layout with known physical lanes per part");

  for (int64_t part = 0; part < *factor; ++part) {
    FailureOr<int64_t> chunks = getVMITypeChunksInPart(type, part);
    if (failed(chunks))
      return fail("requires known physical chunks");
    for (int64_t chunk = 0; chunk < *chunks; ++chunk) {
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPaddingLane(type, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding)
          return fail("found padding lane in physical chunk");
      }
    }
  }

  return success();
}

static FailureOr<int64_t>
getContiguousMaterializationPartCount(Type type, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  if (failed(arity) || failed(factor))
    return fail("requires computable physical arity and assigned layout");

  VMILayoutAttr layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layout = vregType.getLayoutAttr();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layout = maskType.getLayoutAttr();
  else
    return fail("requires VMI data or mask type");

  if (!layout)
    return fail("requires assigned layout");
  if (layout.isContiguous())
    return *arity;
  if (!layout.isDeinterleaved() ||
      (layout.getFactor() != 2 && layout.getFactor() != 4))
    return fail("requires contiguous or deinterleaved=2/4 layout");

  FailureOr<int64_t> chunksPerGroup = getVMITypeChunksInPart(type, 0);
  if (failed(chunksPerGroup))
    return fail("requires known physical chunks per part");
  if (*chunksPerGroup == 0)
    return fail("requires at least one physical chunk per part");

  for (int64_t part = 1; part < *factor; ++part) {
    FailureOr<int64_t> chunks = getVMITypeChunksInPart(type, part);
    if (failed(chunks))
      return fail("requires known physical chunks per part");
    if (*chunks != *chunksPerGroup)
      return fail("requires every deinterleaved part to have the same "
                  "physical chunk count");
  }

  return *arity;
}

static LogicalResult checkLayoutMaterializationShape(Type sourceType,
                                                     Type resultType,
                                                     VMILayoutAttr sourceLayout,
                                                     VMILayoutAttr resultLayout,
                                                     std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(resultArity))
    return fail("requires computable source/result physical arity");
  if (*sourceArity != *resultArity)
    return fail("requires source and result to have the same physical arity");

  if (sourceLayout == resultLayout)
    return success();

  std::string sourceReason;
  std::string resultReason;
  LogicalResult sourceFull =
      checkFullVMIPhysicalChunks(sourceType, &sourceReason);
  LogicalResult resultFull =
      checkFullVMIPhysicalChunks(resultType, &resultReason);
  if (succeeded(sourceFull) && succeeded(resultFull))
    return success();

  std::string sourceMaterializationReason;
  FailureOr<int64_t> sourceMaterializedParts =
      getContiguousMaterializationPartCount(sourceType,
                                            &sourceMaterializationReason);
  std::string resultMaterializationReason;
  FailureOr<int64_t> resultMaterializedParts =
      getContiguousMaterializationPartCount(resultType,
                                            &resultMaterializationReason);
  if (succeeded(sourceMaterializedParts) &&
      succeeded(resultMaterializedParts) &&
      *sourceMaterializedParts == *sourceArity &&
      *resultMaterializedParts == *resultArity)
    return success();

  if (failed(sourceFull))
    return fail(Twine("source ") + sourceReason + "; source materialization " +
                sourceMaterializationReason);
  return fail(Twine("result ") + resultReason + "; result materialization " +
              resultMaterializationReason);
}

static FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                                    int64_t numGroups,
                                                    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };
  if (numGroups <= 0)
    return fail("requires num_groups to be positive");
  if (type.getElementCount() % numGroups != 0)
    return fail("requires num_groups to evenly divide logical lane count");
  return type.getElementCount() / numGroups;
}

static FailureOr<int64_t> getDataLayoutFactor(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return failure();
  return layout.isDeinterleaved() ? layout.getFactor() : 1;
}

static FailureOr<SmallVector<int64_t>>
getPhysicalLogicalBitFootprint(VMIVRegType type) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type.getElementType());
  if (elementBits == 0)
    return failure();

  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  if (failed(factor) || failed(lanesPerPart) || failed(arity) || *factor <= 0)
    return failure();

  SmallVector<int64_t> bits;
  bits.reserve(*arity);
  for (int64_t part = 0; part < *factor; ++part) {
    for (int64_t chunk = 0; chunk < *arity; ++chunk) {
      int64_t activeLanes = 0;
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding = isPaddingLane(type, part, chunk, lane);
        if (failed(padding))
          return failure();
        if (!*padding)
          ++activeLanes;
      }
      if (activeLanes > 0)
        bits.push_back(activeLanes * static_cast<int64_t>(elementBits));
    }
  }
  if (static_cast<int64_t>(bits.size()) != *arity)
    return failure();
  return bits;
}

static FailureOr<VMILayoutMaterializationSupport>
getLayoutMaterializationSupport(VMILayoutAttr sourceLayout,
                                VMILayoutAttr resultLayout,
                                std::string *reason) {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMILayoutMaterializationSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  if (sourceLayout == resultLayout)
    return VMILayoutMaterializationSupport{
        VMILayoutMaterializationSupportKind::Identity};
  if (sourceLayout.isContiguous() && resultLayout.isDeinterleaved() &&
      (resultLayout.getFactor() == 2 || resultLayout.getFactor() == 4))
    return VMILayoutMaterializationSupport{
        VMILayoutMaterializationSupportKind::ContiguousToDeinterleaved};
  if (sourceLayout.isDeinterleaved() && resultLayout.isContiguous() &&
      (sourceLayout.getFactor() == 2 || sourceLayout.getFactor() == 4))
    return VMILayoutMaterializationSupport{
        VMILayoutMaterializationSupportKind::DeinterleavedToContiguous};
  if (sourceLayout.isDeinterleaved() && resultLayout.isDeinterleaved() &&
      (sourceLayout.getFactor() == 2 || sourceLayout.getFactor() == 4) &&
      (resultLayout.getFactor() == 2 || resultLayout.getFactor() == 4))
    return VMILayoutMaterializationSupport{
        VMILayoutMaterializationSupportKind::
            DeinterleavedToDeinterleavedViaContiguous};
  return fail("unsupported source/result layout pair");
}

} // namespace

FailureOr<VMIGroupReduceLayoutFact>
VMILayoutSupport::getPreferredGroupReduceLayoutFact(VMIVRegType sourceType,
                                                    int64_t numGroups,
                                                    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupReduceLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> groupSize =
      getGroupSizeFromNumGroups(sourceType, numGroups, reason);
  if (failed(groupSize))
    return failure();
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart) || *lanesPerPart % 8 != 0)
    return fail("requires element type with known physical VLane width");

  MLIRContext *ctx = sourceType.getContext();
  int64_t vlaneElems = *lanesPerPart / 8;
  VMIGroupReduceLayoutFact fact;
  fact.groupSize = *groupSize;
  fact.lanesPerPart = *lanesPerPart;
  fact.vlaneElems = vlaneElems;

  if (*groupSize == vlaneElems) {
    fact.kind = VMIGroupReduceLayoutKind::OneVLane;
    fact.sourceLayout = VMILayoutAttr::getContiguous(ctx);
    fact.maskLayout = fact.sourceLayout;
    fact.resultLayout =
        VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/8);
    return fact;
  }

  if (*groupSize == 2 * vlaneElems) {
    fact.kind = VMIGroupReduceLayoutKind::TwoVLane;
    fact.sourceLayout =
        VMILayoutAttr::getDeinterleaved(ctx, 2, /*blockElems=*/8);
    fact.maskLayout = fact.sourceLayout;
    fact.resultLayout =
        VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/8);
    return fact;
  }

  if (*groupSize == 4 * vlaneElems) {
    fact.kind = VMIGroupReduceLayoutKind::FourVLane;
    fact.sourceLayout =
        VMILayoutAttr::getDeinterleaved(ctx, 4, /*blockElems=*/8);
    fact.maskLayout = fact.sourceLayout;
    fact.resultLayout =
        VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/8);
    return fact;
  }

  if (*groupSize >= *lanesPerPart && *groupSize % *lanesPerPart == 0) {
    fact.kind = VMIGroupReduceLayoutKind::RowLocal;
    fact.sourceLayout = VMILayoutAttr::getContiguous(ctx);
    fact.maskLayout = fact.sourceLayout;
    fact.resultLayout =
        VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/1);
    return fact;
  }

  return fail("group_reduce layout supports group sizes VLaneElems, "
              "2*VLaneElems, 4*VLaneElems, or full physical chunk multiples");
}

FailureOr<VMICastLayoutFact> VMILayoutSupport::getPreferredCastLayoutFact(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMICastLayoutFact> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0)
    return fail(
        "requires source/result element types with known storage width");
  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("requires source/result lane count to match");

  MLIRContext *ctx = sourceType.getContext();
  VMICastLayoutFact fact;
  fact.sourceBits = sourceBits;
  fact.resultBits = resultBits;

  if (resultBits == 32 && sourceBits == 16) {
    fact.kind = VMICastLayoutKind::Widen2x;
    fact.factor = 2;
    fact.sourceLayout = VMILayoutAttr::getContiguous(ctx);
    fact.resultLayout =
        VMILayoutAttr::getDeinterleaved(ctx, fact.factor, /*blockElems=*/1);
    return fact;
  }

  if (resultBits == 32 && sourceBits == 8) {
    fact.kind = VMICastLayoutKind::Widen4x;
    fact.factor = 4;
    fact.sourceLayout = VMILayoutAttr::getContiguous(ctx);
    fact.resultLayout =
        VMILayoutAttr::getDeinterleaved(ctx, fact.factor, /*blockElems=*/1);
    return fact;
  }

  if (sourceBits == 32 && resultBits == 16) {
    fact.kind = VMICastLayoutKind::Narrow2x;
    fact.factor = 2;
    fact.sourceLayout =
        VMILayoutAttr::getDeinterleaved(ctx, fact.factor, /*blockElems=*/1);
    fact.resultLayout = VMILayoutAttr::getContiguous(ctx);
    return fact;
  }

  if (sourceBits == 32 && resultBits == 8) {
    fact.kind = VMICastLayoutKind::Narrow4x;
    fact.factor = 4;
    fact.sourceLayout =
        VMILayoutAttr::getDeinterleaved(ctx, fact.factor, /*blockElems=*/1);
    fact.resultLayout = VMILayoutAttr::getContiguous(ctx);
    return fact;
  }

  return fail("supports only 8/16-bit <-> 32-bit dense cast layout facts");
}

FailureOr<VMILayoutAttr>
VMILayoutSupport::getWidenSourceLayoutForResultLayout(
    VMIVRegType sourceType, VMIVRegType resultType,
    VMILayoutAttr requestedResultLayout, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMILayoutAttr> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (!requestedResultLayout)
    return fail("requires requested result layout");
  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("requires source/result lane count to match");
  if (requestedResultLayout.isGroupSlots())
    return fail("dense widen relation does not support group_slots layout");
  if (!requestedResultLayout.isContiguous() &&
      (!requestedResultLayout.isDeinterleaved() ||
       requestedResultLayout.getBlockElems() != 1))
    return fail("requires contiguous or deinterleaved result layout with "
                "block_elems=1");

  FailureOr<VMICastLayoutFact> fact =
      getPreferredCastLayoutFact(sourceType, resultType, reason);
  if (failed(fact) || (fact->kind != VMICastLayoutKind::Widen2x &&
                       fact->kind != VMICastLayoutKind::Widen4x))
    return fail("requires supported 8/16-bit to 32-bit widen cast");

  int64_t resultFactor = requestedResultLayout.isDeinterleaved()
                             ? requestedResultLayout.getFactor()
                             : 1;
  if (resultFactor % fact->factor != 0)
    return fail("requested result layout factor is not divisible by widen "
                "factor");

  int64_t sourceFactor = resultFactor / fact->factor;
  if (sourceFactor == 1)
    return VMILayoutAttr::getContiguous(sourceType.getContext());
  if (sourceFactor == 2 || sourceFactor == 4)
    return VMILayoutAttr::getDeinterleaved(sourceType.getContext(),
                                          sourceFactor, /*blockElems=*/1);
  return fail("derived source layout factor is unsupported");
}

FailureOr<VMIContiguousStoreSupport>
VMILayoutSupport::getContiguousStoreSupport(VMIVRegType valueType,
                                            std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIContiguousStoreSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout)
    return fail("requires assigned value layout");
  if (layout.isContiguous())
    return VMIContiguousStoreSupport{
        VMIContiguousStoreSupportKind::ContiguousVsts};
  if (!layout.isDeinterleaved())
    return fail("requires contiguous or deinterleaved value layout");
  if (layout.getBlockElems() != 1)
    return fail("requires block_elems=1 deinterleaved value layout");
  if (failed(checkFullDataPhysicalChunks(valueType, reason)))
    return failure();

  if (layout.getFactor() == 2) {
    if (!hasX2MemoryDistToken(valueType.getElementType()))
      return fail("requires 8/16/32-bit element type for vstsx2 INTLV");
    return VMIContiguousStoreSupport{
        VMIContiguousStoreSupportKind::Deinterleaved2Vstsx2};
  }

  if (layout.getFactor() == 4)
    return VMIContiguousStoreSupport{
        VMIContiguousStoreSupportKind::DeinterleavedMaterializeThenVsts};

  return fail("requires deinterleaved factor 2 or 4");
}

LogicalResult VMILayoutSupport::canFoldContiguousStoreMaterialization(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  if (sourceType.getElementType() != resultType.getElementType())
    return failWithReason("source/result element types must match", reason);
  if (sourceType.getElementCount() != resultType.getElementCount())
    return failWithReason("source/result element counts must match", reason);

  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout || !resultLayout.isContiguous())
    return failWithReason("result layout must be contiguous", reason);

  FailureOr<VMIContiguousStoreSupport> support =
      getContiguousStoreSupport(sourceType, reason);
  if (failed(support))
    return failure();
  if (support->kind == VMIContiguousStoreSupportKind::ContiguousVsts)
    return failWithReason("source layout is already contiguous", reason);

  return success();
}

FailureOr<VMILayoutMaterializationSupport>
VMILayoutSupport::getDataLayoutMaterializationSupport(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMILayoutMaterializationSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementType() != resultType.getElementType())
    return fail("source/result element types must match");
  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("source/result element counts must match");

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  FailureOr<VMILayoutMaterializationSupport> support =
      getLayoutMaterializationSupport(sourceLayout, resultLayout, reason);
  if (failed(support))
    return failure();
  if (failed(checkLayoutMaterializationShape(
          sourceType, resultType, sourceLayout, resultLayout, reason)))
    return failure();
  return support;
}

LogicalResult VMILayoutSupport::canMaterializeDataLayout(
    VMIVRegType sourceType, VMIVRegType resultType, std::string *reason) const {
  if (failed(
          getDataLayoutMaterializationSupport(sourceType, resultType, reason)))
    return failure();
  return success();
}

FailureOr<VMILayoutMaterializationSupport>
VMILayoutSupport::getMaskLayoutMaterializationSupport(
    VMIMaskType sourceType, VMIMaskType resultType, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMILayoutMaterializationSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("source/result mask element counts must match");
  if (sourceType.getGranularity() != resultType.getGranularity())
    return fail("source/result mask granularities must match");

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  FailureOr<VMILayoutMaterializationSupport> support =
      getLayoutMaterializationSupport(sourceLayout, resultLayout, reason);
  if (failed(support))
    return failure();
  if (failed(checkLayoutMaterializationShape(
          sourceType, resultType, sourceLayout, resultLayout, reason)))
    return failure();
  return support;
}

LogicalResult VMILayoutSupport::canMaterializeMaskLayout(
    VMIMaskType sourceType, VMIMaskType resultType, std::string *reason) const {
  if (failed(
          getMaskLayoutMaterializationSupport(sourceType, resultType, reason)))
    return failure();
  return success();
}

FailureOr<VMIMaskGranularityMaterializationSupport>
VMILayoutSupport::getMaskGranularityMaterializationSupport(
    VMIMaskType sourceType, VMIMaskType resultType, std::string *reason) const {
  auto fail = [&](const Twine &message)
      -> FailureOr<VMIMaskGranularityMaterializationSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("source/result mask element counts must match");
  if (sourceType.getLayoutAttr() != resultType.getLayoutAttr())
    return fail("source/result mask layouts must match");
  if (!VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) ||
      !VMIMaskType::isConcreteGranularity(resultType.getGranularity()))
    return fail("requires concrete b8/b16/b32 source and result granularities");
  if (sourceType.getGranularity() == resultType.getGranularity())
    return VMIMaskGranularityMaterializationSupport{
        VMIMaskGranularityMaterializationSupportKind::Identity};

  return VMIMaskGranularityMaterializationSupport{
      VMIMaskGranularityMaterializationSupportKind::PredicateCast};
}

LogicalResult VMILayoutSupport::canMaterializeMaskGranularity(
    VMIMaskType sourceType, VMIMaskType resultType, std::string *reason) const {
  if (failed(getMaskGranularityMaterializationSupport(sourceType, resultType,
                                                      reason)))
    return failure();
  return success();
}

FailureOr<VMIGroupSlotLoadSupport> VMILayoutSupport::getGroupSlotLoadSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupSlotLoadOp op,
    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupSlotLoadSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (resultType.getElementCount() != numGroups)
    return fail("requires result logical lane count to match num_groups");
  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout || !layout.isGroupSlots() ||
      layout.getNumGroups() != numGroups || layout.getSlots() <= 0)
    return fail("requires explicit group_slots result layout matching "
                "num_groups");

  if (layout.getSlots() != 8 && layout.getSlots() != 1)
    return fail("supports only slots=8 or slots=1 group_slot_load layouts");

  if (!capabilities.supportsDirectMemory(op.getSource().getType(), "source")
           .isSupported())
    return fail("requires supported direct memory source");
  if (!isa<PtrType>(op.getSource().getType()))
    return fail("requires !pto.ptr source for vsldb lowering");

  std::optional<int64_t> stride =
      getConstantIndexValue(op.getSourceGroupStride());
  if (layout.getSlots() == 8) {
    if (!stride || *stride != 1)
      return fail("slots=8 group_slot_load requires constant unit "
                  "source_group_stride");
    return VMIGroupSlotLoadSupport{
        VMIGroupSlotLoadSupportKind::Slots8UnitStrideVsldb};
  }

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits == 0 || 256 % elementBits != 0)
    return fail("slots=1 group_slot_load requires an 8/16/32-bit element "
                "type");
  int64_t alignedStrideElems = 256 / elementBits;
  if (!stride || *stride <= 0 || *stride % alignedStrideElems != 0)
    return fail(Twine("slots=1 group_slot_load currently lowers as one "
                      "lane-0 vsldb per group and requires constant "
                      "positive source_group_stride divisible by ") +
                Twine(alignedStrideElems) +
                " elements for 32B load alignment; packed or unaligned "
                "scalar load lowering is not implemented");

  return VMIGroupSlotLoadSupport{
      VMIGroupSlotLoadSupportKind::Slots1AlignedLane0Vsldb};
}

FailureOr<VMIGroupLoadSupport> VMILayoutSupport::getGroupLoadSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupLoadOp op,
    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupLoadSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout || !layout.isDeinterleaved() || layout.getBlockElems() != 8 ||
      !resultType.getElementType().isF32())
    return fail("requires deinterleaved block8 f32 result layout");

  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize))
    return failure();

  if ((*groupSize != 16 || layout.getFactor() != 2) &&
      (*groupSize != 32 || layout.getFactor() != 4))
    return fail("block8 strided group_load requires S=16/factor=2 or "
                "S=32/factor=4");

  if (!capabilities.supportsDirectMemory(op.getSource().getType(), "source")
           .isSupported())
    return fail("requires supported direct memory source");
  if (!isa<PtrType>(op.getSource().getType()))
    return fail("block8 strided group_load requires !pto.ptr source");

  if (op.getNumGroupsAttr().getInt() % 8 != 0)
    return fail("block8 strided group_load requires num_groups multiple of 8");

  std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
  if (!rowStride || *rowStride <= 0 || *rowStride % 8 != 0)
    return fail("block8 strided group_load requires constant positive "
                "row_stride divisible by 8 f32 elements");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return fail(Twine("block8 strided group_load requires full physical "
                      "result chunks; ") +
                fullChunkReason);

  if (*groupSize == 16)
    return VMIGroupLoadSupport{VMIGroupLoadSupportKind::S16Block8Vsldb};
  return VMIGroupLoadSupport{VMIGroupLoadSupportKind::S32Block8Vsldb};
}

FailureOr<VMIGroupSlotsStoreSupport>
VMILayoutSupport::getGroupSlotsStoreSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupStoreOp op,
    std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupSlotsStoreSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (!layout || !layout.isGroupSlots())
    return fail("requires group_slots value layout");

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (layout.getNumGroups() != numGroups)
    return fail("group_slots group_store requires layout num_groups to "
                "match op num_groups");

  VMICapabilityResult elementCapability = capabilities.supportsElementType(
      valueType.getElementType(), VMIElementPurpose::PredicateMask);
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);

  FailureOr<int64_t> arity = getVMIPhysicalArity(valueType);
  if (failed(arity) || *arity < 1)
    return fail("requires computable non-empty physical vreg parts");

  if (layout.getSlots() == 1) {
    if (*arity != numGroups)
      return fail("slots=1 group_store requires one physical part per "
                  "group");
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(valueType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0)
      return fail("slots=1 group_store requires an 8/16/32-bit element "
                  "type");
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueType.getElementType());
    if (rowStride && *rowStride == 1 && succeeded(lanesPerPart) &&
        numGroups <= *lanesPerPart)
      return VMIGroupSlotsStoreSupport{
          VMIGroupSlotsStoreSupportKind::Slots1PackedUnitStrideVsts};
    if (rowStride && *rowStride <= 0)
      return fail("slots=1 group_store requires positive row_stride when "
                  "row_stride is constant");
    return VMIGroupSlotsStoreSupport{
        VMIGroupSlotsStoreSupportKind::Slots1PointVsts};
  }

  if (layout.getSlots() == 8) {
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (!rowStride || *rowStride != 1)
      return fail("slots=8 group_store currently requires constant unit "
                  "row_stride");
    if (*arity != ceilDivNonNegative(numGroups, 8))
      return fail("slots=8 group_store arity must equal ceil(num_groups / "
                  "8)");
    return VMIGroupSlotsStoreSupport{
        VMIGroupSlotsStoreSupportKind::Slots8UnitStrideVsts};
  }

  return fail("group_slots group_store currently supports only slots=1 or "
              "unit-stride slots=8");
}

FailureOr<VMIGroupReduceAddFSupport> getGroupReduceAddSupportImpl(
    const VMITargetCapabilityRegistry &capabilities, Operation *op,
    VMIVRegType sourceType, VMIMaskType maskType, VMIVRegType resultType,
    int64_t numGroups, bool requiresReassoc, VMIReductionKind reductionKind,
    std::string *reason) {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupReduceAddFSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (requiresReassoc && !op->hasAttr("reassoc"))
    return fail("requires reassoc attr for pair-wise floating-point "
                "reduction");

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout)
    return fail("requires assigned source, mask, and result layouts");
  if (!resultLayout.isGroupSlots() || resultLayout.getNumGroups() != numGroups)
    return fail("requires group_slots result layout matching num_groups");
  if (resultLayout.getSlots() != 8 && resultLayout.getSlots() != 1) {
    FailureOr<int64_t> groupSize =
        getGroupSizeFromNumGroups(sourceType, numGroups, reason);
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(sourceType.getElementType());
    int64_t vlaneElems = succeeded(lanesPerPart) && *lanesPerPart % 8 == 0
                             ? *lanesPerPart / 8
                             : -1;
    if (succeeded(groupSize) && resultLayout.getSlots() <= 0 &&
        (*groupSize != vlaneElems && *groupSize != 2 * vlaneElems &&
         *groupSize != 4 * vlaneElems))
      return fail("stable group_reduce_add slots=8 support group "
                  "sizes VLaneElems, 2*VLaneElems, or 4*VLaneElems");
    return fail("stable group_reduce_add layout support currently requires "
                "result layout slots=8 or slots=1");
  }

  VMICapabilityResult elementCapability =
      capabilities.supportsReductionElementType(reductionKind,
                                                sourceType.getElementType());
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);
  if (sourceType.getElementType() != resultType.getElementType())
    return fail("stable group_reduce_add layout support requires matching "
                "source/result element types");
  if (resultType.getElementCount() != numGroups)
    return fail("requires result lane count to match num_groups");

  FailureOr<int64_t> groupSize =
      getGroupSizeFromNumGroups(sourceType, numGroups, reason);
  if (failed(groupSize))
    return failure();
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart) || *lanesPerPart % 8 != 0)
    return fail("requires element type with known physical VLane width");
  int64_t vlaneElems = *lanesPerPart / 8;

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(maskArity) || failed(resultArity))
    return fail("requires computable source/mask/result physical arity");
  if (*sourceArity < 1 || *maskArity != *sourceArity)
    return fail("requires matching non-empty source/mask physical arity");

  if (resultLayout.getSlots() == 1) {
    if (failed(lanesPerPart) || *groupSize < *lanesPerPart ||
        *groupSize % *lanesPerPart != 0)
      return fail("stable group_reduce_add slots=1 support group "
                  "sizes that are multiples of one physical chunk");
    if (!sourceLayout.isContiguous() || !maskLayout.isContiguous())
      return fail("slots=1 group_reduce_add requires contiguous source/mask "
                  "layouts");
    if (*resultArity != numGroups)
      return fail("slots=1 group_reduce_add requires one physical result "
                  "part per group");
    std::string sourceFullReason;
    if (failed(checkFullDataPhysicalChunks(sourceType, &sourceFullReason)))
      return fail(Twine("slots=1 group_reduce_add requires full source "
                        "chunks; ") +
                  sourceFullReason);
    return VMIGroupReduceAddFSupport{
        VMIGroupReduceAddFSupportKind::ContiguousVcaddRows};
  }

  if (*groupSize == vlaneElems) {
    if (!sourceLayout.isContiguous() || !maskLayout.isContiguous())
      return fail("one-vlane group_reduce_add requires contiguous source/mask "
                  "layouts");
    std::string sourceFullReason;
    if (failed(checkFullDataPhysicalChunks(sourceType, &sourceFullReason)))
      return fail(Twine("one-vlane group_reduce_add requires full source "
                        "chunks; ") +
                  sourceFullReason);
    if (*resultArity != *sourceArity)
      return fail("one-vlane group_reduce_add requires source/result physical "
                  "arity to match");
    return VMIGroupReduceAddFSupport{
        VMIGroupReduceAddFSupportKind::OneVLaneVcgadd};
  }

  if (*groupSize == 2 * vlaneElems) {
    if (!sourceLayout.isDeinterleaved() || sourceLayout.getFactor() != 2 ||
        (sourceLayout.getBlockElems() != 1 &&
         sourceLayout.getBlockElems() != 8))
      return fail("two-vlane group_reduce_add requires source layout "
                  "deinterleaved=2 with block_elems=1 or block_elems=8");
    if (!maskLayout.isDeinterleaved() || maskLayout.getFactor() != 2 ||
        maskLayout.getBlockElems() != sourceLayout.getBlockElems())
      return fail("two-vlane group_reduce_add requires matching mask layout "
                  "deinterleaved=2 with the same block_elems");
    int64_t expectedResultArity = ceilDivNonNegative(numGroups, 8);
    if (*resultArity != expectedResultArity || *sourceArity != *resultArity * 2)
      return fail(
          "two-vlane group_reduce_add requires two source/mask parts per "
          "result part");
    return VMIGroupReduceAddFSupport{
        VMIGroupReduceAddFSupportKind::TwoVLaneDeinterleaved2VcgaddVadd};
  }

  if (*groupSize == 4 * vlaneElems) {
    if (!sourceLayout.isDeinterleaved() || sourceLayout.getFactor() != 4 ||
        (sourceLayout.getBlockElems() != 1 &&
         sourceLayout.getBlockElems() != 8))
      return fail("four-vlane group_reduce_add requires source layout "
                  "deinterleaved=4 with block_elems=1 or block_elems=8");
    if (!maskLayout.isDeinterleaved() || maskLayout.getFactor() != 4 ||
        maskLayout.getBlockElems() != sourceLayout.getBlockElems())
      return fail("four-vlane group_reduce_add requires matching mask layout "
                  "deinterleaved=4 with the same block_elems");
    int64_t expectedResultArity = ceilDivNonNegative(numGroups, 8);
    if (*resultArity != expectedResultArity || *sourceArity != *resultArity * 4)
      return fail(
          "four-vlane group_reduce_add requires four source/mask parts per "
          "result part");
    return VMIGroupReduceAddFSupport{
        VMIGroupReduceAddFSupportKind::FourVLaneDeinterleaved4VcgaddTree};
  }

  return fail("stable group_reduce_add slots=8 support group sizes "
              "VLaneElems, 2*VLaneElems, or 4*VLaneElems");
}

FailureOr<VMIGroupReduceAddFSupport>
VMILayoutSupport::getGroupReduceAddFSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupReduceAddFOp op,
    std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      capabilities, op.getOperation(),
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), /*requiresReassoc=*/true,
      VMIReductionKind::GroupAddF, reason);
}

FailureOr<VMIGroupReduceAddFSupport>
VMILayoutSupport::getGroupReduceMaxFSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupReduceMaxFOp op,
    std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      capabilities, op.getOperation(),
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), /*requiresReassoc=*/false,
      VMIReductionKind::GroupMaxF, reason);
}

FailureOr<VMIGroupReduceAddFSupport>
VMILayoutSupport::getGroupReduceAddISupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupReduceAddIOp op,
    std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      capabilities, op.getOperation(),
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), /*requiresReassoc=*/false,
      VMIReductionKind::GroupAddI, reason);
}

FailureOr<VMIGroupReduceAddFSupport>
VMILayoutSupport::getGroupReduceMaxISupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupReduceMaxIOp op,
    std::string *reason) const {
  return getGroupReduceAddSupportImpl(
      capabilities, op.getOperation(),
      cast<VMIVRegType>(op.getSource().getType()),
      cast<VMIMaskType>(op.getMask().getType()),
      cast<VMIVRegType>(op.getResult().getType()),
      op.getNumGroupsAttr().getInt(), /*requiresReassoc=*/false,
      VMIReductionKind::GroupMaxI, reason);
}

FailureOr<VMIGroupBroadcastSupport> VMILayoutSupport::getGroupBroadcastSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupBroadcastOp op,
    std::string *reason) const {
  return getGroupBroadcastSupport(capabilities,
                                  cast<VMIVRegType>(op.getSource().getType()),
                                  cast<VMIVRegType>(op.getResult().getType()),
                                  op.getNumGroupsAttr().getInt(), reason);
}

FailureOr<VMIGroupBroadcastLoadSupport>
VMILayoutSupport::getGroupBroadcastLoadSupport(
    const VMITargetCapabilityRegistry &capabilities,
    VMIGroupBroadcastLoadOp op, std::string *reason) const {
  auto fail =
      [&](const Twine &message) -> FailureOr<VMIGroupBroadcastLoadSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (numGroups <= 0)
    return fail("requires positive num_groups");
  if (resultType.getElementCount() % numGroups != 0)
    return fail("requires num_groups to evenly divide result lane count");
  if (!capabilities.supportsDirectMemory(op.getSource().getType(), "source")
           .isSupported())
    return fail("requires supported direct memory source");
  if (!isa<PtrType>(op.getSource().getType()))
    return fail("requires !pto.ptr source for E2B lowering");

  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (elementBits != 16 && elementBits != 32)
    return fail("E2B lowering currently supports only 16-bit and 32-bit "
                "element types");
  int64_t directGroupSize = 256 / elementBits;

  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout)
    return fail("E2B lowering requires assigned result layout");
  bool contiguousPacketLayout = layout.isContiguous();
  bool splitPacketLayout = layout.isDeinterleaved() && layout.getFactor() == 2 &&
                           layout.getBlockElems() == 1;
  if (!contiguousPacketLayout && !splitPacketLayout)
    return fail("E2B lowering requires contiguous result layout for "
                "direct group size or deinterleaved=2, block_elems=1 "
                "result layout for split group size");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return fail(Twine("requires full result physical chunks; ") +
                fullChunkReason);

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  if (failed(lanesPerPart) || *lanesPerPart != (2048 / elementBits))
    return fail("E2B lowering requires one full 256-byte vreg per physical "
                "part");

  int64_t groupSize = resultType.getElementCount() / numGroups;
  if (contiguousPacketLayout && groupSize != directGroupSize)
    return fail("E2B contiguous lowering requires logical group size matching "
                "the element-width direct packet size");
  if (splitPacketLayout && groupSize != 2 * directGroupSize)
    return fail("E2B deinterleaved=2 lowering requires logical group size "
                "matching the element-width split packet size");
  if (numGroups % 8 != 0)
    return fail("E2B lowering requires num_groups to be a multiple of 8");

  std::optional<int64_t> stride =
      getConstantIndexValue(op.getSourceGroupStride());
  if (!stride || *stride != 1)
    return fail("E2B lowering requires constant unit source_group_stride");

  return VMIGroupBroadcastLoadSupport{
      VMIGroupBroadcastLoadSupportKind::E2BVlds};
}

FailureOr<VMIGroupBroadcastSupport> VMILayoutSupport::getGroupBroadcastSupport(
    const VMITargetCapabilityRegistry &capabilities, VMIVRegType sourceType,
    VMIVRegType resultType, int64_t numGroups, std::string *reason) const {
  (void)capabilities;
  auto fail = [&](const Twine &message) -> FailureOr<VMIGroupBroadcastSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementType() != resultType.getElementType())
    return fail("requires source/result element type to match");
  if (sourceType.getElementCount() != numGroups)
    return fail("requires source lane count to match num_groups");
  if (resultType.getElementCount() % numGroups != 0)
    return fail("requires num_groups to evenly divide result lane count");

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  if (!sourceLayout.isGroupSlots() || sourceLayout.getNumGroups() != numGroups)
    return fail("requires matching num_groups source layout");
  if (resultLayout.isGroupSlots())
    return fail("requires dense result layout");
  if (sourceLayout.getSlots() > 0 && sourceLayout.getSlots() != 8 &&
      sourceLayout.getSlots() != 1)
    return fail("supports only slots=8 or slots=1 group_broadcast source "
                "layouts");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return fail(Twine("requires full result physical chunks; ") +
                fullChunkReason);

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  FailureOr<int64_t> resultLanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  if (failed(lanesPerPart) || failed(resultLanesPerPart) ||
      *lanesPerPart != *resultLanesPerPart)
    return fail("requires matching physical lanes per part");

  FailureOr<int64_t> groupSize =
      getGroupSizeFromNumGroups(resultType, numGroups, reason);
  if (failed(groupSize))
    return failure();
  if (*lanesPerPart % *groupSize != 0 && *groupSize % *lanesPerPart != 0)
    return fail("requires derived group size to divide or be a multiple of "
                "physical lanes per part");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires known result layout factor");
  if (*resultFactor == 1)
    return VMIGroupBroadcastSupport{
        VMIGroupBroadcastSupportKind::GroupSlotsVselr};

  bool blockFragmentSmallGroup =
      resultLayout.isDeinterleaved() && resultLayout.getBlockElems() > 1 &&
      *groupSize < *lanesPerPart &&
      *lanesPerPart % resultLayout.getBlockElems() == 0;
  if (blockFragmentSmallGroup)
    return VMIGroupBroadcastSupport{
        VMIGroupBroadcastSupportKind::GroupSlotsVselr};

  bool deinterleavedSmallGroup =
      resultLayout.isDeinterleaved() && resultLayout.getBlockElems() == 1 &&
      *groupSize < *lanesPerPart && *groupSize >= *resultFactor &&
      *groupSize % *resultFactor == 0 &&
      *lanesPerPart % (*groupSize / *resultFactor) == 0;
  if (deinterleavedSmallGroup)
    return VMIGroupBroadcastSupport{
        VMIGroupBroadcastSupportKind::GroupSlotsVselr};

  int64_t logicalSpanPerResultChunk = *lanesPerPart * *resultFactor;
  if (*groupSize < *lanesPerPart || *groupSize % logicalSpanPerResultChunk != 0)
    return fail("deinterleaved result requires every physical result chunk to "
                "stay within one logical group");

  return VMIGroupBroadcastSupport{
      VMIGroupBroadcastSupportKind::GroupSlotsVselr};
}

FailureOr<VMITruncFSupport>
VMILayoutSupport::getTruncFSupport(VMITruncFOp op, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMITruncFSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (!sourceLayout || !resultLayout || failed(sourceArity) ||
      failed(resultArity))
    return fail("requires assigned source/result layouts and computable "
                "physical arity");

  if (sourceLayout.isGroupSlots() || resultLayout.isGroupSlots()) {
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    if (!sourceLayout.isGroupSlots() || !resultLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != 1 || resultLayout.getSlots() != 1 ||
        !sourceType.getElementType().isF32() || resultBits != 16 ||
        *sourceArity != *resultArity)
      return fail("group-slot truncf requires matching "
                  "group_slots(num_groups=G, slots=1) source/result layouts, "
                  "f32 source, f16 result, and matching physical arity");
    return VMITruncFSupport{VMITruncFSupportKind::GroupSlots1F32ToF16};
  }

  if (!sourceLayout.isDeinterleaved() || !resultLayout.isContiguous() ||
      !sourceType.getElementType().isF32())
    return fail("requires f32 deinterleaved source and contiguous result");

  FailureOr<VMICastLayoutFact> fact =
      getPreferredCastLayoutFact(sourceType, resultType, reason);
  if (failed(fact) || (fact->kind != VMICastLayoutKind::Narrow2x &&
                       fact->kind != VMICastLayoutKind::Narrow4x))
    return fail("unsupported deinterleaved truncf factor, arity, or result "
                "element width");

  if (fact->kind == VMICastLayoutKind::Narrow2x &&
      sourceLayout.getFactor() == fact->factor &&
      *sourceArity == fact->factor * *resultArity)
    return VMITruncFSupport{
        VMITruncFSupportKind::Deinterleaved2F32ToContiguousF16};
  if (fact->kind == VMICastLayoutKind::Narrow4x &&
      sourceLayout.getFactor() == fact->factor &&
      *sourceArity == fact->factor * *resultArity)
    return VMITruncFSupport{
        VMITruncFSupportKind::Deinterleaved4F32ToContiguousF8};

  return fail("unsupported deinterleaved truncf factor, arity, or result "
              "element width");
}

FailureOr<VMIExtFSupport>
VMILayoutSupport::getExtFSupport(VMIExtFOp op, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIExtFSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (!sourceLayout || !resultLayout || failed(sourceArity) ||
      failed(resultArity))
    return fail("requires assigned source/result layouts and computable "
                "physical arity");
  if (!resultLayout.isDeinterleaved() || resultLayout.getBlockElems() != 1 ||
      !(sourceLayout.isContiguous() ||
        (sourceLayout.isDeinterleaved() &&
         sourceLayout.getBlockElems() == 1)) ||
      !resultType.getElementType().isF32())
    return fail("requires contiguous or deinterleaved source layout and "
                "deinterleaved f32 result layout with block_elems=1");

  FailureOr<VMICastLayoutFact> fact =
      getPreferredCastLayoutFact(sourceType, resultType, reason);
  if (failed(fact) || (fact->kind != VMICastLayoutKind::Widen2x &&
                       fact->kind != VMICastLayoutKind::Widen4x))
    return fail("unsupported extf source element width, result factor, or "
                "physical arity");

  int64_t sourceFactor =
      sourceLayout.isDeinterleaved() ? sourceLayout.getFactor() : 1;
  if (resultLayout.getFactor() != sourceFactor * fact->factor ||
      *resultArity != fact->factor * *sourceArity)
    return fail("unsupported extf source/result layout factor or physical "
                "arity");

  if (fact->kind == VMICastLayoutKind::Widen2x)
    return VMIExtFSupport{VMIExtFSupportKind::ContiguousF16ToDeinterleaved2F32};
  if (fact->kind == VMICastLayoutKind::Widen4x)
    return VMIExtFSupport{VMIExtFSupportKind::ContiguousF8ToDeinterleaved4F32};

  return fail("unsupported extf source element width");
}

template <typename OpT>
static FailureOr<VMIExtISupport> getExtISupportImpl(OpT op,
                                                    std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIExtISupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (!sourceLayout || !resultLayout || failed(sourceArity) ||
      failed(resultArity))
    return fail("requires assigned source/result layouts and computable "
                "physical arity");

  if (sourceLayout.isGroupSlots() && resultLayout.isGroupSlots()) {
    if (!isa<IntegerType>(sourceType.getElementType()) ||
        !isa<IntegerType>(resultType.getElementType()))
      return fail("requires integer source/result element types");
    if (sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != 8 || resultLayout.getSlots() != 8)
      return fail("requires matching group_slots(num_groups=G, slots=8) "
                  "source/result layouts");
    if (*sourceArity != *resultArity)
      return fail("group_slots integer extension requires matching physical "
                  "arity");

    unsigned sourceBits = pto::getPTOStorageElemBitWidth(
        sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultType.getElementType());
    if (resultBits != 32)
      return fail("group_slots integer extension requires 32-bit result "
                  "element type");
    if (sourceBits == 16)
      return VMIExtISupport{VMIExtISupportKind::GroupSlotsI16ToI32};
    if (sourceBits == 8)
      return VMIExtISupport{VMIExtISupportKind::GroupSlotsI8ToI32};
    return fail("group_slots integer extension source must be 8-bit or "
                "16-bit");
  }

  if (!resultLayout.isDeinterleaved() || resultLayout.getBlockElems() != 1 ||
      !(sourceLayout.isContiguous() ||
        (sourceLayout.isDeinterleaved() &&
         sourceLayout.getBlockElems() == 1)) ||
      !isa<IntegerType>(sourceType.getElementType()) ||
      !isa<IntegerType>(resultType.getElementType()))
    return fail("requires contiguous or deinterleaved integer source layout "
                "and deinterleaved integer result layout with block_elems=1");

  FailureOr<VMICastLayoutFact> fact =
      VMILayoutSupport().getPreferredCastLayoutFact(sourceType, resultType,
                                                    reason);
  if (failed(fact) || (fact->kind != VMICastLayoutKind::Widen2x &&
                       fact->kind != VMICastLayoutKind::Widen4x))
    return fail("unsupported integer extension source/result element width, "
                "result factor, or physical arity");

  int64_t sourceFactor =
      sourceLayout.isDeinterleaved() ? sourceLayout.getFactor() : 1;
  if (resultLayout.getFactor() != sourceFactor * fact->factor ||
      *resultArity != fact->factor * *sourceArity)
    return fail("unsupported integer extension source/result layout factor or "
                "physical arity");

  if (fact->kind == VMICastLayoutKind::Widen2x)
    return VMIExtISupport{VMIExtISupportKind::ContiguousI16ToDeinterleaved2I32};
  if (fact->kind == VMICastLayoutKind::Widen4x)
    return VMIExtISupport{VMIExtISupportKind::ContiguousI8ToDeinterleaved4I32};

  return fail("unsupported integer extension source/result element width");
}

FailureOr<VMIExtISupport>
VMILayoutSupport::getExtSISupport(VMIExtSIOp op, std::string *reason) const {
  return getExtISupportImpl(op, reason);
}

FailureOr<VMIExtISupport>
VMILayoutSupport::getExtUISupport(VMIExtUIOp op, std::string *reason) const {
  return getExtISupportImpl(op, reason);
}

FailureOr<VMITruncISupport>
VMILayoutSupport::getTruncISupport(VMITruncIOp op, std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMITruncISupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (!sourceLayout || !resultLayout || failed(sourceArity) ||
      failed(resultArity))
    return fail("requires assigned source/result layouts and computable "
                "physical arity");
  if (!isa<IntegerType>(sourceType.getElementType()) ||
      !isa<IntegerType>(resultType.getElementType()))
    return fail("requires integer source and result element types");

  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());

  if (sourceLayout.isGroupSlots() || resultLayout.isGroupSlots()) {
    if (!sourceLayout.isGroupSlots() || !resultLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        sourceBits != 32 || (resultBits != 16 && resultBits != 8) ||
        *sourceArity != *resultArity)
      return fail("group-slot trunci requires matching "
                  "group_slots(num_groups=G, slots=1 or 8) source/result layouts, "
                  "32-bit integer source, 8/16-bit integer result, and "
                  "matching physical arity");
    return VMITruncISupport{VMITruncISupportKind::GroupSlots1I32ToNarrow};
  }

  if (!sourceLayout.isDeinterleaved() || !resultLayout.isContiguous() ||
      *resultArity != 1)
    return fail("requires integer deinterleaved source and contiguous "
                "integer result");

  FailureOr<VMICastLayoutFact> fact =
      getPreferredCastLayoutFact(sourceType, resultType, reason);
  if (failed(fact) || (fact->kind != VMICastLayoutKind::Narrow2x &&
                       fact->kind != VMICastLayoutKind::Narrow4x))
    return fail("unsupported deinterleaved trunci factor, arity, result "
                "element width, or result signedness; 32-bit to 8-bit integer "
                "narrowing requires unsigned i8 result");

  if (fact->kind == VMICastLayoutKind::Narrow2x &&
      sourceLayout.getFactor() == fact->factor && *sourceArity == fact->factor)
    return VMITruncISupport{
        VMITruncISupportKind::Deinterleaved2I32ToContiguousI16};
  if (fact->kind == VMICastLayoutKind::Narrow4x &&
      sourceLayout.getFactor() == fact->factor &&
      *sourceArity == fact->factor &&
      cast<IntegerType>(resultType.getElementType()).isUnsigned())
    return VMITruncISupport{
        VMITruncISupportKind::Deinterleaved4I32ToContiguousI8};

  return fail("unsupported deinterleaved trunci factor, arity, result element "
              "width, or result signedness; 32-bit to 8-bit integer narrowing "
              "requires unsigned i8 result");
}

FailureOr<VMIBitcastSupport>
VMILayoutSupport::getBitcastSupport(VMIBitcastOp op,
                                    std::string *reason) const {
  auto fail = [&](const Twine &message) -> FailureOr<VMIBitcastSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source and result layouts");
  if (sourceLayout != resultLayout)
    return fail("requires matching source and result layouts");

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(resultArity))
    return fail("requires computable source and result physical arity");
  if (*sourceArity != *resultArity)
    return fail("requires source and result to have the same physical arity");

  FailureOr<SmallVector<int64_t>> sourceBits =
      getPhysicalLogicalBitFootprint(sourceType);
  FailureOr<SmallVector<int64_t>> resultBits =
      getPhysicalLogicalBitFootprint(resultType);
  if (failed(sourceBits) || failed(resultBits))
    return fail("requires computable physical logical bit footprints");
  if (sourceBits->size() != resultBits->size())
    return fail("requires source and result physical footprint counts to "
                "match");
  for (auto [source, result] : llvm::zip_equal(*sourceBits, *resultBits)) {
    if (source != result)
      return fail("requires matching logical bit footprint in every physical "
                  "chunk");
  }

  return VMIBitcastSupport{VMIBitcastSupportKind::PerPartVbitcast};
}

template <typename OpTy>
static FailureOr<VMIHistogramSupport>
getHistogramSupportImpl(OpTy op, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIHistogramSupport> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto accType = cast<VMIVRegType>(op.getAcc().getType());
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  VMILayoutAttr accLayout = accType.getLayoutAttr();
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!accLayout || !sourceLayout || !maskLayout || !resultLayout)
    return fail("requires assigned acc/source/mask/result layouts");
  if (!accLayout.isContiguous() || !sourceLayout.isContiguous() ||
      !maskLayout.isContiguous() || !resultLayout.isContiguous())
    return fail("requires contiguous acc, source, mask, and result layouts");
  if (maskType.getGranularity() != "b8")
    return fail("requires b8 mask granularity");
  if (maskType.getElementCount() != sourceType.getElementCount())
    return fail("requires mask lane count to match source lane count");

  auto accElem = dyn_cast<IntegerType>(accType.getElementType());
  auto sourceElem = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!accElem || !accElem.isUnsigned() || accElem.getWidth() != 16 ||
      accType.getElementCount() != 256 || resultType != accType)
    return fail("requires contiguous 256xui16 acc/result");
  if (!sourceElem || !sourceElem.isUnsigned() || sourceElem.getWidth() != 8)
    return fail("requires unsigned 8-bit source elements");

  FailureOr<int64_t> accArity = getVMIPhysicalArity(accType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(accArity) || failed(resultArity) || failed(sourceArity) ||
      failed(maskArity))
    return fail("requires computable physical arity");
  if (*accArity != 2 || *resultArity != 2)
    return fail("requires acc/result to physicalize to two 128xui16 parts");
  if (*sourceArity != *maskArity)
    return fail("requires source and mask physical arity to match");
  if (*sourceArity < 1)
    return fail("requires at least one source physical chunk");

  return VMIHistogramSupport{VMIHistogramSupportKind::Full256BinDhist};
}

FailureOr<VMIHistogramSupport>
VMILayoutSupport::getDhistSupport(VMIDhistOp op, std::string *reason) const {
  return getHistogramSupportImpl(op, reason);
}

FailureOr<VMIHistogramSupport>
VMILayoutSupport::getChistSupport(VMIChistOp op, std::string *reason) const {
  if (reason)
    *reason = "CHISTv2 cumulative high-range semantics are not classified";
  return failure();
}
