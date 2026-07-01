// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutAssignment.cpp - Assign VMI layouts ----------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "PTO/Transforms/VMITargetCapabilities.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILAYOUTASSIGNMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

struct DataNode {
  Value value;
  VMIVRegType type;
  unsigned parent = 0;
  VMILayoutAttr naturalLayout;
};

struct MaskNode {
  Value value;
  VMIMaskType type;
  unsigned parent = 0;
  VMILayoutAttr requestedLayout;
  std::string requestedGranularity;
};

struct DataUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
};

struct MaskUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
  std::string granularity;
};

static unsigned getElementBitWidth(Type type) {
  if (isa<IndexType>(type))
    return 64;
  return pto::getPTOStorageElemBitWidth(type);
}

static StringRef getMaskGranularityForElement(Type elementType) {
  switch (getElementBitWidth(elementType)) {
  case 8:
    return "b8";
  case 16:
    return "b16";
  case 32:
    return "b32";
  default:
    return "";
  }
}

static std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttr.getInt();
  return std::nullopt;
}

static bool isLane0SplatShuffle(VMIShuffleOp op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  ArrayRef<int64_t> indices = op.getIndices();
  return sourceType.getElementCount() == 1 && !indices.empty() &&
         llvm::all_of(indices, [](int64_t index) { return index == 0; });
}

bool containsVMIType(Type type) {
  if (isa<VMIVRegType, VMIMaskType>(type))
    return true;
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(),
                        [](Type result) { return containsVMIType(result); });
  }
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return containsVMIType(shapedType.getElementType());
  return false;
}

struct LayoutSolver {
  explicit LayoutSolver(ModuleOp module,
                        const VMITargetCapabilityRegistry &capabilities)
      : module(module), ctx(module.getContext()), capabilities(capabilities) {}

  unsigned addDataValue(Value value) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return ~0u;
    auto [it, inserted] = dataIds.try_emplace(value, dataNodes.size());
    if (inserted)
      dataNodes.push_back(
          DataNode{value, type, it->second, type.getLayoutAttr()});
    return it->second;
  }

  unsigned addMaskValue(Value value) {
    auto type = dyn_cast<VMIMaskType>(value.getType());
    if (!type)
      return ~0u;
    auto [it, inserted] = maskIds.try_emplace(value, maskNodes.size());
    if (inserted) {
      std::string granularity;
      if (VMIMaskType::isConcreteGranularity(type.getGranularity()))
        granularity = type.getGranularity().str();
      maskNodes.push_back(
          MaskNode{value, type, it->second, type.getLayoutAttr(), granularity});
    }
    return it->second;
  }

  unsigned find(unsigned id) {
    if (dataNodes[id].parent == id)
      return id;
    dataNodes[id].parent = find(dataNodes[id].parent);
    return dataNodes[id].parent;
  }

  unsigned findMask(unsigned id) {
    if (maskNodes[id].parent == id)
      return id;
    maskNodes[id].parent = findMask(maskNodes[id].parent);
    return maskNodes[id].parent;
  }

  LogicalResult unite(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addDataValue(lhs);
    unsigned rhsId = addDataValue(rhs);
    if (lhsId == ~0u || rhsId == ~0u)
      return success();
    unsigned lhsRoot = find(lhsId);
    unsigned rhsRoot = find(rhsId);
    if (lhsRoot == rhsRoot)
      return success();
    dataNodes[rhsRoot].parent = lhsRoot;
    VMILayoutAttr lhsNatural = dataNodes[lhsRoot].naturalLayout;
    VMILayoutAttr rhsNatural = dataNodes[rhsRoot].naturalLayout;
    if (lhsNatural && rhsNatural && lhsNatural != rhsNatural)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting natural layouts "
             << lhsNatural << " and " << rhsNatural;
    if (!lhsNatural)
      dataNodes[lhsRoot].naturalLayout = rhsNatural;
    return success();
  }

  LogicalResult uniteMask(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addMaskValue(lhs);
    unsigned rhsId = addMaskValue(rhs);
    if (lhsId == ~0u || rhsId == ~0u)
      return success();
    unsigned lhsRoot = findMask(lhsId);
    unsigned rhsRoot = findMask(rhsId);
    if (lhsRoot == rhsRoot)
      return success();

    MaskNode &lhsNode = maskNodes[lhsRoot];
    MaskNode &rhsNode = maskNodes[rhsRoot];
    if (lhsNode.requestedLayout && rhsNode.requestedLayout &&
        lhsNode.requestedLayout != rhsNode.requestedLayout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting mask layouts "
             << lhsNode.requestedLayout << " and " << rhsNode.requestedLayout;
    if (!lhsNode.requestedGranularity.empty() &&
        !rhsNode.requestedGranularity.empty() &&
        lhsNode.requestedGranularity != rhsNode.requestedGranularity)
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "conflicting mask granularities "
                             << lhsNode.requestedGranularity << " and "
                             << rhsNode.requestedGranularity;

    rhsNode.parent = lhsRoot;
    if (!lhsNode.requestedLayout)
      lhsNode.requestedLayout = rhsNode.requestedLayout;
    if (lhsNode.requestedGranularity.empty())
      lhsNode.requestedGranularity = rhsNode.requestedGranularity;
    return success();
  }

  LogicalResult setNaturalLayout(Value value, VMILayoutAttr layout,
                                 Operation *op) {
    unsigned id = addDataValue(value);
    if (id == ~0u || !layout)
      return success();
    unsigned root = find(id);
    VMILayoutAttr existing = dataNodes[root].naturalLayout;
    if (existing && existing != layout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting natural layouts "
             << existing << " and " << layout;
    dataNodes[root].naturalLayout = layout;
    return success();
  }

  VMILayoutAttr getContiguousLayout() {
    return VMILayoutAttr::getContiguous(ctx);
  }

  VMILayoutAttr getGroupSlotsLayout(int64_t numGroups) {
    return VMILayoutAttr::getGroupSlots(ctx, numGroups);
  }

  std::optional<int64_t> getVLaneElems(Type elementType) {
    FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(elementType);
    if (failed(lanesPerPart) || *lanesPerPart % 8 != 0)
      return std::nullopt;
    return *lanesPerPart / 8;
  }

  VMILayoutAttr getPreferredGroupSlotsLayout(VMIVRegType type,
                                             int64_t numGroups) {
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    VMILayoutSupport supports;
    FailureOr<VMIGroupReduceLayoutFact> fact =
        supports.getPreferredGroupReduceLayoutFact(type, numGroups);
    if (succeeded(fact))
      return fact->resultLayout;
    return getGroupSlotsLayout(numGroups);
  }

  VMILayoutAttr getPreferredGroupReduceSourceLayout(VMIVRegType type,
                                                    int64_t numGroups) {
    if (VMILayoutAttr existing = type.getLayoutAttr())
      return existing;
    VMILayoutSupport supports;
    FailureOr<VMIGroupReduceLayoutFact> fact =
        supports.getPreferredGroupReduceLayoutFact(type, numGroups);
    if (succeeded(fact))
      return fact->sourceLayout;
    return getContiguousLayout();
  }

  VMILayoutAttr getPreferredGroupSlotLoadLayout(VMIGroupSlotLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    std::optional<int64_t> sourceGroupStride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (sourceGroupStride && *sourceGroupStride == 1)
      return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/8);
    return VMILayoutAttr::getGroupSlots(ctx, numGroups, /*slots=*/1);
  }

  bool isE2BGroupBroadcastLoadCandidate(VMIVRegType type, Type sourceType,
                                        Value sourceGroupStride,
                                        int64_t numGroups) {
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0)
      return false;
    int64_t groupSize = type.getElementCount() / numGroups;
    if (numGroups % 8 != 0)
      return false;

    if (!isa<PtrType>(sourceType))
      return false;
    unsigned elementBits = getElementBitWidth(type.getElementType());
    if (elementBits != 16 && elementBits != 32)
      return false;
    int64_t directGroupSize = 256 / elementBits;
    if (groupSize != directGroupSize && groupSize != 2 * directGroupSize)
      return false;
    std::optional<int64_t> strideValue =
        getConstantIndexValue(sourceGroupStride);
    if (!strideValue || *strideValue != 1)
      return false;

    VMILayoutAttr existing = type.getLayoutAttr();
    if (!existing)
      return true;
    if (groupSize == directGroupSize)
      return existing.isContiguous();
    return existing.isDeinterleaved() && existing.getFactor() == 2 &&
           existing.getBlockElems() == 1;
  }

  bool isE2BGroupBroadcastLoadCandidate(VMIGroupBroadcastLoadOp op) {
    return isE2BGroupBroadcastLoadCandidate(
        cast<VMIVRegType>(op.getResult().getType()), op.getSource().getType(),
        op.getSourceGroupStride(), op.getNumGroupsAttr().getInt());
  }

  VMILayoutAttr getPreferredGroupBroadcastLoadLayout(
      VMIGroupBroadcastLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr())
      return existing;

    if (!isE2BGroupBroadcastLoadCandidate(op))
      return {};
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    int64_t groupSize = type.getElementCount() / numGroups;
    int64_t directGroupSize = 256 / getElementBitWidth(type.getElementType());
    if (groupSize == directGroupSize)
      return getContiguousLayout();
    return VMILayoutAttr::getDeinterleaved(ctx, 2, /*blockElems=*/1);
  }

  VMILayoutAttr getPreferredGroupBroadcastSourceLayout(Value value,
                                                       int64_t numGroups) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return getContiguousLayout();
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    VMILayoutAttr solved = getDataLayout(value);
    if (solved && solved.isGroupSlots() && solved.getNumGroups() == numGroups &&
        solved.getSlots() > 0)
      return solved;
    if (auto load = value.getDefiningOp<VMIGroupSlotLoadOp>())
      return getPreferredGroupSlotLoadLayout(load);
    return getPreferredGroupSlotsLayout(type, numGroups);
  }

  VMILayoutAttr getPreferredGroupLoadResultLayout(VMIGroupLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (VMILayoutAttr existing = type.getLayoutAttr())
      return existing;

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0)
      return getContiguousLayout();

    if (!type.getElementType().isF32())
      return getContiguousLayout();

    int64_t groupSize = type.getElementCount() / numGroups;
    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (!rowStride || *rowStride <= 0 || *rowStride % 8 != 0)
      return getContiguousLayout();

    if (groupSize == 16)
      return VMILayoutAttr::getDeinterleaved(ctx, 2, /*blockElems=*/8);
    if (groupSize == 32)
      return VMILayoutAttr::getDeinterleaved(ctx, 4, /*blockElems=*/8);

    return getContiguousLayout();
  }

  LogicalResult validateGroupLoadLayoutPlan(VMIGroupLoadOp op) {
    auto type = cast<VMIVRegType>(op.getResult().getType());
    if (type.getLayoutAttr())
      return success();

    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (numGroups <= 0 || type.getElementCount() % numGroups != 0)
      return success();
    if (!type.getElementType().isF32())
      return success();

    int64_t groupSize = type.getElementCount() / numGroups;
    if (groupSize != 16 && groupSize != 32)
      return success();

    std::optional<int64_t> rowStride = getConstantIndexValue(op.getRowStride());
    if (rowStride && *rowStride > 0 && *rowStride % 8 == 0)
      return success();

    return op.emitError()
           << kVMIDiagLayoutContractPrefix << "pto.vmi.group_load group_size "
           << groupSize
           << " requires constant positive row_stride divisible by 8 f32 "
              "elements for the block8 stride plan; stable gather fallback is "
              "not implemented";
  }

  VMILayoutAttr getPreferredGroupStoreUseLayout(Value value,
                                                int64_t numGroups) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return getContiguousLayout();
    if (VMILayoutAttr existing = type.getLayoutAttr())
      if (existing.isGroupSlots() && existing.getSlots() > 0)
        return existing;
    VMILayoutAttr solved = getDataLayout(value);
    if (solved && solved.isGroupSlots() && solved.getNumGroups() == numGroups &&
        solved.getSlots() > 0)
      return solved;
    if (value.getDefiningOp<VMIGroupReduceAddFOp>() ||
        value.getDefiningOp<VMIGroupReduceMaxFOp>() ||
        value.getDefiningOp<VMIGroupReduceAddIOp>() ||
        value.getDefiningOp<VMIGroupReduceMaxIOp>())
      return getPreferredGroupSlotsLayout(type, numGroups);
    if (auto load = value.getDefiningOp<VMIGroupSlotLoadOp>())
      return getPreferredGroupSlotLoadLayout(load);
    return getContiguousLayout();
  }

  VMILayoutAttr getPreferredDenseStoreUseLayout(Value value) {
    auto type = dyn_cast<VMIVRegType>(value.getType());
    if (!type)
      return getContiguousLayout();

    VMILayoutAttr layout = getExplicitDataLayout(value);
    if (!layout || !layout.hasDenseLaneStride())
      layout = type.getLayoutAttr();
    if (!layout || !layout.hasDenseLaneStride())
      return getContiguousLayout();

    auto candidateType =
        VMIVRegType::get(ctx, type.getElementCount(), type.getElementType(),
                         layout);
    VMILayoutSupport supports;
    if (succeeded(supports.getContiguousStoreSupport(candidateType)))
      return layout;
    return getContiguousLayout();
  }

  VMILayoutAttr getDataLayout(Value value) {
    unsigned id = addDataValue(value);
    if (id == ~0u)
      return {};
    unsigned root = find(id);
    if (dataNodes[root].naturalLayout)
      return dataNodes[root].naturalLayout;
    return getContiguousLayout();
  }

  VMILayoutAttr getExplicitDataLayout(Value value) {
    unsigned id = addDataValue(value);
    if (id == ~0u)
      return {};
    return dataNodes[find(id)].naturalLayout;
  }

  bool hasCompatibleTruncFUseForGroupReduce(Value value, int64_t groupSize) {
    auto sourceType = dyn_cast<VMIVRegType>(value.getType());
    if (!sourceType || !sourceType.getElementType().isF32())
      return false;

    for (OpOperand &use : value.getUses()) {
      auto truncf = dyn_cast<VMITruncFOp>(use.getOwner());
      if (!truncf || use.getOperandNumber() != 0)
        continue;

      auto resultType = dyn_cast<VMIVRegType>(truncf.getResult().getType());
      if (!resultType)
        continue;
      unsigned resultBits = getElementBitWidth(resultType.getElementType());
      std::optional<int64_t> vlaneElems =
          getVLaneElems(sourceType.getElementType());
      if (vlaneElems && groupSize == 2 * *vlaneElems && resultBits == 16)
        return true;
      if (vlaneElems && groupSize == 4 * *vlaneElems && resultBits == 8)
        return true;
    }
    return false;
  }

  bool isCompatibleGroupReduceSourceLayout(VMIGroupReduceLayoutFact fact,
                                           VMILayoutAttr layout) {
    if (!layout)
      return false;
    if (fact.kind == VMIGroupReduceLayoutKind::OneVLane ||
        fact.kind == VMIGroupReduceLayoutKind::RowLocal)
      return layout.isContiguous();
    int64_t factor = fact.kind == VMIGroupReduceLayoutKind::TwoVLane ? 2 : 4;
    return layout.isDeinterleaved() && layout.getFactor() == factor &&
           (layout.getBlockElems() == 1 || layout.getBlockElems() == 8);
  }

  VMILayoutAttr
  getTruncFCompatibleGroupReduceSourceLayout(VMIGroupReduceLayoutFact fact) {
    if (fact.kind == VMIGroupReduceLayoutKind::TwoVLane)
      return VMILayoutAttr::getDeinterleaved(ctx, 2, /*blockElems=*/1);
    if (fact.kind == VMIGroupReduceLayoutKind::FourVLane)
      return VMILayoutAttr::getDeinterleaved(ctx, 4, /*blockElems=*/1);
    return {};
  }

  LogicalResult requestMask(Value mask, VMILayoutAttr layout,
                            StringRef granularity, Operation *op) {
    unsigned id = addMaskValue(mask);
    if (id == ~0u)
      return success();
    if (!layout || granularity.empty())
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "cannot infer concrete mask layout or granularity";
    MaskNode &node = maskNodes[findMask(id)];
    if (node.requestedLayout && node.requestedLayout != layout)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix << "conflicting mask layouts "
             << node.requestedLayout << " and " << layout;
    if (!node.requestedGranularity.empty() &&
        node.requestedGranularity != granularity)
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "conflicting mask granularities " << node.requestedGranularity
             << " and " << granularity;
    node.requestedLayout = layout;
    node.requestedGranularity = granularity.str();
    return success();
  }

  void requestDataUse(OpOperand &operand, VMILayoutAttr layout) {
    if (isa<VMIVRegType>(operand.get().getType()))
      dataUseRequests.push_back(DataUseRequest{&operand, layout});
  }

  bool canProducerAdoptConsumerLayout(Operation *op) {
    if (!op)
      return false;
    return isa<VMILoadOp, VMIBroadcastOp,
               VMIConstantOp, VMIIotaOp, VMIAddFOp, VMIAddIOp, VMISubFOp,
               VMISubIOp, VMIMulFOp, VMIMulIOp, VMIFmaOp, VMIDivFOp, VMIMinFOp,
               VMIMaxFOp, VMINegFOp, VMIAbsFOp, VMIAbsIOp, VMISqrtOp, VMIExpOp,
               VMILnOp, VMIReluOp, VMIFPToSIOp, VMIAndIOp, VMIOrIOp, VMIXOrIOp,
               VMIShLIOp, VMIShRUIOp, VMINotOp, VMISelectOp, VMIBitcastOp,
               VMIGroupBroadcastLoadOp>(op);
  }

  bool canGroupBroadcastProduceLayout(VMIGroupBroadcastOp broadcast,
                                      VMILayoutAttr resultLayout) {
    if (!resultLayout)
      return false;
    auto sourceType = cast<VMIVRegType>(broadcast.getSource().getType());
    auto resultType = cast<VMIVRegType>(broadcast.getResult().getType());
    int64_t numGroups = broadcast.getNumGroupsAttr().getInt();
    auto assignedSourceType = VMIVRegType::get(
        ctx, sourceType.getElementCount(), sourceType.getElementType(),
        getPreferredGroupSlotsLayout(sourceType, numGroups));
    auto assignedResultType =
        VMIVRegType::get(ctx, resultType.getElementCount(),
                         resultType.getElementType(), resultLayout);
    VMILayoutSupport supports;
    return succeeded(supports.getGroupBroadcastSupport(
        capabilities, assignedSourceType, assignedResultType, numGroups));
  }

  bool canGroupBroadcastLoadProduceLayout(VMIGroupBroadcastLoadOp load,
                                          VMILayoutAttr resultLayout) {
    if (!resultLayout)
      return false;
    auto resultType = cast<VMIVRegType>(load.getResult().getType());
    int64_t numGroups = load.getNumGroupsAttr().getInt();
    unsigned elementBits = getElementBitWidth(resultType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0)
      return false;
    std::optional<int64_t> stride =
        getConstantIndexValue(load.getSourceGroupStride());
    int64_t alignedStrideElems = 256 / elementBits;
    int64_t slots = 0;
    if (stride && *stride == 1)
      slots = 8;
    else if (stride && *stride > 0 && *stride % alignedStrideElems == 0)
      slots = 1;
    else
      return false;

    auto assignedSourceType =
        VMIVRegType::get(ctx, numGroups, resultType.getElementType(),
                         VMILayoutAttr::getGroupSlots(ctx, numGroups, slots));
    auto assignedResultType =
        VMIVRegType::get(ctx, resultType.getElementCount(),
                         resultType.getElementType(), resultLayout);
    VMILayoutSupport supports;
    return succeeded(supports.getGroupBroadcastSupport(
        capabilities, assignedSourceType, assignedResultType, numGroups));
  }

  bool canEquivalenceClassAdoptConsumerLayout(Value value,
                                              VMILayoutAttr requestedLayout) {
    unsigned id = addDataValue(value);
    if (id == ~0u)
      return true;
    unsigned root = find(id);
    for (DataNode &node : dataNodes) {
      if (find(dataIds.lookup(node.value)) != root)
        continue;
      if (auto broadcast = node.value.getDefiningOp<VMIGroupBroadcastOp>()) {
        if (node.value == broadcast.getResult() &&
            !canGroupBroadcastProduceLayout(broadcast, requestedLayout))
          return false;
      }
      if (auto load = node.value.getDefiningOp<VMIGroupBroadcastLoadOp>()) {
        if (node.value == load.getResult() &&
            !canGroupBroadcastLoadProduceLayout(load, requestedLayout))
          return false;
      }
    }
    return true;
  }

  bool isUnsupportedGroupBroadcastResultForLayout(Value value,
                                                  VMILayoutAttr layout) {
    auto broadcast = value.getDefiningOp<VMIGroupBroadcastOp>();
    if (broadcast)
      return !canGroupBroadcastProduceLayout(broadcast, layout);
    auto load = value.getDefiningOp<VMIGroupBroadcastLoadOp>();
    return load && !canGroupBroadcastLoadProduceLayout(load, layout);
  }

  LogicalResult constrainElementwiseBinary(OpOperand &lhs, OpOperand &rhs,
                                           Value result, Operation *op) {
    VMILayoutAttr lhsLayout = getExplicitDataLayout(lhs.get());
    VMILayoutAttr rhsLayout = getExplicitDataLayout(rhs.get());
    VMILayoutAttr fallback = getContiguousLayout();
    if ((lhsLayout &&
         isUnsupportedGroupBroadcastResultForLayout(rhs.get(), lhsLayout)) ||
        (rhsLayout &&
         isUnsupportedGroupBroadcastResultForLayout(lhs.get(), rhsLayout))) {
      requestDataUse(lhs, fallback);
      requestDataUse(rhs, fallback);
      return setNaturalLayout(result, fallback, op);
    }

    if (failed(unite(lhs.get(), rhs.get(), op)))
      return failure();
    return unite(lhs.get(), result, op);
  }

  bool canAdoptConsumerRequestedLayout(Value value,
                                       VMILayoutAttr requestedLayout) {
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp)
      return false;
    if (isa<VMILoadOp>(definingOp)) {
      if (requestedLayout && requestedLayout.hasDenseLaneStride()) {
        auto type = dyn_cast<VMIVRegType>(value.getType());
        if (!type)
          return false;
        auto candidateType =
            VMIVRegType::get(ctx, type.getElementCount(), type.getElementType(),
                             requestedLayout);
        VMILayoutSupport supports;
        if (failed(supports.getContiguousLoadSupport(candidateType)))
          return false;
      }
    } else {
      if (!requestedLayout || requestedLayout.isContiguous())
        return false;
      if (!canProducerAdoptConsumerLayout(definingOp))
        return false;
    }
    if (!canEquivalenceClassAdoptConsumerLayout(value, requestedLayout))
      return false;
    if (value.hasOneUse())
      return true;

    unsigned matchingRequests = 0;
    unsigned totalUses = 0;
    for (OpOperand &use : value.getUses()) {
      ++totalUses;
      bool foundRequest = false;
      for (DataUseRequest request : dataUseRequests) {
        if (request.operand != &use)
          continue;
        if (request.layout != requestedLayout)
          return false;
        foundRequest = true;
      }
      if (!foundRequest)
        return false;
      ++matchingRequests;
    }
    return matchingRequests == totalUses;
  }

  LogicalResult applyConsumerDrivenDataLayouts() {
    for (DataUseRequest request : dataUseRequests) {
      Value value = request.operand->get();
      if (!canAdoptConsumerRequestedLayout(value, request.layout))
        continue;
      unsigned id = addDataValue(value);
      if (id == ~0u)
        continue;
      unsigned root = find(id);
      VMILayoutAttr existing = dataNodes[root].naturalLayout;
      if (existing && existing != request.layout)
        continue;
      dataNodes[root].naturalLayout = request.layout;
    }
    return success();
  }

  LogicalResult requestMaskUse(OpOperand &operand, VMILayoutAttr layout,
                               StringRef granularity, Operation *op) {
    if (!isa<VMIMaskType>(operand.get().getType()))
      return success();
    if (!layout || granularity.empty())
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "cannot infer concrete mask use layout or granularity";
    maskUseRequests.push_back(
        MaskUseRequest{&operand, layout, granularity.str()});
    return success();
  }

  LogicalResult collect() {
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        addDataValue(result);
        addMaskValue(result);
      }
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) {
            addDataValue(arg);
            addMaskValue(arg);
          }
    });
    return success();
  }

  LogicalResult addConstraints() {
    WalkResult result = module.walk([&](Operation *op) -> WalkResult {
      if (auto maskAnd = dyn_cast<VMIMaskAndOp>(op)) {
        if (failed(uniteMask(maskAnd.getLhs(), maskAnd.getRhs(), op)) ||
            failed(uniteMask(maskAnd.getLhs(), maskAnd.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maskOr = dyn_cast<VMIMaskOrOp>(op)) {
        if (failed(uniteMask(maskOr.getLhs(), maskOr.getRhs(), op)) ||
            failed(uniteMask(maskOr.getLhs(), maskOr.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maskXor = dyn_cast<VMIMaskXOrOp>(op)) {
        if (failed(uniteMask(maskXor.getLhs(), maskXor.getRhs(), op)) ||
            failed(uniteMask(maskXor.getLhs(), maskXor.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maskNot = dyn_cast<VMIMaskNotOp>(op)) {
        if (failed(uniteMask(maskNot.getSource(), maskNot.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto addf = dyn_cast<VMIAddFOp>(op)) {
        if (failed(constrainElementwiseBinary(addf.getLhsMutable(),
                                              addf.getRhsMutable(),
                                              addf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto addi = dyn_cast<VMIAddIOp>(op)) {
        if (failed(constrainElementwiseBinary(addi.getLhsMutable(),
                                              addi.getRhsMutable(),
                                              addi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto subf = dyn_cast<VMISubFOp>(op)) {
        if (failed(constrainElementwiseBinary(subf.getLhsMutable(),
                                              subf.getRhsMutable(),
                                              subf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto subi = dyn_cast<VMISubIOp>(op)) {
        if (failed(constrainElementwiseBinary(subi.getLhsMutable(),
                                              subi.getRhsMutable(),
                                              subi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto mulf = dyn_cast<VMIMulFOp>(op)) {
        if (failed(constrainElementwiseBinary(mulf.getLhsMutable(),
                                              mulf.getRhsMutable(),
                                              mulf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto muli = dyn_cast<VMIMulIOp>(op)) {
        if (failed(constrainElementwiseBinary(muli.getLhsMutable(),
                                              muli.getRhsMutable(),
                                              muli.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto fma = dyn_cast<VMIFmaOp>(op)) {
        if (failed(unite(fma.getLhs(), fma.getRhs(), op)) ||
            failed(unite(fma.getLhs(), fma.getAcc(), op)) ||
            failed(unite(fma.getLhs(), fma.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto divf = dyn_cast<VMIDivFOp>(op)) {
        if (failed(constrainElementwiseBinary(divf.getLhsMutable(),
                                              divf.getRhsMutable(),
                                              divf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto minf = dyn_cast<VMIMinFOp>(op)) {
        if (failed(constrainElementwiseBinary(minf.getLhsMutable(),
                                              minf.getRhsMutable(),
                                              minf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto maxf = dyn_cast<VMIMaxFOp>(op)) {
        if (failed(constrainElementwiseBinary(maxf.getLhsMutable(),
                                              maxf.getRhsMutable(),
                                              maxf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto negf = dyn_cast<VMINegFOp>(op)) {
        if (failed(unite(negf.getSource(), negf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto absf = dyn_cast<VMIAbsFOp>(op)) {
        if (failed(unite(absf.getSource(), absf.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto absi = dyn_cast<VMIAbsIOp>(op)) {
        if (failed(unite(absi.getSource(), absi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto sqrt = dyn_cast<VMISqrtOp>(op)) {
        if (failed(unite(sqrt.getSource(), sqrt.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto exp = dyn_cast<VMIExpOp>(op)) {
        if (failed(unite(exp.getSource(), exp.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ln = dyn_cast<VMILnOp>(op)) {
        if (failed(unite(ln.getSource(), ln.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto relu = dyn_cast<VMIReluOp>(op)) {
        if (failed(unite(relu.getSource(), relu.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto fptosi = dyn_cast<VMIFPToSIOp>(op)) {
        if (failed(unite(fptosi.getSource(), fptosi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto sitofp = dyn_cast<VMISIToFPOp>(op)) {
        if (failed(unite(sitofp.getSource(), sitofp.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto andi = dyn_cast<VMIAndIOp>(op)) {
        if (failed(constrainElementwiseBinary(andi.getLhsMutable(),
                                              andi.getRhsMutable(),
                                              andi.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ori = dyn_cast<VMIOrIOp>(op)) {
        if (failed(constrainElementwiseBinary(
                ori.getLhsMutable(), ori.getRhsMutable(), ori.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto xori = dyn_cast<VMIXOrIOp>(op)) {
        if (failed(constrainElementwiseBinary(xori.getLhsMutable(),
                                              xori.getRhsMutable(),
                                              xori.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shli = dyn_cast<VMIShLIOp>(op)) {
        if (failed(constrainElementwiseBinary(shli.getLhsMutable(),
                                              shli.getRhsMutable(),
                                              shli.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shrui = dyn_cast<VMIShRUIOp>(op)) {
        if (failed(constrainElementwiseBinary(shrui.getLhsMutable(),
                                              shrui.getRhsMutable(),
                                              shrui.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto notOp = dyn_cast<VMINotOp>(op)) {
        if (failed(unite(notOp.getSource(), notOp.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
        if (failed(unite(cmpf.getLhs(), cmpf.getRhs(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
        if (failed(unite(cmpi.getLhs(), cmpi.getRhs(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto select = dyn_cast<VMISelectOp>(op)) {
        if (failed(unite(select.getTrueValue(), select.getFalseValue(), op)) ||
            failed(unite(select.getTrueValue(), select.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
        if (failed(setNaturalLayout(activePrefix.getResult(),
                                    getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto compress = dyn_cast<VMICompressOp>(op)) {
        requestDataUse(compress.getSourceMutable(), getContiguousLayout());
        if (failed(setNaturalLayout(compress.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout());
        requestDataUse(reduce.getInitMutable(), getContiguousLayout());
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout());
        requestDataUse(reduce.getInitMutable(), getContiguousLayout());
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout());
        requestDataUse(reduce.getInitMutable(), getContiguousLayout());
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
        requestDataUse(reduce.getSourceMutable(), getContiguousLayout());
        requestDataUse(reduce.getInitMutable(), getContiguousLayout());
        if (failed(setNaturalLayout(reduce.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            getPreferredGroupReduceSourceLayout(sourceType, numGroups);
        VMILayoutAttr solvedSourceLayout =
            getExplicitDataLayout(reduce.getSource());
        if (solvedSourceLayout && succeeded(fact) &&
            isCompatibleGroupReduceSourceLayout(*fact, solvedSourceLayout)) {
          sourceLayout = solvedSourceLayout;
        } else if (!sourceType.getLayoutAttr() && succeeded(fact)) {
          if (hasCompatibleTruncFUseForGroupReduce(reduce.getSource(),
                                                   fact->groupSize)) {
            if (VMILayoutAttr truncLayout =
                    getTruncFCompatibleGroupReduceSourceLayout(*fact))
              sourceLayout = truncLayout;
          }
        }
        requestDataUse(reduce.getSourceMutable(), sourceLayout);
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceLayout,
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            getPreferredGroupReduceSourceLayout(sourceType, numGroups);
        VMILayoutAttr solvedSourceLayout =
            getExplicitDataLayout(reduce.getSource());
        if (solvedSourceLayout && succeeded(fact) &&
            isCompatibleGroupReduceSourceLayout(*fact, solvedSourceLayout)) {
          sourceLayout = solvedSourceLayout;
        } else if (!sourceType.getLayoutAttr() && succeeded(fact)) {
          if (hasCompatibleTruncFUseForGroupReduce(reduce.getSource(),
                                                   fact->groupSize)) {
            if (VMILayoutAttr truncLayout =
                    getTruncFCompatibleGroupReduceSourceLayout(*fact))
              sourceLayout = truncLayout;
          }
        }
        requestDataUse(reduce.getSourceMutable(), sourceLayout);
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceLayout,
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            getPreferredGroupReduceSourceLayout(sourceType, numGroups);
        VMILayoutAttr solvedSourceLayout =
            getExplicitDataLayout(reduce.getSource());
        if (solvedSourceLayout && succeeded(fact) &&
            isCompatibleGroupReduceSourceLayout(*fact, solvedSourceLayout))
          sourceLayout = solvedSourceLayout;
        requestDataUse(reduce.getSourceMutable(), sourceLayout);
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceLayout,
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        auto resultType = cast<VMIVRegType>(reduce.getResult().getType());
        int64_t numGroups = reduce.getNumGroupsAttr().getInt();
        VMILayoutSupport supports;
        FailureOr<VMIGroupReduceLayoutFact> fact =
            supports.getPreferredGroupReduceLayoutFact(sourceType, numGroups);
        VMILayoutAttr sourceLayout =
            getPreferredGroupReduceSourceLayout(sourceType, numGroups);
        VMILayoutAttr solvedSourceLayout =
            getExplicitDataLayout(reduce.getSource());
        if (solvedSourceLayout && succeeded(fact) &&
            isCompatibleGroupReduceSourceLayout(*fact, solvedSourceLayout))
          sourceLayout = solvedSourceLayout;
        requestDataUse(reduce.getSourceMutable(), sourceLayout);
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceLayout,
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                reduce.getResult(),
                succeeded(fact)
                    ? fact->resultLayout
                    : getPreferredGroupSlotsLayout(resultType, numGroups),
                op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
        requestDataUse(
            broadcast.getSourceMutable(),
            getPreferredGroupBroadcastSourceLayout(
                broadcast.getSource(), broadcast.getNumGroupsAttr().getInt()));
        return WalkResult::advance();
      }
      if (auto hist = dyn_cast<VMIDhistOp>(op)) {
        requestDataUse(hist.getAccMutable(), getContiguousLayout());
        requestDataUse(hist.getSourceMutable(), getContiguousLayout());
        if (failed(requestMaskUse(hist.getMaskMutable(), getContiguousLayout(),
                                  "b8", op)))
          return WalkResult::interrupt();
        if (failed(
                setNaturalLayout(hist.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto hist = dyn_cast<VMIChistOp>(op)) {
        requestDataUse(hist.getAccMutable(), getContiguousLayout());
        requestDataUse(hist.getSourceMutable(), getContiguousLayout());
        if (failed(requestMaskUse(hist.getMaskMutable(), getContiguousLayout(),
                                  "b8", op)))
          return WalkResult::interrupt();
        if (failed(
                setNaturalLayout(hist.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto extf = dyn_cast<VMIExtFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(extf.getSource().getType());
        auto resultType = cast<VMIVRegType>(extf.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        if (succeeded(fact) && (fact->kind == VMICastLayoutKind::Widen2x ||
                                fact->kind == VMICastLayoutKind::Widen4x)) {
          requestDataUse(extf.getSourceMutable(), fact->sourceLayout);
          if (failed(
                  setNaturalLayout(extf.getResult(), fact->resultLayout, op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(extsi.getSource().getType());
        auto resultType = cast<VMIVRegType>(extsi.getResult().getType());
        VMILayoutAttr sourceLayout = getDataLayout(extsi.getSource());
        if (sourceLayout && sourceLayout.isGroupSlots() &&
            sourceLayout.getSlots() == 8 &&
            getElementBitWidth(sourceType.getElementType()) < 32 &&
            getElementBitWidth(resultType.getElementType()) == 32) {
          requestDataUse(extsi.getSourceMutable(), sourceLayout);
          if (failed(setNaturalLayout(extsi.getResult(), sourceLayout, op)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        }
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        if (succeeded(fact) && (fact->kind == VMICastLayoutKind::Widen2x ||
                                fact->kind == VMICastLayoutKind::Widen4x)) {
          requestDataUse(extsi.getSourceMutable(), fact->sourceLayout);
          if (failed(
                  setNaturalLayout(extsi.getResult(), fact->resultLayout, op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(extui.getSource().getType());
        auto resultType = cast<VMIVRegType>(extui.getResult().getType());
        VMILayoutAttr sourceLayout = getDataLayout(extui.getSource());
        if (sourceLayout && sourceLayout.isGroupSlots() &&
            sourceLayout.getSlots() == 8 &&
            getElementBitWidth(sourceType.getElementType()) < 32 &&
            getElementBitWidth(resultType.getElementType()) == 32) {
          requestDataUse(extui.getSourceMutable(), sourceLayout);
          if (failed(setNaturalLayout(extui.getResult(), sourceLayout, op)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        }
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        if (succeeded(fact) && (fact->kind == VMICastLayoutKind::Widen2x ||
                                fact->kind == VMICastLayoutKind::Widen4x)) {
          requestDataUse(extui.getSourceMutable(), fact->sourceLayout);
          if (failed(
                  setNaturalLayout(extui.getResult(), fact->resultLayout, op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(truncf.getSource().getType());
        auto resultType = cast<VMIVRegType>(truncf.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        VMILayoutAttr sourceLayout = getDataLayout(truncf.getSource());
        if (succeeded(fact) && fact->kind == VMICastLayoutKind::Narrow2x &&
            sourceLayout && sourceLayout.isGroupSlots() &&
            sourceLayout.getSlots() == 1) {
          requestDataUse(truncf.getSourceMutable(), sourceLayout);
          if (failed(setNaturalLayout(truncf.getResult(), sourceLayout, op)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        }
        VMILayoutAttr resultLayout = getContiguousLayout();
        if (succeeded(fact) && (fact->kind == VMICastLayoutKind::Narrow2x ||
                                fact->kind == VMICastLayoutKind::Narrow4x)) {
          resultLayout = fact->resultLayout;
          requestDataUse(truncf.getSourceMutable(), fact->sourceLayout);
        }
        if (failed(setNaturalLayout(truncf.getResult(), resultLayout, op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(trunci.getSource().getType());
        auto resultType = cast<VMIVRegType>(trunci.getResult().getType());
        VMILayoutSupport supports;
        FailureOr<VMICastLayoutFact> fact =
            supports.getPreferredCastLayoutFact(sourceType, resultType);
        VMILayoutAttr sourceLayout = getDataLayout(trunci.getSource());
        if (succeeded(fact) && sourceLayout && sourceLayout.isGroupSlots() &&
            (sourceLayout.getSlots() == 1 || sourceLayout.getSlots() == 8) &&
            (fact->kind == VMICastLayoutKind::Narrow2x ||
             fact->kind == VMICastLayoutKind::Narrow4x)) {
          requestDataUse(trunci.getSourceMutable(), sourceLayout);
          VMILayoutAttr resultLayout = sourceLayout;
          if (sourceLayout.getSlots() == 8 &&
              fact->kind == VMICastLayoutKind::Narrow4x)
            resultLayout = VMILayoutAttr::getGroupSlots(
                ctx, sourceLayout.getNumGroups(), sourceLayout.getSlots(),
                /*laneStride=*/4);
          if (failed(setNaturalLayout(trunci.getResult(), resultLayout, op)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        }
        VMILayoutAttr resultLayout = getContiguousLayout();
        if (succeeded(fact) && (fact->kind == VMICastLayoutKind::Narrow2x ||
                                fact->kind == VMICastLayoutKind::Narrow4x)) {
          resultLayout = fact->resultLayout;
          requestDataUse(trunci.getSourceMutable(), fact->sourceLayout);
        }
        if (failed(setNaturalLayout(trunci.getResult(), resultLayout, op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
        if (failed(unite(bitcast.getSource(), bitcast.getResult(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIDeinterleaveLoadOp>(op)) {
        if (failed(setNaturalLayout(load.getLow(), getContiguousLayout(), op)) ||
            failed(setNaturalLayout(load.getHigh(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
        requestDataUse(load.getPassthruMutable(), getContiguousLayout());
        if (failed(
                setNaturalLayout(load.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto gather = dyn_cast<VMIGatherOp>(op)) {
        auto resultType = cast<VMIVRegType>(gather.getResult().getType());
        requestDataUse(gather.getIndicesMutable(), getContiguousLayout());
        requestDataUse(gather.getPassthruMutable(), getContiguousLayout());
        if (failed(requestMaskUse(
                gather.getMaskMutable(), getContiguousLayout(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(gather.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
        requestDataUse(load.getPassthruMutable(), getContiguousLayout());
        if (failed(
                setNaturalLayout(load.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
        if (failed(validateGroupLoadLayoutPlan(load)))
          return WalkResult::interrupt();
        if (failed(setNaturalLayout(
                load.getResult(), getPreferredGroupLoadResultLayout(load), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
        if (failed(setNaturalLayout(
                load.getResult(), getPreferredGroupSlotLoadLayout(load), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIGroupBroadcastLoadOp>(op)) {
        if (failed(setNaturalLayout(
                load.getResult(),
                getPreferredGroupBroadcastLoadLayout(load), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIStrideLoadOp>(op)) {
        auto resultType = cast<VMIVRegType>(load.getResult().getType());
        if (failed(
                setNaturalLayout(load.getResult(), getContiguousLayout(), op)))
          return WalkResult::interrupt();
        if (failed(requestMaskUse(
                load.getMaskMutable(), getContiguousLayout(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIStoreOp>(op)) {
        requestDataUse(store.getValueMutable(),
                       getPreferredDenseStoreUseLayout(store.getValue()));
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIInterleaveStoreOp>(op)) {
        requestDataUse(store.getLowMutable(), getContiguousLayout());
        requestDataUse(store.getHighMutable(), getContiguousLayout());
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
        requestDataUse(
            store.getValueMutable(),
            getPreferredGroupStoreUseLayout(store.getValue(),
                                            store.getNumGroupsAttr().getInt()));
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
        auto valueType = cast<VMIVRegType>(store.getValue().getType());
        requestDataUse(store.getValueMutable(), getContiguousLayout());
        if (failed(requestMaskUse(
                store.getMaskMutable(), getContiguousLayout(),
                getMaskGranularityForElement(valueType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMIStrideStoreOp>(op)) {
        auto valueType = cast<VMIVRegType>(store.getValue().getType());
        requestDataUse(store.getValueMutable(), getContiguousLayout());
        if (failed(requestMaskUse(
                store.getMaskMutable(), getContiguousLayout(),
                getMaskGranularityForElement(valueType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
        auto valueType = cast<VMIVRegType>(scatter.getValue().getType());
        requestDataUse(scatter.getValueMutable(), getContiguousLayout());
        requestDataUse(scatter.getIndicesMutable(), getContiguousLayout());
        if (failed(requestMaskUse(
                scatter.getMaskMutable(), getContiguousLayout(),
                getMaskGranularityForElement(valueType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<VMICompressStoreOp>(op)) {
        auto valueType = cast<VMIVRegType>(store.getValue().getType());
        requestDataUse(store.getValueMutable(), getContiguousLayout());
        if (failed(requestMaskUse(
                store.getMaskMutable(), getContiguousLayout(),
                getMaskGranularityForElement(valueType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
        int64_t channels = split.getNumResults();
        VMICapabilityResult capability = capabilities.supportsChannelCount(
            "pto.vmi.channel_split", channels);
        if (!capability.isSupported()) {
          split.emitError() << kVMIDiagUnsupportedPrefix << capability.reason;
          return WalkResult::interrupt();
        }
        requestDataUse(split.getSourceMutable(),
                       VMILayoutAttr::getDeinterleaved(ctx, channels));
        for (Value result : split.getResults())
          if (failed(setNaturalLayout(result, getContiguousLayout(), op)))
            return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
        int64_t channels = merge.getInputs().size();
        VMICapabilityResult capability = capabilities.supportsChannelCount(
            "pto.vmi.channel_merge", channels);
        if (!capability.isSupported()) {
          merge.emitError() << kVMIDiagUnsupportedPrefix << capability.reason;
          return WalkResult::interrupt();
        }
        for (OpOperand &input : merge.getInputsMutable())
          requestDataUse(input, getContiguousLayout());
        if (failed(setNaturalLayout(
                merge.getResult(),
                VMILayoutAttr::getDeinterleaved(ctx, channels), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
        auto sourceType = cast<VMIVRegType>(shuffle.getSource().getType());
        auto resultType = cast<VMIVRegType>(shuffle.getResult().getType());
        if (sourceType.hasLayout() || resultType.hasLayout())
          return WalkResult::advance();

        requestDataUse(shuffle.getSourceMutable(), getContiguousLayout());
        if (isLane0SplatShuffle(shuffle))
          return WalkResult::advance();
        if (failed(setNaturalLayout(shuffle.getResult(), getContiguousLayout(),
                                    op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (failed(addIfConstraints(ifOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto executeRegionOp = dyn_cast<scf::ExecuteRegionOp>(op)) {
        if (failed(addExecuteRegionConstraints(executeRegionOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto indexSwitchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
        if (failed(addIndexSwitchConstraints(indexSwitchOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
        if (failed(addWhileConstraints(whileOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (failed(addForConstraints(forOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto branchOp = dyn_cast<cf::BranchOp>(op)) {
        if (failed(addBranchConstraints(branchOp.getDest(),
                                        branchOp.getDestOperands(), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto condBranchOp = dyn_cast<cf::CondBranchOp>(op)) {
        if (failed(addBranchConstraints(condBranchOp.getTrueDest(),
                                        condBranchOp.getTrueDestOperands(),
                                        op)) ||
            failed(addBranchConstraints(condBranchOp.getFalseDest(),
                                        condBranchOp.getFalseDestOperands(),
                                        op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto switchOp = dyn_cast<cf::SwitchOp>(op)) {
        if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                        switchOp.getDefaultOperands(), op)))
          return WalkResult::interrupt();
        for (auto [dest, operands] : llvm::zip(switchOp.getCaseDestinations(),
                                               switchOp.getCaseOperands())) {
          if (failed(addBranchConstraints(dest, operands, op)))
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (failed(addReturnConstraints(returnOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        if (failed(addCallConstraints(callOp)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (op->getName().getStringRef() == "func.call_indirect") {
        if (hasVMIValueTypes(op)) {
          op->emitError()
              << kVMIDiagLayoutContractPrefix
              << "VMI typed call requires a direct internal callee with a body";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp.empty() && hasVMIFunctionType(funcOp)) {
          funcOp.emitError()
              << kVMIDiagLayoutContractPrefix
              << "VMI typed function declaration requires an explicit "
                 "external ABI materialization plan";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult uniteEquivalentValues(Value lhs, Value rhs, Operation *op) {
    if (failed(unite(lhs, rhs, op)))
      return failure();
    return uniteMask(lhs, rhs, op);
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned resultNo = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty())
          continue;
        auto yieldOp = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yieldOp || resultNo >= yieldOp.getNumOperands())
          continue;
        if (failed(uniteEquivalentValues(result, yieldOp.getOperand(resultNo),
                                         ifOp)))
          return failure();
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yieldOp,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yieldOp.getNumOperands())
        break;
      if (failed(uniteEquivalentValues(result, yieldOp.getOperand(index), op)))
        return failure();
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp executeOp) {
    WalkResult result = executeOp.getRegion().walk([&](scf::YieldOp yieldOp) {
      if (yieldOp->getParentOp() != executeOp.getOperation())
        return WalkResult::advance();
      if (failed(
              addYieldConstraints(executeOp->getResults(), yieldOp, executeOp)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitchOp) {
    auto addBlockTerminator = [&](Block &block) -> LogicalResult {
      auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
      if (!yieldOp)
        return success();
      return addYieldConstraints(indexSwitchOp->getResults(), yieldOp,
                                 indexSwitchOp);
    };

    if (failed(addBlockTerminator(indexSwitchOp.getDefaultBlock())))
      return failure();
    for (unsigned idx = 0, e = indexSwitchOp.getNumCases(); idx < e; ++idx)
      if (failed(addBlockTerminator(indexSwitchOp.getCaseBlock(idx))))
        return failure();
    return success();
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    auto inits = whileOp.getInits();
    auto beforeArgs = whileOp.getBeforeArguments();
    Block &afterBlock = whileOp.getAfter().front();
    auto conditionOp =
        dyn_cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    auto yieldOp = dyn_cast<scf::YieldOp>(afterBlock.getTerminator());

    for (auto [index, init] : llvm::enumerate(inits)) {
      Value anchor = init;
      if (index < beforeArgs.size() &&
          failed(uniteEquivalentValues(anchor, beforeArgs[index], whileOp)))
        return failure();
      if (conditionOp && index < conditionOp.getArgs().size() &&
          failed(uniteEquivalentValues(anchor, conditionOp.getArgs()[index],
                                       whileOp)))
        return failure();
      if (index < afterBlock.getNumArguments() &&
          failed(uniteEquivalentValues(anchor, afterBlock.getArgument(index),
                                       whileOp)))
        return failure();
      if (yieldOp && index < yieldOp.getNumOperands() &&
          failed(uniteEquivalentValues(anchor, yieldOp.getOperand(index),
                                       whileOp)))
        return failure();
      if (index < whileOp.getNumResults() &&
          failed(
              uniteEquivalentValues(anchor, whileOp.getResult(index), whileOp)))
        return failure();
    }
    return success();
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    auto initArgs = forOp.getInitArgs();
    auto regionIterArgs = forOp.getRegionIterArgs();
    auto results = forOp.getResults();
    scf::YieldOp yieldOp = nullptr;
    if (Block *body = forOp.getBody())
      yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());

    for (auto [index, initArg] : llvm::enumerate(initArgs)) {
      Value anchor = initArg;
      if (index < regionIterArgs.size() &&
          failed(uniteEquivalentValues(anchor, regionIterArgs[index], forOp)))
        return failure();
      if (index < results.size() &&
          failed(uniteEquivalentValues(anchor, results[index], forOp)))
        return failure();
      if (yieldOp && index < yieldOp.getNumOperands() &&
          failed(
              uniteEquivalentValues(anchor, yieldOp.getOperand(index), forOp)))
        return failure();
    }
    return success();
  }

  LogicalResult addBranchConstraints(Block *dest, OperandRange operands,
                                     Operation *op) {
    if (!dest)
      return success();
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= dest->getNumArguments())
        break;
      if (failed(uniteEquivalentValues(operand, dest->getArgument(index), op)))
        return failure();
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    auto func = returnOp->getParentOfType<func::FuncOp>();
    if (!func)
      return success();

    auto it = firstReturnOperandsByFunc.find(func);
    if (it == firstReturnOperandsByFunc.end()) {
      SmallVector<Value> operands(returnOp.getOperands());
      firstReturnOperandsByFunc.try_emplace(func, std::move(operands));
      return success();
    }

    ArrayRef<Value> firstOperands = it->second;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= firstOperands.size())
        break;
      if (failed(
              uniteEquivalentValues(firstOperands[index], operand, returnOp)))
        return failure();
    }
    return success();
  }

  bool hasVMIValueTypes(Operation *op) {
    return llvm::any_of(op->getOperandTypes(), containsVMIType) ||
           llvm::any_of(op->getResultTypes(), containsVMIType);
  }

  bool hasVMIFunctionType(func::FuncOp func) {
    FunctionType type = func.getFunctionType();
    return llvm::any_of(type.getInputs(), containsVMIType) ||
           llvm::any_of(type.getResults(), containsVMIType);
  }

  LogicalResult addCallConstraints(func::CallOp callOp) {
    if (!hasVMIValueTypes(callOp))
      return success();

    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || callee.empty())
      return callOp.emitError()
             << kVMIDiagLayoutContractPrefix
             << "VMI typed call requires a direct internal callee with a body";

    for (auto [operand, argument] :
         llvm::zip(callOp.getOperands(), callee.getArguments())) {
      if (failed(uniteEquivalentValues(operand, argument, callOp)))
        return failure();
    }

    SmallVector<func::ReturnOp> returns;
    callee.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
        if (index >= returnOp.getNumOperands())
          break;
        if (failed(uniteEquivalentValues(result, returnOp.getOperand(index),
                                         callOp)))
          return failure();
      }
    }
    return success();
  }

  void rewriteDataTypes() {
    for (DataNode &node : dataNodes) {
      VMILayoutAttr layout = getDataLayout(node.value);
      node.value.setType(VMIVRegType::get(ctx, node.type.getElementCount(),
                                          node.type.getElementType(), layout));
    }
  }

  LogicalResult insertDataUseMaterializations() {
    OpBuilder builder(ctx);
    for (DataUseRequest request : dataUseRequests) {
      Value value = request.operand->get();
      auto sourceType = dyn_cast<VMIVRegType>(value.getType());
      if (!sourceType)
        continue;
      VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
      if (!sourceLayout)
        return request.operand->getOwner()->emitError()
               << kVMIDiagLayoutContractPrefix
               << "data use materialization requires layout-assigned source "
                  "type";
      if (sourceLayout == request.layout)
        continue;

      auto resultType =
          VMIVRegType::get(ctx, sourceType.getElementCount(),
                           sourceType.getElementType(), request.layout);
      builder.setInsertionPoint(request.operand->getOwner());
      auto ensure = builder.create<VMIEnsureLayoutOp>(
          request.operand->getOwner()->getLoc(), resultType, value);
      request.operand->set(ensure.getResult());
    }
    return success();
  }

  LogicalResult inferMaskRequests() {
    WalkResult result = module.walk([&](Operation *op) -> WalkResult {
      if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
        auto lhsType = cast<VMIVRegType>(cmpf.getLhs().getType());
        if (failed(requestMask(
                cmpf.getResult(), lhsType.getLayoutAttr(),
                getMaskGranularityForElement(lhsType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
        auto lhsType = cast<VMIVRegType>(cmpi.getLhs().getType());
        if (failed(requestMask(
                cmpi.getResult(), lhsType.getLayoutAttr(),
                getMaskGranularityForElement(lhsType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto select = dyn_cast<VMISelectOp>(op)) {
        auto resultType = cast<VMIVRegType>(select.getResult().getType());
        if (failed(requestMaskUse(
                select.getMaskMutable(), resultType.getLayoutAttr(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
        auto resultType = cast<VMIVRegType>(activePrefix.getResult().getType());
        if (failed(requestMaskUse(
                activePrefix.getMaskMutable(), resultType.getLayoutAttr(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto compress = dyn_cast<VMICompressOp>(op)) {
        auto resultType = cast<VMIVRegType>(compress.getResult().getType());
        if (failed(requestMaskUse(
                compress.getMaskMutable(), resultType.getLayoutAttr(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto reduce = dyn_cast<VMIGroupReduceMaxIOp>(op)) {
        auto sourceType = cast<VMIVRegType>(reduce.getSource().getType());
        if (failed(requestMaskUse(
                reduce.getMaskMutable(), sourceType.getLayoutAttr(),
                getMaskGranularityForElement(sourceType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
        auto resultType = cast<VMIVRegType>(load.getResult().getType());
        if (failed(requestMaskUse(
                load.getMaskMutable(), resultType.getLayoutAttr(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
        auto resultType = cast<VMIVRegType>(load.getResult().getType());
        if (failed(requestMaskUse(
                load.getMaskMutable(), resultType.getLayoutAttr(),
                getMaskGranularityForElement(resultType.getElementType()), op)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  void rewriteMaskTypes() {
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      VMILayoutAttr layout =
          root.requestedLayout ? root.requestedLayout : getContiguousLayout();
      StringRef granularity = root.requestedGranularity.empty()
                                  ? StringRef("b32")
                                  : StringRef(root.requestedGranularity);
      node.value.setType(VMIMaskType::get(ctx, node.type.getElementCount(),
                                          granularity, layout));
    }
  }

  LogicalResult insertMaskUseMaterializations() {
    OpBuilder builder(ctx);
    for (MaskUseRequest request : maskUseRequests) {
      Value value = request.operand->get();
      auto sourceType = dyn_cast<VMIMaskType>(value.getType());
      if (!sourceType)
        continue;
      VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
      if (!sourceLayout)
        return request.operand->getOwner()->emitError()
               << kVMIDiagLayoutContractPrefix
               << "mask use materialization requires layout-assigned source "
                  "type";

      builder.setInsertionPoint(request.operand->getOwner());
      Value current = value;
      VMIMaskType currentType = sourceType;
      if (sourceLayout != request.layout) {
        auto layoutType =
            VMIMaskType::get(ctx, currentType.getElementCount(),
                             currentType.getGranularity(), request.layout);
        auto ensureLayout = builder.create<VMIEnsureMaskLayoutOp>(
            request.operand->getOwner()->getLoc(), layoutType, current);
        current = ensureLayout.getResult();
        currentType = layoutType;
      }

      if (currentType.getGranularity() != request.granularity) {
        auto granularityType =
            VMIMaskType::get(ctx, currentType.getElementCount(),
                             request.granularity, request.layout);
        auto ensureGranularity = builder.create<VMIEnsureMaskGranularityOp>(
            request.operand->getOwner()->getLoc(), granularityType, current);
        current = ensureGranularity.getResult();
      }

      if (current != value)
        request.operand->set(current);
    }
    return success();
  }

  void rewriteFunctionType() {
    module.walk([&](func::FuncOp func) {
      if (func.empty())
        return;

      SmallVector<Type> inputs;
      inputs.reserve(func.getNumArguments());
      for (BlockArgument arg : func.getArguments())
        inputs.push_back(arg.getType());

      SmallVector<Type> results;
      auto it = firstReturnOperandsByFunc.find(func);
      if (it != firstReturnOperandsByFunc.end()) {
        for (Value operand : it->second)
          results.push_back(operand.getType());
      } else {
        for (Type type : func.getFunctionType().getResults()) {
          if (auto vregType = dyn_cast<VMIVRegType>(type)) {
            results.push_back(VMIVRegType::get(ctx, vregType.getElementCount(),
                                               vregType.getElementType(),
                                               getContiguousLayout()));
          } else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
            results.push_back(VMIMaskType::get(ctx, maskType.getElementCount(),
                                               "b32", getContiguousLayout()));
          } else {
            results.push_back(type);
          }
        }
      }

      func.setFunctionType(FunctionType::get(ctx, inputs, results));
    });
  }

  LogicalResult run() {
    if (failed(collect()))
      return failure();
    if (failed(addConstraints()))
      return failure();
    if (failed(applyConsumerDrivenDataLayouts()))
      return failure();
    rewriteDataTypes();
    if (failed(insertDataUseMaterializations()))
      return failure();
    if (failed(inferMaskRequests()))
      return failure();
    rewriteMaskTypes();
    if (failed(insertMaskUseMaterializations()))
      return failure();
    rewriteFunctionType();
    return validateVMILayoutAssignedIR(module, /*diagOS=*/nullptr,
                                       /*verifyHelperSupport=*/false);
  }

  ModuleOp module;
  MLIRContext *ctx;
  const VMITargetCapabilityRegistry &capabilities;
  DenseMap<Value, unsigned> dataIds;
  DenseMap<Value, unsigned> maskIds;
  DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperandsByFunc;
  SmallVector<DataNode> dataNodes;
  SmallVector<MaskNode> maskNodes;
  SmallVector<DataUseRequest> dataUseRequests;
  SmallVector<MaskUseRequest> maskUseRequests;
};

struct VMILayoutAssignmentPass
    : public mlir::pto::impl::VMILayoutAssignmentBase<VMILayoutAssignmentPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILayoutAssignmentPass)

  void runOnOperation() override {
    VMITargetCapabilityRegistry capabilities;
    if (failed(LayoutSolver(getOperation(), capabilities).run()))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMILayoutAssignmentPass() {
  return std::make_unique<VMILayoutAssignmentPass>();
}
