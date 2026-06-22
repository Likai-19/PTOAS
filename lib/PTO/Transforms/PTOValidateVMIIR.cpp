// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOValidateVMIIR.cpp - VMI boundary verifier ----------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVALIDATEVMIIR
#define GEN_PASS_DEF_PTOVALIDATEVMILAYOUTIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr const char *kVMISelectedPlanAttrName = "vmi.selected_plan";

bool isVMIType(Type type) { return isa<VMIVRegType, VMIMaskType>(type); }

bool isPhysicalVPTOType(Type type) {
  return isa<VRegType, MaskType, AlignType>(type);
}

bool containsVMIOrPhysicalType(Type type) {
  if (isVMIType(type) || isPhysicalVPTOType(type))
    return true;

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), [](Type input) {
             return containsVMIOrPhysicalType(input);
           }) ||
           llvm::any_of(functionType.getResults(), [](Type result) {
             return containsVMIOrPhysicalType(result);
           });
  }

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return containsVMIOrPhysicalType(shapedType.getElementType());

  return false;
}

bool containsVMIOrPhysicalType(Attribute attr) {
  if (!attr)
    return false;

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    if (containsVMIOrPhysicalType(typeAttr.getValue()))
      return true;

  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    if (containsVMIOrPhysicalType(typedAttr.getType()))
      return true;

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr))
    return llvm::any_of(arrayAttr, [](Attribute element) {
      return containsVMIOrPhysicalType(element);
    });

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr))
    return llvm::any_of(dictAttr, [](NamedAttribute namedAttr) {
      return containsVMIOrPhysicalType(namedAttr.getValue());
    });

  return false;
}

bool isSurfaceVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return !vregType.getLayout();
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.isPred() && !maskType.getLayout();
  return false;
}

bool isLayoutAssignedVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return static_cast<bool>(vregType.getLayoutAttr());
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getLayoutAttr() &&
           VMIMaskType::isConcreteGranularity(maskType.getGranularity());
  return false;
}

bool isVMIHelperOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.vmi.ensure_layout" ||
         name == "pto.vmi.ensure_mask_layout" ||
         name == "pto.vmi.ensure_mask_granularity" ||
         name == "pto.vmi.pack" || name == "pto.vmi.unpack";
}

bool isVMILayoutHelperOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "pto.vmi.ensure_layout" ||
         name == "pto.vmi.ensure_mask_layout" ||
         name == "pto.vmi.ensure_mask_granularity";
}

bool isVMISemanticOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name.starts_with("pto.vmi.") && !isVMIHelperOp(op);
}

bool isStructuralOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "builtin.module" || name.starts_with("func.") ||
         name.starts_with("scf.") || name.starts_with("cf.");
}

bool hasVMIOrPhysicalType(Operation *op) {
  auto hasInterestingType = [](Type type) {
    return isVMIType(type) || isPhysicalVPTOType(type);
  };
  if (llvm::any_of(op->getOperandTypes(), hasInterestingType) ||
      llvm::any_of(op->getResultTypes(), hasInterestingType))
    return true;
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (llvm::any_of(block.getArgumentTypes(), hasInterestingType))
        return true;
    }
  }
  return false;
}

void mirrorDiagnostic(llvm::raw_ostream *diagOS, Twine message) {
  if (diagOS)
    *diagOS << message << "\n";
}

LogicalResult emitInvariant(Operation *op, llvm::raw_ostream *diagOS,
                            Twine message) {
  InFlightDiagnostic diag =
      op->emitError() << kVMIDiagPassInvariantPrefix << message;
  (void)diag;
  mirrorDiagnostic(diagOS, Twine(kVMIDiagPassInvariantPrefix) + message);
  return failure();
}

LogicalResult emitLayoutContract(Operation *op, llvm::raw_ostream *diagOS,
                                 Twine message) {
  InFlightDiagnostic diag =
      op->emitError() << kVMIDiagLayoutContractPrefix << message;
  (void)diag;
  mirrorDiagnostic(diagOS, Twine(kVMIDiagLayoutContractPrefix) + message);
  return failure();
}

std::optional<int64_t> getGroupSize(VMIVRegType type, int64_t numGroups) {
  if (!type || numGroups <= 0 || type.getElementCount() % numGroups != 0)
    return std::nullopt;
  return type.getElementCount() / numGroups;
}

bool hasRegisteredGroupReducePlan(VMIGroupReduceAddFOp op) {
  auto sourceType = dyn_cast<VMIVRegType>(op.getSource().getType());
  if (!sourceType)
    return false;
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  if (!sourceLayout)
    return false;

  std::optional<int64_t> groupSize =
      getGroupSize(sourceType, op.getNumGroupsAttr().getInt());
  if (!groupSize)
    return false;

  if (sourceLayout.isContiguous())
    return *groupSize == 8 || *groupSize == 64;

  if (!sourceLayout.isDeinterleaved())
    return false;
  if (*groupSize == 16 && sourceLayout.getFactor() == 2)
    return sourceLayout.getBlockElems() == 1 ||
           sourceLayout.getBlockElems() == 8;
  if (*groupSize == 32 && sourceLayout.getFactor() == 4)
    return sourceLayout.getBlockElems() == 1 ||
           sourceLayout.getBlockElems() == 8;
  return false;
}

bool hasRegisteredGroupLoadPlan(VMIGroupLoadOp op) {
  auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
  if (!resultType)
    return false;
  VMILayoutAttr layout = resultType.getLayoutAttr();
  if (!layout)
    return false;
  if (layout.isContiguous())
    return true;
  if (!layout.isDeinterleaved() || layout.getBlockElems() != 8)
    return false;

  std::optional<int64_t> groupSize =
      getGroupSize(resultType, op.getNumGroupsAttr().getInt());
  if (!groupSize)
    return false;
  return (*groupSize == 16 && layout.getFactor() == 2) ||
         (*groupSize == 32 && layout.getFactor() == 4);
}

bool hasRegisteredGroupSlotLoadPlan(VMIGroupSlotLoadOp op) {
  auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
  if (!resultType)
    return false;
  VMILayoutAttr layout = resultType.getLayoutAttr();
  return layout && layout.isGroupSlots() &&
         layout.getNumGroups() == op.getNumGroupsAttr().getInt() &&
         (layout.getSlots() == 8 || layout.getSlots() == 1);
}

bool hasRegisteredGroupBroadcastPlan(VMIGroupBroadcastOp op) {
  auto sourceType = dyn_cast<VMIVRegType>(op.getSource().getType());
  auto resultType = dyn_cast<VMIVRegType>(op.getResult().getType());
  if (!sourceType || !resultType)
    return false;
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  return sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
         sourceLayout.getNumGroups() == op.getNumGroupsAttr().getInt() &&
         !resultLayout.isGroupSlots() &&
         (sourceLayout.getSlots() == 8 || sourceLayout.getSlots() == 1);
}

bool hasRegisteredGroupSlotTruncFPlan(Operation *op) {
  auto truncf = dyn_cast<VMITruncFOp>(op);
  if (!truncf)
    return false;

  auto sourceType = dyn_cast<VMIVRegType>(truncf.getSource().getType());
  auto resultType = dyn_cast<VMIVRegType>(truncf.getResult().getType());
  if (!sourceType || !resultType)
    return false;

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  return sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
         resultLayout.isGroupSlots() && sourceLayout.getSlots() == 1 &&
         resultLayout.getSlots() == 1 && sourceType.getElementType().isF32() &&
         resultType.getElementType().isF16();
}

bool requiresSelectedPlan(Operation *op) {
  if (auto groupLoad = dyn_cast<VMIGroupLoadOp>(op))
    return hasRegisteredGroupLoadPlan(groupLoad);
  if (auto groupSlotLoad = dyn_cast<VMIGroupSlotLoadOp>(op))
    return hasRegisteredGroupSlotLoadPlan(groupSlotLoad);
  if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op))
    return hasRegisteredGroupReducePlan(reduce);
  if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op))
    return hasRegisteredGroupBroadcastPlan(broadcast);
  return hasRegisteredGroupSlotTruncFPlan(op);
}

LogicalResult verifySelectedPlanContract(Operation *op,
                                         llvm::raw_ostream *diagOS) {
  if (!requiresSelectedPlan(op))
    return success();
  if (op->getAttrOfType<StringAttr>(kVMISelectedPlanAttrName))
    return success();
  return emitLayoutContract(
      op, diagOS,
      Twine(op->getName().getStringRef()) +
          " requires vmi.selected_plan selected by vmi-layout-assignment");
}

LogicalResult verifyBoundaryType(Operation *owner, Type type,
                                 llvm::raw_ostream *diagOS) {
  if (isPhysicalVPTOType(type))
    return emitInvariant(
        owner, diagOS,
        "physical VPTO register type appears before VMI-to-VPTO");

  if (isVMIType(type) && !isSurfaceVMIType(type))
    return emitInvariant(
        owner, diagOS,
        "VMI producer boundary requires surface !pto.vmi.vreg or "
        "!pto.vmi.mask<Nxpred> type");

  return success();
}

LogicalResult verifyBoundaryTypeTree(Operation *owner, Type type,
                                     llvm::raw_ostream *diagOS) {
  if (failed(verifyBoundaryType(owner, type, diagOS)))
    return failure();

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs())
      if (failed(verifyBoundaryTypeTree(owner, input, diagOS)))
        return failure();
    for (Type result : functionType.getResults())
      if (failed(verifyBoundaryTypeTree(owner, result, diagOS)))
        return failure();
  }

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return verifyBoundaryTypeTree(owner, shapedType.getElementType(), diagOS);

  return success();
}

LogicalResult verifyLayoutAssignedType(Operation *owner, Type type,
                                       llvm::raw_ostream *diagOS) {
  if (isPhysicalVPTOType(type))
    return emitInvariant(
        owner, diagOS,
        "physical VPTO register type appears before VMI-to-VPTO");

  if (isVMIType(type) && !isLayoutAssignedVMIType(type))
    return emitInvariant(
        owner, diagOS,
        "layout-assigned VMI IR requires !pto.vmi.vreg with layout and "
        "!pto.vmi.mask with b8/b16/b32 granularity plus layout");

  return success();
}

LogicalResult verifyLayoutAssignedTypeTree(Operation *owner, Type type,
                                           llvm::raw_ostream *diagOS) {
  if (failed(verifyLayoutAssignedType(owner, type, diagOS)))
    return failure();

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs())
      if (failed(verifyLayoutAssignedTypeTree(owner, input, diagOS)))
        return failure();
    for (Type result : functionType.getResults())
      if (failed(verifyLayoutAssignedTypeTree(owner, result, diagOS)))
        return failure();
  }

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return verifyLayoutAssignedTypeTree(owner, shapedType.getElementType(),
                                        diagOS);

  return success();
}

template <typename TypeVerifier>
LogicalResult verifyAttributeTypes(Operation *owner, Attribute attr,
                                   llvm::raw_ostream *diagOS,
                                   TypeVerifier verifyType) {
  if (!attr)
    return success();

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    if (failed(verifyType(owner, typeAttr.getValue(), diagOS)))
      return failure();

  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    if (failed(verifyType(owner, typedAttr.getType(), diagOS)))
      return failure();

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : arrayAttr)
      if (failed(verifyAttributeTypes(owner, element, diagOS, verifyType)))
        return failure();
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute namedAttr : dictAttr)
      if (failed(verifyAttributeTypes(owner, namedAttr.getValue(), diagOS,
                                      verifyType)))
        return failure();
  }

  return success();
}

bool isFunctionTypeAttr(Operation *op, NamedAttribute attr) {
  return isa<func::FuncOp>(op) && attr.getName() == "function_type";
}

LogicalResult verifyNoHiddenVMIAttributeType(Operation *op,
                                             NamedAttribute attr,
                                             llvm::raw_ostream *diagOS) {
  if (isFunctionTypeAttr(op, attr))
    return success();
  if (containsVMIOrPhysicalType(attr.getValue()))
    return emitInvariant(
        op, diagOS,
        "VMI or physical VPTO type appears in a non-signature attribute");
  return success();
}

LogicalResult verifyOperationTypes(Operation *op, llvm::raw_ostream *diagOS) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType functionType = funcOp.getFunctionType();
    for (Type type : functionType.getInputs())
      if (failed(verifyBoundaryTypeTree(op, type, diagOS)))
        return failure();
    for (Type type : functionType.getResults())
      if (failed(verifyBoundaryTypeTree(op, type, diagOS)))
        return failure();
  }

  for (Type type : op->getOperandTypes())
    if (failed(verifyBoundaryTypeTree(op, type, diagOS)))
      return failure();
  for (Type type : op->getResultTypes())
    if (failed(verifyBoundaryTypeTree(op, type, diagOS)))
      return failure();
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Type type : block.getArgumentTypes()) {
        if (failed(verifyBoundaryTypeTree(op, type, diagOS)))
          return failure();
      }
    }
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (failed(verifyNoHiddenVMIAttributeType(op, attr, diagOS)))
      return failure();
    if (failed(verifyAttributeTypes(op, attr.getValue(), diagOS,
                                    verifyBoundaryTypeTree)))
      return failure();
  }
  return success();
}

LogicalResult verifyLayoutAssignedOperationTypes(Operation *op,
                                                 llvm::raw_ostream *diagOS) {
  if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
    FunctionType functionType = funcOp.getFunctionType();
    for (Type type : functionType.getInputs())
      if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS)))
        return failure();
    for (Type type : functionType.getResults())
      if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS)))
        return failure();
  }

  for (Type type : op->getOperandTypes())
    if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS)))
      return failure();
  for (Type type : op->getResultTypes())
    if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS)))
      return failure();
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Type type : block.getArgumentTypes()) {
        if (failed(verifyLayoutAssignedTypeTree(op, type, diagOS)))
          return failure();
      }
    }
  }
  for (NamedAttribute attr : op->getAttrs()) {
    if (failed(verifyNoHiddenVMIAttributeType(op, attr, diagOS)))
      return failure();
    if (failed(verifyAttributeTypes(op, attr.getValue(), diagOS,
                                    verifyLayoutAssignedTypeTree)))
      return failure();
  }
  return success();
}

LogicalResult verifyOperationBoundary(Operation *op,
                                      llvm::raw_ostream *diagOS) {
  if (failed(verifyOperationTypes(op, diagOS)))
    return failure();

  if (!hasVMIOrPhysicalType(op))
    return success();

  if (isVMIHelperOp(op))
    return emitInvariant(
        op, diagOS,
        "VMI helper op appears before layout assignment or VMI-to-VPTO");

  if (isVMISemanticOp(op) || isStructuralOp(op))
    return success();

  return emitInvariant(op, diagOS,
                       "VMI typed value is used by a non-VMI semantic op");
}

LogicalResult verifyLayoutAssignedOperation(Operation *op,
                                            llvm::raw_ostream *diagOS) {
  if (failed(verifyLayoutAssignedOperationTypes(op, diagOS)))
    return failure();

  if (!hasVMIOrPhysicalType(op))
    return success();

  if (failed(verifySelectedPlanContract(op, diagOS)))
    return failure();

  if (isVMIHelperOp(op)) {
    if (isVMILayoutHelperOp(op))
      return success();
    return emitInvariant(
        op, diagOS,
        "VMI pack/unpack helper appears before VMI-to-VPTO physicalization");
  }

  if (isVMISemanticOp(op) || isStructuralOp(op))
    return success();

  return emitInvariant(op, diagOS,
                       "VMI typed value is used by a non-VMI semantic op");
}

struct PTOValidateVMIIRPass
    : public mlir::pto::impl::PTOValidateVMIIRBase<PTOValidateVMIIRPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateVMIIRPass)

  void runOnOperation() override {
    if (failed(validateVMIProducerBoundaryIR(getOperation(), &llvm::errs())))
      signalPassFailure();
  }
};

struct PTOValidateVMILayoutIRPass
    : public mlir::pto::impl::PTOValidateVMILayoutIRBase<
          PTOValidateVMILayoutIRPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOValidateVMILayoutIRPass)

  void runOnOperation() override {
    if (failed(validateVMILayoutAssignedIR(getOperation(), &llvm::errs())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult mlir::pto::validateVMIProducerBoundaryIR(
    ModuleOp module, llvm::raw_ostream *diagOS) {
  WalkResult result = module.walk([&](Operation *op) {
    if (failed(verifyOperationBoundary(op, diagOS)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult mlir::pto::validateVMILayoutAssignedIR(
    ModuleOp module, llvm::raw_ostream *diagOS) {
  WalkResult result = module.walk([&](Operation *op) {
    if (failed(verifyLayoutAssignedOperation(op, diagOS)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

std::unique_ptr<Pass> mlir::pto::createPTOValidateVMIIRPass() {
  return std::make_unique<PTOValidateVMIIRPass>();
}

std::unique_ptr<Pass> mlir::pto::createPTOValidateVMILayoutIRPass() {
  return std::make_unique<PTOValidateVMILayoutIRPass>();
}
