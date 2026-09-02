// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI.cpp - PTO VMI type and attribute support -----------------------===//
//===----------------------------------------------------------------------===//

#include <optional>
#include <set>
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Support/CodeConstants.h"

#include "VMIInternal.h"


using namespace mlir;
using namespace mlir::pto;

// Legacy (old) VMI op verifiers
//===----------------------------------------------------------------------===//

//===--- Legacy (old) VMI op verifiers ---===//

LogicalResult VMIConstantOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto denseAttr = dyn_cast<DenseElementsAttr>(getValue());
  if (!denseAttr) {
    return emitOpError("requires dense elements constant attribute");
  }
  if (denseAttr.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires dense constant element type to match result element type");
  }
  if (denseAttr.getNumElements() != resultType.getElementCount()) {
    return emitOpError("requires dense constant element count to match result "
                       "logical lane count");
  }
  return success();
}

LogicalResult VMIBroadcastOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type valueType = getValue().getType();
  if (valueType == resultType.getElementType()) {
    return success();
  }
  if (auto vregType = dyn_cast<VMIVRegType>(valueType)) {
    if (vregType.getElementCount() != 1) {
      return emitOpError("requires VMI vector input to have one logical lane");
    }
    if (vregType.getElementType() != resultType.getElementType()) {
      return emitOpError("requires VMI vector input element type to match "
                         "result element type");
    }
    return success();
  }
  return emitOpError("requires scalar or VMI vector input element type to "
                     "match result element type");
}

LogicalResult VMIIotaOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = resultType.getElementType();
  if (!isVMIIotaElementType(elementType)) {
    return emitOpError("requires result element type to be integer 8/16/32 "
                       "or f16/f32");
  }
  if (!isCompatibleScalarForSemanticType(elementType, getBase().getType())) {
    return emitOpError("requires base type to match result element type");
  }

  if (std::optional<StringRef> order = getOrder()) {
    if (*order != "ASC" && *order != "DESC") {
      return emitOpError("requires order to be ASC or DESC");
    }
  }
  return success();
}

LogicalResult VMIGroupIotaOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = resultType.getElementType();
  if (!isVMIIotaElementType(elementType)) {
    return emitOpError("requires result element type to be integer 8/16/32 "
                       "or f16/f32");
  }
  if (!isCompatibleScalarForSemanticType(elementType, getBase().getType())) {
    return emitOpError("requires base type to match result element type");
  }
  if (std::optional<StringRef> order = getOrder()) {
    if (*order != "ASC" && *order != "DESC") {
      return emitOpError("requires order to be ASC or DESC");
    }
  }

  int64_t numGroups = getGroupAttr().getInt();
  if (numGroups <= 1) {
    return emitOpError("requires group greater than one");
  }
  if (resultType.getElementCount() % numGroups != 0) {
    return emitOpError("requires group to evenly divide result logical lane "
                       "count");
  }
  int64_t groupSize = resultType.getElementCount() / numGroups;
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(elementType);
  if (succeeded(lanesPerPart) && groupSize % *lanesPerPart != 0 &&
      *lanesPerPart % groupSize != 0) {
    return emitOpError("requires group_size to divide or be a multiple of "
                       "physical lanes per part (")
           << *lanesPerPart << ")";
  }
  if (VMILayoutAttr layout = resultType.getLayoutAttr();
      layout && !layout.isContiguous()) {
    return emitOpError("requires contiguous result layout");
  }
  return success();
}

LogicalResult VMICreateMaskOp::verify() {
  return success();
}

LogicalResult VMICreateGroupMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (numGroups <= 0) {
    return emitOpError("requires positive num_groups");
  }
  if (groupSize <= 0) {
    return emitOpError("requires positive group_size");
  }
  if (resultType.getElementCount() != numGroups * groupSize) {
    return emitOpError("requires result lane count to equal num_groups * "
                       "group_size");
  }
  return success();
}

LogicalResult VMIConstantMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  auto denseAttr = dyn_cast<DenseElementsAttr>(getValue());
  if (!denseAttr) {
    return emitOpError("requires dense elements mask constant attribute");
  }
  if (!denseAttr.getElementType().isInteger(1)) {
    return emitOpError("requires dense mask constant element type to be i1");
  }
  if (denseAttr.getNumElements() != resultType.getElementCount()) {
    return emitOpError("requires dense mask constant element count to match "
                       "result logical lane count");
  }
  return success();
}

LogicalResult VMIMaskAndOp::verify() {
  auto lhsType = cast<VMIMaskType>(getLhs().getType());
  auto rhsType = cast<VMIMaskType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(
      getOperation(), {lhsType, rhsType, resultType});
}

LogicalResult VMIMaskOrOp::verify() {
  auto lhsType = cast<VMIMaskType>(getLhs().getType());
  auto rhsType = cast<VMIMaskType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(
      getOperation(), {lhsType, rhsType, resultType});
}

LogicalResult VMIMaskXOrOp::verify() {
  auto lhsType = cast<VMIMaskType>(getLhs().getType());
  auto rhsType = cast<VMIMaskType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(
      getOperation(), {lhsType, rhsType, resultType});
}

LogicalResult VMIMaskNotOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  return verifyAllSameMaskShapeLayoutAndGranularity(getOperation(),
                                                    {sourceType, resultType});
}

LogicalResult VMIAddFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIAddIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMISubFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMISubIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMulFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMulIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  auto intType = dyn_cast<IntegerType>(lhsType.getElementType());
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  if (!supportedInteger) {
    return emitOpError("requires i16 or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIFmaOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyFloatTernaryVRegOp(getOperation(), lhsType, rhsType, accType,
                                  resultType);
}

//===----------------------------------------------------------------------===//
// Legacy elementwise op verifiers (restored for backward compatibility).
//===----------------------------------------------------------------------===//

LogicalResult VMIDivFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMinFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMaxFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMinIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()) ||
      !isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 integer-like VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMaxIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()) ||
      !isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 integer-like VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMINegFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMINegIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMIAbsFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIAbsIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMISignedI8I16I32Type(sourceType.getElementType())) {
    return emitOpError("requires si8, si16, or si32 VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMISqrtOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIExpOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMILnOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

static LogicalResult verifySignedI32OrF16F32ElementType(Operation *op,
                                                        Type elementType) {
  bool supportedInteger = false;
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    supportedInteger =
        intType.getWidth() == mlir::pto::kValue32 &&
        matchesVMIIntSemantics(intType, VMIIntSignSemantics::Signed);
  }
  if (!supportedInteger && !isVMIF16OrF32Type(elementType)) {
    return op->emitOpError("requires si32, f16, or f32 VMI element type");
  }
  return success();
}

LogicalResult VMIReluOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = sourceType.getElementType();
  if (failed(verifySignedI32OrF16F32ElementType(getOperation(), elementType))) {
    return failure();
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMIAndIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIOrIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIXOrIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIShLIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIShRUIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType || integerType.isSigned()) {
    return emitOpError(
        "requires signless or unsigned integer VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIShRSIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMISignedI8I16I32Type(lhsType.getElementType())) {
    return emitOpError(
        "requires signed i8, i16, or i32 VMI element type");
  }
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMINotOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

//===----------------------------------------------------------------------===//

LogicalResult VMICmpFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType())) {
    return emitOpError("requires floating-point-like VMI element type");
  }
  if (!isVMIF16BF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), resultType, lhsType);
}

LogicalResult VMICmpIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (!isVMIAnyI8I16I32Type(lhsType.getElementType())) {
    return emitOpError("requires i8, i16, or i32 VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), resultType, lhsType);
}

LogicalResult VMISelectOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto trueType = cast<VMIVRegType>(getTrueValue().getType());
  auto falseType = cast<VMIVRegType>(getFalseValue().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {trueType, falseType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

LogicalResult VMIActivePrefixIndexOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto resultIntType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultIntType || !resultIntType.isSignless()) {
    return emitOpError("requires signless integer result element type");
  }
  unsigned resultWidth = resultIntType.getWidth();
  if (resultWidth != mlir::pto::kValue8 && resultWidth != mlir::pto::kValue16 && resultWidth != mlir::pto::kValue32) {
    return emitOpError("requires i8, i16, or i32 result element type");
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

LogicalResult VMICompressOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {sourceType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

LogicalResult VMICompressStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMICompressStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIReduceAddIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI source element type");
  }
  auto sourceIntegerType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!sourceIntegerType || sourceIntegerType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires 32-bit integer source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceAddFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!getOperation()->hasAttr("reassoc")) {
    return emitOpError(
        "requires reassoc attr because VPTO vcadd performs pair-wise "
        "floating-point reduction");
  }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return emitOpError("requires floating-point-like VMI source element type");
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

template <typename OpTy> static LogicalResult verifyReduceMinMaxFOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return op.emitOpError(
        "requires floating-point-like VMI source element type");
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return op.emitOpError("requires f16 or f32 source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op.emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return op.emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(op.getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceMaxFOp::verify() { return verifyReduceMinMaxFOp(*this); }

LogicalResult VMIReduceMinFOp::verify() { return verifyReduceMinMaxFOp(*this); }

template <typename OpTy> static LogicalResult verifyReduceMinMaxIOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto sourceIntegerType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!sourceIntegerType ||
      !isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return op.emitOpError(
        "requires 8-bit, 16-bit, or 32-bit integer source element type");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op.emitOpError("requires source and result element types to match");
  }
  if (resultType.getElementCount() != 1) {
    return op.emitOpError("requires result to be a 1-lane VMI vector");
  }
  return verifyMaskMatchesData(op.getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceMaxIOp::verify() { return verifyReduceMinMaxIOp(*this); }

LogicalResult VMIReduceMinIOp::verify() { return verifyReduceMinMaxIOp(*this); }

template <typename OpTy>
static LogicalResult verifyGroupReduceCommon(OpTy op, VMIVRegType sourceType,
                                             VMIVRegType resultType,
                                             VMIMaskType maskType) {
  int64_t numGroups = op.getNumGroupsAttr().getInt();
  if (resultType.getElementCount() != numGroups) {
    return op.emitOpError(
        "requires result logical lane count to match num_groups");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op.emitOpError("requires source and result element types to match");
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    bool supportedSourceLayout =
        sourceLayout.isContiguous() ||
        (sourceLayout.isDenseSplit() &&
         (sourceLayout.getFactor() == mlir::pto::kValue2 ||
          sourceLayout.getFactor() == mlir::pto::kValue4));
    if (!supportedSourceLayout) {
      return op.emitOpError(
          "requires layout-assigned source to use contiguous layout or "
          "deinterleaved=2/4 or block_deinterleaved=2/4 layout");
    }
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != numGroups) {
      return op.emitOpError() << "requires layout-assigned result to use "
                                 "#pto.vmi.layout<num_groups = "
                              << numGroups << ">";
    }
  }
  if (failed(verifyMaskMatchesData(op.getOperation(), maskType, sourceType))) {
    return failure();
  }
  return verifyNumGroups(op.getOperation(), sourceType, numGroups);
}

template <typename OpTy>
static LogicalResult verifyGroupReduceFloatOp(OpTy op, bool requiresReassoc) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (requiresReassoc && !op->hasAttr("reassoc")) {
    return op.emitOpError(
        "requires reassoc attr because grouped lowering uses pair-wise "
        "floating-point reductions");
  }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return op.emitOpError(
        "requires floating-point-like VMI source element type");
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return op.emitOpError("requires f16 or f32 source element type");
  }
  return verifyGroupReduceCommon(op, sourceType, resultType, maskType);
}

LogicalResult VMIGroupReduceAddFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/true);
}

LogicalResult VMIGroupReduceMaxFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/false);
}

LogicalResult VMIGroupReduceMinFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/false);
}

template <typename OpTy>
static LogicalResult verifyGroupReduceIntegerOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return op.emitOpError("requires integer-like VMI source element type");
  }
  auto intType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!intType || !isVMIAnyI8I16I32Type(sourceType.getElementType())) {
    return op.emitOpError(
        "requires 8-bit, 16-bit, or 32-bit integer source element type");
  }
  return verifyGroupReduceCommon(op, sourceType, resultType, maskType);
}

LogicalResult VMIGroupReduceAddIOp::verify() {
  return verifyGroupReduceIntegerOp(*this);
}

LogicalResult VMIGroupReduceMaxIOp::verify() {
  return verifyGroupReduceIntegerOp(*this);
}

LogicalResult VMIGroupReduceMinIOp::verify() {
  return verifyGroupReduceIntegerOp(*this);
}

//===----------------------------------------------------------------------===//
// Group 5: vcadd / vcmax / vcmin verifiers
//===----------------------------------------------------------------------===//

LogicalResult VMIGroupBroadcastOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  if (sourceType.getElementCount() != numGroups) {
    return emitOpError(
        "requires source logical lane count to match num_groups");
  }
  if (resultType.getElementCount() % numGroups != 0) {
    return emitOpError(
        "requires num_groups to evenly divide result logical lane count");
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result element types to match");
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    if (!sourceLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != numGroups) {
      return emitOpError() << "requires layout-assigned source to use "
                              "#pto.vmi.layout<num_groups = "
                           << numGroups << ">";
    }
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (resultLayout.isGroupSlots()) {
      return emitOpError(
          "requires layout-assigned result to use a dense VMI layout");
    }
  }
  return verifyNumGroups(getOperation(), resultType, numGroups);
}

template <typename OpTy>
static LogicalResult verifyVMIHistogramLayouts(OpTy op, VMIVRegType accType,
                                               VMIVRegType sourceType,
                                               VMIVRegType resultType,
                                               VMIMaskType maskType) {
  if (auto accLayout = accType.getLayoutAttr()) {
    if (!accLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned acc to use contiguous "
                            "layout");
    }
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    if (!sourceLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned source to use contiguous "
                            "layout");
    }
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned result to use "
                            "contiguous layout");
    }
  }
  if (auto maskLayout = maskType.getLayoutAttr()) {
    if (!maskLayout.isContiguous()) {
      return op.emitOpError("requires layout-assigned mask to use contiguous "
                            "layout");
    }
    if (maskType.getGranularity() != "b8") {
      return op.emitOpError("requires layout-assigned mask granularity b8");
    }
  }
  return success();
}

template <typename OpTy> static LogicalResult verifyVMIHistogramOp(OpTy op) {
  auto accType = cast<VMIVRegType>(op.getAcc().getType());
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  auto accElemType = dyn_cast<IntegerType>(accType.getElementType());
  auto sourceElemType = dyn_cast<IntegerType>(sourceType.getElementType());
  int64_t bins = accType.getElementCount();
  if (!accElemType || accElemType.getWidth() != mlir::pto::kValue16 ||
      !matchesVMIIntSemantics(accElemType, VMIIntSignSemantics::Unsigned) ||
      (bins != mlir::pto::kValue128 && bins != mlir::pto::kValue256)) {
    return op.emitOpError("requires acc type to be "
                          "!pto.vmi.vreg<128x{ui16|i16}> (Bin_N0-only) or "
                          "!pto.vmi.vreg<256x{ui16|i16}>");
  }
  if (resultType.getElementCount() != bins) {
    return op.emitOpError("requires result element count to match acc "
                          "(bins must be identical)");
  }
  if (resultType.getLayoutAttr() != accType.getLayoutAttr()) {
    return op.emitOpError("requires result layout attribute to match acc");
  }
  auto resultElemType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultElemType || resultElemType.getWidth() != mlir::pto::kValue16 ||
      !matchesVMIIntSemantics(resultElemType, VMIIntSignSemantics::Unsigned)) {
    return op.emitOpError("requires result element type to be ui16 or i16 "
                          "(interpreted as unsigned)");
  }
  if (!sourceElemType || sourceElemType.getWidth() != mlir::pto::kValue8 ||
      !matchesVMIIntSemantics(sourceElemType, VMIIntSignSemantics::Unsigned)) {
    return op.emitOpError("requires source type to be "
                          "!pto.vmi.vreg<Nx{ui8|i8}> (interpreted as unsigned)");
  }
  if (maskType.getElementCount() != sourceType.getElementCount()) {
    return op.emitOpError("requires mask logical lane count to match source");
  }
  return verifyVMIHistogramLayouts(op, accType, sourceType, resultType,
                                   maskType);
}

LogicalResult VMIVdhistOp::verify() { return verifyVMIHistogramOp(*this); }

LogicalResult VMIVchistOp::verify() { return verifyVMIHistogramOp(*this); }

//===----------------------------------------------------------------------===//
// Group 7: SFU verifiers
//===----------------------------------------------------------------------===//

LogicalResult VMIExtFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type sourceElementType = sourceType.getElementType();
  Type resultElementType = resultType.getElementType();
  bool hasMatchingLaneCount =
      sourceType.getElementCount() == resultType.getElementCount();
  if (!hasMatchingLaneCount) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  if (!isVMIFloatLikeType(sourceElementType) ||
      !isVMIFloatLikeType(resultElementType)) {
    return emitOpError(
        "requires floating-point-like source and result element types");
  }
  if (involvesBF16x2(sourceElementType, resultElementType) &&
      !lookupVMIFpToFpContract(sourceElementType, resultElementType)) {
    return emitOpError(
        "unsupported bf16x2 fp-to-fp conversion element type pair");
}
  if (getVMIElementBitWidth(sourceElementType) >=
      getVMIElementBitWidth(resultElementType)) {
    return emitOpError(
        "requires result element type to be wider than source element type");
}
  return success();
}

static LogicalResult verifyFPToFPRoundingAndSaturate(
    Operation *op, const std::optional<VMIFpToFpContract> &fpContract) {
  if (auto roundingAttr = op->getAttrOfType<StringAttr>("rounding")) {
    StringRef rounding = roundingAttr.getValue();
    if (rounding.size() != 1) {
      return op->emitOpError(
          "rounding attr must be a single-character mode token");
    }
    StringRef allowedRndModes =
        fpContract && !fpContract->allowedRndModes.empty()
            ? fpContract->allowedRndModes
            : StringRef("RAHZ");
    if (!allowedRndModes.contains(rounding)) {
      if (fpContract && !fpContract->allowedRndModes.empty()) {
        return op->emitOpError("rounding attr is not valid for this fp-to-fp "
                               "conversion type pair");
      }
      return op->emitOpError("rounding attr must be R, A, H, or Z");
    }
  }
  auto satAttr = op->getAttrOfType<StringAttr>("saturate");
  if (!fpContract || fpContract->requiresSat) {
    if (!satAttr) {
      return op->emitOpError("'saturate' attribute is required (SAT or NOSAT)");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op->emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
    }
  } else if (satAttr) {
    return op->emitOpError("'saturate' attribute is not valid for this fp-to-fp "
                           "narrow conversion (no saturation)");
  }
  return success();
}

LogicalResult VMITruncFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type sourceElementType = sourceType.getElementType();
  Type resultElementType = resultType.getElementType();
  auto fpContract =
      lookupVMIFpToFpContract(sourceElementType, resultElementType);
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  if (!isVMIFloatLikeType(sourceElementType) ||
      !isVMIFloatLikeType(resultElementType)) {
    return emitOpError(
        "requires floating-point-like source and result element types");
  }
  if (involvesBF16x2(sourceElementType, resultElementType) && !fpContract) {
    return emitOpError(
        "unsupported bf16x2 fp-to-fp conversion element type pair");
  }
  if (involvesVMIPackedFloatCarrier(sourceElementType, resultElementType) &&
      !fpContract) {
    return emitOpError(
        "unsupported packed fp-to-fp conversion element type pair");
  }
  unsigned srcBits = getVMIElementBitWidth(sourceElementType);
  unsigned dstBits = getVMIElementBitWidth(resultElementType);
  if (srcBits < dstBits) {
    return emitOpError(
        "requires result element type to be narrower than or same-width "
        "as source element type");
  }
  if (srcBits == dstBits && !fpContract) {
    return emitOpError("same-width fp-to-fp conversion is not supported "
                       "for this type pair; see lookupVMIFpToFpContract");
  }
  return verifyFPToFPRoundingAndSaturate(*this, fpContract);
}

LogicalResult VMIFPToSIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return emitOpError("requires floating-point-like source element type");
  }
  if (!isVMISignedIntegerType(resultType.getElementType())) {
    return emitOpError("requires signed integer result element type");
  }
  auto contract =
      lookupVMIFpToSiContract(sourceType.getElementType(),
                              resultType.getElementType());
  if (!contract) {
    return emitOpError("unsupported fp-to-si conversion element type pair");
  }
  if (auto roundingAttr = (*this)->getAttrOfType<StringAttr>("rounding")) {
    StringRef rounding = roundingAttr.getValue();
    if (rounding != "R" && rounding != "A" && rounding != "F" &&
        rounding != "C" && rounding != "Z") {
      return emitOpError("rounding attr must be R, A, F, C, or Z");
    }
  }
  if (contract->requiresSat) {
    auto satAttr = (*this)->getAttrOfType<StringAttr>("saturate");
    if (!satAttr) {
      return emitOpError("'saturate' attribute is required for this fp-to-si "
                         "conversion (SAT or NOSAT)");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
    }
  } else {
    if ((*this)->getAttrOfType<StringAttr>("saturate")) {
      return emitOpError("'saturate' attribute is not valid for this fp-to-si "
                         "conversion (no overflow possible)");
    }
  }
  return success();
}

LogicalResult VMIFPToUIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  if (!isVMIFloatLikeType(sourceType.getElementType())) {
    return emitOpError("requires floating-point-like source element type");
  }
  if (!isVMIUnsignedOrSignlessIntegerType(resultType.getElementType())) {
    return emitOpError(
        "requires unsigned or signless integer result element type");
  }
  auto contract =
      lookupVMIFpToUIContract(sourceType.getElementType(),
                              resultType.getElementType());
  if (!contract) {
    return emitOpError("unsupported fp-to-ui conversion element type pair");
  }
  if (auto roundingAttr = (*this)->getAttrOfType<StringAttr>("rounding")) {
    StringRef rounding = roundingAttr.getValue();
    if (rounding != "R" && rounding != "A" && rounding != "F" &&
        rounding != "C" && rounding != "Z") {
      return emitOpError("rounding attr must be R, A, F, C, or Z");
    }
  }
  if (contract->requiresSat) {
    auto satAttr = (*this)->getAttrOfType<StringAttr>("saturate");
    if (!satAttr) {
      return emitOpError("'saturate' attribute is required for this fp-to-ui "
                         "conversion (SAT or NOSAT)");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
    }
  } else {
    if ((*this)->getAttrOfType<StringAttr>("saturate")) {
      return emitOpError("'saturate' attribute is not valid for this fp-to-ui "
                         "conversion (no overflow possible)");
    }
  }
  return success();
}

LogicalResult VMISIToFPOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  if (!isVMISignedIntegerType(sourceType.getElementType())) {
    return emitOpError("requires signed integer source element type");
  }
  if (!isVMIFloatLikeType(resultType.getElementType())) {
    return emitOpError("requires floating-point-like result element type");
  }
  unsigned srcBits = getVMIElementBitWidth(sourceType.getElementType());
  if (srcBits == mlir::pto::kValue32) {
    if (!resultType.getElementType().isF32()) {
      return emitOpError("requires f32 result element type for 32-bit "
                         "integer source");
    }
  } else if (srcBits == mlir::pto::kValue8) {
    if (!resultType.getElementType().isF16()) {
      return emitOpError("requires f16 result element type for 8-bit "
                         "integer source");
    }
  } else {
    return emitOpError("supports only si32 -> f32 or si8 -> f16");
  }
  return success();
}

LogicalResult VMIExtSIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  bool hasMatchingLaneCount =
      sourceType.getElementCount() == resultType.getElementCount();
  if (!hasMatchingLaneCount) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  bool hasSignedTypes =
      isVMISignedIntegerType(sourceType.getElementType()) &&
      isVMISignedIntegerType(resultType.getElementType());
  if (!hasSignedTypes) {
    return emitOpError(
        "requires signed integer source and result element types");
  }
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType())) {
    return emitOpError(
        "requires result element type to be wider than source element type");
  }
  return success();
}

LogicalResult VMIExtUIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  bool hasMatchingLaneCount =
      sourceType.getElementCount() == resultType.getElementCount();
  if (!hasMatchingLaneCount) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  bool hasUnsignedTypes =
      isVMIUnsignedOrSignlessIntegerType(sourceType.getElementType()) &&
      isVMIUnsignedOrSignlessIntegerType(resultType.getElementType());
  if (!hasUnsignedTypes) {
    return emitOpError(
        "requires unsigned or signless integer source and result element "
        "types");
  }
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType())) {
    return emitOpError(
        "requires result element type to be wider than source element type");
  }
  return success();
}

LogicalResult VMITruncIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  if (!isVMIIntegerLikeType(sourceType.getElementType()) ||
      !isVMIIntegerLikeType(resultType.getElementType())) {
    return emitOpError("requires integer source and result element types");
  }
  if (getVMIElementBitWidth(sourceType.getElementType()) <=
      getVMIElementBitWidth(resultType.getElementType())) {
    return emitOpError(
        "requires result element type to be narrower than source element type");
  }
  auto satAttr = (*this)->getAttrOfType<StringAttr>("saturate");
  if (!satAttr) {
    return emitOpError("'saturate' attribute is required (SAT or NOSAT)");
  }
  StringRef satVal = satAttr.getValue();
  if (satVal != "SAT" && satVal != "NOSAT") {
    return emitOpError("saturate attr must be 'SAT' or 'NOSAT'");
  }
  return success();
}

LogicalResult VMIBitcastOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0) {
    return emitOpError(
        "requires integer or floating-point source and result element types");
}
  if (sourceType.getElementCount() * static_cast<int64_t>(sourceBits) !=
      resultType.getElementCount() * static_cast<int64_t>(resultBits)) {
    return emitOpError(
        "requires source and result to carry the same total number of bits");
}

  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return emitOpError(
          "requires either both source and result to carry layout or neither "
          "to carry layout");
}
    if (sourceType.getLayout() != resultType.getLayout()) {
      return emitOpError("requires source and result layouts to match");
    }
  }

  return success();
}

LogicalResult VMILoadOp::verify() {
  if (failed(verifyMemoryElementMatches(
          getOperation(), getSource().getType(),
          cast<VMIVRegType>(getResult().getType()), "source"))) {
    return failure();
  }
  return verifyUBBackedMemory(getOperation(), getSource().getType(), "source");
}

void VMILoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIDeinterleaveLoadOp::verify() {
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {lowType, highType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        lowType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (failed(verifyContiguousIfLayoutAssigned(getOperation(), lowType,
                                              "low result")) ||
      failed(verifyContiguousIfLayoutAssigned(getOperation(), highType,
                                              "high result"))) {
    return failure();
  }
  return success();
}

void VMIDeinterleaveLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  return verifyNumGroups(getOperation(), resultType,
                         getNumGroupsAttr().getInt());
}

void VMIGroupLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupSlotLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  if (resultType.getElementCount() != numGroups) {
    return emitOpError(
        "requires result logical lane count to match num_groups");
  }
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != numGroups) {
      return emitOpError() << "requires layout-assigned result to use "
                              "#pto.vmi.layout<num_groups = "
                           << numGroups << ">";
    }
  }
  return verifyNumGroups(getOperation(), resultType, numGroups);
}

void VMIGroupSlotLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupBroadcastLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  if (numGroups <= 0) {
    return emitOpError("requires num_groups to be positive");
  }
  if (resultType.getElementCount() % numGroups != 0) {
    return emitOpError(
        "requires num_groups to evenly divide result logical lane count");
  }
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (resultLayout.isGroupSlots()) {
      return emitOpError(
          "requires layout-assigned result to use a dense VMI layout");
    }
  }
  return verifyNumGroups(getOperation(), resultType, numGroups);
}

void VMIGroupBroadcastLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIMaskedLoadOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto passthruType = cast<VMIVRegType>(getPassthru().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIMaskedLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

static LogicalResult verifyVMIOffsetIndexType(Operation *op,
                                                  Type indexElementType,
                                                  StringRef valueNoun) {
  auto intType = dyn_cast_or_null<IntegerType>(indexElementType);
  if (!intType || intType.isSigned() ||
      (intType.getWidth() != mlir::pto::kValue16 &&
       intType.getWidth() != mlir::pto::kValue32)) {
    return op->emitOpError(
               "requires signless or unsigned 16-bit or 32-bit integer ")
           << valueNoun;
  }
  return success();
}

LogicalResult VMIGatherOp::verify() {
  auto indicesType = cast<VMIVRegType>(getIndices().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto passthruType = cast<VMIVRegType>(getPassthru().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyGatherMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }

  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (failed(verifyVMIOffsetIndexType(getOperation(), indexElementType,
                                      "indices"))) {
    return failure();
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {indicesType, passthruType, resultType},
          /*requireSameElement=*/false))) {
    return failure();
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }

  if (indexElementType.getWidth() == mlir::pto::kValue16 &&
      !isSupported16BitGatherResult(
          getMemoryElementType(getSource().getType()),
          resultType.getElementType())) {
    return emitOpError(
        "requires i16/ui16/f16/bf16 result and passthru element type when "
        "using ui16 indices, or i8/ui8 -> i16/ui16 integer promotion");
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIExpandLoadOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto passthruType = cast<VMIVRegType>(getPassthru().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIExpandLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIStoreOp::verify() {
  if (failed(verifyMemoryElementMatches(
          getOperation(), getDestination().getType(),
          cast<VMIVRegType>(getValue().getType()), "destination"))) {
    return failure();
  }
  return verifyUBBackedMemory(getOperation(), getDestination().getType(),
                              "destination");
}

void VMIStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIInterleaveStoreOp::verify() {
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {lowType, highType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), lowType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  if (failed(verifyContiguousIfLayoutAssigned(getOperation(), lowType,
                                              "low input")) ||
      failed(verifyContiguousIfLayoutAssigned(getOperation(), highType,
                                              "high input"))) {
    return failure();
  }
  return success();
}

void VMIInterleaveStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIGroupStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  if (!isPackedByteGroupStore(getDestination().getType(), valueType) &&
      failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  return verifyNumGroups(getOperation(), valueType,
                         getNumGroupsAttr().getInt());
}

void VMIGroupStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIStrideLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIStrideLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIMaskedStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIMaskedStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIStrideStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIStrideStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

//===----------------------------------------------------------------------===//

LogicalResult VMIScatterOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto indicesType = cast<VMIVRegType>(getIndices().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }
  if (failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }

  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.isSigned() ||
      (indexElementType.getWidth() != mlir::pto::kValue32 &&
       indexElementType.getWidth() != mlir::pto::kValue16)) {
    return emitOpError(
        "requires signless or unsigned 16-bit or 32-bit integer indices");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {valueType, indicesType},
                                             /*requireSameElement=*/false))) {
    return failure();
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIShuffleOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires result element type to match source element type");
  }
  if (static_cast<int64_t>(getIndices().size()) != resultType.getElementCount()) {
    return emitOpError(
        "requires shuffle index count to match result logical lane count");
  }
  for (int64_t index : getIndices()) {
    if (index < 0 || index >= sourceType.getElementCount()) {
      return emitOpError("requires every shuffle index to select an existing "
                         "source logical lane");
    }
  }
  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return emitOpError("requires either both source and result to carry "
                         "layout or neither to carry layout");
    }
  }
  return success();
}

static LogicalResult verifyChannelSplitLayout(Operation *op,
                                              VMIVRegType sourceType,
                                              ValueRange results) {
  if (!isLayoutAssigned(sourceType)) {
    return op->emitOpError("requires layout-assigned channel_split source when "
                           "any channel result has layout");
  }
  for (Value result : results) {
    auto resultType = cast<VMIVRegType>(result.getType());
    if (!isLayoutAssigned(resultType)) {
      return op->emitOpError("requires every channel_split result to carry "
                             "layout when source has layout");
    }
    if (!cast<VMILayoutAttr>(resultType.getLayout()).isContiguous()) {
      return op->emitOpError(
          "requires layout-assigned channel_split results to be contiguous");
    }
  }
  int64_t channels = results.size();
  if (channels == mlir::pto::kValue2 || channels == mlir::pto::kValue4) {
    auto sourceLayout = cast<VMILayoutAttr>(sourceType.getLayout());
    auto expectedLayout =
        VMILayoutAttr::getDeinterleaved(op->getContext(), channels);
    if (!sourceLayout.isContiguous() && sourceLayout != expectedLayout) {
      return op->emitOpError("requires layout-assigned channel_split source to "
                             "be contiguous or deinterleaved by result count");
    }
  }
  return success();
}

static LogicalResult verifyChannelMergeLayout(Operation *op,
                                              VMIVRegType resultType,
                                              ValueRange inputs) {
  if (!isLayoutAssigned(resultType)) {
    return op->emitOpError("requires layout-assigned channel_merge result when "
                           "any channel input has layout");
  }
  for (Value input : inputs) {
    auto inputType = cast<VMIVRegType>(input.getType());
    if (!isLayoutAssigned(inputType)) {
      return op->emitOpError("requires every channel_merge input to carry layout "
                             "when result has layout");
    }
    if (!cast<VMILayoutAttr>(inputType.getLayout()).isContiguous()) {
      return op->emitOpError(
          "requires layout-assigned channel_merge inputs to be contiguous");
    }
  }
  int64_t channels = inputs.size();
  if (channels == mlir::pto::kValue2 || channels == mlir::pto::kValue4) {
    auto resultLayout = cast<VMILayoutAttr>(resultType.getLayout());
    auto expectedLayout =
        VMILayoutAttr::getDeinterleaved(op->getContext(), channels);
    if (!resultLayout.isContiguous() && resultLayout != expectedLayout) {
      return op->emitOpError("requires layout-assigned channel_merge result to "
                             "be contiguous or deinterleaved by input count");
    }
  }
  return success();
}

LogicalResult VMIChannelSplitOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  if (getResults().size() < mlir::pto::kValue2) {
    return emitOpError("requires at least two channel results");
  }
  auto firstResultType = cast<VMIVRegType>(getResults().front().getType());
  if (sourceType.getElementCount() !=
      static_cast<int64_t>(getResults().size()) *
          firstResultType.getElementCount()) {
    return emitOpError("requires source lane count to equal result count times "
                       "per-channel lane count");
  }
  for (Value result : getResults()) {
    auto resultType = cast<VMIVRegType>(result.getType());
    if (resultType.getElementCount() != firstResultType.getElementCount() ||
        resultType.getElementType() != sourceType.getElementType()) {
      return emitOpError("requires every channel result to have equal lane "
                         "count and source element type");
    }
  }
  if (isLayoutAssigned(sourceType) ||
      llvm::any_of(getResults(), [](Value result) {
        return isLayoutAssigned(cast<VMIVRegType>(result.getType()));
      })) {
    return verifyChannelSplitLayout(getOperation(), sourceType, getResults());
  }
  return success();
}

LogicalResult VMIChannelMergeOp::verify() {
  if (getInputs().size() < mlir::pto::kValue2) {
    return emitOpError("requires at least two channel inputs");
  }
  auto firstInputType = cast<VMIVRegType>(getInputs().front().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  for (Value input : getInputs()) {
    auto inputType = cast<VMIVRegType>(input.getType());
    if (inputType.getElementCount() != firstInputType.getElementCount() ||
        inputType.getElementType() != firstInputType.getElementType()) {
      return emitOpError("requires all channel inputs to have the same lane "
                         "count and element type");
    }
  }
  if (resultType.getElementCount() != static_cast<int64_t>(getInputs().size()) *
                                          firstInputType.getElementCount() ||
      resultType.getElementType() != firstInputType.getElementType()) {
    return emitOpError(
        "requires result lane count and element type to match merged channels");
  }
  if (isLayoutAssigned(resultType) ||
      llvm::any_of(getInputs(), [](Value input) {
        return isLayoutAssigned(cast<VMIVRegType>(input.getType()));
      })) {
    return verifyChannelMergeLayout(getOperation(), resultType, getInputs());
  }
  return success();
}

LogicalResult VMIEnsureLayoutOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount() ||
      sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires source and result to preserve VMI data shape "
                       "and element type");
  }
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
    return emitOpError("requires source and result to be layout-assigned");
  }
  return success();
}

LogicalResult VMIEnsureMaskLayoutOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount() ||
      sourceType.getGranularity() != resultType.getGranularity()) {
    return emitOpError("requires source and result to preserve VMI mask shape "
                       "and granularity");
  }
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
    return emitOpError("requires source and result to be layout-assigned");
  }
  return success();
}

LogicalResult VMIEnsureMaskGranularityOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result to preserve VMI mask lane count");
  }
  if (sourceType.isPred() || resultType.isPred()) {
    return emitOpError(
        "requires concrete source and result mask granularities");
  }
  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return emitOpError("requires either both source and result to carry "
                         "layout or neither to carry layout");
    }
  }
  return success();
}

LogicalResult VMIUnpackOp::verify() {
  return verifyPhysicalParts(getOperation(), getSource().getType(),
                             getParts().getTypes());
}

LogicalResult VMIPackOp::verify() {
  return verifyPhysicalParts(getOperation(), getResult().getType(),
                             getParts().getTypes());
}


enum class CvtDirection { FpWiden, FpNarrow, FpToSi, FpToUi, SiToFp, IntWiden, IntNarrow };

// Shared helper: validates mask-data alignment and pmode value.
// Applicable to all VMI elementwise ops that carry mask + pmode.
static LogicalResult verifyVMIPmodeMask(Operation *op, VMIMaskType maskType,
                                        VMIVRegType dataType,
                                        std::optional<StringRef> pmode) {
  if (failed(verifyMaskMatchesData(op, maskType, dataType))) {
    return failure();
  }
  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("pmode must be \"merge\" or \"zero\", got \"")
             << mode << "\"";
    }
  }
  return success();
}
// Variadic-aware variant: skips mask validation when no mask operand is
// provided (unified v-ops allow an absent mask meaning "all-true").
static LogicalResult verifyVMIVariadicPmodeMask(Operation *op,
                                                ValueRange maskParts,
                                                VMIVRegType dataType,
                                                std::optional<StringRef> pmode) {
  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("pmode must be \"merge\" or \"zero\", got \"")
             << mode << "\"";
    }
  }
  if (maskParts.empty()) {
    return success();
  }
  if (maskParts.size() != 1) {
    return op->emitOpError("expects at most one mask operand");
  }
  return verifyMaskMatchesData(op, cast<VMIMaskType>(maskParts.front().getType()),
                               dataType);
}

//===----------------------------------------------------------------------===//
// VMI vector-scalar op verifiers (vadds/vmuls/vmaxs/vmins/vshls/vshrs)
//===----------------------------------------------------------------------===//

/// Shared verifier for VMI vector-scalar elementwise ops.
static LogicalResult
verifyVMIVectorScalarOp(Operation *op, VMIVRegType srcType,
                        Type scalarType, VMIVRegType resultType,
                        VMIMaskType maskType,
                        std::optional<StringRef> pmode) {
  Type eltTy = srcType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(op, eltTy))) {
    return failure();
}
  if (!isVMIFloatLikeType(eltTy) && !isVMIIntegerLikeType(eltTy)) {
    return op->emitOpError(
        "requires floating-point-like or integer-like VMI element type");
}

  if (scalarType != eltTy) {
    return op->emitOpError(
        "requires scalar type to match vector element type, got scalar ")
           << scalarType << " vs vector element " << eltTy;
}

  if (failed(verifyAllSameVRegShapeAndLayout(op, {srcType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
}

  if (failed(verifyMaskMatchesData(op, maskType, resultType))) {
    return failure();
  }

  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("unsupported pmode '")
             << mode << "'; expected \"merge\" or \"zero\"";
    }
  }

  return success();
}

/// Shared verifier for VMI vector-scalar integer-only shift ops.
static LogicalResult
verifyVMIVectorScalarShiftOp(Operation *op, VMIVRegType srcType,
                             Type scalarType, VMIVRegType resultType,
                             VMIMaskType maskType,
                             std::optional<StringRef> pmode) {
  Type eltTy = srcType.getElementType();
  if (!isVMIIntegerLikeType(eltTy)) {
    return op->emitOpError(
        "requires integer-like VMI element type for shift");
  }
  if (!scalarType.isSignlessInteger(mlir::pto::kValue16)) {
    return op->emitOpError("requires signless i16 shift amount");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(op, {srcType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(op, maskType, resultType))) {
    return failure();
  }
  if (pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return op->emitOpError("unsupported pmode '")
             << mode << "'; expected \"merge\" or \"zero\"";
    }
  }
  return success();
}

LogicalResult VMIAddSOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), srcType.getElementType()))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(srcType.getElementType())) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(getOperation(),
      srcType, getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIMulSOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  Type elementType = srcType.getElementType();
  auto intType = dyn_cast<IntegerType>(elementType);
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  if (!supportedInteger && !isVMIF16OrF32Type(elementType)) {
    return emitOpError("requires i16, i32, f16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(getOperation(),
      srcType, getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIMaxSOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  if (!isVMII8I16I32OrF16BF16F32Type(srcType.getElementType())) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(getOperation(),
      srcType, getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIMinSOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  if (!isVMII8I16I32OrF16BF16F32Type(srcType.getElementType())) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  return verifyVMIVectorScalarOp(getOperation(),
      srcType, getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIShlSOp::verify() {
  return verifyVMIVectorScalarShiftOp(getOperation(),
      cast<VMIVRegType>(getSrc().getType()), getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

LogicalResult VMIShrSOp::verify() {
  return verifyVMIVectorScalarShiftOp(getOperation(),
      cast<VMIVRegType>(getSrc().getType()), getScalar().getType(),
      cast<VMIVRegType>(getResult().getType()),
      cast<VMIMaskType>(getMask().getType()), getPmode());
}

//===----------------------------------------------------------------------===//
// Unified (new) VMI op verifiers
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

/// Returns true if `cmpMode` is a comparison predicate supported by VCMP.
static bool isSupportedVCmpPredicate(StringRef cmpMode) {
  return cmpMode == "eq" || cmpMode == "ne" || cmpMode == "lt" ||
         cmpMode == "le" || cmpMode == "gt" || cmpMode == "ge" ||
         cmpMode == "oeq" || cmpMode == "one" || cmpMode == "olt" ||
         cmpMode == "ole" || cmpMode == "ogt" || cmpMode == "oge";
}

//===----------------------------------------------------------------------===//

static const std::set<StringRef> &validDistModes() {
  static const std::set<StringRef> modes = {"continuous", "dintlv", "brc"};
  return modes;
}

static const std::set<StringRef> &validStoreDistModes() {
  static const std::set<StringRef> modes = {"continuous", "intlv"};
  return modes;
}

static const std::set<StringRef> &validPModes() {
  static const std::set<StringRef> modes = {"zero", "merge"};
  return modes;
}

//===--- Unified (new) VMI op verifiers ---===//

LogicalResult VMIVbrcOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type valueType = getValue().getType();

  if (auto groupAttr = getGroupAttr()) {
    // Group broadcast mode
    int64_t numGroupsVal = groupAttr.getInt();
    auto vregType = dyn_cast<VMIVRegType>(valueType);
    if (!vregType) {
      return emitOpError("requires VMI vector input when num_groups is set");
    }
    if (vregType.getElementCount() != numGroupsVal) {
      return emitOpError()
             << "requires source logical lane count " << vregType.getElementCount()
             << " to match num_groups " << numGroupsVal;
    }
    if (vregType.getElementType() != resultType.getElementType()) {
      return emitOpError("requires source and result element types to match");
    }
    if (auto sourceLayout = vregType.getLayoutAttr()) {
      if (!sourceLayout.isGroupSlots() ||
          sourceLayout.getNumGroups() != numGroupsVal) {
        return emitOpError() << "requires layout-assigned source to use "
                                "#pto.vmi.layout<num_groups = "
                             << numGroupsVal << ">";
      }
    }
    if (auto resultLayout = resultType.getLayoutAttr()) {
      if (resultLayout.isGroupSlots()) {
        return emitOpError(
            "requires layout-assigned result to use a dense VMI layout");
      }
    }
    return verifyNumGroups(getOperation(), resultType, numGroupsVal);
  }

  // Scalar/1-lane broadcast mode (no num_groups)
  if (valueType == resultType.getElementType()) {
    return success();
  }
  if (auto vregType = dyn_cast<VMIVRegType>(valueType)) {
    if (vregType.getElementCount() != 1) {
      return emitOpError("requires VMI vector input to have one logical lane");
    }
    if (vregType.getElementType() != resultType.getElementType()) {
      return emitOpError("requires VMI vector input element type to match "
                         "result element type");
    }
    return success();
  }
  return emitOpError("requires scalar or VMI vector input element type to "
                     "match result element type");
}

LogicalResult VMIVciOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = resultType.getElementType();
  if (!isVMIIotaElementType(elementType)) {
    return emitOpError("requires result element type to be integer 8/16/32 "
                       "or f16/f32");
  }
  if (!isCompatibleScalarForSemanticType(elementType, getBase().getType())) {
    return emitOpError("requires base type to match result element type");
  }

  if (std::optional<StringRef> order = getOrder()) {
    if (*order != "ASC" && *order != "DESC") {
      return emitOpError("requires order to be ASC or DESC");
    }
  }
  if (auto groupAttr = getGroupAttr()) {
    int64_t numGroups = groupAttr.getInt();
    if (numGroups <= 0) {
      return emitOpError("requires group to be positive");
    }
    if (resultType.getElementCount() % numGroups != 0) {
      return emitOpError("requires group to evenly divide result logical lane "
                         "count");
    }
    if (numGroups > 1) {
      int64_t groupSize = resultType.getElementCount() / numGroups;
      FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(elementType);
      if (succeeded(lanesPerPart) && groupSize % *lanesPerPart != 0 &&
          *lanesPerPart % groupSize != 0) {
        return emitOpError("requires group_size to divide or be a multiple of "
                           "physical lanes per part (")
               << *lanesPerPart << ")";
      }
    }
  }
  return success();
}

LogicalResult VMIPsetOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  StringRef pattern = getPattern();
  if (pattern != "PAT_ALL") {
    return emitOpError("requires pattern to be \"PAT_ALL\"");
  }
  if (!resultType.isPred() && !isLayoutAssigned(resultType)) {
    return emitOpError("requires concrete mask result to carry layout");
  }
  return success();
}

LogicalResult VMIPgeOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  StringRef pattern = getPattern();
  if (!pattern.starts_with("PAT_VL")) {
    return emitOpError("requires pattern to start with \"PAT_VL\"");
  }
  int64_t activeLanes;
  if (pattern.drop_front(mlir::pto::kValue6).getAsInteger(mlir::pto::kValue10, activeLanes)) {
    return emitOpError("requires pattern \"PAT_VL<n>\" with integer n");
  }
  if (activeLanes <= 0) {
    return emitOpError("requires positive n in pattern \"PAT_VL<n>\"");
  }
  if (activeLanes > resultType.getElementCount()) {
    return emitOpError("PAT_VL active lanes ") << activeLanes
                                               << " exceeds mask element count " << resultType.getElementCount();
  }
  if (!resultType.isPred() && !isLayoutAssigned(resultType)) {
    return emitOpError("requires concrete mask result to carry layout");
  }
  return success();
}

LogicalResult VMIPltOp::verify() {
  auto resultType = cast<VMIMaskType>(getMask().getType());
  auto scalarType = dyn_cast<IntegerType>(getScalar().getType());
  if (!scalarType || scalarType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires i32 scalar input");
  }
  auto scalarOutType = dyn_cast<IntegerType>(getScalarOut().getType());
  if (!scalarOutType || scalarOutType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires i32 scalar_out result");
  }
  if (!resultType.isPred() && !isLayoutAssigned(resultType)) {
    return emitOpError("requires concrete mask result to carry layout");
  }
  return success();
}

LogicalResult VMIVaddOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), lhsType.getElementType()))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(lhsType.getElementType())) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVsubOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), lhsType.getElementType()))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(lhsType.getElementType())) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVmulOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), lhsType.getElementType()))) {
    return failure();
  }
  if (!isVMII16I32OrF16BF16F32Type(lhsType.getElementType())) {
    return emitOpError(
        "requires i16, i32, f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVdivOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), lhsType.getElementType()))) {
    return failure();
  }
  if (!isVMIF16OrF32Type(lhsType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVminOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = lhsType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), elementType))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(elementType)) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
}
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(), resultType,
                                        getPmode()))) {
    return failure();
}
  return success();
}

LogicalResult VMIVmaxOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = lhsType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), elementType))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(elementType)) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType,
                                     resultType))) {
    return failure();
}
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(), resultType,
                                        getPmode()))) {
    return failure();
}
  return success();
}

LogicalResult VMIVnegOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMII8I16I32OrF16F32Type(sourceType.getElementType())) {
    return emitOpError("requires i8, i16, i32, f16, or f32 VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {sourceType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVabsOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  Type eltTy = sourceType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), eltTy))) {
    return failure();
  }
  bool supportedInteger = isVMISignedI8I16I32Type(eltTy);
  if (!supportedInteger && !isVMIF16BF16OrF32Type(eltTy)) {
    return emitOpError(
        "requires si8, si16, si32, f16, bf16, or f32 VMI element type");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {sourceType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
}

  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                      resultType, getPmode()))) {
    return failure();
}
  return success();
}

LogicalResult VMIVsqrtOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  if (failed(verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVexpOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), sourceType.getElementType()))) {
    return failure();
  }
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  if (failed(verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVlnOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIF16OrF32Type(sourceType.getElementType())) {
    return emitOpError("requires f16 or f32 VMI element type");
  }
  if (failed(verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVreluOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = sourceType.getElementType();
  if (failed(verifySignedI32OrF16F32ElementType(getOperation(), elementType))) {
    return failure();
  }
  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {sourceType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVandOp::verify() {
  if (isa<VMIMaskType>(getLhs().getType())) {
    // Mask logic path: reject predication mask and pmode.
    if (!getMask().empty()) {
      return emitOpError("mask logic op does not support predication mask");
    }
    if (auto pmode = getPmode()) {
      return emitOpError("mask logic op does not support pmode");
    }
    auto lhsType = cast<VMIMaskType>(getLhs().getType());
    auto rhsType = cast<VMIMaskType>(getRhs().getType());
    auto resultType = cast<VMIMaskType>(getResult().getType());
    return verifyAllSameMaskShapeLayoutAndGranularity(
        getOperation(), {lhsType, rhsType, resultType});
  }
  // VReg path.
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVorOp::verify() {
  if (isa<VMIMaskType>(getLhs().getType())) {
    // Mask logic path: reject predication mask and pmode.
    if (!getMask().empty()) {
      return emitOpError("mask logic op does not support predication mask");
    }
    if (auto pmode = getPmode()) {
      return emitOpError("mask logic op does not support pmode");
    }
    auto lhsType = cast<VMIMaskType>(getLhs().getType());
    auto rhsType = cast<VMIMaskType>(getRhs().getType());
    auto resultType = cast<VMIMaskType>(getResult().getType());
    return verifyAllSameMaskShapeLayoutAndGranularity(
        getOperation(), {lhsType, rhsType, resultType});
  }
  // VReg path.
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVxorOp::verify() {
  if (isa<VMIMaskType>(getLhs().getType())) {
    // Mask logic path: reject predication mask and pmode.
    if (!getMask().empty()) {
      return emitOpError("mask logic op does not support predication mask");
    }
    if (auto pmode = getPmode()) {
      return emitOpError("mask logic op does not support pmode");
    }
    auto lhsType = cast<VMIMaskType>(getLhs().getType());
    auto rhsType = cast<VMIMaskType>(getRhs().getType());
    auto resultType = cast<VMIMaskType>(getResult().getType());
    return verifyAllSameMaskShapeLayoutAndGranularity(
        getOperation(), {lhsType, rhsType, resultType});
  }
  // VReg path.
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVshlOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVshrOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType) {
    return emitOpError("requires integer VMI element type");
  }
  if (failed(verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVnotOp::verify() {
  if (isa<VMIMaskType>(getSource().getType())) {
    // Mask logic path: reject predication mask and pmode.
    if (!getMask().empty()) {
      return emitOpError("mask logic op does not support predication mask");
    }
    if (auto pmode = getPmode()) {
      return emitOpError("mask logic op does not support pmode");
    }
    auto sourceType = cast<VMIMaskType>(getSource().getType());
    auto resultType = cast<VMIMaskType>(getResult().getType());
    return verifyAllSameMaskShapeLayoutAndGranularity(
        getOperation(), {sourceType, resultType});
  }
  // VReg path.
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType())) {
    return emitOpError("requires integer-like VMI element type");
  }
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {sourceType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIvSelOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto trueType = cast<VMIVRegType>(getTrueValue().getType());
  auto falseType = cast<VMIVRegType>(getFalseValue().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {trueType, falseType, resultType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }
  if (auto pmode = getPmode(); pmode.has_value()) {
    StringRef mode = pmode.value();
    if (mode != "merge" && mode != "zero") {
      return emitOpError("pmode must be \"merge\" or \"zero\", got \"")
             << mode << "\"";
    }
  }
  return success();
}

static LogicalResult verifyVCReductionElementAndMask(Operation *op,
                                                     VMIVRegType sourceType,
                                                     VMIMaskType maskType,
                                                     bool &isFloat) {
  Type elemTy = sourceType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(op, elemTy))) {
    return failure();
  }
  isFloat = isVMIFloatLikeType(elemTy);
  bool isInt = isVMIIntegerLikeType(elemTy);
  if (!isFloat && !isInt) {
    return op->emitOpError("requires integer-like or floating-point-like VMI "
                           "source element type");
  }
  return verifyMaskMatchesData(op, maskType, sourceType);
}

LogicalResult VMIvcaddOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  bool isFloat = false;
  if (failed(verifyVCReductionElementAndMask(getOperation(), sourceType,
                                             maskType, isFloat))) {
    return failure();
  }
  // Floating-point vcadd MUST carry reassoc
  if (isFloat && !getReassoc()) {
    return emitOpError("floating add-reduction requires reassoc attr");
  }
  return verifyReductionGroupAndPmode(getOperation(), sourceType, resultType,
                                      getGroupAttr(), getPmode());
}

LogicalResult VMIvcmaxOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  bool isFloat = false;
  if (failed(verifyVCReductionElementAndMask(getOperation(), sourceType,
                                             maskType, isFloat))) {
    return failure();
  }
  return verifyReductionGroupAndPmode(getOperation(), sourceType, resultType,
                                      getGroupAttr(), getPmode());
}

LogicalResult VMIvcminOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  bool isFloat = false;
  if (failed(verifyVCReductionElementAndMask(getOperation(), sourceType,
                                             maskType, isFloat))) {
    return failure();
  }
  return verifyReductionGroupAndPmode(getOperation(), sourceType, resultType,
                                      getGroupAttr(), getPmode());
}

LogicalResult VMIVgatherOp::verify() {
  auto offsetsType = cast<VMIVRegType>(getOffsets().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }

  if (failed(verifyGatherMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }

  auto indexElementType =
      dyn_cast<IntegerType>(offsetsType.getElementType());
  if (failed(verifyVMIOffsetIndexType(getOperation(), indexElementType,
                                      "offsets"))) {
    return failure();
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {offsetsType, resultType},
          /*requireSameElement=*/false))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }

  // 16-bit offsets address the pto.vgather2 / b16 mask path.  It supports
  // i16/ui16/f16/bf16 same-width gather and i8/ui8 -> i16/ui16 promotion.
  if (indexElementType.getWidth() == mlir::pto::kValue16 &&
      !isSupported16BitGatherResult(
          getMemoryElementType(getSource().getType()),
          resultType.getElementType())) {
    return emitOpError(
        "requires i16/ui16/f16/bf16 result element type when using ui16 "
        "offsets, or i8/ui8 -> i16/ui16 integer promotion");
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

void VMIVgatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIVgatherbOp::verify() {
  auto offsetsType = cast<VMIVRegType>(getOffsets().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyUBBackedMemory(getOperation(), getSource().getType(),
                                  "source"))) {
    return failure();
  }

  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source"))) {
    return failure();
  }

  auto indexElementType =
      dyn_cast<IntegerType>(offsetsType.getElementType());
  if (failed(verifyVMIOffsetIndexType(getOperation(), indexElementType,
                                      "offsets"))) {
    return failure();
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

void VMIVgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIVscatterOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto offsetsType = cast<VMIVRegType>(getOffsets().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());

  if (failed(verifyUBBackedMemory(getOperation(),
                                  getDestination().getType(), "destination"))) {
    return failure();
  }

  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination"))) {
    return failure();
  }

  auto indexElementType =
      dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!indexElementType || indexElementType.isSigned() ||
      (indexElementType.getWidth() != mlir::pto::kValue32 &&
       indexElementType.getWidth() != mlir::pto::kValue16)) {
    return emitOpError(
        "requires signless or unsigned 16-bit or 32-bit integer offsets");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {valueType, offsetsType},
                                             /*requireSameElement=*/false))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, valueType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

void VMIVscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIVexpdifOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto maxType = cast<VMIVRegType>(getMax().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (!isVMIF16OrF32Type(xType.getElementType())) {
    return emitOpError("requires x element type to be f16 or f32");
  }

  if (maxType.getElementType() != xType.getElementType()) {
    return emitOpError("requires x and max to have the same element type");
  }

  auto resultElemType = dyn_cast<FloatType>(resultType.getElementType());
  if (!resultElemType || resultElemType.getWidth() != mlir::pto::kValue32) {
    return emitOpError("requires result element type to be f32");
  }

  if (xType.getElementCount() != maxType.getElementCount() ||
      xType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires x, max, and result logical lane counts to match");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, xType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

LogicalResult VMIVaxpyOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), xType.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(xType.getElementType())) {
    return emitOpError("requires vector element type to be f16 or f32");
  }

  if (xType != accType || accType != resultType) {
    return emitOpError(
        "requires x, acc, and result to have identical VMI vreg types");
  }

  auto alphaType = cast<FloatType>(getAlpha().getType());
  if (alphaType != xType.getElementType()) {
    return emitOpError("requires alpha scalar type to match vector element type");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

LogicalResult VMIVlreluOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), xType.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(xType.getElementType())) {
    return emitOpError("requires vector element type to be f16 or f32");
  }

  if (xType != resultType) {
    return emitOpError("requires x and result to have identical VMI vreg types");
  }

  auto slopeType = cast<FloatType>(getSlope().getType());
  if (slopeType != xType.getElementType()) {
    return emitOpError(
        "requires slope scalar type to match vector element type");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

LogicalResult VMIVpreluOp::verify() {
  auto xType = cast<VMIVRegType>(getX().getType());
  auto alphaType = cast<VMIVRegType>(getAlpha().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  if (failed(verifyBF16x2ComputeElementType(
          getOperation(), xType.getElementType()))) {
    return failure();
  }

  if (!isVMIFloatLikeType(xType.getElementType())) {
    return emitOpError("requires vector element type to be f16 or f32");
  }

  if (xType != alphaType || alphaType != resultType) {
    return emitOpError(
        "requires x, alpha, and result to have identical VMI vreg types");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, resultType))) {
    return failure();
  }

  if (auto pmode = getPmode()) {
    if (pmode.value() != "merge" && pmode.value() != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

template <typename CarryOp>
static LogicalResult verifyVMIAddCarryOp(CarryOp op, VMIMaskType carryInType,
                                         bool hasCarryIn) {
  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  auto rhsType = cast<VMIVRegType>(op.getRhs().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto carryType = cast<VMIMaskType>(op.getCarry().getType());

  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType || integerType.getWidth() != mlir::pto::kValue32) {
    return op.emitOpError("requires 32-bit integer vector element types");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(
          op.getOperation(), {lhsType, rhsType, resultType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyMaskMatchesData(op.getOperation(), maskType, lhsType)) ||
      failed(verifyMaskMatchesData(op.getOperation(), carryType, lhsType))) {
    return failure();
  }
  if (hasCarryIn &&
      failed(verifyMaskMatchesData(op.getOperation(), carryInType, lhsType))) {
    return failure();
  }

  SmallVector<VMIMaskType> masks{maskType, carryType};
  if (hasCarryIn) {
    masks.push_back(carryInType);
  }
  return verifyAllSameMaskShapeLayoutAndGranularity(op.getOperation(), masks);
}

LogicalResult VMIVaddcOp::verify() {
  return verifyVMIAddCarryOp(*this, VMIMaskType{}, /*hasCarryIn=*/false);
}

LogicalResult VMIVaddcsOp::verify() {
  return verifyVMIAddCarryOp(*this, cast<VMIMaskType>(getCarryIn().getType()),
                             /*hasCarryIn=*/true);
}

LogicalResult VMIVmullOp::verify() {
  auto aType = cast<VMIVRegType>(getA().getType());
  auto bType = cast<VMIVRegType>(getB().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());

  auto isLegalElementType = [](Type type) {
    auto integerType = dyn_cast<IntegerType>(type);
    return integerType && integerType.getWidth() == mlir::pto::kValue32 &&
           (integerType.isSignless() || integerType.isUnsigned());
  };
  if (!isLegalElementType(aType.getElementType()) ||
      !isLegalElementType(bType.getElementType()) ||
      !isLegalElementType(lowType.getElementType()) ||
      !isLegalElementType(highType.getElementType())) {
    return emitOpError(
        "requires a, b, low, and high element types to be exactly i32 or ui32");
  }

  if (aType != bType || aType != lowType || aType != highType) {
    return emitOpError(
        "requires a, b, low, and high to have identical VMI vreg types");
  }

  int64_t lanes = aType.getElementCount();
  if (lanes != mlir::pto::kValue64 && lanes != mlir::pto::kValue128 && lanes != mlir::pto::kValue256) {
    return emitOpError("requires logical lane count to be 64, 128, or 256");
  }

  if (failed(verifyMaskMatchesData(getOperation(), maskType, aType))) {
    return failure();
  }

  if (auto pmode = getPmode(); pmode && pmode.value() != "zero") {
    return emitOpError("pmode must be 'zero' when specified");
  }
  return success();
}

LogicalResult VMIVmulaOp::verify() {
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());

  Type eltTy = accType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), eltTy))) {
    return failure();
}
  if (!isVMIFloatLikeType(eltTy) && !isVMIIntegerLikeType(eltTy)) {
    return emitOpError(
        "requires floating-point-like or integer-like VMI element type");
}

  if (accType != lhsType || lhsType != rhsType || rhsType != resultType) {
    return emitOpError(
        "requires acc, lhs, rhs, and result to have identical VMI vreg types");
}

  if (failed(verifyVMIVariadicPmodeMask(getOperation(), getMask(),
                                        resultType, getPmode()))) {
    return failure();
}
  return success();
}

static LogicalResult classifyCvtDirection(
    VMICvtOp op, Type srcElem, Type dstElem, unsigned srcBits, unsigned dstBits,
    bool srcFp, bool dstFp, bool srcInt, bool dstInt,
    const std::optional<VMIFpToFpContract> &fpContract, CvtDirection &dir) {
  if (srcFp && dstFp) {
    if (dstBits > srcBits) {
      dir = CvtDirection::FpWiden;
    } else if (dstBits < srcBits) {
      dir = CvtDirection::FpNarrow;
    } else {
      // Same-width fp→fp (e.g. bf16 → f16): only allowed for VMI fp-to-fp
      // contract pairs, routed through FpNarrow (1:1 TruncF).
      if (!fpContract) {
        return op.emitOpError(
            "same-width fp-to-fp conversion is not supported for this type "
            "pair; see lookupVMIFpToFpContract");
      }
      dir = CvtDirection::FpNarrow;
    }
  } else if (srcFp && dstInt) {
    if (isVMIUnsignedOrSignlessIntegerType(dstElem)) {
      dir = CvtDirection::FpToUi;
    } else {
      dir = CvtDirection::FpToSi;
    }
  } else if (srcInt && dstFp) {
    if (!isVMISignedIntegerType(srcElem)) {
      return op.emitOpError(
          "int-to-fp conversion requires explicitly signed integer source "
          "element type");
    }
    dir = CvtDirection::SiToFp;
  } else if (srcInt && dstInt) {
    if (dstBits > srcBits) {
      dir = CvtDirection::IntWiden;
    } else if (dstBits < srcBits) {
      dir = CvtDirection::IntNarrow;
    } else {
      return op.emitOpError(
          "int-to-int conversion must change element bit-width");
    }
  } else {
    return op.emitOpError(
        "unsupported element type combination for vcvt");
  }
  return success();
}

// Validate the rounding attribute for the given conversion direction.
static LogicalResult verifyCvtRounding(VMICvtOp op, CvtDirection dir,
                                       Type srcElem, Type dstElem,
                                       const std::optional<VMIFpToFpContract> &fpContract) {
  auto roundingAttr = op->getAttrOfType<StringAttr>("rounding");
  if (!roundingAttr) {
    return success();
  }
  if (dir != CvtDirection::FpNarrow && dir != CvtDirection::FpToSi &&
      dir != CvtDirection::FpToUi) {
    return op.emitOpError("'rounding' attribute is only valid for floating-point "
                          "narrowing or floating-point-to-integer conversions");
  }
  StringRef rnd = roundingAttr.getValue();
  if (rnd.size() != 1) {
    return op.emitOpError("rounding must be a single-character mode token");
  }
  if (dir == CvtDirection::FpNarrow) {
    StringRef allowedRndModes =
        fpContract && !fpContract->allowedRndModes.empty()
            ? fpContract->allowedRndModes
            : StringRef("RAHZ");
    if (!allowedRndModes.contains(rnd)) {
      if (fpContract && !fpContract->allowedRndModes.empty()) {
        return op.emitOpError(
            "rounding is not valid for this fp-to-fp conversion type pair");
      }
      return op.emitOpError("rounding must be 'R' (nearest-even), "
                            "'A' (away-from-zero), 'H' (half-up), "
                            "or 'Z' (toward-zero)");
    }
  } else if (rnd != "R" && rnd != "A" && rnd != "F" && rnd != "C" &&
             rnd != "Z") {
    return op.emitOpError("rounding must be 'R', 'A', 'F', 'C', or 'Z' for "
                          "floating-point-to-integer conversions");
  }
  return success();
}

// Validate the saturate attribute for FpToSi conversions.
static LogicalResult verifyCvtSaturateFpToSi(VMICvtOp op, Type srcElem,
                                             Type dstElem,
                                             StringAttr satAttr) {
  auto contract = lookupVMIFpToSiContract(srcElem, dstElem);
  if (!contract) {
    return op.emitOpError("unsupported fp-to-si conversion element type pair");
  }
  if (contract->requiresSat) {
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for this "
                            "fp-to-si conversion; write 'SAT' or 'NOSAT'");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate must be 'SAT' or 'NOSAT'");
    }
  } else if (satAttr) {
    return op.emitOpError("'saturate' attribute is not valid for this "
                          "fp-to-si conversion (no overflow possible)");
  }
  return success();
}

// Validate the saturate attribute for FpToUi conversions.
static LogicalResult verifyCvtSaturateFpToUi(VMICvtOp op, Type srcElem,
                                             Type dstElem,
                                             StringAttr satAttr) {
  auto contract = lookupVMIFpToUIContract(srcElem, dstElem);
  if (!contract) {
    return op.emitOpError("unsupported fp-to-ui conversion element type pair");
  }
  if (contract->requiresSat) {
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for this "
                            "fp-to-ui conversion; write 'SAT' or 'NOSAT'");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate must be 'SAT' or 'NOSAT'");
    }
  } else if (satAttr) {
    return op.emitOpError("'saturate' attribute is not valid for this "
                          "fp-to-ui conversion (no overflow possible)");
  }
  return success();
}

// Validate the saturate attribute for narrow (FpNarrow / IntNarrow) conversions.
static LogicalResult verifyCvtSaturateNarrow(
    VMICvtOp op, CvtDirection dir, unsigned srcBits, unsigned dstBits,
    Type srcElem, Type dstElem, StringAttr satAttr,
    const std::optional<VMIFpToFpContract> &fpContract) {
  // Fp-narrow: default to requiring a saturate attribute, but consult the
  // fp-to-fp contract when one exists (e.g. bf16x2->f4x2 narrows with
  // requiresSat=false and must NOT carry saturate).
  bool needSat = (dir == CvtDirection::IntNarrow);
  if (dir == CvtDirection::FpNarrow) {
    needSat = !fpContract || fpContract->requiresSat;
  }
  if (needSat) {
    if (!satAttr) {
      return op.emitOpError("'saturate' attribute is required for fp-narrow / "
                            "int-narrow conversions; write 'SAT' or 'NOSAT'");
    }
    StringRef satVal = satAttr.getValue();
    if (satVal != "SAT" && satVal != "NOSAT") {
      return op.emitOpError("saturate must be 'SAT' or 'NOSAT'");
    }
    // si32 -> si8 IntNarrow has no native hardware form.  Lowering aliases
    // it through ui32 -> ui8 (bit-pattern equal ONLY under NOSAT).  Reject
    // SAT here because ui32 -> ui8 SAT clamps to [0, 255], which does NOT
    // match the expected si32 -> si8 SAT clamp to [-128, 127].
    if (dir == CvtDirection::IntNarrow && satVal == "SAT" &&
        srcBits == mlir::pto::kValue32 && dstBits == mlir::pto::kValue8 &&
        isa<IntegerType>(srcElem) &&
        cast<IntegerType>(srcElem).isSigned() &&
        isa<IntegerType>(dstElem) &&
        cast<IntegerType>(dstElem).isSigned()) {
      return op.emitOpError("si32 -> si8 int-narrow does not support "
                            "saturate=\"SAT\" (no native hardware form; "
                            "only saturate=\"NOSAT\" is allowed)");
    }
  } else if (satAttr && dir == CvtDirection::FpNarrow) {
    return op.emitOpError("'saturate' attribute is not valid for this fp-to-fp "
                          "narrow conversion (no saturation)");
  } else if (satAttr) {
    return op.emitOpError("'saturate' attribute is only valid for fp-narrow / "
                          "int-narrow conversions");
  }
  return success();
}

// Dispatch saturate validation to the correct sub-verifier.
static LogicalResult verifyCvtSaturate(VMICvtOp op, CvtDirection dir,
                                       unsigned srcBits, unsigned dstBits,
                                       Type srcElem, Type dstElem,
                                       StringAttr satAttr,
                                       const std::optional<VMIFpToFpContract> &fpContract) {
  if (dir == CvtDirection::FpToSi) {
    return verifyCvtSaturateFpToSi(op, srcElem, dstElem, satAttr);
  }
  if (dir == CvtDirection::FpToUi) {
    return verifyCvtSaturateFpToUi(op, srcElem, dstElem, satAttr);
  }
  return verifyCvtSaturateNarrow(op, dir, srcBits, dstBits, srcElem, dstElem,
                                 satAttr, fpContract);
}

LogicalResult VMICvtOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires source and result logical lane counts to match");
  }
  Type srcElem = sourceType.getElementType();
  Type dstElem = resultType.getElementType();
unsigned srcBits = getVMIElementBitWidth(srcElem), dstBits = getVMIElementBitWidth(dstElem);
bool srcFp = isVMIFloatLikeType(srcElem), dstFp = isVMIFloatLikeType(dstElem);
bool srcInt = isVMIIntegerLikeType(srcElem), dstInt = isVMIIntegerLikeType(dstElem);
  auto fpContract = srcFp && dstFp ? lookupVMIFpToFpContract(srcElem, dstElem)
                                   : std::nullopt;
  if (involvesBF16x2(srcElem, dstElem) && !fpContract) {
    return emitOpError("unsupported conversion involving bf16x2 element type");
  }
  if (srcFp && dstFp && involvesVMIPackedFloatCarrier(srcElem, dstElem) &&
      !fpContract) {
    return emitOpError(
        "unsupported packed fp-to-fp conversion element type pair");
  }
  CvtDirection dir = CvtDirection::FpNarrow;
  if (failed(classifyCvtDirection(*this, srcElem, dstElem, srcBits, dstBits,
                                  srcFp, dstFp, srcInt, dstInt, fpContract,
                                  dir))) {
    return failure();
  }
  if (failed(verifyCvtRounding(*this, dir, srcElem, dstElem, fpContract))) {
    return failure();
  }
  auto satAttr = (*this)->getAttrOfType<StringAttr>("saturate");
  if (failed(verifyCvtSaturate(*this, dir, srcBits, dstBits, srcElem, dstElem,
                               satAttr, fpContract))) {
    return failure();
  }
  if (auto pmodeAttr = (*this)->getAttrOfType<StringAttr>("pmode")) {
    StringRef pmode = pmodeAttr.getValue();
    if (pmode != "merge" && pmode != "zero") {
      return emitOpError("pmode must be 'merge' or 'zero'");
    }
  }
  return success();
}

LogicalResult VMIVinterpretCastOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned resultBits =
      pto::getPTOStorageElemBitWidth(resultType.getElementType());
  if (sourceBits == 0 || resultBits == 0) {
    return emitOpError(
        "requires integer or floating-point source and result element types");
}
  if (sourceType.getElementCount() * static_cast<int64_t>(sourceBits) !=
      resultType.getElementCount() * static_cast<int64_t>(resultBits)) {
    return emitOpError(
        "requires source and result to carry the same total number of bits");
}

  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType)) {
      return emitOpError(
          "requires either both source and result to carry layout or neither "
          "to carry layout");
}
    if (sourceType.getLayout() != resultType.getLayout()) {
      return emitOpError("requires source and result layouts to match");
    }
  }

  return success();
}

// Parse pre-bracket operands: value(s) + optional [offset].
static ParseResult parseVStorePreBracket(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &operand,
    OpAsmParser::UnresolvedOperand &offsetOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preBracketOperands) {
  if (parser.parseOperand(operand)) {
    return failure();
  }
  preBracketOperands.push_back(operand);
  bool consumedLSquare = false;
  while (!consumedLSquare) {
    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseOperand(offsetOperand) || parser.parseRSquare()) {
        return failure();
      }
      consumedLSquare = true;
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseOperand(offsetOperand) || parser.parseRSquare()) {
        return failure();
      }
      consumedLSquare = true;
      break;
    }
    if (parser.parseOperand(operand)) {
      return failure();
    }
    preBracketOperands.push_back(operand);
  }
  return success();
}

// Disambiguate post-bracket operands into stride/block/mask.
static void disambiguateVStorePostBracket(
    bool hasGroup, size_t numPostOps, size_t nValues, size_t nTypes,
    bool &hasStride, bool &hasBlock, bool &hasMask,
    int &strideIdx, int &blockIdx, int &maskIdx) {
  hasStride = false;
  hasBlock = false;
  hasMask = false;
  strideIdx = -1;
  blockIdx = -1;
  maskIdx = -1;
  if (hasGroup) {
    // Group mode: post-bracket ops are stride[, mask]
    if (numPostOps >= 1) {
      hasStride = true;
      strideIdx = 0;
    }
    if (numPostOps >= mlir::pto::kValue2) {
      hasMask = true;
      maskIdx = 1;
    }
  } else if (numPostOps == mlir::pto::kValue2) {
    // Block-stride mode with a mask: block_stride, mask.
    hasBlock = true;
    hasMask = true;
    blockIdx = 0;
    maskIdx = 1;
  } else if (numPostOps == 1) {
    // The type list includes mask types, but never block-stride types.
    if (nTypes == nValues + mlir::pto::kValue2) {
      hasMask = true;
      maskIdx = 0;
    } else {
      hasBlock = true;
      blockIdx = 0;
    }
  }
}

// Resolve all parsed operands against their types.
static ParseResult resolveVStoreOperands(
    OpAsmParser &parser, OperationState &result,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preBracketOperands,
    OpAsmParser::UnresolvedOperand offsetOperand,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &postBracketOps,
    SmallVectorImpl<Type> &types, size_t nValues, bool hasStride,
    bool hasBlock, bool hasMask, int strideIdx, int blockIdx,
    int maskIdx) {
  size_t expectedTypes = nValues + 1 + (hasMask ? 1 : 0);
  size_t nTypes = types.size();
  if (nTypes != expectedTypes) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected " << expectedTypes << " types (" << nValues
           << " value(s), 1 destination" << (hasMask ? ", 1 mask" : "")
           << "), got " << nTypes;
  }
  for (size_t i = 0; i < nValues; ++i) {
    if (parser.resolveOperand(preBracketOperands[i], types[i], result.operands)) {
      return failure();
    }
  }
  Type destType = types[nValues];
  if (parser.resolveOperand(preBracketOperands[nValues], destType,
                            result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(offsetOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasStride &&
      parser.resolveOperand(postBracketOps[strideIdx],
                            parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasBlock &&
      parser.resolveOperand(postBracketOps[blockIdx],
                            parser.getBuilder().getIntegerType(mlir::pto::kValue16),
                            result.operands)) {
    return failure();
  }
  if (hasMask) {
    Type maskType = types.back();
    if (parser.resolveOperand(postBracketOps[maskIdx], maskType,
                              result.operands)) {
      return failure();
    }
  }
  return success();
}

ParseResult VMIvStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue4> preBracketOperands;
  OpAsmParser::UnresolvedOperand operand, offsetOperand;
  SmallVector<OpAsmParser::UnresolvedOperand, mlir::pto::kValue3> postBracketOps;
  if (failed(parseVStorePreBracket(parser, operand, offsetOperand,
                                   preBracketOperands))) {
    return failure();
  }
  if (preBracketOperands.empty()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least one value and one destination");
  }
  // Optional post-bracket operands: stride, block_stride, and/or mask.
  // Up to 2, disambiguated after parsing attrs and the type list.
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand postOp;
    if (parser.parseOperand(postOp)) {
      return failure();
    }
    postBracketOps.push_back(postOp);
    if (postBracketOps.size() >= mlir::pto::kValue3) {
      break;
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  SmallVector<Type, mlir::pto::kValue6> types;
  if (parser.parseColon() || parser.parseTypeList(types)) {
    return failure();
  }
  bool hasGroup = result.attributes.get("group") != nullptr;
  bool hasStride, hasBlock, hasMask;
  int strideIdx, blockIdx, maskIdx;
  disambiguateVStorePostBracket(hasGroup, postBracketOps.size(),
                                preBracketOperands.size() - 1, types.size(),
                                hasStride, hasBlock, hasMask,
                                strideIdx, blockIdx, maskIdx);
  size_t nValues = preBracketOperands.size() - 1;
  if (failed(resolveVStoreOperands(parser, result, preBracketOperands,
                                   offsetOperand, postBracketOps, types,
                                   nValues, hasStride, hasBlock, hasMask,
                                   strideIdx, blockIdx, maskIdx))) {
    return failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(nValues), 1, 1,
                           hasStride ? 1 : 0, hasBlock ? 1 : 0,
                           hasMask ? 1 : 0}));
  return success();
}

void VMIvStoreOp::print(OpAsmPrinter &p) {
  for (auto val : getValues()) {
    p << ' ' << val << ", ";
  }
  p << getDestination() << '[';
  p.printOperand(getOffset());
  p << ']';
  if (getStride()) {
    p << ", ";
    p.printOperand(getStride());
  }
  if (getBlockStride()) {
    p << ", ";
    p.printOperand(getBlockStride());
  }
  if (!getMask().empty()) {
    p << ", ";
    p.printOperand(getMask()[0]);
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  p << " : ";
  for (auto val : getValues()) {
    p << val.getType() << ", ";
  }
  p << getDestination().getType();
  if (!getMask().empty()) {
    p << ", " << getMask()[0].getType();
  }
}

static LogicalResult verifyBlockStrideExclusions(
    Operation *op, bool hasBlock, std::optional<StringRef> distMode,
    bool hasGroup) {
  if (hasBlock) {
    if (distMode) {
      return op->emitOpError(
          "block_stride and dist_mode are mutually exclusive");
    }
    if (hasGroup) {
      return op->emitOpError("block_stride and group are mutually exclusive");
    }
  }
  return success();
}

static LogicalResult verifyVStoreGroupAndBlockModes(
    Operation *op, bool hasGroup, std::optional<StringRef> distMode,
    bool hasStride, bool hasBlock, bool hasMask, int64_t numGroups,
    size_t nValues, ValueRange values) {
  if (hasGroup && distMode) {
    return op->emitOpError("group and dist_mode are mutually exclusive");
  }
  if (hasGroup && !hasStride) {
    return op->emitOpError("group requires a stride operand");
  }
  if (!hasGroup && hasStride) {
    return op->emitOpError("stride operand is only valid with group");
  }
  if (hasGroup && hasMask) {
    return op->emitOpError("group mode does not support mask operand");
  }
  if (hasGroup) {
    if (numGroups <= 0) {
      return op->emitOpError("group must be positive, got ") << numGroups;
    }
    if (nValues != 1) {
      return op->emitOpError("group mode requires exactly 1 value");
    }
    auto valueType = cast<VMIVRegType>(values[0].getType());
    if (failed(verifyNumGroups(op, valueType, numGroups))) {
      return failure();
    }
  }
  if (failed(verifyBlockStrideExclusions(op, hasBlock, distMode, hasGroup))) {
    return failure();
  }
  if (hasBlock && nValues != 1) {
    return op->emitOpError("block-stride mode requires exactly 1 value");
  }
  return success();
}

static LogicalResult verifyVStoreDistModeAndPmode(
    Operation *op, std::optional<StringRef> distMode, size_t nValues,
    size_t maskCount, std::optional<StringRef> pmode) {
  if (distMode && !validStoreDistModes().count(*distMode)) {
    return op->emitOpError("invalid dist-mode: \"") << *distMode << "\"";
  }
  bool isIntlv = distMode && *distMode == "intlv";
  if (nValues < 1) {
    return op->emitOpError("requires at least 1 value");
  }
  if (isIntlv && nValues != mlir::pto::kValue2) {
    return op->emitOpError("dist-mode \"intlv\" requires exactly 2 values");
  }
  if (!isIntlv && nValues != 1) {
    return op->emitOpError("requires exactly 1 value for dist-mode \"")
           << (distMode ? *distMode : "continuous") << "\"";
  }
  if (maskCount > 1) {
    return op->emitOpError("at most one mask allowed");
  }
  if (pmode && !validPModes().count(*pmode)) {
    return op->emitOpError("invalid pmode: \"") << *pmode << "\"";
  }
  if (pmode && *pmode != "zero") {
    return op->emitOpError("pmode \"merge\" is not supported for stores: the "
                           "legacy store lowering is mask-governed only and "
                           "cannot retain prior destination contents on inactive "
                           "lanes; omit pmode (defaults to \"zero\")");
  }
  return success();
}

static LogicalResult verifyVStoreValueMaskTypes(Operation *op, bool hasGroup,
                                                ValueRange values,
                                                Value destination,
                                                ValueRange mask) {
  auto valueType = cast<VMIVRegType>(values[0].getType());
  bool isPackedGroupStore =
      hasGroup &&
      isPackedByteGroupStore(destination.getType(), valueType);
  if (!isPackedGroupStore &&
      failed(verifyMemoryElementMatches(op, destination.getType(),
                                        valueType, "destination"))) {
    return failure();
  }
  if (values.size() == mlir::pto::kValue2) {
    auto loType = cast<VMIVRegType>(values[0].getType());
    auto hiType = cast<VMIVRegType>(values[1].getType());
    if (failed(verifyAllSameVRegShapeAndLayout(
            op, {loType, hiType}, /*requireSameElement=*/true))) {
      return failure();
    }
    if (failed(verifyContiguousIfLayoutAssigned(op, loType,
                                                "low input")) ||
        failed(verifyContiguousIfLayoutAssigned(op, hiType,
                                                "high input"))) {
      return failure();
    }
  }
  if (!mask.empty()) {
    auto maskType = cast<VMIMaskType>(mask[0].getType());
    if (failed(verifyMaskMatchesData(op, maskType, valueType))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMIvStoreOp::verify() {
  bool hasGroup = static_cast<bool>(getGroup());
  auto distMode = getDistMode();
  bool hasStride = static_cast<bool>(getStride());
  bool hasBlock = static_cast<bool>(getBlockStride());
  bool hasMask = !getMask().empty();
  size_t nValues = getValues().size();
  if (failed(verifyVStoreGroupAndBlockModes(
          getOperation(), hasGroup, distMode, hasStride, hasBlock, hasMask,
          hasGroup ? getGroupAttr().getInt() : 0, nValues, getValues()))) {
    return failure();
  }
  if (failed(verifyVStoreDistModeAndPmode(getOperation(), distMode, nValues,
                                         getMask().size(), getPmode()))) {
    return failure();
  }
  return verifyVStoreValueMaskTypes(getOperation(), hasGroup, getValues(),
                                   getDestination(), getMask());
}

void VMIvStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIVsstbOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination")) ||
      failed(verifyUBBackedMemory(getOperation(), getDestination().getType(),
                                  "destination"))) {
    return failure();
  }
  if (auto pmode = getPmode(); pmode && !validPModes().count(*pmode)) {
    return emitOpError("invalid pmode: \"") << *pmode << "\"";
  }
  if (auto pmode = getPmode(); pmode && *pmode != "zero") {
    return emitOpError("pmode \"merge\" is not supported for stores: the "
                       "legacy store lowering is mask-governed only and "
                       "cannot retain prior destination contents on inactive "
                       "blocks; omit pmode (defaults to \"zero\")");
  }
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIVsstbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIVselrOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto indexType = cast<VMIVRegType>(getIndex().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires result element type to match source element type");
  }

  if (indexType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires index lane count to match result lane count");
  }

  if (!isa<IntegerType>(indexType.getElementType())) {
    return emitOpError("requires index element type to be integer");
  }

  unsigned sourceBits =
      pto::getPTOStorageElemBitWidth(sourceType.getElementType());
  unsigned indexBits =
      pto::getPTOStorageElemBitWidth(indexType.getElementType());
  if (sourceBits != mlir::pto::kValue8 && sourceBits != mlir::pto::kValue16 && sourceBits != mlir::pto::kValue32) {
    return emitOpError(
        "requires source/result element storage width to be 8, 16, or 32 bits");
  }
  if (indexBits != sourceBits) {
    return emitOpError(
        "requires index element storage width to match source element storage width");
  }

  bool sourceHasLayout = isLayoutAssigned(sourceType);
  bool indexHasLayout = isLayoutAssigned(indexType);
  bool resultHasLayout = isLayoutAssigned(resultType);
  if (sourceHasLayout != resultHasLayout) {
    return emitOpError("requires source and result to both carry layout or "
                       "neither carry layout");
  }
  if (indexHasLayout && !sourceHasLayout) {
    return emitOpError(
        "requires index to carry layout only when source does");
  }

  return success();
}

LogicalResult VMIVintlvOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayoutPresence(
          getOperation(), {lhsType, rhsType, lowType, highType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIPmodeMask(getOperation(),
                                cast<VMIMaskType>(getMask().getType()),
                                lhsType, getPmode()))) {
    return failure();
  }
  return success();
}

LogicalResult VMIVdintlvOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto lowType = cast<VMIVRegType>(getLow().getType());
  auto highType = cast<VMIVRegType>(getHigh().getType());
  if (failed(verifyAllSameVRegShapeAndLayoutPresence(
          getOperation(), {lhsType, rhsType, lowType, highType},
          /*requireSameElement=*/true))) {
    return failure();
  }
  if (failed(verifyVMIPmodeMask(getOperation(),
                                cast<VMIMaskType>(getMask().getType()),
                                lhsType, getPmode()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// VMIVabsOp verifier (unified fp/int abs, replaces absf/absi)
//===----------------------------------------------------------------------===//
// VMIVcmpOp / VMIVcmpsOp verifiers
LogicalResult VMIVcmpOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto seedType = cast<VMIMaskType>(getSeed().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());

  Type eltTy = lhsType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), eltTy))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(eltTy)) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  bool usesOrderedFloatPredicate = getCmp().starts_with("o");
  bool invalidIntegerPredicate =
      isVMIIntegerLikeType(eltTy) && usesOrderedFloatPredicate;
  if (invalidIntegerPredicate) {
    return emitOpError("requires integer compare predicate eq/ne/lt/le/gt/ge; "
                       "signedness is selected by the integer element type");
  }

  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true))) {
    return failure();
  }

  // Validate cmp predicate.
  if (!isSupportedVCmpPredicate(getCmp())) {
    return emitOpError("unsupported compare predicate '")
           << getCmp() << "'; expected eq/ne/lt/le/gt/ge, "
           << "or oeq/one/olt/ole/ogt/oge";
  }

  // Validate pmode.
  if (auto pmode = getPmode()) {
    bool unsupportedPmode =
        pmode.value() != "zeroing" && pmode.value() != "merge";
    if (unsupportedPmode) {
      return emitOpError("unsupported pmode '")
             << pmode.value() << "'; expected \"zeroing\" or \"merge\"";
    }
  }

  // Seed mask must match data shape.
  if (failed(verifyMaskMatchesData(getOperation(), seedType, lhsType))) {
    return failure();
  }

  // Result mask must match seed mask.
  if (seedType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires result mask lane count to match seed mask lane count");
  }

  return success();
}

LogicalResult VMIVcmpsOp::verify() {
  auto srcType = cast<VMIVRegType>(getSrc().getType());
  auto seedType = cast<VMIMaskType>(getSeed().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());

  Type eltTy = srcType.getElementType();
  if (failed(verifyBF16x2ComputeElementType(getOperation(), eltTy))) {
    return failure();
  }
  if (!isVMII8I16I32OrF16BF16F32Type(eltTy)) {
    return emitOpError(
        "requires i8, i16, i32, f16, bf16, or f32 VMI element type");
  }
  bool usesOrderedFloatPredicate = getCmp().starts_with("o");
  bool invalidIntegerPredicate =
      isVMIIntegerLikeType(eltTy) && usesOrderedFloatPredicate;
  if (invalidIntegerPredicate) {
    return emitOpError("requires integer compare predicate eq/ne/lt/le/gt/ge; "
                       "signedness is selected by the integer element type");
  }

  // Scalar type must match vector element type.
  Type scalarTy = getScalar().getType();
  if (scalarTy != eltTy) {
    return emitOpError("requires scalar type to match vector element type, "
                       "got scalar ")
           << scalarTy << " vs vector element " << eltTy;
  }

  // Validate cmp predicate.
  if (!isSupportedVCmpPredicate(getCmp())) {
    return emitOpError("unsupported compare predicate '")
           << getCmp() << "'; expected eq/ne/lt/le/gt/ge, "
           << "or oeq/one/olt/ole/ogt/oge";
  }

  // Validate pmode.
  if (auto pmode = getPmode()) {
    bool unsupportedPmode =
        pmode.value() != "zeroing" && pmode.value() != "merge";
    if (unsupportedPmode) {
      return emitOpError("unsupported pmode '")
             << pmode.value() << "'; expected \"zeroing\" or \"merge\"";
    }
  }

  // Seed mask must match data shape.
  if (failed(verifyMaskMatchesData(getOperation(), seedType, srcType))) {
    return failure();
  }

  // Result mask must match seed mask.
  if (seedType.getElementCount() != resultType.getElementCount()) {
    return emitOpError(
        "requires result mask lane count to match seed mask lane count");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// VMICvtOp — unified elementwise type conversion
//===----------------------------------------------------------------------===//
// VMIvLoadOp
static ParseResult parseVLoadOptionalPostOperand(
    OpAsmParser &parser, OperationState &result, int &numPostBracket,
    OpAsmParser::UnresolvedOperand &postOp1) {
  // Optional comma-separated post-bracket operands.
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(postOp1)) {
      return failure();
    }
    numPostBracket = 1;
  }
  return success();
}

static void disambiguateVLoadPostOperands(
    OperationState &result, int numPostBracket,
    OpAsmParser::UnresolvedOperand postOp1,
    OpAsmParser::UnresolvedOperand &strideOperand,
    OpAsmParser::UnresolvedOperand &blockStrideOperand, bool &hasStride,
    bool &hasBlock) {
  // Disambiguate post-bracket operands
  // 1 operand + group attr → stride; otherwise → block_stride.
  if (numPostBracket == 1) {
    if (result.attributes.get("group")) {
      hasStride = true;
      strideOperand = postOp1;
    } else {
      hasBlock = true;
      blockStrideOperand = postOp1;
    }
  }
}

static ParseResult resolveVLoadOperandsAndFinalize(
    OpAsmParser &parser, OperationState &result,
    OpAsmParser::UnresolvedOperand sourceOperand,
    OpAsmParser::UnresolvedOperand offsetOperand,
    OpAsmParser::UnresolvedOperand strideOperand,
    OpAsmParser::UnresolvedOperand blockStrideOperand, bool hasStride,
    bool hasBlock, Type sourceType, ArrayRef<Type> resultTypes) {
  if (parser.resolveOperand(sourceOperand, sourceType, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(offsetOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasStride &&
      parser.resolveOperand(strideOperand, parser.getBuilder().getIndexType(),
                            result.operands)) {
    return failure();
  }
  if (hasBlock &&
      parser.resolveOperand(blockStrideOperand,
                            parser.getBuilder().getIntegerType(mlir::pto::kValue16),
                            result.operands)) {
    return failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, hasStride ? 1 : 0, hasBlock ? 1 : 0}));
  result.addTypes(resultTypes);
  return success();
}

ParseResult VMIvLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand sourceOperand;
  OpAsmParser::UnresolvedOperand offsetOperand;
  OpAsmParser::UnresolvedOperand strideOperand;
  OpAsmParser::UnresolvedOperand blockStrideOperand;
  // Parse: %source[%offset]
  if (parser.parseOperand(sourceOperand) || parser.parseLSquare() ||
      parser.parseOperand(offsetOperand) || parser.parseRSquare()) {
    return failure();
  }
  int numPostBracket = 0;
  OpAsmParser::UnresolvedOperand postOp1;
  if (failed(parseVLoadOptionalPostOperand(parser, result, numPostBracket,
                                           postOp1))) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  Type sourceType;
  if (parser.parseColonType(sourceType)) {
    return failure();
  }
  if (parser.parseArrow()) {
    return failure();
  }
  SmallVector<Type, mlir::pto::kValue2> resultTypes;
  if (parser.parseTypeList(resultTypes)) {
    return failure();
  }
  bool hasStride = false;
  bool hasBlock = false;
  disambiguateVLoadPostOperands(result, numPostBracket, postOp1, strideOperand,
                                blockStrideOperand, hasStride, hasBlock);
  return resolveVLoadOperandsAndFinalize(
      parser, result, sourceOperand, offsetOperand, strideOperand,
      blockStrideOperand, hasStride, hasBlock, sourceType, resultTypes);
}


void VMIvLoadOp::print(OpAsmPrinter &p) {
  p << ' ' << getSource() << '[';
  p.printOperand(getOffset());
  p << ']';
  if (getStride()) {
    p << ", ";
    p.printOperand(getStride());
  }
  if (getBlockStride()) {
    p << ", ";
    p.printOperand(getBlockStride());
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  p << " : " << getSource().getType() << " -> " << getResults().getTypes();
}

static LogicalResult verifyVLoadGroupAndBlockModes(
    Operation *op, bool hasGroup, std::optional<StringRef> distMode,
    bool hasStride, bool hasBlock, int64_t numGroups, size_t nResults) {
  if (hasGroup && distMode && *distMode != "brc") {
    return op->emitOpError("group and dist_mode are mutually exclusive");
  }
  if (hasGroup && !hasStride) {
    return op->emitOpError("group requires a stride operand");
  }
  if (!hasGroup && hasStride) {
    return op->emitOpError("stride operand is only valid with group");
  }
  if (hasGroup) {
    if (numGroups <= 0) {
      return op->emitOpError("group must be positive, got ") << numGroups;
    }
    if (nResults != 1) {
      return op->emitOpError("group mode requires exactly 1 result");
    }
  }
  if (failed(verifyBlockStrideExclusions(op, hasBlock, distMode, hasGroup))) {
    return failure();
  }
  if (hasBlock && nResults != 1) {
    return op->emitOpError("block-stride mode requires exactly 1 result");
  }
  return success();
}

static LogicalResult verifyVLoadDistModeAndPmode(
    Operation *op, std::optional<StringRef> distMode, size_t nResults,
    std::optional<StringRef> pmode) {
  if (distMode && !validDistModes().count(*distMode)) {
    return op->emitOpError("invalid dist-mode: \"") << *distMode << "\"";
  }
  bool isDintlv = distMode && *distMode == "dintlv";
  if (isDintlv && nResults != mlir::pto::kValue2) {
    return op->emitOpError("dist-mode \"dintlv\" requires exactly 2 results");
  }
  if (!isDintlv && nResults != 1) {
    return op->emitOpError("requires exactly 1 result for dist-mode \"")
           << (distMode ? *distMode : "continuous") << "\"";
  }
  if (pmode && !validPModes().count(*pmode)) {
    return op->emitOpError("invalid pmode: \"") << *pmode << "\"";
  }
  return success();
}

static LogicalResult verifyVLoadModeAndResultCounts(
    Operation *op, bool hasGroup, std::optional<StringRef> distMode,
    bool hasStride, bool hasBlock, int64_t numGroups, size_t nResults,
    std::optional<StringRef> pmode) {
  if (failed(verifyVLoadGroupAndBlockModes(op, hasGroup, distMode, hasStride,
                                           hasBlock, numGroups, nResults))) {
    return failure();
  }
  return verifyVLoadDistModeAndPmode(op, distMode, nResults, pmode);
}

static LogicalResult verifyVLoadResultTypes(
    Operation *op, ValueRange results, Type sourceType,
    std::optional<StringRef> distMode) {
  bool isDintlv = distMode && *distMode == "dintlv";
  for (auto res : results) {
    auto resType = cast<VMIVRegType>(res.getType());
    if (failed(verifyMemoryElementMatches(op, sourceType, resType,
                                          "source"))) {
      return failure();
    }
    if (isDintlv &&
        failed(verifyContiguousIfLayoutAssigned(op, resType, "result"))) {
      return failure();
    }
  }
  return success();
}

LogicalResult VMIvLoadOp::verify() {
  bool hasGroup = static_cast<bool>(getGroup());
  auto distMode = getDistMode();
  bool hasStride = static_cast<bool>(getStride());
  bool hasBlock = static_cast<bool>(getBlockStride());
  size_t nResults = getResults().size();
  if (failed(verifyVLoadModeAndResultCounts(
          getOperation(), hasGroup, distMode, hasStride, hasBlock,
          hasGroup ? getGroupAttr().getInt() : 0, nResults, getPmode()))) {
    return failure();
  }
  if (hasGroup) {
    auto resultType = cast<VMIVRegType>(getResults()[0].getType());
    if (failed(verifyNumGroups(getOperation(), resultType,
                               getGroupAttr().getInt()))) {
      return failure();
    }
  }
  return verifyVLoadResultTypes(getOperation(), getResults(),
                                getSource().getType(), distMode);
}

void VMIvLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

//===----------------------------------------------------------------------===//
// VMIvStoreOp

Type mlir::pto::getVMIPhysicalDataElementType(VMIVRegType type) {
  // lane_stride describes where logical elements reside inside a physical
  // vector register; it does not change their element type.  Packed group
  // slots therefore use the same sparse logical-element carrier as dense
  // lane-stride values.
  return type.getElementType();
}

FailureOr<int64_t> mlir::pto::getDataLanesPerPart(Type elementType) {
  unsigned elementBitWidth = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBitWidth == 0) {
    return failure();
  }
  constexpr int64_t kPhysicalVRegBits = mlir::pto::kValue256 * mlir::pto::kValue8;
  if (kPhysicalVRegBits % elementBitWidth != 0) {
    return failure();
  }
  return kPhysicalVRegBits / elementBitWidth;
}

FailureOr<int64_t> mlir::pto::getMaskLanesPerPart(StringRef granularity) {
  if (granularity == "b8") {
    return mlir::pto::kValue256;
  }
  if (granularity == "b16") {
    return mlir::pto::kValue128;
  }
  if (granularity == "b32") {
    return mlir::pto::kValue64;
  }
  return failure();
}

FailureOr<int64_t> mlir::pto::getVMILayoutBlockElems(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout)) {
    return failure();
  }
  if (!(*layout).isBlockDeinterleaved()) {
    return 1;
  }

  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  constexpr int64_t kVCGBlocksPerPart = mlir::pto::kValue8;
  if (failed(lanesPerPart) || *lanesPerPart <= 0 ||
      *lanesPerPart % kVCGBlocksPerPart != 0) {
    return failure();
  }
  return *lanesPerPart / kVCGBlocksPerPart;
}

FailureOr<int64_t> mlir::pto::getVMIPhysicalArity(Type type) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(elementCount) || failed(lanesPerPart) || failed(layout)) {
    return failure();
  }

  if ((*layout).isGroupSlots() && (*layout).getSlots() > 0) {
    return divideCeilNonNegative((*layout).getNumGroups(),
                                 (*layout).getSlots());
  }

  int64_t factor = (*layout).isDenseSplit() ? (*layout).getFactor() : 1;
  FailureOr<int64_t> blockElems = getVMILayoutBlockElems(type);
  if (failed(blockElems)) {
    return failure();
  }
  int64_t laneStride =
      isa<VMIMaskType>(type) ? 1
                             : ((*layout).isDense() ? (*layout).getLaneStride()
                                                    : 1);
  int64_t arity = 0;
  for (int64_t part = 0; part < factor; ++part) {
    int64_t lanesInPart =
        getDenseLogicalLanesInPart(*elementCount, factor, *blockElems, part);
    int64_t requiredPhysicalLanes =
        lanesInPart == 0 ? 0 : (lanesInPart - 1) * laneStride + 1;
    arity += divideCeilNonNegative(requiredPhysicalLanes, *lanesPerPart);
  }
  return arity;
}

FailureOr<VMIPhysicalLane>
mlir::pto::mapLogicalLaneToPhysical(Type type, int64_t logicalLane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> laneStride = getDenseLaneStride(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(laneStride) || failed(lanesPerPart)) {
    return failure();
  }
  if (logicalLane < 0 || logicalLane >= *elementCount) {
    return failure();
  }

  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (succeeded(layout) && (*layout).isGroupSlots() &&
      (*layout).getSlots() > 0) {
    int64_t slots = (*layout).getSlots();
    int64_t lane = logicalLane % slots;
    if (lane >= *lanesPerPart) {
      return failure();
    }
    return VMIPhysicalLane{/*part=*/0, logicalLane / slots, lane};
  }

  int64_t part = 0;
  std::optional<int64_t> indexInPart = mapDenseLogicalLaneToPartIndex(
      *elementCount, *factor, *blockElems, logicalLane, part);
  if (!indexInPart) {
    return failure();
  }
  int64_t physicalIndex = *indexInPart * *laneStride;
  return VMIPhysicalLane{part, physicalIndex / *lanesPerPart,
                         physicalIndex % *lanesPerPart};
}

FailureOr<int64_t> mlir::pto::mapPhysicalLaneToLogical(Type type, int64_t part,
                                                       int64_t chunk,
                                                       int64_t lane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> laneStride = getDenseLaneStride(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(laneStride) || failed(lanesPerPart)) {
    return failure();
  }
  if (part < 0 || part >= *factor || chunk < 0 || lane < 0 ||
      lane >= *lanesPerPart) {
    return failure();
  }

  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (succeeded(layout) && (*layout).isGroupSlots() &&
      (*layout).getSlots() > 0) {
    int64_t slots = (*layout).getSlots();
    if (part != 0 || lane >= slots) {
      return failure();
    }
    int64_t logicalLane = chunk * slots + lane;
    if (logicalLane >= *elementCount) {
      return failure();
    }
    return logicalLane;
  }

  int64_t physicalIndexInPart = chunk * *lanesPerPart + lane;
  if (physicalIndexInPart % *laneStride != 0) {
    return failure();
  }
  int64_t indexInPart = physicalIndexInPart / *laneStride;
  std::optional<int64_t> logicalLane = mapDensePartIndexToLogicalLane(
      *elementCount, *factor, *blockElems, part, indexInPart);
  if (!logicalLane) {
    return failure();
  }
  return *logicalLane;
}

FailureOr<bool> mlir::pto::isPaddingLane(Type type, int64_t part, int64_t chunk,
                                         int64_t lane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> laneStride = getDenseLaneStride(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(laneStride) || failed(lanesPerPart)) {
    return failure();
  }
  if (part < 0 || part >= *factor || chunk < 0 || lane < 0 ||
      lane >= *lanesPerPart) {
    return failure();
  }

  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (succeeded(layout) && (*layout).isGroupSlots() &&
      (*layout).getSlots() > 0) {
    int64_t slots = (*layout).getSlots();
    if (part != 0) {
      return true;
    }
    if (lane >= slots) {
      return true;
    }
    return chunk * slots + lane >= *elementCount;
  }

  int64_t lanesInPart =
      getDenseLogicalLanesInPart(*elementCount, *factor, *blockElems, part);
  int64_t physicalIndexInPart = chunk * *lanesPerPart + lane;
  if (physicalIndexInPart % *laneStride != 0) {
    return true;
  }
  int64_t indexInPart = physicalIndexInPart / *laneStride;
  return indexInPart >= lanesInPart;
}
