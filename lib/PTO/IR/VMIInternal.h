// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Internal shared helper definitions for VMI.
// This header is internal to lib/PTO/IR and not installed.
// Each including translation unit gets its own internal copy of these helpers.

#ifndef PTO_IR_VMI_INTERNAL_H
#define PTO_IR_VMI_INTERNAL_H

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


using namespace mlir;
using namespace mlir::pto;

namespace {
[[maybe_unused]] static std::string formatVMIVRegType(int64_t elementCount, Type elementType,
                                     Attribute layout) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "!pto.vmi.vreg<" << elementCount << "x" << elementType;
  if (layout) {
    os << ", " << layout;
  }
  os << ">";
  return result;
}

[[maybe_unused]] static std::string formatVMIMaskType(int64_t elementCount,
                                     StringRef granularity, Attribute layout) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "!pto.vmi.mask<" << elementCount << "x" << granularity;
  if (layout) {
    os << ", " << layout;
  }
  os << ">";
  return result;
}

[[maybe_unused]] static bool isSupportedVMIElementType(Type type) {
  return isa<IntegerType, FloatType, IndexType>(type) ||
         pto::isPTOLowPrecisionType(type);
}

[[maybe_unused]] static bool isVMIFloatLikeType(Type type) {
  return isa<FloatType>(type) || pto::isPTOLowPrecisionType(type);
}

[[maybe_unused]] static bool involvesBF16x2(Type sourceType, Type resultType) {
  return pto::isPTOBF16x2Type(sourceType) ||
         pto::isPTOBF16x2Type(resultType);
}

[[maybe_unused]] static bool isVMIPackedFloatCarrierType(Type type) {
  return pto::isPTOHiFloat8x2Type(type) ||
         pto::isPTOFloat4PackedType(type) ||
         pto::isPTOBF16x2Type(type);
}

[[maybe_unused]] static bool involvesVMIPackedFloatCarrier(Type sourceType, Type resultType) {
  return isVMIPackedFloatCarrierType(sourceType) ||
         isVMIPackedFloatCarrierType(resultType);
}

[[maybe_unused]] static LogicalResult verifyBF16x2ComputeElementType(Operation *op, Type type) {
  if (pto::isPTOBF16x2Type(type)) {
    return op->emitOpError(
        "does not support bf16x2 VMI element type; bf16x2 is conversion-only");
}
  return success();
}

[[maybe_unused]] static bool isVMIIntegerLikeType(Type type) {
  return isa<IntegerType, IndexType>(type);
}

[[maybe_unused]] static bool isVMIF16OrF32Type(Type type) {
  return type.isF16() || type.isF32();
}

[[maybe_unused]] static bool isVMIF16BF16OrF32Type(Type type) {
  return type.isF16() || type.isBF16() || type.isF32();
}

[[maybe_unused]] static bool isVMIPredicateMaskableElementType(Type type) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(type);
  return elementBits == mlir::pto::kValue8 || elementBits == mlir::pto::kValue16 || elementBits == mlir::pto::kValue32;
}

[[maybe_unused]] static bool isVMIAnyI8I16I32Type(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType) {
    return false;
  }
  return integerType.getWidth() == mlir::pto::kValue8 || integerType.getWidth() == mlir::pto::kValue16 ||
         integerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] static bool isVMII8I16I32OrF16BF16F32Type(Type type) {
  return isVMIAnyI8I16I32Type(type) || isVMIF16BF16OrF32Type(type);
}

[[maybe_unused]] static bool isVMII16I32OrF16BF16F32Type(Type type) {
  auto intType = dyn_cast<IntegerType>(type);
  bool supportedInteger =
      intType && (intType.getWidth() == mlir::pto::kValue16 || intType.getWidth() == mlir::pto::kValue32);
  return supportedInteger || isVMIF16BF16OrF32Type(type);
}

[[maybe_unused]] static bool isVMII8I16I32OrF16F32Type(Type type) {
  return isVMIAnyI8I16I32Type(type) || isVMIF16OrF32Type(type);
}

[[maybe_unused]] static bool isVMISignedI8I16I32Type(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  if (!integerType || !integerType.isSigned()) {
    return false;
  }
  return integerType.getWidth() == mlir::pto::kValue8 || integerType.getWidth() == mlir::pto::kValue16 ||
         integerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] static bool isVMISignedIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.isSigned();
}

[[maybe_unused]] static bool isVMIUnsignedOrSignlessIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && (integerType.isUnsigned() || integerType.isSignless());
}

// ---------------------------------------------------------------------------
// VMI integer element type sign-semantics helper
//
// CONVENTION: VMI op verifiers that need "unsigned semantics" or "signed
// semantics" on an integer element type MUST route the sign check through
// matchesVMIIntSemantics(...) instead of calling IntegerType::isUnsigned()
// / isSigned() directly.
//
// Signless integers are treated as equivalent to UNSIGNED only. They are
// NOT accepted for signed semantics: signed hardware ops require an
// explicitly signed integer type, to avoid silent sign-extension bugs when
// a producer happens to emit a signless value.
//
// Width / kind / IntegerType-cast checks stay inline at each callsite;
// only the sign-semantics decision is centralized here.
// ---------------------------------------------------------------------------

enum class VMIIntSignSemantics { Unsigned, Signed, Any };

[[maybe_unused]] static bool matchesVMIIntSemantics(IntegerType intType,
                                   VMIIntSignSemantics semantics) {
  switch (semantics) {
  case VMIIntSignSemantics::Unsigned:
    return intType.isUnsigned() || intType.isSignless();
  case VMIIntSignSemantics::Signed:
    return intType.isSigned();
  case VMIIntSignSemantics::Any:
    return true;
  }
  llvm_unreachable("bad VMIIntSignSemantics");
}

[[maybe_unused]] static bool isVMIIotaElementType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() == mlir::pto::kValue8 || intType.getWidth() == mlir::pto::kValue16 ||
           intType.getWidth() == mlir::pto::kValue32;
  }
  return type.isF16() || type.isF32();
}

[[maybe_unused]] static bool isCompatibleScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType) {
    return true;
  }

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt ||
      semanticInt.getWidth() != scalarInt.getWidth()) {
    return false;
  }

  if (semanticInt.isSigned()) {
    return scalarInt.isSigned() || scalarInt.isSignless();
  }
  if (semanticInt.isUnsigned()) {
    return scalarInt.isUnsigned() || scalarInt.isSignless();
  }
  return scalarInt.isSignless();
}

[[maybe_unused]] static unsigned getVMIElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

[[maybe_unused]] static int64_t divideCeilNonNegative(int64_t value, int64_t divisor) {
  if (divisor <= 0) {
    return 0;
  }
  return value == 0 ? 0 : (value + divisor - 1) / divisor;
}

[[maybe_unused]] static LogicalResult parseOptionalVMILayout(AsmParser &parser,
                                            Attribute &layout) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }

  if (failed(parser.parseAttribute(layout))) {
    return failure();
  }
  if (!mlir::isa<VMILayoutAttr>(layout)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected #pto.vmi.layout attribute");
  }
  return success();
}

[[maybe_unused]] static FailureOr<int64_t> getVMIElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return vregType.getElementCount();
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    return maskType.getElementCount();
  }
  return failure();
}

[[maybe_unused]] static FailureOr<VMILayoutAttr> getAssignedVMILayout(Type type) {
  Attribute layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    layout = vregType.getLayout();
  }
  else if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    layout = maskType.getLayout();
  }
  else {
    return failure();
  }

  auto layoutAttr = dyn_cast_or_null<VMILayoutAttr>(layout);
  if (!layoutAttr) {
    return failure();
  }
  return layoutAttr;
}

[[maybe_unused]] static FailureOr<int64_t> getLayoutFactor(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout)) {
    return failure();
  }
  return (*layout).isDenseSplit() ? (*layout).getFactor() : 1;
}

[[maybe_unused]] static FailureOr<int64_t> getLayoutBlockElems(Type type) {
  return getVMILayoutBlockElems(type);
}

[[maybe_unused]] static int64_t getMaskGranularityBitWidth(StringRef granularity) {
  if (granularity == "b8") {
    return mlir::pto::kValue8;
  }
  if (granularity == "b16") {
    return mlir::pto::kValue16;
  }
  if (granularity == "b32") {
    return mlir::pto::kValue32;
  }
  return 0;
}

[[maybe_unused]] static StringRef getMaskGranularityForBitWidth(int64_t bits) {
  switch (bits) {
  case mlir::pto::kValue8:
    return "b8";
  case mlir::pto::kValue16:
    return "b16";
  case mlir::pto::kValue32:
    return "b32";
  default:
    return "";
  }
}

[[maybe_unused]] static FailureOr<StringRef> getVMIMaskPhysicalGranularity(VMIMaskType type) {
  int64_t bits = getMaskGranularityBitWidth(type.getGranularity());
  if (bits == 0) {
    return failure();
  }

  VMILayoutAttr layout = type.getLayoutAttr();
  int64_t laneStride = layout && layout.hasLaneStride() ? layout.getLaneStride()
                                                        : 1;
  StringRef physicalGranularity =
      getMaskGranularityForBitWidth(bits * laneStride);
  if (physicalGranularity.empty()) {
    return failure();
  }
  return physicalGranularity;
}

[[maybe_unused]] static FailureOr<int64_t> getPhysicalLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type)) {
    return getDataLanesPerPart(getVMIPhysicalDataElementType(vregType));
  }
  if (auto maskType = dyn_cast<VMIMaskType>(type)) {
    FailureOr<StringRef> physicalGranularity =
        getVMIMaskPhysicalGranularity(maskType);
    if (failed(physicalGranularity)) {
      return failure();
    }
    return getMaskLanesPerPart(*physicalGranularity);
  }
  return failure();
}

[[maybe_unused]] static FailureOr<int64_t> getDenseLaneStride(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout)) {
    return failure();
  }
  if (isa<VMIMaskType>(type)) {
    return 1;
  }
  return (*layout).isDense() ? (*layout).getLaneStride() : 1;
}

[[maybe_unused]] static bool isLayoutAssigned(VMIVRegType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

[[maybe_unused]] static bool isLayoutAssigned(VMIMaskType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

[[maybe_unused]] static LogicalResult
verifyAllSameVRegShapeAndLayout(Operation *op, ArrayRef<VMIVRegType> types,
                                bool requireSameElement) {
  if (types.empty()) {
    return success();
  }

  VMIVRegType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIVRegType type) { return isLayoutAssigned(type); });

  for (VMIVRegType type : types) {
    if (type.getElementCount() != first.getElementCount()) {
      return op->emitOpError(
          "requires all VMI data values to have the same logical lane count");
    }
    if (requireSameElement && type.getElementType() != first.getElementType()) {
      return op->emitOpError(
          "requires all VMI data values to have the same element type");
    }
    if (anyLayout && !isLayoutAssigned(type)) {
      return op->emitOpError(
          "requires either all or no VMI data values to carry layout");
    }
    if (anyLayout && type.getLayout() != first.getLayout()) {
      return op->emitOpError("requires all layout-assigned VMI data values to "
                             "have the same layout");
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyAllSameVRegShapeAndLayoutPresence(
    Operation *op, ArrayRef<VMIVRegType> types, bool requireSameElement) {
  if (types.empty()) {
    return success();
  }

  VMIVRegType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIVRegType type) { return isLayoutAssigned(type); });

  for (VMIVRegType type : types) {
    if (type.getElementCount() != first.getElementCount()) {
      return op->emitOpError(
          "requires all VMI data values to have the same logical lane count");
    }
    if (requireSameElement && type.getElementType() != first.getElementType()) {
      return op->emitOpError(
          "requires all VMI data values to have the same element type");
    }
    if (anyLayout && !isLayoutAssigned(type)) {
      return op->emitOpError(
          "requires either all or no VMI data values to carry layout");
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyElementwiseVRegOp(Operation *op, VMIVRegType lhs,
                                             VMIVRegType rhs,
                                             VMIVRegType result) {
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, result},
                                         /*requireSameElement=*/true);
}

[[maybe_unused]] static LogicalResult verifyFloatUnaryVRegOp(Operation *op, VMIVRegType source,
                                            VMIVRegType result) {
  if (failed(
          verifyBF16x2ComputeElementType(op, source.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(source.getElementType())) {
    return op->emitOpError("requires floating-point-like VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(op, {source, result},
                                         /*requireSameElement=*/true);
}

[[maybe_unused]] static LogicalResult verifyFloatTernaryVRegOp(Operation *op, VMIVRegType lhs,
                                              VMIVRegType rhs, VMIVRegType acc,
                                              VMIVRegType result) {
  if (failed(verifyBF16x2ComputeElementType(op, lhs.getElementType()))) {
    return failure();
  }
  if (!isVMIFloatLikeType(lhs.getElementType())) {
    return op->emitOpError("requires floating-point-like VMI element type");
  }
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, acc, result},
                                         /*requireSameElement=*/true);
}

[[maybe_unused]] static LogicalResult
verifyAllSameMaskShapeLayoutAndGranularity(Operation *op,
                                           ArrayRef<VMIMaskType> types) {
  if (types.empty()) {
    return success();
  }

  VMIMaskType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIMaskType type) { return isLayoutAssigned(type); });

  for (VMIMaskType type : types) {
    if (type.getElementCount() != first.getElementCount()) {
      return op->emitOpError(
          "requires all VMI mask values to have the same logical lane count");
    }
    if (type.getGranularity() != first.getGranularity()) {
      return op->emitOpError(
          "requires all VMI mask values to have the same granularity");
    }
    if (anyLayout && !isLayoutAssigned(type)) {
      return op->emitOpError(
          "requires either all or no VMI mask values to carry layout");
    }
    if (anyLayout && type.getLayout() != first.getLayout()) {
      return op->emitOpError(
          "requires all layout-assigned VMI mask values to have the same "
          "layout");
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyMaskMatchesData(Operation *op, VMIMaskType maskType,
                                           VMIVRegType dataType) {
  if (maskType.getElementCount() != dataType.getElementCount()) {
    return op->emitOpError(
        "requires mask logical lane count to match data lane count");
  }

  if (isLayoutAssigned(maskType) || isLayoutAssigned(dataType)) {
    if (!isLayoutAssigned(maskType) || !isLayoutAssigned(dataType)) {
      return op->emitOpError("requires either both mask and data to carry "
                             "layout or neither to carry layout");
    }
    if (maskType.getLayout() != dataType.getLayout()) {
      return op->emitOpError("requires mask layout to match data layout");
    }
  }

  if (maskType.isPred()) {
    return success();
  }

  unsigned elementBitWidth = getVMIElementBitWidth(dataType.getElementType());
  int64_t maskBitWidth = getMaskGranularityBitWidth(maskType.getGranularity());
  if (elementBitWidth != 0 && maskBitWidth != 0 &&
      elementBitWidth != static_cast<unsigned>(maskBitWidth)) {
    return op->emitOpError(
        "requires mask granularity to match data element width");
  }

  return success();
}


[[maybe_unused]] static Type getMemoryElementType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getElementType();
  }
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    return memrefType.getElementType();
  }
  return {};
}

[[maybe_unused]] static bool isUBBackedMemoryType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace().getAddressSpace() == AddressSpace::VEC;
  }

  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    return false;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (auto addressSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    return addressSpace.getAddressSpace() == AddressSpace::VEC;
  }
  if (auto integerSpace = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    return integerSpace.getInt() == static_cast<int64_t>(AddressSpace::VEC);
  }
  return false;
}

[[maybe_unused]] static LogicalResult verifyUBBackedMemory(Operation *op, Type memoryType,
                                          StringRef role) {
  if (isUBBackedMemoryType(memoryType)) {
    return success();
  }
  return op->emitOpError() << "requires memory " << role
                           << " to be UB-backed";
}

[[maybe_unused]] static LogicalResult verifyMemoryElementMatches(Operation *op, Type memoryType,
                                                VMIVRegType dataType,
                                                StringRef role) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType) {
    return success();
  }
  if (memoryElementType != dataType.getElementType()) {
    return op->emitOpError() << "requires memory " << role
                             << " element type to match VMI data element type";
  }
  return success();
}

// 8->16 gather promotion is a zero-extension (unsigned) operation. signless
// i8/i16 are accepted and treated as unsigned bytes; sign-extension is not
// supported (see VMIVgatherOp / Vgather2Op description).
[[maybe_unused]] static bool isVMI8To16GatherPair(Type sourceElemType, Type resultElemType) {
  auto srcInt = dyn_cast<IntegerType>(sourceElemType);
  auto resInt = dyn_cast<IntegerType>(resultElemType);
  if (!srcInt || !resInt ||
      srcInt.getWidth() != mlir::pto::kValue8 ||
      resInt.getWidth() != mlir::pto::kValue16) {
    return false;
  }
  if (srcInt.isUnsigned()) {
    return resInt.isUnsigned();
  }
  return !resInt.isUnsigned();
}

[[maybe_unused]] static LogicalResult verifyGatherMemoryElementMatches(
    Operation *op, Type memoryType, VMIVRegType dataType, StringRef role) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType) {
    return success();
  }
  if (memoryElementType == dataType.getElementType()) {
    return success();
  }
  if (isVMI8To16GatherPair(memoryElementType, dataType.getElementType())) {
    return success();
  }
  return op->emitOpError()
         << "requires memory " << role
         << " element type to match VMI data element type"
            " or be an 8-bit integer promoted to a matching 16-bit integer";
}

[[maybe_unused]] static bool isSameWidth16BitGatherPair(Type sourceElemType,
                                       Type resultElemType) {
  // Existing VMI f16/bf16 path: same-width 16-bit float gather.
  if (sourceElemType == resultElemType &&
      (sourceElemType.isF16() || sourceElemType.isBF16())) {
    return true;
  }
  auto srcInt = dyn_cast<IntegerType>(sourceElemType);
  auto resInt = dyn_cast<IntegerType>(resultElemType);
  // Existing VMI ui16/i16 path: same-width 16-bit integer gather with matching
  // integer semantics (signless i16 / i16 is accepted as the non-unsigned side).
  if (!srcInt || !resInt ||
      srcInt.getWidth() != mlir::pto::kValue16 ||
      resInt.getWidth() != mlir::pto::kValue16) {
    return false;
  }
  if (srcInt.isUnsigned()) {
    return resInt.isUnsigned();
  }
  return !resInt.isUnsigned();
}

[[maybe_unused]] static bool isSupported16BitGatherResult(Type sourceElemType,
                                         Type resultElemType) {
  // New 8 -> 16 path: i8/ui8 -> i16/ui16 with matching integer semantics.
  if (isVMI8To16GatherPair(sourceElemType, resultElemType)) {
    return true;
  }
  return isSameWidth16BitGatherPair(sourceElemType, resultElemType);
}

[[maybe_unused]] static LogicalResult verifyContiguousIfLayoutAssigned(Operation *op,
                                                      VMIVRegType type,
                                                      StringRef role) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (layout && !layout.isContiguous()) {
    return op->emitOpError()
           << "requires layout-assigned " << role
           << " to use #pto.vmi.layout<contiguous>";
  }
  return success();
}

[[maybe_unused]] static bool isPackedByteGroupStore(Type memoryType, VMIVRegType dataType) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType) {
    return false;
  }
  auto memoryIntegerType = dyn_cast<IntegerType>(memoryElementType);
  auto dataIntegerType = dyn_cast<IntegerType>(dataType.getElementType());
  return memoryIntegerType && dataIntegerType &&
         memoryIntegerType.getWidth() == mlir::pto::kValue8 && dataIntegerType.getWidth() == mlir::pto::kValue32;
}

[[maybe_unused]] static LogicalResult verifyNumGroups(Operation *op, VMIVRegType type,
                                     int64_t numGroups) {
  if (numGroups <= 0) {
    return op->emitOpError("requires num_groups to be positive");
  }
  if (type.getElementCount() % numGroups != 0) {
    return op->emitOpError()
           << "requires num_groups to evenly divide VMI logical lane count "
           << type.getElementCount();
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyPhysicalVRegParts(Operation *op,
                                             VMIVRegType vregType,
                                             TypeRange physicalTypes) {
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(vregType);
  Type physicalElementType = getVMIPhysicalDataElementType(vregType);
  if (failed(lanesPerPart)) {
    return op->emitOpError(
        "requires data element type with known physical lane count");
  }
  for (Type physicalType : physicalTypes) {
    auto partType = dyn_cast<VRegType>(physicalType);
    if (!partType) {
      return op->emitOpError("requires physical data parts to be !pto.vreg");
    }
    if (partType.getElementCount() != *lanesPerPart ||
        partType.getElementType() != physicalElementType) {
      return op->emitOpError(
          "requires physical data part type to match VMI lane-map helper");
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyPhysicalMaskParts(Operation *op,
                                             VMIMaskType maskType,
                                             TypeRange physicalTypes) {
  if (maskType.isPred()) {
    return op->emitOpError(
        "requires layout-assigned mask with concrete granularity");
  }
  FailureOr<StringRef> physicalGranularity =
      getVMIMaskPhysicalGranularity(maskType);
  if (failed(physicalGranularity)) {
    return op->emitOpError(
        "requires mask type with supported physical carrier granularity");
  }
  for (Type physicalType : physicalTypes) {
    auto partType = dyn_cast<MaskType>(physicalType);
    if (!partType) {
      return op->emitOpError("requires physical mask parts to be !pto.mask");
    }
    if (partType.getGranularity() != *physicalGranularity) {
      return op->emitOpError(
          "requires physical mask part granularity to match VMI mask carrier");
    }
  }
  return success();
}

[[maybe_unused]] static LogicalResult verifyPhysicalParts(Operation *op, Type vmiType,
                                         TypeRange physicalTypes) {
  FailureOr<int64_t> expectedArity = getVMIPhysicalArity(vmiType);
  if (failed(expectedArity)) {
    return op->emitOpError(
        "requires a layout-assigned VMI type with computable physical arity");
  }
  if (static_cast<int64_t>(physicalTypes.size()) != *expectedArity) {
    return op->emitOpError() << "requires " << *expectedArity
                             << " physical parts, got " << physicalTypes.size();
  }
  if (auto vregType = dyn_cast<VMIVRegType>(vmiType)) {
    return verifyPhysicalVRegParts(op, vregType, physicalTypes);
  }
  auto maskType = dyn_cast<VMIMaskType>(vmiType);
  if (!maskType) {
    return op->emitOpError("requires VMI data or mask type");
  }
  return verifyPhysicalMaskParts(op, maskType, physicalTypes);
}

[[maybe_unused]] static std::optional<int64_t>
mapDenseLogicalLaneToPartIndex(int64_t elementCount, int64_t factor,
                               int64_t blockElems, int64_t logicalLane,
                               int64_t &part) {
  if (logicalLane < 0 || logicalLane >= elementCount || factor <= 0 ||
      blockElems <= 0) {
    return std::nullopt;
  }
  int64_t block = logicalLane / blockElems;
  int64_t inBlockLane = logicalLane % blockElems;
  part = block % factor;
  int64_t partBlock = block / factor;
  return partBlock * blockElems + inBlockLane;
}

[[maybe_unused]] static std::optional<int64_t>
mapDensePartIndexToLogicalLane(int64_t elementCount, int64_t factor,
                               int64_t blockElems, int64_t part,
                               int64_t indexInPart) {
  if (part < 0 || part >= factor || indexInPart < 0 || factor <= 0 ||
      blockElems <= 0) {
    return std::nullopt;
  }
  int64_t partBlock = indexInPart / blockElems;
  int64_t inBlockLane = indexInPart % blockElems;
  int64_t logicalBlock = partBlock * factor + part;
  int64_t logicalLane = logicalBlock * blockElems + inBlockLane;
  if (logicalLane >= elementCount) {
    return std::nullopt;
  }
  return logicalLane;
}

[[maybe_unused]] static int64_t getDenseLogicalLanesInPart(int64_t elementCount, int64_t factor,
                                          int64_t blockElems, int64_t part) {
  int64_t maxIndex = -1;
  for (int64_t lane = 0; lane < elementCount; ++lane) {
    int64_t lanePart = 0;
    std::optional<int64_t> index = mapDenseLogicalLaneToPartIndex(
        elementCount, factor, blockElems, lane, lanePart);
    if (index && lanePart == part) {
      maxIndex = std::max(maxIndex, *index);
    }
  }
  return maxIndex + 1;
}

[[maybe_unused]] static LogicalResult verifyReductionGroupAndPmode(
    Operation *op, VMIVRegType sourceType, VMIVRegType resultType,
    IntegerAttr groupAttr, std::optional<StringRef> pmode) {
  if (groupAttr) {
    int64_t C = groupAttr.getInt();
    if (C <= 0) {
      return op->emitOpError("group count must be positive");
    }
    if (sourceType.getElementCount() % C != 0) {
      return op->emitOpError("group count ") << C
                                            << " must divide source lane count "
                                            << sourceType.getElementCount();
    }
    if (resultType.getElementCount() != C) {
      return op->emitOpError("result lane count must equal group count ")
             << C << ", got " << resultType.getElementCount();
    }
    if (auto resultLayout = resultType.getLayoutAttr()) {
      if (!resultLayout.isGroupSlots() ||
          resultLayout.getNumGroups() != C) {
        return op->emitOpError()
               << "layout-assigned result must use "
                  "#pto.vmi.layout<num_groups = "
               << C << ">";
      }
    }
  } else if (resultType.getElementCount() != 1) {
    return op->emitOpError("full reduction (no group) requires 1-lane result, got ")
           << resultType.getElementCount();
  }
  if (sourceType.getElementType() != resultType.getElementType()) {
    return op->emitOpError("source and result element types must match");
  }
  if (pmode) {
    StringRef val = *pmode;
    if (val != "zero" && val != "merge") {
      return op->emitOpError("pmode must be \"zero\" or \"merge\", got \"")
             << val << "\"";
    }
  }
  return success();
}

} // namespace

#endif // PTO_IR_VMI_INTERNAL_H
