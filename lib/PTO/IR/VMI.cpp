// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMI.cpp - PTO VMI type and attribute support -----------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {

static std::string formatVMIVRegType(int64_t elementCount, Type elementType,
                                     Attribute layout) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "!pto.vmi.vreg<" << elementCount << "x" << elementType;
  if (layout)
    os << ", " << layout;
  os << ">";
  return result;
}

static std::string formatVMIMaskType(int64_t elementCount,
                                     StringRef granularity, Attribute layout) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "!pto.vmi.mask<" << elementCount << "x" << granularity;
  if (layout)
    os << ", " << layout;
  os << ">";
  return result;
}

static bool isSupportedVMIElementType(Type type) {
  return isa<IntegerType, FloatType, IndexType>(type) ||
         pto::isPTOLowPrecisionType(type);
}

static bool isVMIFloatLikeType(Type type) {
  return isa<FloatType>(type) || pto::isPTOLowPrecisionType(type);
}

static bool isVMIIntegerLikeType(Type type) {
  return isa<IntegerType, IndexType>(type);
}

static bool isVMISignedOrSignlessIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && !integerType.isUnsigned();
}

static bool isVMIUnsignedIntegerType(Type type) {
  auto integerType = dyn_cast<IntegerType>(type);
  return integerType && integerType.isUnsigned();
}

static bool isVMIIotaElementType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() == 8 || intType.getWidth() == 16 ||
           intType.getWidth() == 32;
  return type.isF16() || type.isF32();
}

static bool isCompatibleScalarForSemanticType(Type semanticType,
                                              Type scalarType) {
  if (semanticType == scalarType)
    return true;

  auto semanticInt = dyn_cast<IntegerType>(semanticType);
  auto scalarInt = dyn_cast<IntegerType>(scalarType);
  if (!semanticInt || !scalarInt ||
      semanticInt.getWidth() != scalarInt.getWidth())
    return false;

  if (semanticInt.isSigned())
    return scalarInt.isSigned() || scalarInt.isSignless();
  if (semanticInt.isUnsigned())
    return scalarInt.isUnsigned() || scalarInt.isSignless();
  return scalarInt.isSignless();
}

static unsigned getVMIElementBitWidth(Type type) {
  if (isa<IndexType>(type))
    return 64;
  return pto::getPTOStorageElemBitWidth(type);
}

static std::optional<unsigned> getVMIIntegerOrFloatBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return std::nullopt;
}

static int64_t divideCeilNonNegative(int64_t value, int64_t divisor) {
  return value == 0 ? 0 : (value + divisor - 1) / divisor;
}

static LogicalResult parseOptionalVMILayout(AsmParser &parser,
                                            Attribute &layout) {
  if (failed(parser.parseOptionalComma()))
    return success();

  if (failed(parser.parseAttribute(layout)))
    return failure();
  if (!mlir::isa<VMILayoutAttr>(layout))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected #pto.vmi.layout attribute");
  return success();
}

static FailureOr<int64_t> getVMIElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return vregType.getElementCount();
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getElementCount();
  return failure();
}

static FailureOr<VMILayoutAttr> getAssignedVMILayout(Type type) {
  Attribute layout;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layout = vregType.getLayout();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layout = maskType.getLayout();
  else
    return failure();

  auto layoutAttr = dyn_cast_or_null<VMILayoutAttr>(layout);
  if (!layoutAttr)
    return failure();
  return layoutAttr;
}

static FailureOr<int64_t> getLayoutFactor(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout))
    return failure();
  return (*layout).isDeinterleaved() ? (*layout).getFactor() : 1;
}

static FailureOr<int64_t> getLayoutBlockElems(Type type) {
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(layout))
    return failure();
  return (*layout).isDeinterleaved() ? (*layout).getBlockElems() : 1;
}

static FailureOr<int64_t> getPhysicalLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return getDataLanesPerPart(vregType.getElementType());
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return getMaskLanesPerPart(maskType.getGranularity());
  return failure();
}

static int64_t getMaskGranularityBitWidth(StringRef granularity) {
  if (granularity == "b8")
    return 8;
  if (granularity == "b16")
    return 16;
  if (granularity == "b32")
    return 32;
  return 0;
}

static bool isLayoutAssigned(VMIVRegType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

static bool isLayoutAssigned(VMIMaskType type) {
  return static_cast<bool>(type.getLayoutAttr());
}

static LogicalResult
verifyAllSameVRegShapeAndLayout(Operation *op, ArrayRef<VMIVRegType> types,
                                bool requireSameElement) {
  if (types.empty())
    return success();

  VMIVRegType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIVRegType type) { return isLayoutAssigned(type); });

  for (VMIVRegType type : types) {
    if (type.getElementCount() != first.getElementCount())
      return op->emitOpError(
          "requires all VMI data values to have the same logical lane count");
    if (requireSameElement && type.getElementType() != first.getElementType())
      return op->emitOpError(
          "requires all VMI data values to have the same element type");
    if (anyLayout && !isLayoutAssigned(type))
      return op->emitOpError(
          "requires either all or no VMI data values to carry layout");
    if (anyLayout && type.getLayout() != first.getLayout())
      return op->emitOpError("requires all layout-assigned VMI data values to "
                             "have the same layout");
  }
  return success();
}

static LogicalResult verifyElementwiseVRegOp(Operation *op, VMIVRegType lhs,
                                             VMIVRegType rhs,
                                             VMIVRegType result) {
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, result},
                                         /*requireSameElement=*/true);
}

static LogicalResult verifyFloatUnaryVRegOp(Operation *op, VMIVRegType source,
                                            VMIVRegType result) {
  if (!isVMIFloatLikeType(source.getElementType()))
    return op->emitOpError("requires floating-point-like VMI element type");
  return verifyAllSameVRegShapeAndLayout(op, {source, result},
                                         /*requireSameElement=*/true);
}

static LogicalResult verifyFloatTernaryVRegOp(Operation *op, VMIVRegType lhs,
                                              VMIVRegType rhs, VMIVRegType acc,
                                              VMIVRegType result) {
  if (!isVMIFloatLikeType(lhs.getElementType()))
    return op->emitOpError("requires floating-point-like VMI element type");
  return verifyAllSameVRegShapeAndLayout(op, {lhs, rhs, acc, result},
                                         /*requireSameElement=*/true);
}

static LogicalResult
verifyAllSameMaskShapeLayoutAndGranularity(Operation *op,
                                           ArrayRef<VMIMaskType> types) {
  if (types.empty())
    return success();

  VMIMaskType first = types.front();
  bool anyLayout = llvm::any_of(
      types, [](VMIMaskType type) { return isLayoutAssigned(type); });

  for (VMIMaskType type : types) {
    if (type.getElementCount() != first.getElementCount())
      return op->emitOpError(
          "requires all VMI mask values to have the same logical lane count");
    if (type.getGranularity() != first.getGranularity())
      return op->emitOpError(
          "requires all VMI mask values to have the same granularity");
    if (anyLayout && !isLayoutAssigned(type))
      return op->emitOpError(
          "requires either all or no VMI mask values to carry layout");
    if (anyLayout && type.getLayout() != first.getLayout())
      return op->emitOpError(
          "requires all layout-assigned VMI mask values to have the same "
          "layout");
  }
  return success();
}

static LogicalResult verifyMaskMatchesData(Operation *op, VMIMaskType maskType,
                                           VMIVRegType dataType) {
  if (maskType.getElementCount() != dataType.getElementCount())
    return op->emitOpError(
        "requires mask logical lane count to match data lane count");

  if (isLayoutAssigned(maskType) || isLayoutAssigned(dataType)) {
    if (!isLayoutAssigned(maskType) || !isLayoutAssigned(dataType))
      return op->emitOpError("requires either both mask and data to carry "
                             "layout or neither to carry layout");
    if (maskType.getLayout() != dataType.getLayout())
      return op->emitOpError("requires mask layout to match data layout");
  }

  if (maskType.isPred())
    return success();

  unsigned elementBitWidth = getVMIElementBitWidth(dataType.getElementType());
  int64_t maskBitWidth = getMaskGranularityBitWidth(maskType.getGranularity());
  if (elementBitWidth != 0 && maskBitWidth != 0 &&
      elementBitWidth != static_cast<unsigned>(maskBitWidth))
    return op->emitOpError(
        "requires mask granularity to match data element width");

  return success();
}

static Type getMemoryElementType(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getElementType();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getElementType();
  return {};
}

static LogicalResult verifyMemoryElementMatches(Operation *op, Type memoryType,
                                                VMIVRegType dataType,
                                                StringRef role) {
  Type memoryElementType = getMemoryElementType(memoryType);
  if (!memoryElementType)
    return success();
  if (memoryElementType != dataType.getElementType())
    return op->emitOpError() << "requires memory " << role
                             << " element type to match VMI data element type";
  return success();
}

static LogicalResult verifyNumGroups(Operation *op, VMIVRegType type,
                                     int64_t numGroups) {
  if (numGroups <= 0)
    return op->emitOpError("requires num_groups to be positive");
  if (type.getElementCount() % numGroups != 0)
    return op->emitOpError()
           << "requires num_groups to evenly divide VMI logical lane count "
           << type.getElementCount();
  return success();
}

static LogicalResult verifyPhysicalParts(Operation *op, Type vmiType,
                                         TypeRange physicalTypes) {
  FailureOr<int64_t> expectedArity = getVMIPhysicalArity(vmiType);
  if (failed(expectedArity))
    return op->emitOpError(
        "requires a layout-assigned VMI type with computable physical arity");
  if (static_cast<int64_t>(physicalTypes.size()) != *expectedArity)
    return op->emitOpError() << "requires " << *expectedArity
                             << " physical parts, got " << physicalTypes.size();

  if (auto vregType = dyn_cast<VMIVRegType>(vmiType)) {
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(vregType.getElementType());
    if (failed(lanesPerPart))
      return op->emitOpError(
          "requires data element type with known physical lane count");
    for (Type physicalType : physicalTypes) {
      auto partType = dyn_cast<VRegType>(physicalType);
      if (!partType)
        return op->emitOpError("requires physical data parts to be !pto.vreg");
      if (partType.getElementCount() != *lanesPerPart ||
          partType.getElementType() != vregType.getElementType())
        return op->emitOpError(
            "requires physical data part type to match VMI lane-map helper");
    }
    return success();
  }

  auto maskType = dyn_cast<VMIMaskType>(vmiType);
  if (!maskType)
    return op->emitOpError("requires VMI data or mask type");
  if (maskType.isPred())
    return op->emitOpError(
        "requires layout-assigned mask with concrete granularity");

  for (Type physicalType : physicalTypes) {
    auto partType = dyn_cast<MaskType>(physicalType);
    if (!partType)
      return op->emitOpError("requires physical mask parts to be !pto.mask");
    if (partType.getGranularity() != maskType.getGranularity())
      return op->emitOpError(
          "requires physical mask part granularity to match VMI mask");
  }
  return success();
}

static std::optional<int64_t>
mapDenseLogicalLaneToPartIndex(int64_t elementCount, int64_t factor,
                               int64_t blockElems, int64_t logicalLane,
                               int64_t &part) {
  if (logicalLane < 0 || logicalLane >= elementCount || factor <= 0 ||
      blockElems <= 0)
    return std::nullopt;
  int64_t block = logicalLane / blockElems;
  int64_t inBlockLane = logicalLane % blockElems;
  part = block % factor;
  int64_t partBlock = block / factor;
  return partBlock * blockElems + inBlockLane;
}

static std::optional<int64_t>
mapDensePartIndexToLogicalLane(int64_t elementCount, int64_t factor,
                               int64_t blockElems, int64_t part,
                               int64_t indexInPart) {
  if (part < 0 || part >= factor || indexInPart < 0 || factor <= 0 ||
      blockElems <= 0)
    return std::nullopt;
  int64_t partBlock = indexInPart / blockElems;
  int64_t inBlockLane = indexInPart % blockElems;
  int64_t logicalBlock = partBlock * factor + part;
  int64_t logicalLane = logicalBlock * blockElems + inBlockLane;
  if (logicalLane >= elementCount)
    return std::nullopt;
  return logicalLane;
}

static int64_t getDenseLogicalLanesInPart(int64_t elementCount, int64_t factor,
                                          int64_t blockElems, int64_t part) {
  int64_t maxIndex = -1;
  for (int64_t lane = 0; lane < elementCount; ++lane) {
    int64_t lanePart = 0;
    std::optional<int64_t> index = mapDenseLogicalLaneToPartIndex(
        elementCount, factor, blockElems, lane, lanePart);
    if (index && lanePart == part)
      maxIndex = std::max(maxIndex, *index);
  }
  return maxIndex + 1;
}

} // namespace

VMILayoutAttr VMILayoutAttr::getContiguous(MLIRContext *context) {
  return VMILayoutAttr::get(context, "contiguous", 1, 1, 0);
}

VMILayoutAttr VMILayoutAttr::getDeinterleaved(MLIRContext *context,
                                              int64_t factor,
                                              int64_t blockElems) {
  return VMILayoutAttr::get(context, "deinterleaved", factor, blockElems, 0);
}

VMILayoutAttr VMILayoutAttr::getGroupSlots(MLIRContext *context,
                                           int64_t numGroups, int64_t slots) {
  return VMILayoutAttr::get(context, "num_groups", numGroups, 1, slots);
}

Attribute VMILayoutAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef kind;
  int64_t factor = 1;
  int64_t blockElems = 1;
  int64_t slots = 0;

  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&kind)))
    return {};

  if (kind == "contiguous") {
    factor = 1;
  } else if (kind == "deinterleaved") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor)))
      return {};
    if (succeeded(parser.parseOptionalComma())) {
      StringRef field;
      if (failed(parser.parseKeyword(&field)) || field != "block_elems" ||
          failed(parser.parseEqual()) ||
          failed(parser.parseInteger(blockElems))) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected 'block_elems = <integer>'");
        return {};
      }
    }
  } else if (kind == "num_groups") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor)))
      return {};
    if (succeeded(parser.parseOptionalComma())) {
      StringRef field;
      if (failed(parser.parseKeyword(&field)) || field != "slots" ||
          failed(parser.parseEqual()) || failed(parser.parseInteger(slots))) {
        parser.emitError(parser.getCurrentLocation(),
                         "expected 'slots = <integer>'");
        return {};
      }
    }
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected VMI layout kind 'contiguous' or "
                     "'deinterleaved' or 'num_groups'");
    return {};
  }

  if (failed(parser.parseGreater()))
    return {};

  return parser.getChecked<VMILayoutAttr>(loc, parser.getContext(), kind,
                                          factor, blockElems, slots);
}

void VMILayoutAttr::print(AsmPrinter &printer) const {
  printer << "<" << getKind();
  if (isDeinterleaved()) {
    printer << " = " << getFactor();
    if (getBlockElems() != 1)
      printer << ", block_elems = " << getBlockElems();
  } else if (isGroupSlots()) {
    printer << " = " << getFactor();
    if (getSlots() != 0)
      printer << ", slots = " << getSlots();
  }
  printer << ">";
}

LogicalResult
VMILayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      StringRef kind, int64_t factor, int64_t blockElems,
                      int64_t slots) {
  if (kind == "contiguous") {
    if (factor != 1 || blockElems != 1 || slots != 0)
      return emitError()
             << "#pto.vmi.layout<contiguous> requires factor, block_elems, "
                "and slots to be their defaults";
    return success();
  }

  if (kind == "deinterleaved") {
    if (factor != 2 && factor != 4)
      return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                         << "> expected factor to be 2 or 4";
    if (blockElems <= 0)
      return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                         << ", block_elems = " << blockElems
                         << "> requires block_elems to be positive";
    if (slots != 0)
      return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                         << "> requires slots to be omitted";
    return success();
  }

  if (kind == "num_groups") {
    if (factor <= 0)
      return emitError() << "#pto.vmi.layout<num_groups = " << factor
                         << "> requires num_groups to be positive";
    if (blockElems != 1)
      return emitError() << "#pto.vmi.layout<num_groups = " << factor
                         << "> requires block_elems to be 1";
    if (slots < 0)
      return emitError() << "#pto.vmi.layout<num_groups = " << factor
                         << ", slots = " << slots
                         << "> requires slots to be omitted or positive";
    return success();
  }

  return emitError() << "expected VMI layout kind to be 'contiguous' or "
                        "'deinterleaved' or 'num_groups'";
}

Type VMIVRegType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  Type elementType;
  Attribute layout;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseType(elementType)) ||
      failed(parseOptionalVMILayout(parser, layout)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<VMIVRegType>(loc, parser.getContext(), shape.front(),
                                        elementType, layout);
}

void VMIVRegType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  if (getLayout())
    printer << ", " << getLayout();
  printer << ">";
}

LogicalResult VMIVRegType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  int64_t elementCount, Type elementType,
                                  Attribute layout) {
  if (elementCount <= 0)
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected a positive element count";

  if (!isSupportedVMIElementType(elementType))
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected an integer, index, floating-point, or "
                          "PTO low-precision element type";

  if (layout && !mlir::isa<VMILayoutAttr>(layout))
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected layout to be #pto.vmi.layout";
  if (auto layoutAttr = llvm::dyn_cast_or_null<VMILayoutAttr>(layout)) {
    if (layoutAttr.isGroupSlots() &&
        elementCount % layoutAttr.getNumGroups() != 0)
      return emitError() << "'"
                         << formatVMIVRegType(elementCount, elementType, layout)
                         << "' expected num_groups layout to evenly divide "
                            "the VMI logical lane count";
  }

  return success();
}

bool VMIMaskType::isSupportedGranularity(StringRef granularity) {
  return granularity == "pred" || isConcreteGranularity(granularity);
}

bool VMIMaskType::isConcreteGranularity(StringRef granularity) {
  return granularity == "b8" || granularity == "b16" || granularity == "b32";
}

Type VMIMaskType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  StringRef granularity;
  Attribute layout;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseKeyword(&granularity)) ||
      failed(parseOptionalVMILayout(parser, layout)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<VMIMaskType>(loc, parser.getContext(), shape.front(),
                                        granularity, layout);
}

void VMIMaskType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x" << getGranularity();
  if (getLayout())
    printer << ", " << getLayout();
  printer << ">";
}

LogicalResult VMIMaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  int64_t elementCount, StringRef granularity,
                                  Attribute layout) {
  if (elementCount <= 0)
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected a positive element count";

  if (!isSupportedGranularity(granularity))
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected granularity to be one of pred, b8, b16, "
                          "b32";

  if (layout && !mlir::isa<VMILayoutAttr>(layout))
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected layout to be #pto.vmi.layout";
  if (auto layoutAttr = llvm::dyn_cast_or_null<VMILayoutAttr>(layout)) {
    if (layoutAttr.isGroupSlots())
      return emitError() << "'"
                         << formatVMIMaskType(elementCount, granularity, layout)
                         << "' mask type must not carry num_groups layout";
  }

  if (granularity == "pred" && layout)
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' pred mask must not carry layout";

  if (granularity != "pred" && !layout)
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' concrete mask granularity requires layout";

  return success();
}

LogicalResult VMIConstantOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto denseAttr = dyn_cast<DenseElementsAttr>(getValue());
  if (!denseAttr)
    return emitOpError("requires dense elements constant attribute");
  if (denseAttr.getElementType() != resultType.getElementType())
    return emitOpError(
        "requires dense constant element type to match result element type");
  if (denseAttr.getNumElements() != resultType.getElementCount())
    return emitOpError("requires dense constant element count to match result "
                       "logical lane count");
  return success();
}

LogicalResult VMIBroadcastOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type valueType = getValue().getType();
  if (valueType == resultType.getElementType())
    return success();
  if (auto vregType = dyn_cast<VMIVRegType>(valueType)) {
    if (vregType.getElementCount() != 1)
      return emitOpError("requires VMI vector input to have one logical lane");
    if (vregType.getElementType() != resultType.getElementType())
      return emitOpError("requires VMI vector input element type to match "
                         "result element type");
    return success();
  }
  return emitOpError("requires scalar or VMI vector input element type to "
                     "match result element type");
}

LogicalResult VMIIotaOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  Type elementType = resultType.getElementType();
  if (!isVMIIotaElementType(elementType))
    return emitOpError("requires result element type to be integer 8/16/32 "
                       "or f16/f32");
  if (!isCompatibleScalarForSemanticType(elementType, getBase().getType()))
    return emitOpError("requires base type to match result element type");

  if (std::optional<StringRef> order = getOrder()) {
    if (*order != "ASC" && *order != "DESC")
      return emitOpError("requires order to be ASC or DESC");
  }
  return success();
}

LogicalResult VMICreateMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!resultType.isPred() && !isLayoutAssigned(resultType))
    return emitOpError("requires concrete mask result to carry layout");
  return success();
}

LogicalResult VMICreateGroupMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  int64_t numGroups = getNumGroupsAttr().getInt();
  int64_t groupSize = getGroupSizeAttr().getInt();
  if (numGroups <= 0)
    return emitOpError("requires positive num_groups");
  if (groupSize <= 0)
    return emitOpError("requires positive group_size");
  if (resultType.getElementCount() != numGroups * groupSize)
    return emitOpError("requires result lane count to equal num_groups * "
                       "group_size");
  if (!resultType.isPred() && !isLayoutAssigned(resultType))
    return emitOpError("requires concrete mask result to carry layout");
  return success();
}

LogicalResult VMIConstantMaskOp::verify() {
  auto resultType = cast<VMIMaskType>(getResult().getType());
  auto denseAttr = dyn_cast<DenseElementsAttr>(getValue());
  if (!denseAttr)
    return emitOpError("requires dense elements mask constant attribute");
  if (!denseAttr.getElementType().isInteger(1))
    return emitOpError("requires dense mask constant element type to be i1");
  if (denseAttr.getNumElements() != resultType.getElementCount())
    return emitOpError("requires dense mask constant element count to match "
                       "result logical lane count");
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
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIAddIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMISubFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMISubIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMulFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMulIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIFmaOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto accType = cast<VMIVRegType>(getAcc().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatTernaryVRegOp(getOperation(), lhsType, rhsType, accType,
                                  resultType);
}

LogicalResult VMIDivFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMinFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIMaxFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMINegFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIAbsFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIAbsIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMISqrtOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIExpOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMILnOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIReluOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  return verifyFloatUnaryVRegOp(getOperation(), sourceType, resultType);
}

LogicalResult VMIAndIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIOrIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIXOrIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIShLIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMIShRUIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto integerType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!integerType || integerType.isSigned())
    return emitOpError(
        "requires signless or unsigned integer VMI element type");
  return verifyElementwiseVRegOp(getOperation(), lhsType, rhsType, resultType);
}

LogicalResult VMINotOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  return verifyAllSameVRegShapeAndLayout(getOperation(),
                                         {sourceType, resultType},
                                         /*requireSameElement=*/true);
}

LogicalResult VMICmpFOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!isVMIFloatLikeType(lhsType.getElementType()))
    return emitOpError("requires floating-point-like VMI element type");
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), resultType, lhsType);
}

LogicalResult VMICmpIOp::verify() {
  auto lhsType = cast<VMIVRegType>(getLhs().getType());
  auto rhsType = cast<VMIVRegType>(getRhs().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (!isVMIIntegerLikeType(lhsType.getElementType()))
    return emitOpError("requires integer-like VMI element type");
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(), {lhsType, rhsType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), resultType, lhsType);
}

LogicalResult VMISelectOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto trueType = cast<VMIVRegType>(getTrueValue().getType());
  auto falseType = cast<VMIVRegType>(getFalseValue().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {trueType, falseType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

LogicalResult VMIActivePrefixIndexOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  auto resultIntType = dyn_cast<IntegerType>(resultType.getElementType());
  if (!resultIntType || !resultIntType.isSignless())
    return emitOpError("requires signless integer result element type");
  unsigned resultWidth = resultIntType.getWidth();
  if (resultWidth != 8 && resultWidth != 16 && resultWidth != 32)
    return emitOpError("requires i8, i16, or i32 result element type");
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

LogicalResult VMICompressOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {sourceType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

LogicalResult VMICompressStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination")))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMICompressStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIReduceAddIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto initType = cast<VMIVRegType>(getInit().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType()))
    return emitOpError("requires integer-like VMI source element type");
  if (sourceType.getElementType() != initType.getElementType() ||
      sourceType.getElementType() != resultType.getElementType())
    return emitOpError(
        "requires source, init, and result element types to match");
  if (initType.getElementCount() != 1 || resultType.getElementCount() != 1)
    return emitOpError("requires init and result to be rank-0 VMI vectors");
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {initType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceAddFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto initType = cast<VMIVRegType>(getInit().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!getOperation()->hasAttr("reassoc"))
    return emitOpError(
        "requires reassoc attr because VPTO vcadd performs pair-wise "
        "floating-point reduction");
  if (!isVMIFloatLikeType(sourceType.getElementType()))
    return emitOpError("requires floating-point-like VMI source element type");
  if (sourceType.getElementType() != initType.getElementType() ||
      sourceType.getElementType() != resultType.getElementType())
    return emitOpError(
        "requires source, init, and result element types to match");
  if (initType.getElementCount() != 1 || resultType.getElementCount() != 1)
    return emitOpError("requires init and result to be rank-0 VMI vectors");
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {initType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, sourceType);
}

template <typename OpTy> LogicalResult verifyReduceMinMaxFOp(OpTy op) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto initType = cast<VMIVRegType>(op.getInit().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!isVMIFloatLikeType(sourceType.getElementType()))
    return op.emitOpError(
        "requires floating-point-like VMI source element type");
  if (sourceType.getElementType() != initType.getElementType() ||
      sourceType.getElementType() != resultType.getElementType())
    return op.emitOpError(
        "requires source, init, and result element types to match");
  if (initType.getElementCount() != 1 || resultType.getElementCount() != 1)
    return op.emitOpError("requires init and result to be rank-0 VMI vectors");
  if (failed(verifyAllSameVRegShapeAndLayout(op.getOperation(),
                                             {initType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(op.getOperation(), maskType, sourceType);
}

LogicalResult VMIReduceMaxFOp::verify() { return verifyReduceMinMaxFOp(*this); }

LogicalResult VMIReduceMinFOp::verify() { return verifyReduceMinMaxFOp(*this); }

template <typename OpTy>
static LogicalResult verifyGroupReduceFloatOp(OpTy op, bool requiresReassoc) {
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (requiresReassoc && !op->hasAttr("reassoc"))
    return op.emitOpError(
        "requires reassoc attr because grouped lowering uses pair-wise "
        "floating-point reductions");
  if (!isVMIFloatLikeType(sourceType.getElementType()))
    return op.emitOpError(
        "requires floating-point-like VMI source element type");
  if (sourceType.getElementCount() != resultType.getElementCount())
    return op.emitOpError(
        "requires source and result logical lane counts to match");
  if (sourceType.getElementType() != resultType.getElementType())
    return op.emitOpError("requires source and result element types to match");
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    bool supportedSourceLayout =
        sourceLayout.isContiguous() ||
        (sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 2 &&
         (sourceLayout.getBlockElems() == 1 ||
          sourceLayout.getBlockElems() == 8)) ||
        (sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 4 &&
         (sourceLayout.getBlockElems() == 1 ||
          sourceLayout.getBlockElems() == 8));
    if (!supportedSourceLayout)
      return op.emitOpError(
          "requires layout-assigned source to use contiguous layout or "
          "deinterleaved=2/4 layout with block_elems=1 or block_elems=8");
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != op.getNumGroupsAttr().getInt())
      return op.emitOpError() << "requires layout-assigned result to use "
                                 "#pto.vmi.layout<num_groups = "
                              << op.getNumGroupsAttr().getInt() << ">";
  }
  if (failed(verifyMaskMatchesData(op.getOperation(), maskType, sourceType)))
    return failure();
  return verifyNumGroups(op.getOperation(), sourceType,
                         op.getNumGroupsAttr().getInt());
}

LogicalResult VMIGroupReduceAddFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/true);
}

LogicalResult VMIGroupReduceMaxFOp::verify() {
  return verifyGroupReduceFloatOp(*this, /*requiresReassoc=*/false);
}

LogicalResult VMIGroupReduceAddIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (!isVMIIntegerLikeType(sourceType.getElementType()))
    return emitOpError("requires integer-like VMI source element type");
  auto intType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!intType || intType.getWidth() != 32)
    return emitOpError(
        "requires i32 accumulator element type; cast i8/i16 storage to i32 "
        "before grouped reduction because integer reduction widens narrow "
        "inputs");
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("requires source and result element types to match");
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    bool supportedSourceLayout =
        sourceLayout.isContiguous() ||
        (sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 2 &&
         (sourceLayout.getBlockElems() == 1 ||
          sourceLayout.getBlockElems() == 8)) ||
        (sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 4 &&
         (sourceLayout.getBlockElems() == 1 ||
          sourceLayout.getBlockElems() == 8));
    if (!supportedSourceLayout)
      return emitOpError(
          "requires layout-assigned source to use contiguous layout or "
          "deinterleaved=2/4 layout with block_elems=1 or block_elems=8");
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != getNumGroupsAttr().getInt())
      return emitOpError() << "requires layout-assigned result to use "
                              "#pto.vmi.layout<num_groups = "
                           << getNumGroupsAttr().getInt() << ">";
  }
  if (failed(verifyMaskMatchesData(getOperation(), maskType, sourceType)))
    return failure();
  return verifyNumGroups(getOperation(), sourceType,
                         getNumGroupsAttr().getInt());
}

LogicalResult VMIGroupBroadcastOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError("requires source and result element types to match");
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    if (!sourceLayout.isGroupSlots() ||
        sourceLayout.getNumGroups() != getNumGroupsAttr().getInt())
      return emitOpError() << "requires layout-assigned source to use "
                              "#pto.vmi.layout<num_groups = "
                           << getNumGroupsAttr().getInt() << ">";
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (resultLayout.isGroupSlots())
      return emitOpError(
          "requires layout-assigned result to use a dense VMI layout");
  }
  return verifyNumGroups(getOperation(), sourceType,
                         getNumGroupsAttr().getInt());
}

template <typename OpTy> static LogicalResult verifyVMIHistogramOp(OpTy op) {
  auto accType = cast<VMIVRegType>(op.getAcc().getType());
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());

  auto accElemType = dyn_cast<IntegerType>(accType.getElementType());
  auto sourceElemType = dyn_cast<IntegerType>(sourceType.getElementType());
  if (!accElemType || !accElemType.isUnsigned() ||
      accElemType.getWidth() != 16 || accType.getElementCount() != 256)
    return op.emitOpError("requires acc type to be "
                          "!pto.vmi.vreg<256xui16>");
  if (resultType != accType)
    return op.emitOpError("requires result type to match acc type");
  if (!sourceElemType || !sourceElemType.isUnsigned() ||
      sourceElemType.getWidth() != 8)
    return op.emitOpError("requires source type to be "
                          "!pto.vmi.vreg<Nxui8>");
  if (maskType.getElementCount() != sourceType.getElementCount())
    return op.emitOpError("requires mask logical lane count to match source");

  if (auto accLayout = accType.getLayoutAttr()) {
    if (!accLayout.isContiguous())
      return op.emitOpError("requires layout-assigned acc to use contiguous "
                            "layout");
  }
  if (auto sourceLayout = sourceType.getLayoutAttr()) {
    if (!sourceLayout.isContiguous())
      return op.emitOpError("requires layout-assigned source to use contiguous "
                            "layout");
  }
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isContiguous())
      return op.emitOpError("requires layout-assigned result to use "
                            "contiguous layout");
  }
  if (auto maskLayout = maskType.getLayoutAttr()) {
    if (!maskLayout.isContiguous())
      return op.emitOpError("requires layout-assigned mask to use contiguous "
                            "layout");
    if (maskType.getGranularity() != "b8")
      return op.emitOpError("requires layout-assigned mask granularity b8");
  }
  return success();
}

LogicalResult VMIDhistOp::verify() { return verifyVMIHistogramOp(*this); }

LogicalResult VMIChistOp::verify() { return verifyVMIHistogramOp(*this); }

LogicalResult VMIExtFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (!isVMIFloatLikeType(sourceType.getElementType()) ||
      !isVMIFloatLikeType(resultType.getElementType()))
    return emitOpError(
        "requires floating-point-like source and result element types");
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType()))
    return emitOpError(
        "requires result element type to be wider than source element type");
  return success();
}

LogicalResult VMITruncFOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (!isVMIFloatLikeType(sourceType.getElementType()) ||
      !isVMIFloatLikeType(resultType.getElementType()))
    return emitOpError(
        "requires floating-point-like source and result element types");
  if (getVMIElementBitWidth(sourceType.getElementType()) <=
      getVMIElementBitWidth(resultType.getElementType()))
    return emitOpError(
        "requires result element type to be narrower than source element type");
  return success();
}

LogicalResult VMIExtSIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (!isVMISignedOrSignlessIntegerType(sourceType.getElementType()) ||
      !isVMISignedOrSignlessIntegerType(resultType.getElementType()))
    return emitOpError(
        "requires signed or signless integer source and result element types");
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType()))
    return emitOpError(
        "requires result element type to be wider than source element type");
  return success();
}

LogicalResult VMIExtUIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (!isVMIUnsignedIntegerType(sourceType.getElementType()) ||
      !isVMIUnsignedIntegerType(resultType.getElementType()))
    return emitOpError(
        "requires unsigned integer source and result element types");
  if (getVMIElementBitWidth(sourceType.getElementType()) >=
      getVMIElementBitWidth(resultType.getElementType()))
    return emitOpError(
        "requires result element type to be wider than source element type");
  return success();
}

LogicalResult VMITruncIOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result logical lane counts to match");
  if (!isVMIIntegerLikeType(sourceType.getElementType()) ||
      !isVMIIntegerLikeType(resultType.getElementType()))
    return emitOpError("requires integer source and result element types");
  if (getVMIElementBitWidth(sourceType.getElementType()) <=
      getVMIElementBitWidth(resultType.getElementType()))
    return emitOpError(
        "requires result element type to be narrower than source element type");
  return success();
}

LogicalResult VMIBitcastOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  std::optional<unsigned> sourceBits =
      getVMIIntegerOrFloatBitWidth(sourceType.getElementType());
  std::optional<unsigned> resultBits =
      getVMIIntegerOrFloatBitWidth(resultType.getElementType());
  if (!sourceBits || !resultBits)
    return emitOpError(
        "requires integer or floating-point source and result element types");
  if (sourceType.getElementCount() * static_cast<int64_t>(*sourceBits) !=
      resultType.getElementCount() * static_cast<int64_t>(*resultBits))
    return emitOpError(
        "requires source and result to carry the same total number of bits");

  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType))
      return emitOpError(
          "requires either both source and result to carry layout or neither "
          "to carry layout");
    if (sourceType.getLayout() != resultType.getLayout())
      return emitOpError("requires source and result layouts to match");
  }

  return success();
}

LogicalResult VMILoadOp::verify() {
  return verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                    cast<VMIVRegType>(getResult().getType()),
                                    "source");
}

void VMILoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGroupLoadOp::verify() {
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source")))
    return failure();
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
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source")))
    return failure();
  if (auto resultLayout = resultType.getLayoutAttr()) {
    if (!resultLayout.isGroupSlots() ||
        resultLayout.getNumGroups() != getNumGroupsAttr().getInt())
      return emitOpError() << "requires layout-assigned result to use "
                              "#pto.vmi.layout<num_groups = "
                           << getNumGroupsAttr().getInt() << ">";
  }
  return verifyNumGroups(getOperation(), resultType,
                         getNumGroupsAttr().getInt());
}

void VMIGroupSlotLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIMaskedLoadOp::verify() {
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto passthruType = cast<VMIVRegType>(getPassthru().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source")))
    return failure();
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIMaskedLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIGatherOp::verify() {
  auto indicesType = cast<VMIVRegType>(getIndices().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  auto passthruType = cast<VMIVRegType>(getPassthru().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (failed(verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                        resultType, "source")))
    return failure();

  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.getWidth() != 32 ||
      indexElementType.isSigned())
    return emitOpError("requires signless or unsigned 32-bit integer indices");

  if (failed(verifyAllSameVRegShapeAndLayout(
          getOperation(), {indicesType, passthruType, resultType},
          /*requireSameElement=*/false)))
    return failure();
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
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
                                        resultType, "source")))
    return failure();
  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {passthruType, resultType},
                                             /*requireSameElement=*/true)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, resultType);
}

void VMIExpandLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMIStoreOp::verify() {
  return verifyMemoryElementMatches(getOperation(), getDestination().getType(),
                                    cast<VMIVRegType>(getValue().getType()),
                                    "destination");
}

void VMIStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIGroupStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination")))
    return failure();
  return verifyNumGroups(getOperation(), valueType,
                         getNumGroupsAttr().getInt());
}

void VMIGroupStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIMaskedStoreOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination")))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIMaskedStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIScatterOp::verify() {
  auto valueType = cast<VMIVRegType>(getValue().getType());
  auto indicesType = cast<VMIVRegType>(getIndices().getType());
  auto maskType = cast<VMIMaskType>(getMask().getType());
  if (failed(verifyMemoryElementMatches(getOperation(),
                                        getDestination().getType(), valueType,
                                        "destination")))
    return failure();

  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.getWidth() != 32 ||
      indexElementType.isSigned())
    return emitOpError("requires signless or unsigned 32-bit integer indices");

  if (failed(verifyAllSameVRegShapeAndLayout(getOperation(),
                                             {valueType, indicesType},
                                             /*requireSameElement=*/false)))
    return failure();
  return verifyMaskMatchesData(getOperation(), maskType, valueType);
}

void VMIScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMITileReadOp::verify() {
  return verifyMemoryElementMatches(getOperation(), getSource().getType(),
                                    cast<VMIVRegType>(getResult().getType()),
                                    "source");
}

void VMITileReadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VMITileWriteOp::verify() {
  return verifyMemoryElementMatches(getOperation(), getDestination().getType(),
                                    cast<VMIVRegType>(getValue().getType()),
                                    "destination");
}

void VMITileWriteOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VMIShuffleOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType())
    return emitOpError(
        "requires result element type to match source element type");
  if (static_cast<int64_t>(getIndices().size()) != resultType.getElementCount())
    return emitOpError(
        "requires shuffle index count to match result logical lane count");
  for (int64_t index : getIndices()) {
    if (index < 0 || index >= sourceType.getElementCount())
      return emitOpError("requires every shuffle index to select an existing "
                         "source logical lane");
  }
  if (isLayoutAssigned(sourceType) || isLayoutAssigned(resultType)) {
    if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType))
      return emitOpError("requires either both source and result to carry "
                         "layout or neither to carry layout");
  }
  return success();
}

LogicalResult VMIChannelSplitOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  if (getResults().size() < 2)
    return emitOpError("requires at least two channel results");
  auto firstResultType = cast<VMIVRegType>(getResults().front().getType());
  if (sourceType.getElementCount() !=
      static_cast<int64_t>(getResults().size()) *
          firstResultType.getElementCount())
    return emitOpError("requires source lane count to equal result count times "
                       "per-channel lane count");
  for (Value result : getResults()) {
    auto resultType = cast<VMIVRegType>(result.getType());
    if (resultType.getElementCount() != firstResultType.getElementCount() ||
        resultType.getElementType() != sourceType.getElementType())
      return emitOpError("requires every channel result to have equal lane "
                         "count and source element type");
  }
  bool anyLayout = isLayoutAssigned(sourceType);
  for (Value result : getResults())
    anyLayout |= isLayoutAssigned(cast<VMIVRegType>(result.getType()));
  if (anyLayout) {
    if (!isLayoutAssigned(sourceType))
      return emitOpError("requires layout-assigned channel_split source when "
                         "any channel result has layout");
    for (Value result : getResults()) {
      auto resultType = cast<VMIVRegType>(result.getType());
      if (!isLayoutAssigned(resultType))
        return emitOpError("requires every channel_split result to carry "
                           "layout when source has layout");
      if (!cast<VMILayoutAttr>(resultType.getLayout()).isContiguous())
        return emitOpError(
            "requires layout-assigned channel_split results to be contiguous");
    }
    int64_t channels = getResults().size();
    if (channels == 2 || channels == 4) {
      auto sourceLayout = cast<VMILayoutAttr>(sourceType.getLayout());
      auto expectedLayout =
          VMILayoutAttr::getDeinterleaved(getContext(), channels);
      if (!sourceLayout.isContiguous() && sourceLayout != expectedLayout)
        return emitOpError("requires layout-assigned channel_split source to "
                           "be contiguous or deinterleaved by result count");
    }
  }
  return success();
}

LogicalResult VMIChannelMergeOp::verify() {
  if (getInputs().size() < 2)
    return emitOpError("requires at least two channel inputs");
  auto firstInputType = cast<VMIVRegType>(getInputs().front().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  for (Value input : getInputs()) {
    auto inputType = cast<VMIVRegType>(input.getType());
    if (inputType.getElementCount() != firstInputType.getElementCount() ||
        inputType.getElementType() != firstInputType.getElementType())
      return emitOpError("requires all channel inputs to have the same lane "
                         "count and element type");
  }
  if (resultType.getElementCount() != static_cast<int64_t>(getInputs().size()) *
                                          firstInputType.getElementCount() ||
      resultType.getElementType() != firstInputType.getElementType())
    return emitOpError(
        "requires result lane count and element type to match merged channels");
  bool anyLayout = isLayoutAssigned(resultType);
  for (Value input : getInputs())
    anyLayout |= isLayoutAssigned(cast<VMIVRegType>(input.getType()));
  if (anyLayout) {
    if (!isLayoutAssigned(resultType))
      return emitOpError("requires layout-assigned channel_merge result when "
                         "any channel input has layout");
    for (Value input : getInputs()) {
      auto inputType = cast<VMIVRegType>(input.getType());
      if (!isLayoutAssigned(inputType))
        return emitOpError("requires every channel_merge input to carry layout "
                           "when result has layout");
      if (!cast<VMILayoutAttr>(inputType.getLayout()).isContiguous())
        return emitOpError(
            "requires layout-assigned channel_merge inputs to be contiguous");
    }
    int64_t channels = getInputs().size();
    if (channels == 2 || channels == 4) {
      auto resultLayout = cast<VMILayoutAttr>(resultType.getLayout());
      auto expectedLayout =
          VMILayoutAttr::getDeinterleaved(getContext(), channels);
      if (!resultLayout.isContiguous() && resultLayout != expectedLayout)
        return emitOpError("requires layout-assigned channel_merge result to "
                           "be contiguous or deinterleaved by input count");
    }
  }
  return success();
}

LogicalResult VMIEnsureLayoutOp::verify() {
  auto sourceType = cast<VMIVRegType>(getSource().getType());
  auto resultType = cast<VMIVRegType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount() ||
      sourceType.getElementType() != resultType.getElementType())
    return emitOpError("requires source and result to preserve VMI data shape "
                       "and element type");
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType))
    return emitOpError("requires source and result to be layout-assigned");
  return success();
}

LogicalResult VMIEnsureMaskLayoutOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount() ||
      sourceType.getGranularity() != resultType.getGranularity())
    return emitOpError("requires source and result to preserve VMI mask shape "
                       "and granularity");
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType))
    return emitOpError("requires source and result to be layout-assigned");
  return success();
}

LogicalResult VMIEnsureMaskGranularityOp::verify() {
  auto sourceType = cast<VMIMaskType>(getSource().getType());
  auto resultType = cast<VMIMaskType>(getResult().getType());
  if (sourceType.getElementCount() != resultType.getElementCount())
    return emitOpError(
        "requires source and result to preserve VMI mask lane count");
  if (!isLayoutAssigned(sourceType) || !isLayoutAssigned(resultType))
    return emitOpError("requires source and result to be layout-assigned");
  if (sourceType.getLayout() != resultType.getLayout())
    return emitOpError("requires source and result mask layouts to match");
  if (sourceType.isPred() || resultType.isPred())
    return emitOpError(
        "requires concrete source and result mask granularities");
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

FailureOr<int64_t> mlir::pto::getDataLanesPerPart(Type elementType) {
  unsigned elementBitWidth = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBitWidth == 0)
    return failure();
  constexpr int64_t kPhysicalVRegBits = 256 * 8;
  if (kPhysicalVRegBits % elementBitWidth != 0)
    return failure();
  return kPhysicalVRegBits / elementBitWidth;
}

FailureOr<int64_t> mlir::pto::getMaskLanesPerPart(StringRef granularity) {
  if (granularity == "b8")
    return 256;
  if (granularity == "b16")
    return 128;
  if (granularity == "b32")
    return 64;
  return failure();
}

FailureOr<int64_t> mlir::pto::getVMIPhysicalArity(Type type) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  FailureOr<VMILayoutAttr> layout = getAssignedVMILayout(type);
  if (failed(elementCount) || failed(lanesPerPart) || failed(layout))
    return failure();

  if ((*layout).isGroupSlots() && (*layout).getSlots() > 0)
    return divideCeilNonNegative((*layout).getNumGroups(),
                                 (*layout).getSlots());

  int64_t factor = (*layout).isDeinterleaved() ? (*layout).getFactor() : 1;
  int64_t blockElems =
      (*layout).isDeinterleaved() ? (*layout).getBlockElems() : 1;
  int64_t arity = 0;
  for (int64_t part = 0; part < factor; ++part) {
    int64_t lanesInPart =
        getDenseLogicalLanesInPart(*elementCount, factor, blockElems, part);
    arity += divideCeilNonNegative(lanesInPart, *lanesPerPart);
  }
  return arity;
}

FailureOr<VMIPhysicalLane>
mlir::pto::mapLogicalLaneToPhysical(Type type, int64_t logicalLane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(lanesPerPart))
    return failure();
  if (logicalLane < 0 || logicalLane >= *elementCount)
    return failure();

  int64_t part = 0;
  std::optional<int64_t> indexInPart = mapDenseLogicalLaneToPartIndex(
      *elementCount, *factor, *blockElems, logicalLane, part);
  if (!indexInPart)
    return failure();
  return VMIPhysicalLane{part, *indexInPart / *lanesPerPart,
                         *indexInPart % *lanesPerPart};
}

FailureOr<int64_t> mlir::pto::mapPhysicalLaneToLogical(Type type, int64_t part,
                                                       int64_t chunk,
                                                       int64_t lane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(lanesPerPart))
    return failure();
  if (part < 0 || part >= *factor || chunk < 0 || lane < 0 ||
      lane >= *lanesPerPart)
    return failure();

  int64_t indexInPart = chunk * *lanesPerPart + lane;
  std::optional<int64_t> logicalLane = mapDensePartIndexToLogicalLane(
      *elementCount, *factor, *blockElems, part, indexInPart);
  if (!logicalLane)
    return failure();
  return *logicalLane;
}

FailureOr<bool> mlir::pto::isPaddingLane(Type type, int64_t part, int64_t chunk,
                                         int64_t lane) {
  FailureOr<int64_t> elementCount = getVMIElementCount(type);
  FailureOr<int64_t> factor = getLayoutFactor(type);
  FailureOr<int64_t> blockElems = getLayoutBlockElems(type);
  FailureOr<int64_t> lanesPerPart = getPhysicalLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(blockElems) ||
      failed(lanesPerPart))
    return failure();
  if (part < 0 || part >= *factor || chunk < 0 || lane < 0 ||
      lane >= *lanesPerPart)
    return failure();

  int64_t lanesInPart =
      getDenseLogicalLanesInPart(*elementCount, *factor, *blockElems, part);
  int64_t indexInPart = chunk * *lanesPerPart + lane;
  return indexInPart >= lanesInPart;
}
