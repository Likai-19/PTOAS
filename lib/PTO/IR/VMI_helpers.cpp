// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMI_helpers.cpp - VMI internal helper implementations ---------------===//
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


// ---------------------------------------------------------------------------
// FpToSi hardware contract (mirrors VPTO lookupVcvtContract fp→int rows)
// ---------------------------------------------------------------------------

namespace mlir::pto {
std::optional<VMIFpToSiContract>
lookupVMIFpToSiContract(Type srcElem, Type dstElem) {
  // Must be float → explicitly signed integer.
  if (!isVMIFloatLikeType(srcElem)) {
    return std::nullopt;
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || !dstInt.isSigned()) {
    return std::nullopt;
  }

  bool srcF32 = srcElem.isF32();
  bool srcF16 = srcElem.isF16();
  bool srcBF16 = srcElem.isBF16();
  unsigned dstBits = dstInt.getWidth();
  // f32 → s32: same width, rnd, sat, no part
  if (srcF32 && dstBits == 32) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/false};
  }
  // f32 → s16: narrow 2×, rnd, sat, part (EvenOdd)
  if (srcF32 && dstBits == 16) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }
  // f16 → s32: widen 2×, rnd, NO sat, part (EvenOdd)
  if (srcF16 && dstBits == 32) {
    return VMIFpToSiContract{/*requiresSat=*/false, /*requiresPart=*/true};
  }
  // f16 → s16: same width, rnd, sat, no part
  if (srcF16 && dstBits == 16) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/false};
  }
  // f16 → s8: narrow 2×, rnd, sat, part (EvenOdd)
  if (srcF16 && dstBits == 8) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }
  // bf16 → s32: widen 2×, rnd, sat, part (EvenOdd)
  if (srcBF16 && dstBits == 32) {
    return VMIFpToSiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// FpToUi hardware contract (mirrors VPTO lookupVcvtContract fp→uint rows)
// ---------------------------------------------------------------------------

std::optional<VMIFpToUiContract>
lookupVMIFpToUIContract(Type srcElem, Type dstElem) {
  // Must be float → unsigned integer. VMI signless integers carry
  // unsigned semantics.
  if (!isVMIFloatLikeType(srcElem)) {
    return std::nullopt;
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || dstInt.isSigned()) {
    return std::nullopt;
  }

  bool srcF16 = srcElem.isF16();
  unsigned dstBits = dstInt.getWidth();
  // f16 → u8: narrow 2×, rnd, sat, part (EvenOdd)
  if (srcF16 && dstBits == 8) {
    return VMIFpToUiContract{/*requiresSat=*/true, /*requiresPart=*/true};
  }

  return std::nullopt;
}

// ---------------------------------------------------------------------------
// FpToFp hardware contract (VMI-owned; may diverge from VPTO).
// Enumerates same-width fp->fp whitelist entries plus the fp->fp narrow
// paths whose sat/rounding semantics differ from the generic truncf default
// (e.g. bf16x2->f4x2 narrows with NO saturation).
// ---------------------------------------------------------------------------

std::optional<VMIFpToFpContract>
lookupVMIFpToFpContract(Type srcElem, Type dstElem) {
  if (!isVMIFloatLikeType(srcElem) || !isVMIFloatLikeType(dstElem)) {
    return std::nullopt;
  }
  unsigned srcBits = pto::getPTOStorageElemBitWidth(srcElem);
  unsigned dstBits = pto::getPTOStorageElemBitWidth(dstElem);
  // bf16x2 -> f4x2 (32->8 narrow): Packed4, rnd, NO sat. Mirrors the VPTO
  // bf16->f4 contract row (requiresSat=false) so the VMI verifier does not
  // force a saturate attribute that the physical pto.vcvt would reject.
  if (pto::isPTOBF16x2Type(srcElem) && pto::isPTOFloat4PackedType(dstElem)) {
    return VMIFpToFpContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                            /*requiresPart=*/true,
                            /*allowedRndModes=*/"RAFZC"};
}
  // f4x2 -> bf16x2 (8->32 widen): Packed4, no rnd, no sat. Mirrors the VPTO
  // f4->bf16 contract row (requiresSat=false, requiresRnd=false). Widen has
  // no rounding/saturate semantics by construction; the contract exists so
  // the involvesBF16x2 / packed-carrier gates in the VMI verifiers pass.
  if (pto::isPTOFloat4PackedType(srcElem) && pto::isPTOBF16x2Type(dstElem)) {
    return VMIFpToFpContract{/*requiresRnd=*/false, /*requiresSat=*/false,
                            /*requiresPart=*/true,
                            /*allowedRndModes=*/StringRef()};
  }
  if (srcBits != dstBits) {
    return std::nullopt;
  }
  // bf16 -> f16: same-width, rnd, sat, no part.
  if (srcElem.isBF16() && dstElem.isF16()) {
    return VMIFpToFpContract{/*requiresRnd=*/true, /*requiresSat=*/true,
                            /*requiresPart=*/false,
                            /*allowedRndModes=*/StringRef()};
  }
  // f16 -> bf16: same-width, rnd, NO sat, no part.
  bool srcIsF16 = srcElem.isF16();
  bool dstIsBF16 = dstElem.isBF16();
  if (srcIsF16 && dstIsBF16) {
    return VMIFpToFpContract{/*requiresRnd=*/true, /*requiresSat=*/false,
                            /*requiresPart=*/false,
                            /*allowedRndModes=*/StringRef()};
  }
  return std::nullopt;
}

} // namespace mlir::pto

VMILayoutAttr VMILayoutAttr::getContiguous(MLIRContext *context,
                                           int64_t laneStride) {
  return VMILayoutAttr::get(context, "contiguous", 1, 1, 0, laneStride);
}

VMILayoutAttr VMILayoutAttr::getDeinterleaved(MLIRContext *context,
                                              int64_t factor,
                                              int64_t laneStride) {
  return VMILayoutAttr::get(context, "deinterleaved", factor, 1, 0,
                            laneStride);
}

VMILayoutAttr VMILayoutAttr::getBlockDeinterleaved(MLIRContext *context,
                                                   int64_t factor) {
  return VMILayoutAttr::get(context, "block_deinterleaved", factor, 1, 0, 1);
}

VMILayoutAttr VMILayoutAttr::getGroupSlots(MLIRContext *context,
                                           int64_t numGroups, int64_t slots,
                                           int64_t laneStride) {
  return VMILayoutAttr::get(context, "num_groups", numGroups, 1, slots,
                            laneStride);
}

static ParseResult parseContiguousLayoutFields(AsmParser &parser,
                                               int64_t &laneStride) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef field;
    if (failed(parser.parseKeyword(&field)) || failed(parser.parseEqual()) ||
        field != "lane_stride" || failed(parser.parseInteger(laneStride))) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'lane_stride = <integer>'");
      return failure();
    }
  }
  return success();
}

static ParseResult parseDeinterleavedLayoutFields(AsmParser &parser,
                                                  int64_t &laneStride) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef field;
    if (failed(parser.parseKeyword(&field)) || failed(parser.parseEqual())) {
      return failure();
    }
    if (field == "lane_stride") {
      if (failed(parser.parseInteger(laneStride))) {
        return failure();
      }
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'lane_stride = <integer>'");
      return failure();
    }
  }
  return success();
}

static ParseResult parseNumGroupsLayoutFields(AsmParser &parser,
                                              int64_t &slots,
                                              int64_t &laneStride) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef field;
    if (failed(parser.parseKeyword(&field)) || failed(parser.parseEqual())) {
      return failure();
    }
    if (field == "slots") {
      if (failed(parser.parseInteger(slots))) {
        return failure();
      }
    } else if (field == "lane_stride") {
      if (failed(parser.parseInteger(laneStride))) {
        return failure();
      }
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "expected 'slots = <integer>' or "
                       "'lane_stride = <integer>'");
      return failure();
    }
  }
  return success();
}

Attribute VMILayoutAttr::parse(AsmParser &parser, Type) {
  SMLoc loc = parser.getCurrentLocation();
  StringRef kind;
  int64_t factor = 1;
  int64_t blockElems = 1;
  int64_t slots = 0;
  int64_t laneStride = 1;

  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&kind))) {
    return {};
  }

  if (kind == "contiguous") {
    factor = 1;
    if (failed(parseContiguousLayoutFields(parser, laneStride))) {
      return {};
    }
  } else if (kind == "deinterleaved") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor)) ||
        failed(parseDeinterleavedLayoutFields(parser, laneStride))) {
      return {};
    }
  } else if (kind == "block_deinterleaved") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor))) {
      return {};
    }
  } else if (kind == "num_groups") {
    if (failed(parser.parseEqual()) || failed(parser.parseInteger(factor)) ||
        failed(parseNumGroupsLayoutFields(parser, slots, laneStride))) {
      return {};
    }
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected VMI layout kind 'contiguous' or "
                     "'deinterleaved' or 'block_deinterleaved' or "
                     "'num_groups'");
    return {};
  }

  if (failed(parser.parseGreater())) {
    return {};
  }
  return parser.getChecked<VMILayoutAttr>(loc, parser.getContext(), kind,
                                          factor, blockElems, slots,
                                          laneStride);
}

void VMILayoutAttr::print(AsmPrinter &printer) const {
  printer << "<" << getKind();
  if (isContiguous()) {
    if (getLaneStride() != 1) {
      printer << ", lane_stride = " << getLaneStride();
    }
  } else if (isDeinterleaved()) {
    printer << " = " << getFactor();
    if (getLaneStride() != 1) {
      printer << ", lane_stride = " << getLaneStride();
    }
  } else if (isBlockDeinterleaved()) {
    printer << " = " << getFactor();
  } else if (isGroupSlots()) {
    printer << " = " << getFactor();
    if (getSlots() != 0) {
      printer << ", slots = " << getSlots();
    }
    if (getLaneStride() != 1) {
      printer << ", lane_stride = " << getLaneStride();
    }
  }
  printer << ">";
}

static LogicalResult verifyContiguousLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots) {
  if (factor != 1 || blockElems != 1 || slots != 0) {
    return emitError()
           << "#pto.vmi.layout<contiguous> requires factor, block_elems, "
              "and slots to be their defaults";
  }
  return success();
}

static LogicalResult verifyDeinterleavedLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots) {
  if (factor != mlir::pto::kValue2 && factor != mlir::pto::kValue4) {
    return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                       << "> expected factor to be 2 or 4";
  }
  if (blockElems != 1) {
    return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                       << ", block_elems = " << blockElems
                       << "> requires block_elems to be omitted";
  }
  if (slots != 0) {
    return emitError() << "#pto.vmi.layout<deinterleaved = " << factor
                       << "> requires slots to be omitted";
  }
  return success();
}

static LogicalResult verifyBlockDeinterleavedLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots, int64_t laneStride) {
  if (factor != mlir::pto::kValue2 && factor != mlir::pto::kValue4) {
    return emitError() << "#pto.vmi.layout<block_deinterleaved = " << factor
                       << "> expected factor to be 2 or 4";
  }
  if (blockElems != 1 || slots != 0 || laneStride != 1) {
    return emitError()
           << "#pto.vmi.layout<block_deinterleaved = " << factor
           << "> does not accept block_elems, slots, or lane_stride";
  }
  return success();
}

static LogicalResult verifyNumGroupsLayout(
    function_ref<InFlightDiagnostic()> emitError, int64_t factor,
    int64_t blockElems, int64_t slots) {
  if (factor <= 0) {
    return emitError() << "#pto.vmi.layout<num_groups = " << factor
                       << "> requires num_groups to be positive";
  }
  if (blockElems != 1) {
    return emitError() << "#pto.vmi.layout<num_groups = " << factor
                       << "> requires block_elems to be omitted";
  }
  if (slots < 0) {
    return emitError() << "#pto.vmi.layout<num_groups = " << factor
                       << ", slots = " << slots
                       << "> requires slots to be omitted or positive";
  }
  return success();
}

LogicalResult
VMILayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      StringRef kind, int64_t factor, int64_t blockElems,
                      int64_t slots, int64_t laneStride) {
  if (laneStride <= 0) {
    return emitError() << "#pto.vmi.layout<" << kind
                       << "> requires lane_stride to be positive";
  }
  if (kind == "contiguous") {
    return verifyContiguousLayout(emitError, factor, blockElems, slots);
  }
  if (kind == "deinterleaved") {
    return verifyDeinterleavedLayout(emitError, factor, blockElems, slots);
  }
  if (kind == "block_deinterleaved") {
    return verifyBlockDeinterleavedLayout(emitError, factor, blockElems, slots,
                                          laneStride);
  }
  if (kind == "num_groups") {
    return verifyNumGroupsLayout(emitError, factor, blockElems, slots);
  }
  return emitError() << "expected VMI layout kind to be 'contiguous' or "
                        "'deinterleaved' or 'block_deinterleaved' or "
                        "'num_groups'";
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
      failed(parser.parseGreater())) {
    return {};
  }

  return parser.getChecked<VMIVRegType>(loc, parser.getContext(), shape.front(),
                                        elementType, layout);
}

void VMIVRegType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  if (getLayout()) {
    printer << ", " << getLayout();
  }
  printer << ">";
}

LogicalResult VMIVRegType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  int64_t elementCount, Type elementType,
                                  Attribute layout) {
  if (elementCount <= 0) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected a positive element count";
}

  if (!isSupportedVMIElementType(elementType)) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected an integer, index, floating-point, or "
                          "PTO low-precision element type";
}
  if (!isVMIPredicateMaskableElementType(elementType)) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected an 8-bit, 16-bit, or 32-bit logical "
                          "element type";
}
  if (layout && !mlir::isa<VMILayoutAttr>(layout)) {
    return emitError() << "'"
                       << formatVMIVRegType(elementCount, elementType, layout)
                       << "' expected layout to be #pto.vmi.layout";
}
  if (auto layoutAttr = llvm::dyn_cast_or_null<VMILayoutAttr>(layout)) {
    if (layoutAttr.isGroupSlots() &&
        elementCount != layoutAttr.getNumGroups()) {
      return emitError() << "'"
                         << formatVMIVRegType(elementCount, elementType, layout)
                         << "' expected num_groups layout to describe exactly "
                            "one logical result lane per group";
}
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
      failed(parser.parseGreater())) {
    return {};
  }

  return parser.getChecked<VMIMaskType>(loc, parser.getContext(), shape.front(),
                                        granularity, layout);
}

void VMIMaskType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x" << getGranularity();
  if (getLayout()) {
    printer << ", " << getLayout();
  }
  printer << ">";
}

LogicalResult VMIMaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  int64_t elementCount, StringRef granularity,
                                  Attribute layout) {
  if (elementCount <= 0) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected a positive element count";
  }

  if (!isSupportedGranularity(granularity)) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected granularity to be one of pred, b8, b16, "
                          "b32";
  }

  if (layout && !mlir::isa<VMILayoutAttr>(layout)) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' expected layout to be #pto.vmi.layout";
  }

  if (granularity == "pred" && layout) {
    return emitError() << "'"
                       << formatVMIMaskType(elementCount, granularity, layout)
                       << "' pred mask must not carry layout";
  }

  return success();
}

//===----------------------------------------------------------------------===//
