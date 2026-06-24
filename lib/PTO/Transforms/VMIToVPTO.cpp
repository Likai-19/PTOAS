// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMIToVPTO.cpp - Convert VMI to physical VPTO IR -------------------===//
//===----------------------------------------------------------------------===//

// https://discourse.llvm.org/t/matchandrewrite-hiding-virtual-functions/84933/8
#pragma GCC diagnostic ignored "-Woverloaded-virtual"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMILayoutSupport.h"
#include "PTO/Transforms/VMITargetCapabilities.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <type_traits>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMITOVPTO
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

bool isVMIType(Type type) { return isa<VMIVRegType, VMIMaskType>(type); }

bool containsVMIType(Type type) {
  if (isVMIType(type))
    return true;

  if (auto functionType = dyn_cast<FunctionType>(type))
    return llvm::any_of(functionType.getInputs(),
                        [](Type input) { return containsVMIType(input); }) ||
           llvm::any_of(functionType.getResults(),
                        [](Type result) { return containsVMIType(result); });

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return containsVMIType(shapedType.getElementType());

  return false;
}

bool hasVMIType(TypeRange types) {
  return llvm::any_of(types, [](Type type) { return containsVMIType(type); });
}

bool hasVMIType(FunctionType type) {
  return hasVMIType(type.getInputs()) || hasVMIType(type.getResults());
}

bool hasVMIType(Attribute attr) {
  if (!attr)
    return false;

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    if (containsVMIType(typeAttr.getValue()))
      return true;

  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    if (containsVMIType(typedAttr.getType()))
      return true;

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr))
    return llvm::any_of(arrayAttr,
                        [](Attribute element) { return hasVMIType(element); });

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr))
    return llvm::any_of(dictAttr, [](NamedAttribute namedAttr) {
      return hasVMIType(namedAttr.getValue());
    });

  return false;
}

bool hasVMIType(Operation *op) {
  if (auto func = dyn_cast<func::FuncOp>(op))
    if (hasVMIType(func.getFunctionType()))
      return true;
  if (hasVMIType(op->getOperandTypes()) || hasVMIType(op->getResultTypes()))
    return true;
  for (Region &region : op->getRegions())
    for (Block &block : region)
      if (hasVMIType(block.getArgumentTypes()))
        return true;
  for (NamedAttribute attr : op->getAttrs())
    if (hasVMIType(attr.getValue()))
      return true;
  return false;
}

bool isVMIOp(Operation *op) {
  return op->getName().getStringRef().starts_with("pto.vmi.");
}

bool isLayoutAssignedVMIType(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return static_cast<bool>(vregType.getLayoutAttr());
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getLayoutAttr() &&
           VMIMaskType::isConcreteGranularity(maskType.getGranularity());
  return true;
}

LogicalResult verifyLayoutAssignedVMITypeTree(Operation *op, Type type) {
  if (!isLayoutAssignedVMIType(type))
    return op->emitError() << kVMIDiagPassInvariantPrefix
                           << "vmi-to-vpto requires layout-assigned VMI types";

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs())
      if (failed(verifyLayoutAssignedVMITypeTree(op, input)))
        return failure();
    for (Type result : functionType.getResults())
      if (failed(verifyLayoutAssignedVMITypeTree(op, result)))
        return failure();
  }

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return verifyLayoutAssignedVMITypeTree(op, shapedType.getElementType());

  return success();
}

LogicalResult verifyVMIToVPTOInputAttribute(Operation *op, Attribute attr) {
  if (!attr)
    return success();

  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    if (failed(verifyLayoutAssignedVMITypeTree(op, typeAttr.getValue())))
      return failure();

  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    if (failed(verifyLayoutAssignedVMITypeTree(op, typedAttr.getType())))
      return failure();

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute element : arrayAttr)
      if (failed(verifyVMIToVPTOInputAttribute(op, element)))
        return failure();
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    for (NamedAttribute namedAttr : dictAttr)
      if (failed(verifyVMIToVPTOInputAttribute(op, namedAttr.getValue())))
        return failure();
  }

  return success();
}

LogicalResult verifyVMIToVPTOInputTypes(Operation *op) {
  for (Type type : op->getOperandTypes())
    if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
      return failure();
  for (Type type : op->getResultTypes())
    if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
      return failure();
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    for (Type type : func.getFunctionType().getInputs())
      if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
        return failure();
    for (Type type : func.getFunctionType().getResults())
      if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
        return failure();
  }
  for (Region &region : op->getRegions())
    for (Block &block : region)
      for (Type type : block.getArgumentTypes())
        if (failed(verifyLayoutAssignedVMITypeTree(op, type)))
          return failure();
  for (NamedAttribute attr : op->getAttrs())
    if (failed(verifyVMIToVPTOInputAttribute(op, attr.getValue())))
      return failure();
  return success();
}

LogicalResult verifyVMIToVPTOInputIR(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (failed(verifyVMIToVPTOInputTypes(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

static std::optional<Value> materializeVPTOToVMI(OpBuilder &builder,
                                                 Type resultType,
                                                 ValueRange inputs,
                                                 Location loc) {
  if (!isVMIType(resultType))
    return std::nullopt;
  return builder.create<VMIPackOp>(loc, resultType, inputs).getResult();
}

static std::optional<SmallVector<Value>>
materializeVMIToVPTO(OpBuilder &builder, TypeRange resultTypes, Value input,
                     Location loc) {
  if (!isVMIType(input.getType()))
    return std::nullopt;
  auto unpackOp = builder.create<VMIUnpackOp>(loc, resultTypes, input);
  return SmallVector<Value>(unpackOp->getResults());
}

class VMIToVPTOTypeConverter final : public OneToNTypeConverter {
public:
  VMIToVPTOTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(
        [](VMIVRegType type, SmallVectorImpl<Type> &results) -> LogicalResult {
          FailureOr<int64_t> arity = getVMIPhysicalArity(type);
          FailureOr<int64_t> lanesPerPart =
              getDataLanesPerPart(type.getElementType());
          if (failed(arity) || failed(lanesPerPart))
            return failure();
          for (int64_t i = 0; i < *arity; ++i)
            results.push_back(VRegType::get(type.getContext(), *lanesPerPart,
                                            type.getElementType()));
          return success();
        });
    addConversion(
        [](VMIMaskType type, SmallVectorImpl<Type> &results) -> LogicalResult {
          FailureOr<int64_t> arity = getVMIPhysicalArity(type);
          if (failed(arity))
            return failure();
          for (int64_t i = 0; i < *arity; ++i)
            results.push_back(
                MaskType::get(type.getContext(), type.getGranularity()));
          return success();
        });
    TypeConverter::addSourceMaterialization(materializeVPTOToVMI);
    TypeConverter::addArgumentMaterialization(materializeVPTOToVMI);
    OneToNTypeConverter::addTargetMaterialization(materializeVMIToVPTO);
  }
};

FailureOr<Value> createAllTrueMaskForVReg(Location loc, VRegType vregType,
                                          PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(vregType.getElementType());
  if (elementBits == 8)
    return rewriter
        .create<PsetB8Op>(loc, MaskType::get(ctx, "b8"),
                          rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  if (elementBits == 16)
    return rewriter
        .create<PsetB16Op>(loc, MaskType::get(ctx, "b16"),
                           rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  if (elementBits == 32)
    return rewriter
        .create<PsetB32Op>(loc, MaskType::get(ctx, "b32"),
                           rewriter.getStringAttr("PAT_ALL"))
        .getResult();
  return failure();
}

FailureOr<MaskType> getMaskTypeForVReg(VRegType vregType, MLIRContext *ctx) {
  unsigned elementBits =
      pto::getPTOStorageElemBitWidth(vregType.getElementType());
  if (elementBits == 8)
    return MaskType::get(ctx, "b8");
  if (elementBits == 16)
    return MaskType::get(ctx, "b16");
  if (elementBits == 32)
    return MaskType::get(ctx, "b32");
  return failure();
}

FailureOr<Value> createAllTrueMask(Location loc, MaskType maskType,
                                   PatternRewriter &rewriter) {
  StringAttr pattern = rewriter.getStringAttr("PAT_ALL");
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8())
    return rewriter.create<PsetB8Op>(loc, MaskType::get(ctx, "b8"), pattern)
        .getResult();
  if (maskType.isB16())
    return rewriter.create<PsetB16Op>(loc, MaskType::get(ctx, "b16"), pattern)
        .getResult();
  if (maskType.isB32())
    return rewriter.create<PsetB32Op>(loc, MaskType::get(ctx, "b32"), pattern)
        .getResult();
  return failure();
}

FailureOr<Value> createPatternMask(Location loc, MaskType maskType,
                                   StringRef pattern,
                                   PatternRewriter &rewriter) {
  StringAttr patternAttr = rewriter.getStringAttr(pattern);
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8())
    return rewriter.create<PsetB8Op>(loc, MaskType::get(ctx, "b8"), patternAttr)
        .getResult();
  if (maskType.isB16())
    return rewriter
        .create<PsetB16Op>(loc, MaskType::get(ctx, "b16"), patternAttr)
        .getResult();
  if (maskType.isB32())
    return rewriter
        .create<PsetB32Op>(loc, MaskType::get(ctx, "b32"), patternAttr)
        .getResult();
  return failure();
}

FailureOr<Value> createPrefixMask(Location loc, MaskType maskType,
                                  StringRef pattern,
                                  PatternRewriter &rewriter) {
  StringAttr patternAttr = rewriter.getStringAttr(pattern);
  MLIRContext *ctx = rewriter.getContext();
  if (maskType.isB8())
    return rewriter.create<PgeB8Op>(loc, MaskType::get(ctx, "b8"), patternAttr)
        .getResult();
  if (maskType.isB16())
    return rewriter
        .create<PgeB16Op>(loc, MaskType::get(ctx, "b16"), patternAttr)
        .getResult();
  if (maskType.isB32())
    return rewriter
        .create<PgeB32Op>(loc, MaskType::get(ctx, "b32"), patternAttr)
        .getResult();
  return failure();
}

FailureOr<std::pair<Value, Value>>
createRuntimePrefixMask(Location loc, MaskType maskType, Value activeLanes,
                        PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  Type scalarType = activeLanes.getType();
  if (maskType.isB8()) {
    auto op = rewriter.create<PltB8Op>(loc, MaskType::get(ctx, "b8"),
                                       scalarType, activeLanes);
    return std::make_pair(Value(op.getMask()), Value(op.getScalarOut()));
  }
  if (maskType.isB16()) {
    auto op = rewriter.create<PltB16Op>(loc, MaskType::get(ctx, "b16"),
                                        scalarType, activeLanes);
    return std::make_pair(Value(op.getMask()), Value(op.getScalarOut()));
  }
  if (maskType.isB32()) {
    auto op = rewriter.create<PltB32Op>(loc, MaskType::get(ctx, "b32"),
                                        scalarType, activeLanes);
    return std::make_pair(Value(op.getMask()), Value(op.getScalarOut()));
  }
  return failure();
}

LogicalResult
checkSupportedMaskableVReg(const VMITargetCapabilityRegistry &capabilities,
                           VMIVRegType type, std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMICapabilityResult elementCapability = capabilities.supportsElementType(
      type.getElementType(), VMIElementPurpose::PredicateMask);
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  if (failed(lanesPerPart) || failed(arity) || *arity < 1)
    return fail("requires computable non-empty physical vreg parts");

  return success();
}

LogicalResult
checkSupportedTargetElementVReg(const VMITargetCapabilityRegistry &capabilities,
                                VMIVRegType type, VMIElementPurpose purpose,
                                StringRef elementContract,
                                std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (failed(checkSupportedMaskableVReg(capabilities, type, reason)))
    return failure();

  VMICapabilityResult elementCapability =
      capabilities.supportsElementType(type.getElementType(), purpose);
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);

  return success();
}

Value createI32Constant(Location loc, int64_t value,
                        PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 32);
}

Value createI16Constant(Location loc, int64_t value,
                        PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, value, 16);
}

FailureOr<Value> createPrefixMaskForActiveLanes(Location loc, MaskType maskType,
                                                int64_t activeLanes,
                                                PatternRewriter &rewriter) {
  if (activeLanes <= 0)
    return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);

  switch (activeLanes) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
  case 128:
    return createPrefixMask(
        loc, maskType, (Twine("PAT_VL") + Twine(activeLanes)).str(), rewriter);
  default: {
    FailureOr<std::pair<Value, Value>> dynamicMask = createRuntimePrefixMask(
        loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
    if (failed(dynamicMask))
      return failure();
    return dynamicMask->first;
  }
  }
}

Value clampDynamicActiveLanes(Location loc, Value activeLanes,
                              int64_t maxActiveLanes,
                              PatternRewriter &rewriter) {
  Value activeI32 = rewriter.create<arith::IndexCastOp>(
      loc, rewriter.getI32Type(), activeLanes);
  Value zeroI32 = createI32Constant(loc, 0, rewriter);
  Value nonNegative = rewriter.create<arith::MaxSIOp>(loc, activeI32, zeroI32);
  Value maxI32 = createI32Constant(loc, maxActiveLanes, rewriter);
  return rewriter.create<arith::MinUIOp>(loc, nonNegative, maxI32);
}

Value createPartitionActiveLanes(Location loc, Value activeLanesI32,
                                 int64_t factor, int64_t part,
                                 PatternRewriter &rewriter) {
  if (factor == 1)
    return activeLanesI32;
  int64_t bias = factor - 1 - part;
  Value biased = activeLanesI32;
  if (bias != 0)
    biased = rewriter.create<arith::AddIOp>(
        loc, biased, createI32Constant(loc, bias, rewriter));
  return rewriter.create<arith::DivUIOp>(
      loc, biased, createI32Constant(loc, factor, rewriter));
}

std::optional<int64_t> getPowerOfTwoLog2(int64_t value) {
  if (value <= 0 || (value & (value - 1)) != 0)
    return std::nullopt;
  int64_t log2 = 0;
  while (value > 1) {
    value >>= 1;
    ++log2;
  }
  return log2;
}

std::optional<std::string> getPrefixPattern(int64_t activeLanes,
                                            int64_t lanesPerPart) {
  if (activeLanes <= 0)
    return std::string("PAT_ALLF");
  if (activeLanes >= lanesPerPart)
    return std::string("PAT_ALL");
  switch (activeLanes) {
  case 1:
  case 2:
  case 3:
  case 4:
  case 8:
  case 16:
  case 32:
  case 64:
  case 128:
    return std::string("PAT_VL") + std::to_string(activeLanes);
  default:
    return std::nullopt;
  }
}

FailureOr<Value> getSingleValue(Operation *op, ValueRange values,
                                StringRef description,
                                PatternRewriter &rewriter) {
  if (values.size() != 1) {
    (void)rewriter.notifyMatchFailure(op, description);
    return failure();
  }
  return values.front();
}

static int64_t ceilDivNonNegative(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

FailureOr<int64_t> getDataLayoutFactor(VMIVRegType type) {
  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return failure();
  return layout.isDeinterleaved() ? layout.getFactor() : 1;
}

FailureOr<int64_t> getDataChunksInPart(VMIVRegType type, int64_t part) {
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(factor) || failed(lanesPerPart) || part < 0 || part >= *factor)
    return failure();

  int64_t logicalLanesInPart =
      (type.getElementCount() + *factor - 1 - part) / *factor;
  return ceilDivNonNegative(logicalLanesInPart, *lanesPerPart);
}

FailureOr<int64_t> getDataFlatPartIndex(VMIVRegType type, int64_t part,
                                        int64_t chunk) {
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor) || part < 0 || part >= *factor || chunk < 0)
    return failure();

  int64_t flatIndex = 0;
  for (int64_t currentPart = 0; currentPart < part; ++currentPart) {
    FailureOr<int64_t> chunks = getDataChunksInPart(type, currentPart);
    if (failed(chunks))
      return failure();
    flatIndex += *chunks;
  }

  FailureOr<int64_t> chunks = getDataChunksInPart(type, part);
  if (failed(chunks) || chunk >= *chunks)
    return failure();
  return flatIndex + chunk;
}

FailureOr<int64_t> checkFullDataPhysicalChunks(VMIVRegType type,
                                               std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known physical lanes per part");

  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor))
    return fail("requires assigned layout");

  for (int64_t part = 0; part < *factor; ++part) {
    FailureOr<int64_t> chunks = getDataChunksInPart(type, part);
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

  return *lanesPerPart;
}

FailureOr<int64_t> getVMITypeLayoutFactor(Type type) {
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
  return layoutAttr.isDeinterleaved() ? layoutAttr.getFactor() : 1;
}

FailureOr<int64_t> getVMITypeElementCount(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return vregType.getElementCount();
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return maskType.getElementCount();
  return failure();
}

FailureOr<int64_t> getVMITypeLanesPerPart(Type type) {
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    return getDataLanesPerPart(vregType.getElementType());
  if (auto maskType = dyn_cast<VMIMaskType>(type))
    return getMaskLanesPerPart(maskType.getGranularity());
  return failure();
}

FailureOr<int64_t> getVMITypeChunksInPart(Type type, int64_t part) {
  FailureOr<int64_t> elementCount = getVMITypeElementCount(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  FailureOr<int64_t> lanesPerPart = getVMITypeLanesPerPart(type);
  if (failed(elementCount) || failed(factor) || failed(lanesPerPart) ||
      part < 0 || part >= *factor)
    return failure();

  int64_t logicalLanesInPart = (*elementCount + *factor - 1 - part) / *factor;
  return ceilDivNonNegative(logicalLanesInPart, *lanesPerPart);
}

LogicalResult checkFullVMIPhysicalChunks(Type type, std::string *reason) {
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

FailureOr<int64_t> getContiguousMaterializationPartCount(Type type,
                                                         std::string *reason);

LogicalResult checkSupportedLayoutMaterialization(
    const VMITargetCapabilityRegistry &capabilities, Type sourceType,
    Type resultType, VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMICapabilityResult layoutCapability =
      capabilities.supportsLayoutConversion(sourceLayout, resultLayout, Type{});
  if (!layoutCapability.isSupported())
    return fail(layoutCapability.reason);

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

FailureOr<int64_t> getContiguousMaterializationPartCount(Type type,
                                                         std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> arity = getVMIPhysicalArity(type);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(type);
  if (failed(arity) || failed(factor))
    return fail("requires computable physical arity and assigned layout");

  Attribute layoutAttr;
  if (auto vregType = dyn_cast<VMIVRegType>(type))
    layoutAttr = vregType.getLayout();
  else if (auto maskType = dyn_cast<VMIMaskType>(type))
    layoutAttr = maskType.getLayout();
  else
    return fail("requires VMI data or mask type");

  auto layout = dyn_cast_or_null<VMILayoutAttr>(layoutAttr);
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

LogicalResult checkCanMaterializeToContiguous(Type type, std::string *reason) {
  return succeeded(getContiguousMaterializationPartCount(type, reason))
             ? success()
             : failure();
}

std::optional<int64_t> getConstantIndexValue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
    return constant.value();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integerAttr = dyn_cast<IntegerAttr>(constant.getValue()))
      return integerAttr.getInt();
  }
  return std::nullopt;
}

FailureOr<int64_t> getStaticMemRefElementCount(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType || !memrefType.hasStaticShape())
    return failure();

  int64_t elements = 1;
  for (int64_t dim : memrefType.getShape())
    elements *= dim;
  return elements;
}

enum class VMIMemoryValidMaskKind {
  AllTrue,
  ExplicitMask,
};

enum class VMIMemoryWriteMaskKind {
  AllTrue,
  ExplicitMask,
};

enum class VMIMemoryPermutationKind {
  Identity,
};

enum class VMIMemoryFallbackDecisionKind {
  NotRequired,
  RequiredUnavailable,
};

struct VMIMemoryLogicalShape {
  int64_t elementCount = 0;
};

struct VMIMemoryLaneAddressMap {
  VMIMemoryPermutationKind permutation = VMIMemoryPermutationKind::Identity;
  int64_t baseElementOffset = 0;
  int64_t elementStride = 1;
  int64_t physicalLaneFootprint = 0;

  int64_t getExclusiveEndElement() const {
    return baseElementOffset + physicalLaneFootprint * elementStride;
  }
};

struct VMIMemoryFallbackDecision {
  VMIMemoryFallbackDecisionKind kind =
      VMIMemoryFallbackDecisionKind::NotRequired;
  std::string reason = "not required";

  static VMIMemoryFallbackDecision notRequired() { return {}; }

  static VMIMemoryFallbackDecision requiredUnavailable(const Twine &reason) {
    VMIMemoryFallbackDecision decision;
    decision.kind = VMIMemoryFallbackDecisionKind::RequiredUnavailable;
    decision.reason = reason.str();
    return decision;
  }
};

struct VMIMemorySafeReadProof {
  bool proven = false;
  std::string reason;
  std::optional<int64_t> constantOffset;
  std::optional<int64_t> staticElementCount;
  std::optional<VMIMemoryLaneAddressMap> laneAddressMap;
  int64_t physicalFootprint = 0;
};

struct VMIMemoryAccessPlan {
  Type baseType;
  VMIVRegType valueType;
  std::optional<int64_t> constantOffset;
  VMIMemoryLogicalShape logicalShape;
  VMIMemoryValidMaskKind validMask = VMIMemoryValidMaskKind::AllTrue;
  VMIMemoryPermutationKind permutation = VMIMemoryPermutationKind::Identity;
  std::optional<VMIMemoryLaneAddressMap> laneAddressMap;
  Attribute paddingValue;
  VMIMemoryWriteMaskKind writeMask = VMIMemoryWriteMaskKind::AllTrue;
  VMIMemorySafeReadProof safeReadProof;
  VMICapabilityResult targetCapability;
  VMICapabilityResult trueMaskedLoadCapability;
  VMICapabilityResult scratchFallbackCapability;
  VMICapabilityResult guardedFallbackCapability;
  VMIMemoryFallbackDecision fallbackDecision;
};

FailureOr<VMIMemoryLaneAddressMap>
buildContiguousIdentityLaneAddressMap(int64_t constantOffset,
                                      VMIVRegType resultType,
                                      std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> FailureOr<VMIMemoryLaneAddressMap> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(resultType.getElementType());
  FailureOr<int64_t> arity = getVMIPhysicalArity(resultType);
  if (failed(lanesPerPart) || failed(arity))
    return fail("requires computable physical read footprint");

  VMIMemoryLaneAddressMap map;
  map.baseElementOffset = constantOffset;
  map.physicalLaneFootprint = *arity * *lanesPerPart;
  return map;
}

VMICapabilityResult requireIdentityMemRefLayout(Type memoryType, StringRef role,
                                                Value memoryValue = {}) {
  auto memrefType = dyn_cast<MemRefType>(memoryType);
  if (!memrefType || memrefType.getLayout().isIdentity())
    return VMICapabilityResult::supported();
  std::string reason =
      (Twine(role) +
       " memref layout is non-identity; current VMI memory access plan "
       "supports only contiguous identity lane-to-address maps")
          .str();
  if (memoryValue && memoryValue.getDefiningOp<memref::SubViewOp>())
    reason += "; memref.subview requires normalized base/offset/stride "
              "lane-to-address planning";
  return VMICapabilityResult::missingCapability(reason);
}

VMIMemorySafeReadProof
computeSafeFullReadProof(Type sourceType, std::optional<int64_t> constantOffset,
                         VMIVRegType resultType) {
  VMIMemorySafeReadProof proof;
  proof.constantOffset = constantOffset;

  auto fail = [&](const Twine &message) {
    proof.proven = false;
    proof.reason = message.str();
    return proof;
  };

  if (!constantOffset)
    return fail("requires constant index offset");

  FailureOr<int64_t> staticElements = getStaticMemRefElementCount(sourceType);
  if (failed(staticElements))
    return fail("requires statically shaped memref source");
  int64_t elements = *staticElements;
  proof.staticElementCount = elements;

  if (*constantOffset < 0)
    return fail("requires non-negative offset");

  std::string addressMapReason;
  FailureOr<VMIMemoryLaneAddressMap> addressMap =
      buildContiguousIdentityLaneAddressMap(*constantOffset, resultType,
                                            &addressMapReason);
  if (failed(addressMap))
    return fail(addressMapReason);
  proof.laneAddressMap = *addressMap;

  proof.physicalFootprint = addressMap->physicalLaneFootprint;
  if (addressMap->getExclusiveEndElement() > elements)
    return fail(Twine("full physical read footprint [") +
                Twine(addressMap->baseElementOffset) + ", " +
                Twine(addressMap->getExclusiveEndElement()) +
                ") exceeds static memref element count " + Twine(elements));

  proof.proven = true;
  return proof;
}

VMIMemoryAccessPlan
buildReadAccessPlan(const VMITargetCapabilityRegistry &capabilities,
                    Value source, Type sourceType, VMIVRegType resultType,
                    std::optional<int64_t> constantOffset,
                    VMIMemoryValidMaskKind validMask) {
  VMIMemoryAccessPlan plan;
  plan.baseType = sourceType;
  plan.valueType = resultType;
  plan.constantOffset = constantOffset;
  plan.logicalShape.elementCount = resultType.getElementCount();
  plan.validMask = validMask;
  plan.permutation = VMIMemoryPermutationKind::Identity;
  plan.writeMask = VMIMemoryWriteMaskKind::AllTrue;
  plan.safeReadProof =
      computeSafeFullReadProof(sourceType, constantOffset, resultType);
  plan.laneAddressMap = plan.safeReadProof.laneAddressMap;
  plan.targetCapability =
      capabilities.supportsDirectMemory(sourceType, "source");
  if (plan.targetCapability.isSupported())
    plan.targetCapability =
        requireIdentityMemRefLayout(sourceType, "source", source);
  if (validMask == VMIMemoryValidMaskKind::ExplicitMask)
    plan.trueMaskedLoadCapability =
        capabilities.supportsTrueMaskedLoad(sourceType, resultType, Type{});
  plan.scratchFallbackCapability = capabilities.supportsFallbackResource(
      VMIFallbackResourceKind::ScratchMemory);
  plan.guardedFallbackCapability = capabilities.supportsFallbackResource(
      VMIFallbackResourceKind::GuardedControlFlow);
  return plan;
}

VMIMemoryAccessPlan
buildWriteAccessPlan(const VMITargetCapabilityRegistry &capabilities,
                     Value destination, Type destinationType,
                     VMIVRegType valueType, VMIMemoryWriteMaskKind writeMask) {
  VMIMemoryAccessPlan plan;
  plan.baseType = destinationType;
  plan.valueType = valueType;
  plan.logicalShape.elementCount = valueType.getElementCount();
  plan.validMask = VMIMemoryValidMaskKind::AllTrue;
  plan.permutation = VMIMemoryPermutationKind::Identity;
  plan.writeMask = writeMask;
  plan.targetCapability =
      capabilities.supportsDirectMemory(destinationType, "destination");
  if (plan.targetCapability.isSupported())
    plan.targetCapability = requireIdentityMemRefLayout(
        destinationType, "destination", destination);
  return plan;
}

void requireUnavailableReadFallback(VMIMemoryAccessPlan &plan) {
  std::string maskedLoadReason;
  if (plan.validMask == VMIMemoryValidMaskKind::ExplicitMask &&
      !plan.trueMaskedLoadCapability.isSupported())
    maskedLoadReason =
        (Twine("; ") + plan.trueMaskedLoadCapability.reason).str();
  std::string scratchReason;
  if (!plan.scratchFallbackCapability.isSupported())
    scratchReason = (Twine("; ") + plan.scratchFallbackCapability.reason).str();
  std::string guardedReason;
  if (!plan.guardedFallbackCapability.isSupported())
    guardedReason = (Twine("; ") + plan.guardedFallbackCapability.reason).str();
  plan.fallbackDecision = VMIMemoryFallbackDecision::requiredUnavailable(
      Twine("partial/tail read needs a scratch, guarded, or true "
            "masked/non-faulting load fallback, but no such fallback resource "
            "plan is implemented") +
      maskedLoadReason + scratchReason + guardedReason);
}

FailureOr<int64_t> verifyFullOrSafeReadVRegChunks(Operation *op,
                                                  VMIVRegType type,
                                                  Type sourceType, Value offset,
                                                  PatternRewriter &rewriter) {
  std::string fullChunkReason;
  FailureOr<int64_t> lanesPerPart =
      checkFullDataPhysicalChunks(type, &fullChunkReason);
  if (succeeded(lanesPerPart))
    return *lanesPerPart;

  VMIMemorySafeReadProof safeReadProof =
      computeSafeFullReadProof(sourceType, getConstantIndexValue(offset), type);
  if (safeReadProof.proven) {
    lanesPerPart = getDataLanesPerPart(type.getElementType());
    if (succeeded(lanesPerPart))
      return *lanesPerPart;
  }

  (void)rewriter.notifyMatchFailure(
      op, Twine("memory lowering ") + fullChunkReason +
              "; safe full-read proof failed: " + safeReadProof.reason);
  return failure();
}

LogicalResult
checkSupportedLoadShape(const VMITargetCapabilityRegistry &capabilities,
                        VMIVRegType type, Value source, Type sourceType,
                        std::optional<int64_t> constantOffset,
                        std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMIMemoryAccessPlan accessPlan =
      buildReadAccessPlan(capabilities, source, sourceType, type,
                          constantOffset, VMIMemoryValidMaskKind::AllTrue);
  if (!accessPlan.targetCapability.isSupported())
    return fail(accessPlan.targetCapability.reason);

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(type, &fullChunkReason)))
    return success();

  if (accessPlan.safeReadProof.proven)
    return success();
  requireUnavailableReadFallback(accessPlan);
  return fail(Twine(fullChunkReason) +
              "; safe-read proof failed: " + accessPlan.safeReadProof.reason +
              "; fallback decision: " + accessPlan.fallbackDecision.reason);
}

LogicalResult
checkSupportedStoreShape(const VMITargetCapabilityRegistry &capabilities,
                         VMIVRegType type, Value destination,
                         Type destinationType, std::string *reason) {
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(capabilities, destination, destinationType, type,
                           VMIMemoryWriteMaskKind::AllTrue);
  if (!accessPlan.targetCapability.isSupported()) {
    if (reason)
      *reason = accessPlan.targetCapability.reason;
    return failure();
  }

  if (failed(checkSupportedMaskableVReg(capabilities, type, reason)))
    return failure();

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(type, &fullChunkReason)))
    return success();

  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return fail("requires assigned layout");
  if (failed(getDataLanesPerPart(type.getElementType())))
    return fail("requires known physical lanes per part");
  if (layout.isContiguous())
    return success();

  std::string materializationReason;
  if (succeeded(checkCanMaterializeToContiguous(type, &materializationReason)))
    return success();
  return fail(Twine("partial/tail store requires contiguous layout or "
                    "deinterleaved layout that can materialize to contiguous; "
                    "value ") +
              fullChunkReason + ", materialization " + materializationReason);
}

FailureOr<int64_t> getGroupSizeFromNumGroups(VMIVRegType type,
                                             int64_t numGroups,
                                             std::string *reason = nullptr) {
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

LogicalResult checkSupportedGroupChunkShape(VMIVRegType type, int64_t groupSize,
                                            std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return fail("requires assigned contiguous layout");
  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(type, &fullChunkReason)))
    return fail(Twine("requires full physical chunks; ") + fullChunkReason);
  FailureOr<int64_t> lanesPerPart = getDataLanesPerPart(type.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known physical lanes per part");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("requires derived group size to evenly divide logical lane "
                "count");
  if (groupSize % *lanesPerPart != 0)
    return fail("currently requires group size to be a multiple of physical "
                "lanes per part");
  return success();
}

LogicalResult
checkSupportedGroupLoadShape(const VMITargetCapabilityRegistry &capabilities,
                             VMIGroupLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout)
    return fail("requires assigned result layout");
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      resultType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize))
    return failure();

  if (resultLayout.isContiguous()) {
    if (failed(checkSupportedLoadShape(capabilities, resultType, op.getSource(),
                                       op.getSource().getType(), std::nullopt,
                                       reason)))
      return failure();
    return checkSupportedGroupChunkShape(resultType, *groupSize, reason);
  }

  if (resultLayout.isDeinterleaved() && resultLayout.getBlockElems() == 8 &&
      resultType.getElementType().isF32()) {
    VMILayoutSupport supports;
    if (failed(supports.getGroupLoadSupport(capabilities, op, reason)))
      return failure();
    return success();
  }

  return fail("requires contiguous layout or deinterleaved block8 f32 layout");
}

LogicalResult checkSupportedGroupSlotLoadShape(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupSlotLoadOp op,
    std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getGroupSlotLoadSupport(capabilities, op, reason)))
    return failure();
  return success();
}

LogicalResult
checkSupportedGroupStoreShape(const VMITargetCapabilityRegistry &capabilities,
                              VMIGroupStoreOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  VMILayoutAttr layout = valueType.getLayoutAttr();
  if (layout && layout.isGroupSlots()) {
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    if (layout.getNumGroups() != numGroups)
      return fail("group_slots group_store requires layout num_groups to "
                  "match op num_groups");

    VMIMemoryAccessPlan accessPlan = buildWriteAccessPlan(
        capabilities, op.getDestination(), op.getDestination().getType(),
        valueType, VMIMemoryWriteMaskKind::AllTrue);
    if (!accessPlan.targetCapability.isSupported())
      return fail(accessPlan.targetCapability.reason);

    VMILayoutSupport supports;
    if (failed(supports.getGroupSlotsStoreSupport(capabilities, op, reason)))
      return failure();
    return success();
  }

  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      valueType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize))
    return failure();
  if (failed(checkSupportedStoreShape(capabilities, valueType,
                                      op.getDestination(),
                                      op.getDestination().getType(), reason)))
    return failure();
  return checkSupportedGroupChunkShape(valueType, *groupSize, reason);
}

LogicalResult
checkSupportedMaskedLoadShape(const VMITargetCapabilityRegistry &capabilities,
                              VMIMaskedLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      capabilities, op.getSource(), op.getSource().getType(), resultType,
      getConstantIndexValue(op.getOffset()),
      VMIMemoryValidMaskKind::ExplicitMask);
  if (!accessPlan.targetCapability.isSupported())
    return fail(accessPlan.targetCapability.reason);
  if (!resultLayout || !passthruLayout || !maskLayout)
    return fail("requires assigned result, passthru, and mask layouts");
  if (!resultLayout.isContiguous() || !passthruLayout.isContiguous() ||
      !maskLayout.isContiguous())
    return fail("requires contiguous result, passthru, and mask layouts");

  std::string fullChunkReason;
  if (succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return success();

  if (accessPlan.safeReadProof.proven)
    return success();
  requireUnavailableReadFallback(accessPlan);
  return fail(Twine("partial/tail masked_load requires statically safe "
                    "full-read footprint; value ") +
              fullChunkReason + ", safe-read proof " +
              accessPlan.safeReadProof.reason +
              "; fallback decision: " + accessPlan.fallbackDecision.reason);
}

LogicalResult
checkSupportedGatherShape(const VMITargetCapabilityRegistry &capabilities,
                          VMIGatherOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto indicesType = cast<VMIVRegType>(op.getIndices().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!resultLayout || !indicesLayout || !passthruLayout || !maskLayout)
    return fail("requires assigned result, indices, passthru, and mask "
                "layouts");
  if (!resultLayout.isContiguous() || !indicesLayout.isContiguous() ||
      !passthruLayout.isContiguous() || !maskLayout.isContiguous())
    return fail("requires contiguous result, indices, passthru, and mask "
                "layouts");

  VMICapabilityResult sourceCapability = capabilities.supportsUBPointerMemory(
      op.getSource().getType(), "source", "pto.vgather2_bc",
      "pto.vgather2_bc reads only UB");
  if (!sourceCapability.isSupported())
    return fail(sourceCapability.reason);

  if (pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 32)
    return fail("currently requires 32-bit result element type so physical "
                "offset and result lane counts match pto.vgather2_bc");
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.getWidth() != 32 ||
      indexElementType.isSigned())
    return fail("requires signless or unsigned 32-bit indices");
  if (maskType.getGranularity() != "b32")
    return fail("requires b32 mask granularity");

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(resultArity) || failed(indicesArity) || failed(passthruArity) ||
      failed(maskArity))
    return fail("requires computable physical arity");
  if (*resultArity != *indicesArity || *resultArity != *passthruArity ||
      *resultArity != *maskArity)
    return fail("requires result, indices, passthru, and mask to have the "
                "same physical arity");

  std::string resultReason;
  std::string indicesReason;
  std::string passthruReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultReason)))
    return fail(Twine("result requires full physical chunks; ") + resultReason);
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason)))
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason)))
    return fail(Twine("passthru requires full physical chunks; ") +
                passthruReason);
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason)))
    return fail(Twine("mask requires full physical chunks; ") + maskReason);

  return success();
}

LogicalResult
checkSupportedScatterShape(const VMITargetCapabilityRegistry &capabilities,
                           VMIScatterOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto indicesType = cast<VMIVRegType>(op.getIndices().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr indicesLayout = indicesType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !indicesLayout || !maskLayout)
    return fail("requires assigned value, indices, and mask layouts");
  if (!valueLayout.isContiguous() || !indicesLayout.isContiguous() ||
      !maskLayout.isContiguous())
    return fail("requires contiguous value, indices, and mask layouts");

  VMICapabilityResult destinationCapability =
      capabilities.supportsUBPointerMemory(op.getDestination().getType(),
                                           "destination", "pto.vscatter",
                                           "pto.vscatter writes only UB");
  if (!destinationCapability.isSupported())
    return fail(destinationCapability.reason);

  if (pto::getPTOStorageElemBitWidth(valueType.getElementType()) != 32)
    return fail("currently requires 32-bit value element type so physical "
                "index and value lane counts match pto.vscatter");
  auto indexElementType = dyn_cast<IntegerType>(indicesType.getElementType());
  if (!indexElementType || indexElementType.getWidth() != 32 ||
      indexElementType.isSigned())
    return fail("requires signless or unsigned 32-bit indices");
  if (maskType.getGranularity() != "b32")
    return fail("requires b32 mask granularity");

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> indicesArity = getVMIPhysicalArity(indicesType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(indicesArity) || failed(maskArity))
    return fail("requires computable physical arity");
  if (*valueArity != *indicesArity || *valueArity != *maskArity)
    return fail("requires value, indices, and mask to have the same physical "
                "arity");

  std::string valueReason;
  std::string indicesReason;
  std::string maskReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &valueReason)))
    return fail(Twine("value requires full physical chunks; ") + valueReason);
  if (failed(checkFullDataPhysicalChunks(indicesType, &indicesReason)))
    return fail(Twine("indices require full physical chunks; ") +
                indicesReason);
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskReason)))
    return fail(Twine("mask requires full physical chunks; ") + maskReason);

  return success();
}

Value stripMaskMaterialization(Value value) {
  while (true) {
    if (auto ensure = value.getDefiningOp<VMIEnsureMaskLayoutOp>()) {
      value = ensure.getSource();
      continue;
    }
    if (auto ensure = value.getDefiningOp<VMIEnsureMaskGranularityOp>()) {
      value = ensure.getSource();
      continue;
    }
    return value;
  }
}

bool isStaticAllActiveMask(Value mask, int64_t expectedLanes,
                           std::string *reason = nullptr) {
  mask = stripMaskMaterialization(mask);
  auto fail = [&](const Twine &message) {
    if (reason)
      *reason = message.str();
    return false;
  };

  if (auto createMask = mask.getDefiningOp<VMICreateMaskOp>()) {
    auto activeConstant =
        createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant)
      return fail("create_mask active_lanes is dynamic");
    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr)
      return fail("create_mask active_lanes is not an integer constant");
    return activeAttr.getInt() >= expectedLanes
               ? true
               : fail("create_mask active_lanes is smaller than the logical "
                      "lane count");
  }

  if (auto constantMask = mask.getDefiningOp<VMIConstantMaskOp>()) {
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantMask.getValue());
    if (!denseAttr)
      return fail("constant_mask is not a dense integer mask");
    if (denseAttr.getNumElements() != expectedLanes)
      return fail("constant_mask element count does not match the logical "
                  "lane count");
    auto values = denseAttr.getValues<bool>();
    for (bool value : values)
      if (!value)
        return fail("constant_mask contains an inactive lane");
    return true;
  }

  return fail("mask is not a static all-active create_mask or constant_mask");
}

LogicalResult
checkSupportedExpandLoadShape(const VMITargetCapabilityRegistry &capabilities,
                              VMIExpandLoadOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  auto passthruType = cast<VMIVRegType>(op.getPassthru().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  VMILayoutAttr passthruLayout = passthruType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMIMemoryAccessPlan accessPlan = buildReadAccessPlan(
      capabilities, op.getSource(), op.getSource().getType(), resultType,
      getConstantIndexValue(op.getOffset()),
      VMIMemoryValidMaskKind::ExplicitMask);
  if (!accessPlan.targetCapability.isSupported())
    return fail(accessPlan.targetCapability.reason);
  if (!resultLayout || !passthruLayout || !maskLayout)
    return fail("requires assigned result, passthru, and mask layouts");
  if (!resultLayout.isContiguous() || !passthruLayout.isContiguous() ||
      !maskLayout.isContiguous())
    return fail("requires contiguous result, passthru, and mask layouts");

  std::string maskReason;
  bool staticAllActive = isStaticAllActiveMask(
      op.getMask(), resultType.getElementCount(), &maskReason);

  std::string fullChunkReason;
  if (staticAllActive &&
      succeeded(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return success();

  if (staticAllActive && accessPlan.safeReadProof.proven)
    return success();

  std::string allActivePathReason;
  if (!staticAllActive) {
    allActivePathReason =
        maskReason.empty() ? "requires static all-active mask" : maskReason;
  } else {
    requireUnavailableReadFallback(accessPlan);
    allActivePathReason =
        (Twine("requires full physical chunks or statically safe full-read "
               "footprint; value ") +
         fullChunkReason + ", safe-read proof " +
         accessPlan.safeReadProof.reason +
         "; fallback decision: " + accessPlan.fallbackDecision.reason)
            .str();
  }

  VMICapabilityResult sourceCapability = capabilities.supportsUBPointerMemory(
      op.getSource().getType(), "source", "pto.vgather2_bc",
      "pto.vgather2_bc reads only UB");
  if (!sourceCapability.isSupported()) {
    if (!isa<PtrType>(op.getSource().getType()))
      return fail(Twine("runtime-mask path ") + sourceCapability.reason +
                  "; all-active path " + allActivePathReason);
    return fail(Twine("runtime-mask path ") + sourceCapability.reason);
  }
  if (pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 32)
    return fail("runtime-mask path currently requires 32-bit result element "
                "type so prefix indices and gather result lane counts match");
  if (maskType.getGranularity() != "b32")
    return fail("runtime-mask path requires b32 mask granularity");

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  FailureOr<int64_t> passthruArity = getVMIPhysicalArity(passthruType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(resultArity) || failed(passthruArity) || failed(maskArity))
    return fail("runtime-mask path requires computable physical arity");
  if (*resultArity != 1 || *passthruArity != 1 || *maskArity != 1)
    return fail("runtime-mask path currently supports only one physical "
                "chunk because prefix indices must not reset across chunks");

  std::string passthruReason;
  std::string maskFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &fullChunkReason)))
    return fail(Twine("runtime-mask result requires full physical chunks; ") +
                fullChunkReason);
  if (failed(checkFullDataPhysicalChunks(passthruType, &passthruReason)))
    return fail(Twine("runtime-mask passthru requires full physical chunks; ") +
                passthruReason);
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason)))
    return fail(Twine("runtime-mask mask requires full physical chunks; ") +
                maskFullReason);

  return success();
}

LogicalResult
checkSupportedMaskedStoreShape(const VMITargetCapabilityRegistry &capabilities,
                               VMIVRegType valueType, VMIMaskType maskType,
                               Value destination, Type destinationType,
                               std::string *reason) {
  VMIMemoryAccessPlan accessPlan =
      buildWriteAccessPlan(capabilities, destination, destinationType,
                           valueType, VMIMemoryWriteMaskKind::ExplicitMask);
  if (!accessPlan.targetCapability.isSupported()) {
    if (reason)
      *reason = accessPlan.targetCapability.reason;
    return failure();
  }

  std::string valueReason;
  std::string maskReason;
  if (succeeded(checkFullDataPhysicalChunks(valueType, &valueReason)) &&
      succeeded(checkFullVMIPhysicalChunks(maskType, &maskReason)))
    return success();

  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout)
    return fail("requires assigned value and mask layouts");

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(maskArity) || *valueArity != *maskArity)
    return fail("requires matching value/mask physical arity");

  std::string valueMaterializationReason;
  FailureOr<int64_t> valueParts = getContiguousMaterializationPartCount(
      valueType, &valueMaterializationReason);
  if (failed(valueParts))
    return fail(Twine("value cannot materialize to contiguous; value ") +
                valueReason + ", materialization " +
                valueMaterializationReason);

  std::string maskMaterializationReason;
  FailureOr<int64_t> maskParts = getContiguousMaterializationPartCount(
      maskType, &maskMaterializationReason);
  if (failed(maskParts))
    return fail(Twine("mask cannot materialize to contiguous; mask ") +
                maskReason + ", materialization " + maskMaterializationReason);
  if (*valueParts != *maskParts)
    return fail(
        "requires value/mask contiguous materialization arity to match");
  return success();
}

FailureOr<int64_t> getContiguousActiveDataLanes(VMIVRegType vmiType,
                                                int64_t chunk) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  int64_t remaining = vmiType.getElementCount() - chunk * *lanesPerPart;
  return std::clamp<int64_t>(remaining, 0, *lanesPerPart);
}

FailureOr<Value> createContiguousStoreMask(Location loc, VMIVRegType vmiType,
                                           int64_t chunk, VRegType vregType,
                                           PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  FailureOr<int64_t> activeLanes = getContiguousActiveDataLanes(vmiType, chunk);
  if (failed(activeLanes))
    return failure();
  if (*activeLanes == *lanesPerPart)
    return createAllTrueMaskForVReg(loc, vregType, rewriter);

  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(vregType, rewriter.getContext());
  if (failed(maskType))
    return failure();
  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, *maskType, createI32Constant(loc, *activeLanes, rewriter), rewriter);
  if (failed(maskAndRemaining))
    return failure();
  return maskAndRemaining->first;
}

FailureOr<Value> createMaskedStorePredicate(Location loc, VMIVRegType vmiType,
                                            int64_t chunk, Value userMask,
                                            VRegType vregType,
                                            PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(vmiType.getElementType());
  if (failed(lanesPerPart))
    return failure();

  FailureOr<int64_t> activeLanes = getContiguousActiveDataLanes(vmiType, chunk);
  if (failed(activeLanes))
    return failure();
  if (*activeLanes == *lanesPerPart)
    return userMask;

  auto maskType = dyn_cast<MaskType>(userMask.getType());
  if (!maskType)
    return failure();
  FailureOr<Value> tailMask =
      createContiguousStoreMask(loc, vmiType, chunk, vregType, rewriter);
  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  if (failed(tailMask) || failed(allTrue))
    return failure();
  return rewriter.create<PandOp>(loc, maskType, userMask, *tailMask, *allTrue)
      .getResult();
}

FailureOr<SmallVector<int64_t>>
computeShuffleForwardingSourceParts(VMIShuffleOp op, std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<int64_t>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known lanes per physical part");

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty())
    return fail("requires non-empty indices");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires assigned result layout");

  SmallVector<int64_t> sourceFlatIndices;
  for (int64_t resultPart = 0; resultPart < *resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getDataChunksInPart(resultType, resultPart);
    if (failed(resultChunks))
      return fail("requires known result physical chunks");

    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      std::optional<int64_t> sourcePart;
      std::optional<int64_t> sourceChunk;
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultType, resultPart, resultChunk, lane);
        if (failed(padding))
          return fail("failed to classify result padding lanes");
        if (*padding)
          continue;

        FailureOr<int64_t> resultLogicalLane =
            mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
        if (failed(resultLogicalLane) ||
            *resultLogicalLane >= static_cast<int64_t>(indices.size()))
          return fail("failed to map result lane");

        FailureOr<VMIPhysicalLane> sourcePhysical =
            mapLogicalLaneToPhysical(sourceType, indices[*resultLogicalLane]);
        if (failed(sourcePhysical))
          return fail("failed to map source lane");
        if (sourcePhysical->lane != lane)
          return fail("requires same-lane physical chunks");

        if (!sourcePart) {
          sourcePart = sourcePhysical->part;
          sourceChunk = sourcePhysical->chunk;
          continue;
        }
        if (*sourcePart != sourcePhysical->part ||
            *sourceChunk != sourcePhysical->chunk)
          return fail("requires one source chunk per result chunk");
      }

      if (!sourcePart || !sourceChunk)
        return fail("requires at least one logical lane per result chunk");
      FailureOr<int64_t> sourceFlatIndex =
          getDataFlatPartIndex(sourceType, *sourcePart, *sourceChunk);
      if (failed(sourceFlatIndex))
        return fail("source part range is out of bounds");
      sourceFlatIndices.push_back(*sourceFlatIndex);
    }
  }

  return sourceFlatIndices;
}

struct ShuffleVselrPlan {
  int64_t sourceFlatIndex = 0;
  int64_t baseLane = 0;
  bool descending = false;
};

FailureOr<int64_t> computeShuffleLane0SplatSourcePart(VMIShuffleOp op,
                                                      std::string *reason) {
  auto fail = [&](const Twine &message) -> FailureOr<int64_t> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty())
    return fail("requires non-empty indices");
  if (!llvm::all_of(indices, [](int64_t index) { return index == 0; }))
    return fail("requires every result lane to select source lane 0");

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  FailureOr<VMIPhysicalLane> sourceLane =
      mapLogicalLaneToPhysical(sourceType, 0);
  if (failed(sourceLane))
    return fail("failed to map source lane 0");
  FailureOr<int64_t> sourceFlatIndex =
      getDataFlatPartIndex(sourceType, sourceLane->part, sourceLane->chunk);
  if (failed(sourceFlatIndex))
    return fail("source lane 0 part range is out of bounds");
  return *sourceFlatIndex;
}

FailureOr<SmallVector<ShuffleVselrPlan>>
computeShuffleVselrPlans(VMIShuffleOp op, std::string *reason) {
  auto fail =
      [&](const Twine &message) -> FailureOr<SmallVector<ShuffleVselrPlan>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  FailureOr<int64_t> lanesPerPart =
      getDataLanesPerPart(sourceType.getElementType());
  if (failed(lanesPerPart))
    return fail("requires known lanes per physical part");

  ArrayRef<int64_t> indices = op.getIndices();
  if (indices.empty())
    return fail("requires non-empty indices");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires assigned result layout");

  SmallVector<ShuffleVselrPlan> plans;
  for (int64_t resultPart = 0; resultPart < *resultFactor; ++resultPart) {
    FailureOr<int64_t> resultChunks =
        getDataChunksInPart(resultType, resultPart);
    if (failed(resultChunks))
      return fail("requires known result physical chunks");

    for (int64_t resultChunk = 0; resultChunk < *resultChunks; ++resultChunk) {
      std::optional<int64_t> sourcePart;
      std::optional<int64_t> sourceChunk;
      std::optional<int64_t> baseLane;
      std::optional<bool> descending;
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultType, resultPart, resultChunk, lane);
        if (failed(padding) || *padding)
          return fail("requires full physical result chunks");

        FailureOr<int64_t> resultLogicalLane =
            mapPhysicalLaneToLogical(resultType, resultPart, resultChunk, lane);
        if (failed(resultLogicalLane) ||
            *resultLogicalLane >= static_cast<int64_t>(indices.size()))
          return fail("failed to map result lane");

        FailureOr<VMIPhysicalLane> sourcePhysical =
            mapLogicalLaneToPhysical(sourceType, indices[*resultLogicalLane]);
        if (failed(sourcePhysical))
          return fail("failed to map source lane");

        if (!sourcePart) {
          sourcePart = sourcePhysical->part;
          sourceChunk = sourcePhysical->chunk;
          baseLane = sourcePhysical->lane;
          continue;
        }

        if (*sourcePart != sourcePhysical->part ||
            *sourceChunk != sourcePhysical->chunk)
          return fail("requires one source chunk per result chunk");

        int64_t ascExpected = *baseLane + lane;
        int64_t descExpected = *baseLane - lane;
        bool asc = sourcePhysical->lane == ascExpected;
        bool desc = sourcePhysical->lane == descExpected;
        if (!asc && !desc)
          return fail("requires ASC or DESC affine source lane indices");

        bool laneDescending = desc && !asc;
        if (!descending) {
          descending = laneDescending;
          continue;
        }
        if (*descending != laneDescending)
          return fail("requires one index order per result chunk");
      }

      FailureOr<int64_t> sourceFlatIndex =
          getDataFlatPartIndex(sourceType, *sourcePart, *sourceChunk);
      if (failed(sourceFlatIndex))
        return fail("source part range is out of bounds");
      plans.push_back(ShuffleVselrPlan{*sourceFlatIndex, *baseLane,
                                       descending.value_or(false)});
    }
  }

  return plans;
}

struct ConstantMaskChunkMaterialization {
  SmallVector<int8_t> activeLanes;
};

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeConstantMaskMaterialization(VMIConstantMaskOp op, std::string *reason) {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<ConstantMaskChunkMaterialization>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto denseAttr = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!denseAttr)
    return fail("only dense integer mask constants are supported");

  auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout ||
      !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity()))
    return fail("requires concrete layout and granularity");

  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(resultVMIType.getGranularity());
  if (failed(lanesPerPart))
    return fail("requires known physical mask lanes per part");

  auto boolValues = denseAttr.getValues<bool>();
  int64_t factor = layout.isDeinterleaved() ? layout.getFactor() : 1;
  SmallVector<ConstantMaskChunkMaterialization> materializations;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      ConstantMaskChunkMaterialization materialization;
      materialization.activeLanes.reserve(*lanesPerPart);
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding) {
          materialization.activeLanes.push_back(0);
          continue;
        }
        anyLane = true;

        FailureOr<int64_t> logicalLane =
            mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
        if (failed(logicalLane))
          return fail("failed to map physical lane");
        materialization.activeLanes.push_back(boolValues[*logicalLane] ? 1 : 0);
      }
      if (!anyLane)
        break;
      materializations.push_back(std::move(materialization));
    }
  }

  return materializations;
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterializationForType(VMICreateGroupMaskOp op,
                                       VMIMaskType resultVMIType,
                                       std::string *reason) {
  auto fail = [&](const Twine &message)
      -> FailureOr<SmallVector<ConstantMaskChunkMaterialization>> {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto activeConstant =
      op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
  if (!activeConstant)
    return fail("requires constant active_elems_per_group");
  auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
  if (!activeAttr)
    return fail("active_elems_per_group must be an integer constant");

  VMILayoutAttr layout = resultVMIType.getLayoutAttr();
  if (!layout ||
      !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity()))
    return fail("requires concrete layout and granularity");

  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(resultVMIType.getGranularity());
  if (failed(lanesPerPart))
    return fail("requires known physical mask lanes per part");

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  if (numGroups <= 0 || groupSize <= 0 ||
      resultVMIType.getElementCount() != numGroups * groupSize)
    return fail("requires result lane count to match num_groups * group_size");

  int64_t activeElems = activeAttr.getInt();
  if (activeElems < 0)
    activeElems = 0;
  if (activeElems > groupSize)
    activeElems = groupSize;

  int64_t factor = layout.isDeinterleaved() ? layout.getFactor() : 1;
  SmallVector<ConstantMaskChunkMaterialization> materializations;
  for (int64_t part = 0; part < factor; ++part) {
    for (int64_t chunk = 0;; ++chunk) {
      bool anyLane = false;
      ConstantMaskChunkMaterialization materialization;
      materialization.activeLanes.reserve(*lanesPerPart);
      for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
        FailureOr<bool> padding =
            isPaddingLane(resultVMIType, part, chunk, lane);
        if (failed(padding))
          return fail("failed to map physical padding lane");
        if (*padding) {
          materialization.activeLanes.push_back(0);
          continue;
        }
        anyLane = true;

        FailureOr<int64_t> logicalLane =
            mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
        if (failed(logicalLane))
          return fail("failed to map physical lane");
        int64_t laneInGroup = *logicalLane % groupSize;
        materialization.activeLanes.push_back(laneInGroup < activeElems ? 1
                                                                        : 0);
      }
      if (!anyLane)
        break;
      materializations.push_back(std::move(materialization));
    }
  }

  return materializations;
}

FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
computeGroupMaskMaterialization(VMICreateGroupMaskOp op, std::string *reason) {
  return computeGroupMaskMaterializationForType(
      op, cast<VMIMaskType>(op.getResult().getType()), reason);
}

FailureOr<SmallVector<Value>> materializeDynamicContiguousGroupMask(
    VMICreateGroupMaskOp op, Value activeElemsPerGroup,
    VMIMaskType contiguousVMIType, TypeRange resultTypes,
    PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  VMILayoutAttr layout = contiguousVMIType.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return fail("dynamic create_group_mask requires contiguous seed layout");
  if (contiguousVMIType.getGranularity() != "b32")
    return fail("dynamic create_group_mask currently requires b32 "
                "granularity");

  int64_t numGroups = op.getNumGroupsAttr().getInt();
  int64_t groupSize = op.getGroupSizeAttr().getInt();
  if (numGroups <= 0 || groupSize <= 0 ||
      contiguousVMIType.getElementCount() != numGroups * groupSize)
    return fail("dynamic create_group_mask requires result lane count to "
                "match num_groups * group_size");

  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(contiguousVMIType.getGranularity());
  FailureOr<int64_t> arity = getVMIPhysicalArity(contiguousVMIType);
  if (failed(lanesPerPart) || failed(arity) || *arity < 1)
    return fail("dynamic create_group_mask requires computable physical "
                "mask chunks");
  if (static_cast<int64_t>(resultTypes.size()) != *arity)
    return fail("dynamic create_group_mask physical result count mismatch");
  if (groupSize > *lanesPerPart || (*lanesPerPart % groupSize) != 0)
    return fail("dynamic create_group_mask currently requires group_size to "
                "divide one physical b32 predicate chunk");

  std::optional<int64_t> shift = getPowerOfTwoLog2(groupSize);
  if (!shift)
    return fail("dynamic create_group_mask currently requires power-of-two "
                "group_size");

  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();
  Type i32 = rewriter.getI32Type();
  auto indexVectorType = VRegType::get(ctx, *lanesPerPart, i32);
  Value activeI32 =
      clampDynamicActiveLanes(loc, activeElemsPerGroup, groupSize, rewriter);

  SmallVector<Value> results;
  results.reserve(resultTypes.size());
  for (Type resultType : resultTypes) {
    auto maskType = dyn_cast<MaskType>(resultType);
    if (!maskType || !maskType.isB32())
      return fail("dynamic create_group_mask result must be b32 mask");

    FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
    if (failed(allMask))
      return fail("failed to create dynamic create_group_mask all mask");

    Value zero = createI32Constant(loc, 0, rewriter);
    Value lane =
        rewriter.create<VciOp>(loc, indexVectorType, zero, StringAttr{})
            .getResult();

    Value col = lane;
    if (groupSize != *lanesPerPart) {
      Value shiftScalar = createI16Constant(loc, *shift, rewriter);
      Value group = rewriter
                        .create<VshrsOp>(loc, indexVectorType, lane,
                                         shiftScalar, *allMask)
                        .getResult();
      Value groupBase = rewriter
                            .create<VshlsOp>(loc, indexVectorType, group,
                                             shiftScalar, *allMask)
                            .getResult();
      col = rewriter
                .create<VsubOp>(loc, indexVectorType, lane, groupBase, *allMask)
                .getResult();
    }

    results.push_back(rewriter
                          .create<VcmpsOp>(loc, maskType, col, activeI32,
                                           *allMask,
                                           rewriter.getStringAttr("lt"))
                          .getResult());
  }

  return results;
}

std::optional<int64_t> getPrefixActiveLaneCount(ArrayRef<int8_t> activeLanes) {
  bool seenInactive = false;
  int64_t activeCount = 0;
  for (int8_t active : activeLanes) {
    if (active) {
      if (seenInactive)
        return std::nullopt;
      ++activeCount;
      continue;
    }
    seenInactive = true;
  }
  return activeCount;
}

FailureOr<Value> materializePrefixMask(Location loc, MaskType maskType,
                                       int64_t activeLanes,
                                       int64_t lanesPerPart,
                                       PatternRewriter &rewriter) {
  std::optional<std::string> pattern =
      getPrefixPattern(activeLanes, lanesPerPart);
  if (pattern)
    return createPatternMask(loc, maskType, *pattern, rewriter);

  FailureOr<std::pair<Value, Value>> maskAndRemaining = createRuntimePrefixMask(
      loc, maskType, createI32Constant(loc, activeLanes, rewriter), rewriter);
  if (failed(maskAndRemaining))
    return failure();
  return maskAndRemaining->first;
}

FailureOr<Value> materializeConstantMaskChunk(Location loc, MaskType maskType,
                                              ArrayRef<int8_t> activeLanes,
                                              PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  if (failed(lanesPerPart) ||
      static_cast<int64_t>(activeLanes.size()) != *lanesPerPart)
    return failure();

  if (std::optional<int64_t> prefixCount =
          getPrefixActiveLaneCount(activeLanes))
    return materializePrefixMask(loc, maskType, *prefixCount, *lanesPerPart,
                                 rewriter);

  FailureOr<Value> allTrue = createAllTrueMask(loc, maskType, rewriter);
  if (failed(allTrue))
    return failure();

  Value result;
  int64_t lane = 0;
  while (lane < *lanesPerPart) {
    while (lane < *lanesPerPart && !activeLanes[lane])
      ++lane;
    if (lane >= *lanesPerPart)
      break;

    int64_t runBegin = lane;
    while (lane < *lanesPerPart && activeLanes[lane])
      ++lane;
    int64_t runEnd = lane;

    FailureOr<Value> prefixEnd =
        materializePrefixMask(loc, maskType, runEnd, *lanesPerPart, rewriter);
    if (failed(prefixEnd))
      return failure();

    Value runMask = *prefixEnd;
    if (runBegin != 0) {
      FailureOr<Value> prefixBegin = materializePrefixMask(
          loc, maskType, runBegin, *lanesPerPart, rewriter);
      if (failed(prefixBegin))
        return failure();
      Value notPrefixBegin =
          rewriter.create<PnotOp>(loc, maskType, *prefixBegin, *allTrue)
              .getResult();
      runMask = rewriter
                    .create<PandOp>(loc, maskType, *prefixEnd, notPrefixBegin,
                                    *allTrue)
                    .getResult();
    }

    if (!result) {
      result = runMask;
      continue;
    }
    result = rewriter.create<PorOp>(loc, maskType, result, runMask, *allTrue)
                 .getResult();
  }

  if (result)
    return result;
  return materializePrefixMask(loc, maskType, 0, *lanesPerPart, rewriter);
}

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter);

Value createChunkOffset(Location loc, Value baseOffset, int64_t laneOffset,
                        PatternRewriter &rewriter) {
  if (laneOffset == 0)
    return baseOffset;
  Value delta = rewriter.create<arith::ConstantIndexOp>(loc, laneOffset);
  return rewriter.create<arith::AddIOp>(loc, baseOffset, delta).getResult();
}

Value createGroupChunkOffset(Location loc, Value baseOffset, Value rowStride,
                             int64_t group, int64_t inGroupLaneOffset,
                             PatternRewriter &rewriter) {
  Value offset = baseOffset;
  if (group != 0) {
    Value groupIndex = rewriter.create<arith::ConstantIndexOp>(loc, group);
    Value rowOffset =
        rewriter.create<arith::MulIOp>(loc, rowStride, groupIndex).getResult();
    offset = rewriter.create<arith::AddIOp>(loc, offset, rowOffset).getResult();
  }
  return createChunkOffset(loc, offset, inGroupLaneOffset, rewriter);
}

LogicalResult checkContiguousFullGroupChunks(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t *lanesPerPart,
    int64_t *groupCount, int64_t *chunksPerGroup, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isContiguous())
    return fail("group op requires contiguous VMI layout");
  if (failed(checkFullDataPhysicalChunks(type, nullptr)))
    return fail("group op requires full physical chunks");
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes))
    return fail("group op requires known physical lanes per part");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("group op requires derived group size to evenly divide lane "
                "count");
  if (groupSize % *lanes != 0)
    return fail("group op currently requires group size to be a multiple of "
                "physical lanes per part");

  *lanesPerPart = *lanes;
  *groupCount = type.getElementCount() / groupSize;
  *chunksPerGroup = groupSize / *lanes;
  return success();
}

LogicalResult checkFullGroupSlotSourceShape(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t numGroups,
    int64_t *lanesPerPart, int64_t *groupCount, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout || !layout.isGroupSlots() || layout.getNumGroups() != numGroups)
    return fail("group slot op requires matching num_groups VMI layout");
  if (failed(checkFullDataPhysicalChunks(type, nullptr)))
    return fail("group slot op requires full physical chunks");
  FailureOr<int64_t> lanes = getDataLanesPerPart(type.getElementType());
  if (failed(lanes))
    return fail("group slot op requires known physical lanes per part");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("group slot op requires derived group size to evenly divide "
                "lane count");
  if (*lanes % groupSize != 0 && groupSize % *lanes != 0)
    return fail("group slot op requires group size to divide or be a "
                "multiple of physical lanes per part");

  *lanesPerPart = *lanes;
  *groupCount = type.getElementCount() / groupSize;
  return success();
}

LogicalResult checkFullGroupBroadcastResultShape(
    Operation *op, VMIVRegType type, int64_t groupSize, int64_t lanesPerPart,
    int64_t *layoutFactor, int64_t *groupCount, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) {
    return rewriter.notifyMatchFailure(op, message);
  };

  VMILayoutAttr layout = type.getLayoutAttr();
  if (!layout)
    return fail("group_broadcast result requires assigned VMI layout");
  if (layout.isGroupSlots())
    return fail("group_broadcast result requires a dense VMI layout");
  if (failed(checkFullDataPhysicalChunks(type, nullptr)))
    return fail("group_broadcast result requires full physical chunks");
  FailureOr<int64_t> resultLanes = getDataLanesPerPart(type.getElementType());
  if (failed(resultLanes) || *resultLanes != lanesPerPart)
    return fail("group_broadcast result requires matching physical lanes");
  if (groupSize <= 0 || type.getElementCount() % groupSize != 0)
    return fail("group_broadcast result requires derived group size to evenly "
                "divide lane count");
  FailureOr<int64_t> factor = getDataLayoutFactor(type);
  if (failed(factor))
    return fail("group_broadcast result requires known layout factor");

  if (*factor == 1) {
    if (lanesPerPart % groupSize != 0 && groupSize % lanesPerPart != 0)
      return fail("group_broadcast contiguous result requires group size to "
                  "divide or be a multiple of physical lanes per part");
  } else {
    bool blockFragmentSmallGroup =
        layout.isDeinterleaved() && layout.getBlockElems() > 1 &&
        groupSize < lanesPerPart && lanesPerPart % layout.getBlockElems() == 0;
    int64_t logicalSpanPerResultChunk = lanesPerPart * *factor;
    if (!blockFragmentSmallGroup &&
        (groupSize < lanesPerPart ||
         groupSize % logicalSpanPerResultChunk != 0))
      return fail("group_broadcast deinterleaved result requires every "
                  "physical result chunk to stay within one logical group");
  }

  *layoutFactor = *factor;
  *groupCount = type.getElementCount() / groupSize;
  return success();
}

FailureOr<Value> createZeroVector(Location loc, VRegType type,
                                  PatternRewriter &rewriter) {
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, type.getElementType(), 0, rewriter);
  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, type, rewriter);
  if (failed(zero) || failed(mask))
    return failure();
  return rewriter
      .create<VdupOp>(loc, type, *zero, *mask,
                      /*position=*/nullptr)
      .getResult();
}

FailureOr<Value> createLaneRangeMask(Location loc, MaskType maskType,
                                     int64_t begin, int64_t end,
                                     PatternRewriter &rewriter) {
  FailureOr<int64_t> lanesPerPart =
      getMaskLanesPerPart(maskType.getGranularity());
  if (failed(lanesPerPart) || begin < 0 || begin > end || end > *lanesPerPart)
    return failure();
  SmallVector<int8_t> active(*lanesPerPart, 0);
  for (int64_t lane = begin; lane < end; ++lane)
    active[lane] = 1;
  return materializeConstantMaskChunk(loc, maskType, active, rewriter);
}

FailureOr<Value> createGroupSlotIndexVector(Location loc, VRegType indexType,
                                            int64_t groupSize,
                                            int64_t baseGroupSlot,
                                            PatternRewriter &rewriter) {
  int64_t lanesPerPart = indexType.getElementCount();
  FailureOr<Value> baseScalar = createScalarOffsetConstant(
      loc, indexType.getElementType(), baseGroupSlot, rewriter);
  FailureOr<MaskType> maskType =
      getMaskTypeForVReg(indexType, rewriter.getContext());
  FailureOr<Value> allMask = createAllTrueMaskForVReg(loc, indexType, rewriter);
  if (failed(baseScalar) || failed(maskType) || failed(allMask))
    return failure();
  Value result = rewriter
                     .create<VdupOp>(loc, indexType, *baseScalar, *allMask,
                                     /*position=*/nullptr)
                     .getResult();
  if (groupSize >= lanesPerPart)
    return result;
  if (lanesPerPart % groupSize != 0)
    return failure();

  int64_t groupsPerChunk = lanesPerPart / groupSize;
  for (int64_t localGroup = 1; localGroup < groupsPerChunk; ++localGroup) {
    FailureOr<Value> groupScalar = createScalarOffsetConstant(
        loc, indexType.getElementType(), baseGroupSlot + localGroup, rewriter);
    FailureOr<Value> laneMask =
        createLaneRangeMask(loc, *maskType, localGroup * groupSize,
                            (localGroup + 1) * groupSize, rewriter);
    if (failed(groupScalar) || failed(laneMask))
      return failure();
    Value splat = rewriter
                      .create<VdupOp>(loc, indexType, *groupScalar, *allMask,
                                      /*position=*/nullptr)
                      .getResult();
    result = rewriter.create<VselOp>(loc, indexType, splat, result, *laneMask)
                 .getResult();
  }
  return result;
}

std::optional<std::string> getX2MemoryDistToken(Type elementType,
                                                StringRef prefix) {
  unsigned elementBits = pto::getPTOStorageElemBitWidth(elementType);
  if (elementBits != 8 && elementBits != 16 && elementBits != 32)
    return std::nullopt;
  return (Twine(prefix) + "_B" + Twine(elementBits)).str();
}

std::optional<StringRef> getVPTOCmpMode(StringRef predicate) {
  if (predicate == "eq" || predicate == "ne" || predicate == "lt" ||
      predicate == "le" || predicate == "gt" || predicate == "ge")
    return predicate;
  if (predicate == "oeq")
    return StringRef("eq");
  if (predicate == "one")
    return StringRef("ne");
  if (predicate == "olt")
    return StringRef("lt");
  if (predicate == "ole")
    return StringRef("le");
  if (predicate == "ogt")
    return StringRef("gt");
  if (predicate == "oge")
    return StringRef("ge");
  if (predicate == "slt")
    return StringRef("lt");
  if (predicate == "sle")
    return StringRef("le");
  if (predicate == "sgt")
    return StringRef("gt");
  if (predicate == "sge")
    return StringRef("ge");
  return std::nullopt;
}

LogicalResult checkSupportedComparePredicate(Operation *op,
                                             StringRef predicate) {
  if (getVPTOCmpMode(predicate))
    return success();
  return op->emitError()
         << kVMIDiagUnsupportedPrefix << "compare predicate " << predicate
         << " cannot be lowered to pto.vcmp; supported predicates are "
            "eq/ne/lt/le/gt/ge, ordered FP forms oeq/one/olt/ole/ogt/oge, "
            "and signed integer forms slt/sle/sgt/sge";
}

struct OneToNVMIUnpackOpPattern : OneToNOpConversionPattern<VMIUnpackOp> {
  using OneToNOpConversionPattern<VMIUnpackOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIUnpackOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    if (sourceParts.size() != op->getNumResults())
      return rewriter.notifyMatchFailure(
          op, "converted source part count must match unpack results");
    rewriter.replaceOp(op, sourceParts, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIPackOpPattern : OneToNOpConversionPattern<VMIPackOp> {
  using OneToNOpConversionPattern<VMIPackOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIPackOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<int64_t> arity = getVMIPhysicalArity(op.getResult().getType());
    if (failed(arity) ||
        static_cast<int64_t>(adaptor.getFlatOperands().size()) != *arity)
      return rewriter.notifyMatchFailure(
          op, "pack part count must match converted VMI result arity");
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());
    return success();
  }
};

LogicalResult verifyIdentityPartForwarding(Operation *op,
                                           ValueRange sourceParts,
                                           TypeRange resultTypes,
                                           PatternRewriter &rewriter) {
  if (sourceParts.size() != resultTypes.size())
    return rewriter.notifyMatchFailure(
        op, "source and result physical arity mismatch");
  for (auto [part, resultType] : llvm::zip_equal(sourceParts, resultTypes)) {
    if (part.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "helper requires non-identity physical materialization");
  }
  return success();
}

FailureOr<SmallVector<Value>> materializeDataLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "layout materialization requires assigned source/result layouts");
    return failure();
  }

  if (sourceLayout == resultLayout) {
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  bool deint2ToContiguous = sourceLayout.isDeinterleaved() &&
                            sourceLayout.getFactor() == 2 &&
                            resultLayout.isContiguous();
  bool contiguousToDeint2 = sourceLayout.isContiguous() &&
                            resultLayout.isDeinterleaved() &&
                            resultLayout.getFactor() == 2;
  if (deint2ToContiguous || contiguousToDeint2) {
    if (sourceParts.size() != resultTypes.size() || sourceParts.empty() ||
        sourceParts.size() % 2 != 0) {
      (void)rewriter.notifyMatchFailure(
          op, "deinterleaved=2 layout materialization requires 2*N parts");
      return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();

    int64_t groups = sourceParts.size() / 2;
    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    if (deint2ToContiguous) {
      for (int64_t i = 0; i < groups; ++i) {
        auto materialize = rewriter.create<VintlvOp>(
            op->getLoc(), resultTypes[2 * i], resultTypes[2 * i + 1],
            sourceParts[i], sourceParts[groups + i]);
        results.append({materialize.getLow(), materialize.getHigh()});
      }
    } else {
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      part0.reserve(groups);
      part1.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        auto materialize = rewriter.create<VdintlvOp>(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            sourceParts[2 * i], sourceParts[2 * i + 1]);
        part0.push_back(materialize.getLow());
        part1.push_back(materialize.getHigh());
      }
      results.append(part0);
      results.append(part1);
    }
    return results;
  }

  bool deint4ToContiguous = sourceLayout.isDeinterleaved() &&
                            sourceLayout.getFactor() == 4 &&
                            resultLayout.isContiguous();
  bool contiguousToDeint4 = sourceLayout.isContiguous() &&
                            resultLayout.isDeinterleaved() &&
                            resultLayout.getFactor() == 4;
  if (deint4ToContiguous || contiguousToDeint4) {
    if (sourceParts.size() != resultTypes.size() || sourceParts.empty() ||
        sourceParts.size() % 4 != 0) {
      (void)rewriter.notifyMatchFailure(
          op, "deinterleaved=4 layout materialization requires 4*N parts");
      return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();

    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    int64_t groups = sourceParts.size() / 4;
    if (deint4ToContiguous) {
      for (int64_t i = 0; i < groups; ++i) {
        Value p0 = sourceParts[i];
        Value p1 = sourceParts[groups + i];
        Value p2 = sourceParts[2 * groups + i];
        Value p3 = sourceParts[3 * groups + i];
        auto even = rewriter.create<VintlvOp>(op->getLoc(), resultTypes[4 * i],
                                              resultTypes[4 * i + 1], p0, p2);
        auto odd = rewriter.create<VintlvOp>(op->getLoc(), resultTypes[4 * i],
                                             resultTypes[4 * i + 1], p1, p3);
        auto low = rewriter.create<VintlvOp>(op->getLoc(), resultTypes[4 * i],
                                             resultTypes[4 * i + 1],
                                             even.getLow(), odd.getLow());
        auto high = rewriter.create<VintlvOp>(
            op->getLoc(), resultTypes[4 * i + 2], resultTypes[4 * i + 3],
            even.getHigh(), odd.getHigh());
        results.append(
            {low.getLow(), low.getHigh(), high.getLow(), high.getHigh()});
      }
    } else {
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      SmallVector<Value> part2;
      SmallVector<Value> part3;
      part0.reserve(groups);
      part1.reserve(groups);
      part2.reserve(groups);
      part3.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        auto low = rewriter.create<VdintlvOp>(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            sourceParts[4 * i], sourceParts[4 * i + 1]);
        auto high = rewriter.create<VdintlvOp>(
            op->getLoc(), resultTypes[2 * groups + i],
            resultTypes[3 * groups + i], sourceParts[4 * i + 2],
            sourceParts[4 * i + 3]);
        auto even = rewriter.create<VdintlvOp>(op->getLoc(), resultTypes[i],
                                               resultTypes[2 * groups + i],
                                               low.getLow(), high.getLow());
        auto odd = rewriter.create<VdintlvOp>(
            op->getLoc(), resultTypes[groups + i], resultTypes[3 * groups + i],
            low.getHigh(), high.getHigh());
        part0.push_back(even.getLow());
        part1.push_back(odd.getLow());
        part2.push_back(even.getHigh());
        part3.push_back(odd.getHigh());
      }
      results.append(part0);
      results.append(part1);
      results.append(part2);
      results.append(part3);
    }
    return results;
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI data layout materialization");
  return failure();
}

FailureOr<std::pair<Value, Value>>
createPredicateDintlv(Location loc, Type lowType, Type highType, Value lhs,
                      Value rhs, PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(lowType);
  if (!maskType || highType != lowType)
    return failure();
  if (maskType.isB8()) {
    auto op = rewriter.create<PdintlvB8Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB16()) {
    auto op = rewriter.create<PdintlvB16Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB32()) {
    auto op = rewriter.create<PdintlvB32Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  return failure();
}

FailureOr<std::pair<Value, Value>>
createPredicateIntlv(Location loc, Type lowType, Type highType, Value lhs,
                     Value rhs, PatternRewriter &rewriter) {
  auto maskType = dyn_cast<MaskType>(lowType);
  if (!maskType || highType != lowType)
    return failure();
  if (maskType.isB8()) {
    auto op = rewriter.create<PintlvB8Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB16()) {
    auto op = rewriter.create<PintlvB16Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  if (maskType.isB32()) {
    auto op = rewriter.create<PintlvB32Op>(loc, lowType, highType, lhs, rhs);
    return std::make_pair(op.getLow(), op.getHigh());
  }
  return failure();
}

FailureOr<SmallVector<Value>> materializeMaskLayoutConversion(
    Operation *op, ValueRange sourceParts, TypeRange resultTypes,
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
    PatternRewriter &rewriter) {
  if (!sourceLayout || !resultLayout) {
    (void)rewriter.notifyMatchFailure(
        op, "mask layout materialization requires assigned source/result "
            "layouts");
    return failure();
  }

  if (sourceLayout == resultLayout) {
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    return SmallVector<Value>(sourceParts.begin(), sourceParts.end());
  }

  bool deint2ToContiguous = sourceLayout.isDeinterleaved() &&
                            sourceLayout.getFactor() == 2 &&
                            resultLayout.isContiguous();
  bool contiguousToDeint2 = sourceLayout.isContiguous() &&
                            resultLayout.isDeinterleaved() &&
                            resultLayout.getFactor() == 2;
  if (deint2ToContiguous || contiguousToDeint2) {
    if (sourceParts.size() != resultTypes.size() || sourceParts.empty() ||
        sourceParts.size() % 2 != 0) {
      (void)rewriter.notifyMatchFailure(
          op, "deinterleaved=2 mask layout materialization requires 2*N "
              "parts");
      return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();

    int64_t groups = sourceParts.size() / 2;
    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    if (deint2ToContiguous) {
      for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> materialize = createPredicateIntlv(
            op->getLoc(), resultTypes[2 * i], resultTypes[2 * i + 1],
            sourceParts[i], sourceParts[groups + i], rewriter);
        if (failed(materialize))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate intlv mask type");
        results.append({materialize->first, materialize->second});
      }
    } else {
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      part0.reserve(groups);
      part1.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> materialize = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            sourceParts[2 * i], sourceParts[2 * i + 1], rewriter);
        if (failed(materialize))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate dintlv mask type");
        part0.push_back(materialize->first);
        part1.push_back(materialize->second);
      }
      results.append(part0);
      results.append(part1);
    }
    return results;
  }

  bool deint4ToContiguous = sourceLayout.isDeinterleaved() &&
                            sourceLayout.getFactor() == 4 &&
                            resultLayout.isContiguous();
  bool contiguousToDeint4 = sourceLayout.isContiguous() &&
                            resultLayout.isDeinterleaved() &&
                            resultLayout.getFactor() == 4;
  if (deint4ToContiguous || contiguousToDeint4) {
    if (sourceParts.size() != resultTypes.size() || sourceParts.empty() ||
        sourceParts.size() % 4 != 0) {
      (void)rewriter.notifyMatchFailure(
          op, "deinterleaved=4 mask layout materialization requires 4*N "
              "parts");
      return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();

    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    int64_t groups = sourceParts.size() / 4;
    if (deint4ToContiguous) {
      for (int64_t i = 0; i < groups; ++i) {
        Value p0 = sourceParts[i];
        Value p1 = sourceParts[groups + i];
        Value p2 = sourceParts[2 * groups + i];
        Value p3 = sourceParts[3 * groups + i];
        FailureOr<std::pair<Value, Value>> even =
            createPredicateIntlv(op->getLoc(), resultTypes[4 * i],
                                 resultTypes[4 * i + 1], p0, p2, rewriter);
        FailureOr<std::pair<Value, Value>> odd =
            createPredicateIntlv(op->getLoc(), resultTypes[4 * i],
                                 resultTypes[4 * i + 1], p1, p3, rewriter);
        if (failed(even) || failed(odd))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate intlv mask type");
        FailureOr<std::pair<Value, Value>> low = createPredicateIntlv(
            op->getLoc(), resultTypes[4 * i], resultTypes[4 * i + 1],
            even->first, odd->first, rewriter);
        FailureOr<std::pair<Value, Value>> high = createPredicateIntlv(
            op->getLoc(), resultTypes[4 * i + 2], resultTypes[4 * i + 3],
            even->second, odd->second, rewriter);
        if (failed(low) || failed(high))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate intlv mask type");
        results.append({low->first, low->second, high->first, high->second});
      }
    } else {
      SmallVector<Value> part0;
      SmallVector<Value> part1;
      SmallVector<Value> part2;
      SmallVector<Value> part3;
      part0.reserve(groups);
      part1.reserve(groups);
      part2.reserve(groups);
      part3.reserve(groups);
      for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> low = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[groups + i],
            sourceParts[4 * i], sourceParts[4 * i + 1], rewriter);
        FailureOr<std::pair<Value, Value>> high = createPredicateDintlv(
            op->getLoc(), resultTypes[2 * groups + i],
            resultTypes[3 * groups + i], sourceParts[4 * i + 2],
            sourceParts[4 * i + 3], rewriter);
        if (failed(low) || failed(high))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate dintlv mask type");
        FailureOr<std::pair<Value, Value>> even = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[2 * groups + i],
            low->first, high->first, rewriter);
        FailureOr<std::pair<Value, Value>> odd = createPredicateDintlv(
            op->getLoc(), resultTypes[groups + i], resultTypes[3 * groups + i],
            low->second, high->second, rewriter);
        if (failed(even) || failed(odd))
          return rewriter.notifyMatchFailure(
              op, "unsupported predicate dintlv mask type");
        part0.push_back(even->first);
        part1.push_back(odd->first);
        part2.push_back(even->second);
        part3.push_back(odd->second);
      }
      results.append(part0);
      results.append(part1);
      results.append(part2);
      results.append(part3);
    }
    return results;
  }

  (void)rewriter.notifyMatchFailure(
      op, "unsupported VMI mask layout materialization");
  return failure();
}

int getMaskGranularityRank(StringRef granularity) {
  if (granularity == "b8")
    return 0;
  if (granularity == "b16")
    return 1;
  if (granularity == "b32")
    return 2;
  return -1;
}

StringRef getMaskGranularityForRank(int rank) {
  switch (rank) {
  case 0:
    return "b8";
  case 1:
    return "b16";
  case 2:
    return "b32";
  default:
    return "";
  }
}

LogicalResult checkSupportedMaskGranularityMaterialization(
    const VMITargetCapabilityRegistry &capabilities, VMIMaskType sourceType,
    VMIMaskType resultType, std::string *reason) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (sourceType.getElementCount() != resultType.getElementCount())
    return fail("requires source and result mask lane counts to match");
  if (sourceType.getLayoutAttr() != resultType.getLayoutAttr())
    return fail("requires source and result mask layouts to match");

  VMICapabilityResult granularityCapability =
      capabilities.supportsMaskGranularityConversion(
          sourceType.getGranularity(), resultType.getGranularity());
  if (!granularityCapability.isSupported())
    return fail(granularityCapability.reason);

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(resultArity))
    return fail("requires computable source/result physical arity");
  if (*sourceArity < 1 || *resultArity < 1)
    return fail("requires non-empty source/result physical arity");

  return success();
}

FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityConversion(
    Operation *op, VMIMaskType sourceType, VMIMaskType resultType,
    ValueRange sourceParts, PatternRewriter &rewriter) {
  auto fail = [&](const Twine &message) -> FailureOr<SmallVector<Value>> {
    (void)rewriter.notifyMatchFailure(op, message);
    return failure();
  };

  int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  if (std::abs(sourceRank - resultRank) != 1)
    return fail("mask granularity conversion must be adjacent");

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> factor = getVMITypeLayoutFactor(sourceType);
  if (failed(sourceArity) || failed(factor) ||
      static_cast<int64_t>(sourceParts.size()) != *sourceArity)
    return fail("source mask part count does not match source VMI type");

  MLIRContext *ctx = op->getContext();
  auto partAttr = [&](StringRef part) { return StringAttr::get(ctx, part); };
  auto resultMaskType = MaskType::get(ctx, resultType.getGranularity());
  SmallVector<Value> results;

  int64_t sourceOffset = 0;
  for (int64_t part = 0; part < *factor; ++part) {
    FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
    FailureOr<int64_t> resultChunks = getVMITypeChunksInPart(resultType, part);
    if (failed(sourceChunks) || failed(resultChunks))
      return fail("requires computable source/result chunks per layout part");

    if (resultRank > sourceRank) {
      int64_t produced = 0;
      for (int64_t chunk = 0; chunk < *sourceChunks && produced < *resultChunks;
           ++chunk) {
        Value source = sourceParts[sourceOffset + chunk];
        results.push_back(rewriter
                              .create<PunpackOp>(op->getLoc(), resultMaskType,
                                                 source, partAttr("LOWER"))
                              .getResult());
        ++produced;
        if (produced >= *resultChunks)
          break;
        results.push_back(rewriter
                              .create<PunpackOp>(op->getLoc(), resultMaskType,
                                                 source, partAttr("HIGHER"))
                              .getResult());
        ++produced;
      }
      if (produced != *resultChunks)
        return fail("widening mask granularity conversion produced the wrong "
                    "number of result chunks");
    } else {
      Value allTrue;
      int64_t consumed = 0;
      for (int64_t chunk = 0; chunk < *resultChunks; ++chunk) {
        if (consumed >= *sourceChunks)
          return fail("narrowing mask granularity conversion ran out of "
                      "source chunks");
        Value lowerSource = sourceParts[sourceOffset + consumed++];
        Value packed = rewriter
                           .create<PpackOp>(op->getLoc(), resultMaskType,
                                            lowerSource, partAttr("LOWER"))
                           .getResult();
        if (consumed < *sourceChunks) {
          Value higherSource = sourceParts[sourceOffset + consumed++];
          Value higher = rewriter
                             .create<PpackOp>(op->getLoc(), resultMaskType,
                                              higherSource, partAttr("HIGHER"))
                             .getResult();
          if (!allTrue) {
            FailureOr<Value> mask =
                createAllTrueMask(op->getLoc(), resultMaskType, rewriter);
            if (failed(mask))
              return fail("failed to create all-true mask for ppack merge");
            allTrue = *mask;
          }
          packed = rewriter
                       .create<PorOp>(op->getLoc(), resultMaskType, packed,
                                      higher, allTrue)
                       .getResult();
        }
        results.push_back(packed);
      }
      if (consumed != *sourceChunks)
        return fail("narrowing mask granularity conversion left unused source "
                    "chunks");
    }

    sourceOffset += *sourceChunks;
  }

  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(resultArity) ||
      static_cast<int64_t>(results.size()) != *resultArity)
    return fail("mask granularity conversion result count mismatch");
  return results;
}

FailureOr<SmallVector<Value>> materializeMaskGranularityConversion(
    Operation *op, const VMITargetCapabilityRegistry &capabilities,
    VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts,
    PatternRewriter &rewriter) {
  std::string reason;
  if (failed(checkSupportedMaskGranularityMaterialization(
          capabilities, sourceType, resultType, &reason))) {
    (void)rewriter.notifyMatchFailure(op, reason);
    return failure();
  }

  int currentRank = getMaskGranularityRank(sourceType.getGranularity());
  int resultRank = getMaskGranularityRank(resultType.getGranularity());
  VMIMaskType currentType = sourceType;
  SmallVector<Value> currentParts(sourceParts.begin(), sourceParts.end());

  while (currentRank != resultRank) {
    currentRank += currentRank < resultRank ? 1 : -1;
    StringRef nextGranularity = getMaskGranularityForRank(currentRank);
    if (nextGranularity.empty()) {
      (void)rewriter.notifyMatchFailure(op,
                                        "invalid target mask granularity rank");
      return failure();
    }
    VMIMaskType nextType =
        VMIMaskType::get(op->getContext(), currentType.getElementCount(),
                         nextGranularity, currentType.getLayoutAttr());
    FailureOr<SmallVector<Value>> nextParts =
        materializeAdjacentMaskGranularityConversion(op, currentType, nextType,
                                                     currentParts, rewriter);
    if (failed(nextParts))
      return failure();
    currentType = nextType;
    currentParts = std::move(*nextParts);
  }

  return currentParts;
}

struct OneToNVMIEnsureLayoutOpPattern
    : OneToNOpConversionPattern<VMIEnsureLayoutOp> {
  using OneToNOpConversionPattern<VMIEnsureLayoutOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureLayoutOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(supports.canMaterializeDataLayout(sourceType, resultType,
                                                 &supportReason)))
      return rewriter.notifyMatchFailure(
          op,
          Twine("ensure_layout has no registered materialization support: ") +
              supportReason);
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    if (!sourceLayout || !resultLayout)
      return rewriter.notifyMatchFailure(
          op, "ensure_layout requires assigned source/result layouts");

    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(results))
      return failure();
    rewriter.replaceOp(op, *results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIEnsureMaskLayoutOpPattern
    : OneToNOpConversionPattern<VMIEnsureMaskLayoutOp> {
  using OneToNOpConversionPattern<
      VMIEnsureMaskLayoutOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIEnsureMaskLayoutOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(supports.canMaterializeMaskLayout(sourceType, resultType,
                                                 &supportReason)))
      return rewriter.notifyMatchFailure(
          op, Twine("ensure_mask_layout has no registered materialization "
                    "support: ") +
                  supportReason);
    if (sourceType.getGranularity() != resultType.getGranularity())
      return rewriter.notifyMatchFailure(
          op, "mask layout helper cannot also change granularity");
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();

    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(results))
      return failure();
    rewriter.replaceOp(op, *results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIEnsureMaskGranularityOpPattern
    : OneToNOpConversionPattern<VMIEnsureMaskGranularityOp> {
  OneToNVMIEnsureMaskGranularityOpPattern(
      TypeConverter &typeConverter, MLIRContext *context,
      const VMITargetCapabilityRegistry &capabilities)
      : OneToNOpConversionPattern<VMIEnsureMaskGranularityOp>(typeConverter,
                                                              context),
        capabilities(capabilities) {}

  LogicalResult
  matchAndRewrite(VMIEnsureMaskGranularityOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceType = cast<VMIMaskType>(op.getSource().getType());
    auto resultType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutSupport supports;
    std::string supportReason;
    if (failed(supports.canMaterializeMaskGranularity(sourceType, resultType,
                                                      &supportReason)))
      return rewriter.notifyMatchFailure(
          op, Twine("ensure_mask_granularity has no registered materialization "
                    "support: ") +
                  supportReason);
    if (sourceType.getLayout() != resultType.getLayout())
      return rewriter.notifyMatchFailure(
          op, "mask granularity helper cannot also change layout");

    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceType.getGranularity() != resultType.getGranularity()) {
      FailureOr<SmallVector<Value>> results =
          materializeMaskGranularityConversion(
              op, capabilities, sourceType, resultType, sourceParts, rewriter);
      if (failed(results))
        return failure();
      if (results->size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "mask granularity result arity mismatch");
      for (auto [result, type] : llvm::zip_equal(*results, resultTypes))
        if (result.getType() != type)
          return rewriter.notifyMatchFailure(
              op, "mask granularity result type mismatch");
      rewriter.replaceOp(op, *results, adaptor.getResultMapping());
      return success();
    }

    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes,
                                            rewriter)))
      return failure();
    rewriter.replaceOp(op, sourceParts, adaptor.getResultMapping());
    return success();
  }

private:
  const VMITargetCapabilityRegistry &capabilities;
};

struct OneToNVMIBroadcastOpPattern : OneToNOpConversionPattern<VMIBroadcastOp> {
  using OneToNOpConversionPattern<VMIBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange inputParts = adaptor.getValue();
    if (inputParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "broadcast input must convert to one value");
    bool inputIsVReg = isa<VMIVRegType>(op.getValue().getType());

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "broadcast result must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for broadcast mask");
      StringAttr position =
          inputIsVReg ? rewriter.getStringAttr("LOWEST") : StringAttr{};
      results.push_back(rewriter
                            .create<VdupOp>(op.getLoc(), resultType,
                                            inputParts.front(), *mask, position)
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

FailureOr<Value> createScalarOffsetConstant(Location loc, Type type,
                                            int64_t value,
                                            PatternRewriter &rewriter) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return rewriter
        .create<arith::ConstantOp>(loc, IntegerAttr::get(intType, value))
        .getResult();
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return rewriter
        .create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(floatType, static_cast<double>(value)))
        .getResult();
  }
  return failure();
}

FailureOr<Value> createIotaChunkBase(Location loc, Value base,
                                     int64_t laneOffset, StringRef order,
                                     PatternRewriter &rewriter) {
  if (laneOffset == 0)
    return base;

  FailureOr<Value> offset =
      createScalarOffsetConstant(loc, base.getType(), laneOffset, rewriter);
  if (failed(offset))
    return failure();

  if (isa<IntegerType>(base.getType())) {
    if (order == "DESC")
      return rewriter.create<arith::SubIOp>(loc, base, *offset).getResult();
    return rewriter.create<arith::AddIOp>(loc, base, *offset).getResult();
  }
  if (isa<FloatType>(base.getType())) {
    if (order == "DESC")
      return rewriter.create<arith::SubFOp>(loc, base, *offset).getResult();
    return rewriter.create<arith::AddFOp>(loc, base, *offset).getResult();
  }

  return failure();
}

FailureOr<Value> createIotaContiguousChunk(Location loc, Type resultType,
                                           Value base, int64_t laneOffset,
                                           StringAttr orderAttr,
                                           PatternRewriter &rewriter) {
  StringRef order = orderAttr ? orderAttr.getValue() : StringRef("ASC");
  FailureOr<Value> chunkBase =
      createIotaChunkBase(loc, base, laneOffset, order, rewriter);
  if (failed(chunkBase))
    return failure();
  return rewriter.create<VciOp>(loc, resultType, *chunkBase, orderAttr)
      .getResult();
}

FailureOr<Value> createIotaDeinterleavedChunk(Location loc, Type resultType,
                                              Value base, int64_t factor,
                                              int64_t part, int64_t chunk,
                                              int64_t lanesPerPart,
                                              StringAttr orderAttr,
                                              PatternRewriter &rewriter) {
  auto vregType = dyn_cast<VRegType>(resultType);
  if (!vregType)
    return failure();

  FailureOr<Value> mask = createAllTrueMaskForVReg(loc, vregType, rewriter);
  FailureOr<Value> zero =
      createScalarOffsetConstant(loc, base.getType(), 0, rewriter);
  FailureOr<Value> factorScalar =
      createScalarOffsetConstant(loc, base.getType(), factor, rewriter);
  if (failed(mask) || failed(zero) || failed(factorScalar))
    return failure();

  Value local =
      rewriter.create<VciOp>(loc, resultType, *zero, StringAttr{}).getResult();
  Value scaled =
      rewriter.create<VmulsOp>(loc, resultType, local, *factorScalar, *mask)
          .getResult();

  StringRef order = orderAttr ? orderAttr.getValue() : StringRef("ASC");
  int64_t partOffset = part + factor * chunk * lanesPerPart;
  FailureOr<Value> biasedBase =
      createIotaChunkBase(loc, base, partOffset, order, rewriter);
  if (failed(biasedBase))
    return failure();

  if (order == "DESC") {
    Value baseVector = rewriter
                           .create<VdupOp>(loc, resultType, *biasedBase, *mask,
                                           /*position=*/nullptr)
                           .getResult();
    return rewriter.create<VsubOp>(loc, resultType, baseVector, scaled, *mask)
        .getResult();
  }

  return rewriter.create<VaddsOp>(loc, resultType, scaled, *biasedBase, *mask)
      .getResult();
}

struct OneToNVMIIotaOpPattern : OneToNOpConversionPattern<VMIIotaOp> {
  using OneToNOpConversionPattern<VMIIotaOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIIotaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout)
      return rewriter.notifyMatchFailure(op, "iota requires assigned layout");

    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "iota requires known physical lanes per part");

    FailureOr<Value> base = getSingleValue(
        op, adaptor.getBase(), "iota base must convert to one value", rewriter);
    if (failed(base))
      return failure();

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());

    if (layout.isContiguous()) {
      for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
        if (!isa<VRegType>(resultType))
          return rewriter.notifyMatchFailure(op, "iota result must be vreg");
        FailureOr<Value> result = createIotaContiguousChunk(
            op.getLoc(), resultType, *base,
            static_cast<int64_t>(index) * *lanesPerPart, op.getOrderAttr(),
            rewriter);
        if (failed(result))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize contiguous iota chunk");
        results.push_back(*result);
      }
      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    int64_t factor = layout.getFactor();
    if (resultTypes.size() % factor != 0)
      return rewriter.notifyMatchFailure(
          op, "deinterleaved iota physical result count does not match "
              "layout factor");
    int64_t chunksPerPart = resultTypes.size() / factor;
    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
        Type resultType = resultTypes[part * chunksPerPart + chunk];
        FailureOr<Value> result = createIotaDeinterleavedChunk(
            op.getLoc(), resultType, *base, factor, part, chunk, *lanesPerPart,
            op.getOrderAttr(), rewriter);
        if (failed(result))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize deinterleaved iota chunk");
        results.push_back(*result);
      }
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIConstantOpPattern : OneToNOpConversionPattern<VMIConstantOp> {
  using OneToNOpConversionPattern<VMIConstantOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIConstantOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr || !denseAttr.isSplat())
      return rewriter.notifyMatchFailure(
          op, "only splat dense data constants are supported");
    auto splatAttr = dyn_cast<TypedAttr>(denseAttr.getSplatValue<Attribute>());
    if (!splatAttr)
      return rewriter.notifyMatchFailure(op, "splat constant must be typed");

    Value scalar =
        rewriter.create<arith::ConstantOp>(op.getLoc(), splatAttr).getResult();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "constant result must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for constant mask");
      results.push_back(rewriter
                            .create<VdupOp>(op.getLoc(), resultType, scalar,
                                            *mask,
                                            /*position=*/nullptr)
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIConstantMaskOpPattern
    : OneToNOpConversionPattern<VMIConstantMaskOp> {
  using OneToNOpConversionPattern<VMIConstantMaskOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIConstantMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeConstantMaskMaterialization(op, &reason);
    if (failed(materializations))
      return rewriter.notifyMatchFailure(op, Twine("constant_mask ") + reason);

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         *materializations) {
      if (results.size() >= resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "constant_mask produced too many physical masks");
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType)
        return rewriter.notifyMatchFailure(op,
                                           "constant_mask result must be mask");
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize constant_mask physical chunk");
      results.push_back(*mask);
    }

    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "constant_mask physical result count mismatch");
    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMICreateMaskOpPattern
    : OneToNOpConversionPattern<VMICreateMaskOp> {
  using OneToNOpConversionPattern<VMICreateMaskOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICreateMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto activeConstant =
        op.getActiveLanes().getDefiningOp<arith::ConstantOp>();
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout ||
        !VMIMaskType::isConcreteGranularity(resultVMIType.getGranularity()))
      return rewriter.notifyMatchFailure(
          op, "create_mask requires concrete layout and granularity");
    FailureOr<int64_t> lanesPerPart =
        getMaskLanesPerPart(resultVMIType.getGranularity());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "create_mask requires known physical mask lanes per part");

    if (!activeConstant) {
      FailureOr<Value> active = getSingleValue(
          op, adaptor.getActiveLanes(),
          "create_mask active_lanes must convert to one value", rewriter);
      if (failed(active))
        return failure();

      TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
      int64_t factor = layout.isDeinterleaved() ? layout.getFactor() : 1;
      if (resultTypes.size() % factor != 0)
        return rewriter.notifyMatchFailure(
            op, "dynamic create_mask physical result count does not match "
                "layout factor");
      int64_t chunksPerPart = resultTypes.size() / factor;
      Value activeI32 = clampDynamicActiveLanes(
          op.getLoc(), *active, resultVMIType.getElementCount(), rewriter);

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (int64_t part = 0; part < factor; ++part) {
        Value remaining = createPartitionActiveLanes(op.getLoc(), activeI32,
                                                     factor, part, rewriter);
        for (int64_t chunk = 0; chunk < chunksPerPart; ++chunk) {
          Type resultType = resultTypes[part * chunksPerPart + chunk];
          auto maskType = dyn_cast<MaskType>(resultType);
          if (!maskType)
            return rewriter.notifyMatchFailure(
                op, "create_mask result must be mask");
          FailureOr<std::pair<Value, Value>> maskAndRemaining =
              createRuntimePrefixMask(op.getLoc(), maskType, remaining,
                                      rewriter);
          if (failed(maskAndRemaining))
            return rewriter.notifyMatchFailure(
                op, "unsupported mask type for dynamic create_mask");
          results.push_back(maskAndRemaining->first);
          remaining = maskAndRemaining->second;
        }
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    auto activeAttr = dyn_cast<IntegerAttr>(activeConstant.getValue());
    if (!activeAttr)
      return rewriter.notifyMatchFailure(
          op, "create_mask active_lanes must be an integer constant");

    int64_t activeLanes = activeAttr.getInt();
    if (activeLanes < 0)
      activeLanes = 0;
    if (activeLanes > resultVMIType.getElementCount())
      activeLanes = resultVMIType.getElementCount();

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    int64_t factor = layout.isDeinterleaved() ? layout.getFactor() : 1;
    SmallVector<Value> results;
    results.reserve(resultTypes.size());

    for (int64_t part = 0; part < factor; ++part) {
      for (int64_t chunk = 0;; ++chunk) {
        bool anyLane = false;
        int64_t activeInChunk = 0;
        for (int64_t lane = 0; lane < *lanesPerPart; ++lane) {
          FailureOr<bool> padding =
              isPaddingLane(resultVMIType, part, chunk, lane);
          if (failed(padding))
            return rewriter.notifyMatchFailure(
                op, "failed to map create_mask physical padding lane");
          if (*padding)
            continue;
          anyLane = true;
          FailureOr<int64_t> logicalLane =
              mapPhysicalLaneToLogical(resultVMIType, part, chunk, lane);
          if (failed(logicalLane))
            return rewriter.notifyMatchFailure(
                op, "failed to map create_mask physical lane");
          if (*logicalLane < activeLanes)
            ++activeInChunk;
        }
        if (!anyLane)
          break;

        if (results.size() >= resultTypes.size())
          return rewriter.notifyMatchFailure(
              op, "create_mask produced too many physical masks");
        auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
        if (!maskType)
          return rewriter.notifyMatchFailure(op,
                                             "create_mask result must be mask");
        std::optional<std::string> pattern =
            getPrefixPattern(activeInChunk, *lanesPerPart);
        if (pattern) {
          FailureOr<Value> mask =
              createPrefixMask(op.getLoc(), maskType, *pattern, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "unsupported mask type for create_mask");
          results.push_back(*mask);
          continue;
        }

        FailureOr<std::pair<Value, Value>> maskAndRemaining =
            createRuntimePrefixMask(
                op.getLoc(), maskType,
                createI32Constant(op.getLoc(), activeInChunk, rewriter),
                rewriter);
        if (failed(maskAndRemaining))
          return rewriter.notifyMatchFailure(
              op, "unsupported mask type for create_mask plt fallback");
        results.push_back(maskAndRemaining->first);
      }
    }

    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "create_mask physical result count mismatch");
    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMICreateGroupMaskOpPattern
    : OneToNOpConversionPattern<VMICreateGroupMaskOp> {
  using OneToNOpConversionPattern<
      VMICreateGroupMaskOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICreateGroupMaskOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    auto resultVMIType = cast<VMIMaskType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 4 && resultLayout.getBlockElems() == 8) {
      VMILayoutAttr contiguousLayout =
          VMILayoutAttr::getContiguous(op.getContext());
      auto contiguousType =
          VMIMaskType::get(op.getContext(), resultVMIType.getElementCount(),
                           resultVMIType.getGranularity(), contiguousLayout);
      SmallVector<Value> contiguousParts;
      auto activeConstant =
          op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
      if (activeConstant) {
        std::string contiguousReason;
        FailureOr<SmallVector<ConstantMaskChunkMaterialization>>
            contiguousMaterializations = computeGroupMaskMaterializationForType(
                op, contiguousType, &contiguousReason);
        if (failed(contiguousMaterializations))
          return rewriter.notifyMatchFailure(op, Twine("create_group_mask ") +
                                                     contiguousReason);

        contiguousParts.reserve(contiguousMaterializations->size());
        for (const ConstantMaskChunkMaterialization &materialization :
             *contiguousMaterializations) {
          if (contiguousParts.size() >= resultTypes.size())
            return rewriter.notifyMatchFailure(
                op, "create_group_mask produced too many contiguous masks");
          auto maskType =
              dyn_cast<MaskType>(resultTypes[contiguousParts.size()]);
          if (!maskType)
            return rewriter.notifyMatchFailure(
                op, "create_group_mask result must be mask");
          FailureOr<Value> mask = materializeConstantMaskChunk(
              op.getLoc(), maskType, materialization.activeLanes, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "failed to materialize create_group_mask contiguous chunk");
          contiguousParts.push_back(*mask);
        }
      } else {
        FailureOr<Value> active = getSingleValue(
            op, adaptor.getActiveElemsPerGroup(),
            "create_group_mask active_elems_per_group must convert to one "
            "value",
            rewriter);
        if (failed(active))
          return failure();
        FailureOr<SmallVector<Value>> dynamicParts =
            materializeDynamicContiguousGroupMask(op, *active, contiguousType,
                                                  resultTypes, rewriter);
        if (failed(dynamicParts))
          return failure();
        contiguousParts = std::move(*dynamicParts);
      }

      if (contiguousParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "create_group_mask contiguous physical result count mismatch");
      FailureOr<SmallVector<Value>> results = materializeMaskLayoutConversion(
          op, contiguousParts, resultTypes, contiguousLayout, resultLayout,
          rewriter);
      if (failed(results))
        return failure();
      rewriter.replaceOp(op, *results, adaptor.getResultMapping());
      return success();
    }

    auto activeConstant =
        op.getActiveElemsPerGroup().getDefiningOp<arith::ConstantOp>();
    if (!activeConstant && resultLayout && resultLayout.isContiguous()) {
      FailureOr<Value> active = getSingleValue(
          op, adaptor.getActiveElemsPerGroup(),
          "create_group_mask active_elems_per_group must convert to one value",
          rewriter);
      if (failed(active))
        return failure();
      FailureOr<SmallVector<Value>> results =
          materializeDynamicContiguousGroupMask(op, *active, resultVMIType,
                                                resultTypes, rewriter);
      if (failed(results))
        return failure();
      rewriter.replaceOp(op, *results, adaptor.getResultMapping());
      return success();
    }

    std::string reason;
    FailureOr<SmallVector<ConstantMaskChunkMaterialization>> materializations =
        computeGroupMaskMaterialization(op, &reason);
    if (failed(materializations))
      return rewriter.notifyMatchFailure(op,
                                         Twine("create_group_mask ") + reason);

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (const ConstantMaskChunkMaterialization &materialization :
         *materializations) {
      if (results.size() >= resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "create_group_mask produced too many physical masks");
      auto maskType = dyn_cast<MaskType>(resultTypes[results.size()]);
      if (!maskType)
        return rewriter.notifyMatchFailure(
            op, "create_group_mask result must be mask");
      FailureOr<Value> mask = materializeConstantMaskChunk(
          op.getLoc(), maskType, materialization.activeLanes, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize create_group_mask physical chunk");
      results.push_back(*mask);
    }

    if (results.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(
          op, "create_group_mask physical result count mismatch");
    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMILoadOpPattern : OneToNOpConversionPattern<VMILoadOp> {
  using OneToNOpConversionPattern<VMILoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMILoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "load source must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "load offset must convert to one value", rewriter);
    if (failed(source) || failed(offset))
      return failure();
    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, op.getSource().getType(), *offset, rewriter);
    if (failed(lanesPerPart))
      return failure();

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 2) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
      if (dist && !resultTypes.empty() && resultTypes.size() % 2 == 0) {
        int64_t groups = resultTypes.size() / 2;
        SmallVector<Value> lows;
        SmallVector<Value> highs;
        lows.reserve(groups);
        highs.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
          Type lowType = resultTypes[group];
          Type highType = resultTypes[groups + group];
          if (lowType != highType)
            return rewriter.notifyMatchFailure(
                op, "vldsx2 requires matching low/high result types");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), *offset, group * 2 * *lanesPerPart, rewriter);
          auto load = rewriter.create<Vldsx2Op>(op.getLoc(), lowType, highType,
                                                *source, chunkOffset,
                                                rewriter.getStringAttr(*dist));
          lows.push_back(load.getLow());
          highs.push_back(load.getHigh());
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        results.append(lows);
        results.append(highs);
        rewriter.replaceOp(op, results, adaptor.getResultMapping());
        return success();
      }
    }

    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 4 && resultLayout.getBlockElems() == 1) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
      if (dist && !resultTypes.empty() && resultTypes.size() % 4 == 0) {
        int64_t groups = resultTypes.size() / 4;
        SmallVector<Value> part0;
        SmallVector<Value> part1;
        SmallVector<Value> part2;
        SmallVector<Value> part3;
        part0.reserve(groups);
        part1.reserve(groups);
        part2.reserve(groups);
        part3.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
          Type part0Type = resultTypes[group];
          Type part1Type = resultTypes[groups + group];
          Type part2Type = resultTypes[2 * groups + group];
          Type part3Type = resultTypes[3 * groups + group];
          if (part0Type != part1Type || part0Type != part2Type ||
              part0Type != part3Type)
            return rewriter.notifyMatchFailure(
                op, "vldsx2 deinterleaved=4 load requires matching part "
                    "types");

          Value firstOffset = createChunkOffset(
              op.getLoc(), *offset, group * 4 * *lanesPerPart, rewriter);
          Value secondOffset = createChunkOffset(
              op.getLoc(), *offset, (group * 4 + 2) * *lanesPerPart, rewriter);
          auto first = rewriter.create<Vldsx2Op>(
              op.getLoc(), part0Type, part1Type, *source, firstOffset,
              rewriter.getStringAttr(*dist));
          auto second = rewriter.create<Vldsx2Op>(
              op.getLoc(), part2Type, part3Type, *source, secondOffset,
              rewriter.getStringAttr(*dist));

          auto even =
              rewriter.create<VdintlvOp>(op.getLoc(), part0Type, part2Type,
                                         first.getLow(), second.getLow());
          auto odd =
              rewriter.create<VdintlvOp>(op.getLoc(), part1Type, part3Type,
                                         first.getHigh(), second.getHigh());
          part0.push_back(even.getLow());
          part1.push_back(odd.getLow());
          part2.push_back(even.getHigh());
          part3.push_back(odd.getHigh());
        }

        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        results.append(part0);
        results.append(part1);
        results.append(part2);
        results.append(part3);
        rewriter.replaceOp(op, results, adaptor.getResultMapping());
        return success();
      }
    }

    SmallVector<Value> contiguousParts;
    contiguousParts.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "load result must be vreg");
      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      contiguousParts.push_back(rewriter
                                    .create<VldsOp>(op.getLoc(), resultType,
                                                    /*updated_base=*/Type{},
                                                    *source, chunkOffset,
                                                    /*dist=*/nullptr)
                                    .getResult());
    }

    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, contiguousParts, resultTypes,
        VMILayoutAttr::getContiguous(rewriter.getContext()),
        resultVMIType.getLayoutAttr(), rewriter);
    if (failed(results))
      return failure();

    rewriter.replaceOp(op, *results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIGroupLoadOpPattern : OneToNOpConversionPattern<VMIGroupLoadOp> {
  using OneToNOpConversionPattern<VMIGroupLoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "group_load source must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "group_load offset must convert to one value", rewriter);
    FailureOr<Value> rowStride = getSingleValue(
        op, adaptor.getRowStride(),
        "group_load row_stride must convert to one value", rewriter);
    if (failed(source) || failed(offset) || failed(rowStride))
      return failure();

    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getBlockElems() == 8 &&
        resultVMIType.getElementType().isF32()) {
      FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
          resultVMIType, op.getNumGroupsAttr().getInt());
      if (failed(groupSize))
        return rewriter.notifyMatchFailure(
            op, "group_load requires num_groups to evenly divide lane count");
      if ((*groupSize != 16 || resultLayout.getFactor() != 2) &&
          (*groupSize != 32 || resultLayout.getFactor() != 4))
        return rewriter.notifyMatchFailure(
            op, "block8 group_load requires S=16/factor=2 or S=32/factor=4");
      if (op.getNumGroupsAttr().getInt() % 8 != 0)
        return rewriter.notifyMatchFailure(
            op, "block8 group_load requires num_groups multiple of 8");
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      if (!constantRowStride || *constantRowStride <= 0 ||
          *constantRowStride % 8 != 0)
        return rewriter.notifyMatchFailure(
            op, "block8 group_load requires constant positive row_stride "
                "divisible by 8 f32 elements");
      if (!isa<PtrType>((*source).getType()))
        return rewriter.notifyMatchFailure(
            op, "block8 group_load requires !pto.ptr source");

      TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
      int64_t factor = resultLayout.getFactor();
      FailureOr<int64_t> chunksPerPart = getDataChunksInPart(resultVMIType, 0);
      if (failed(chunksPerPart) || *chunksPerPart <= 0)
        return rewriter.notifyMatchFailure(
            op, "block8 group_load requires known chunks per part");
      for (int64_t part = 1; part < factor; ++part) {
        FailureOr<int64_t> currentChunks =
            getDataChunksInPart(resultVMIType, part);
        if (failed(currentChunks) || *currentChunks != *chunksPerPart)
          return rewriter.notifyMatchFailure(
              op, "block8 group_load requires uniform chunks per part");
      }
      if (static_cast<int64_t>(resultTypes.size()) != factor * *chunksPerPart)
        return rewriter.notifyMatchFailure(op,
                                           "block8 group_load arity mismatch");

      auto makeI16 = [&](int64_t value) -> Value {
        return rewriter.create<arith::ConstantIntOp>(op.getLoc(), value, 16);
      };
      Value blockStride = makeI16(*constantRowStride / 8);
      Value zeroI16 = makeI16(0);
      auto makePtr = [&](Value elementOffset) -> Value {
        return rewriter
            .create<AddPtrOp>(op.getLoc(), (*source).getType(), *source,
                              elementOffset)
            .getResult();
      };

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      constexpr int64_t kGroupsPerBlock8Load = 8;
      for (int64_t part = 0; part < factor; ++part) {
        for (int64_t chunk = 0; chunk < *chunksPerPart; ++chunk) {
          int64_t flatIndex = part * *chunksPerPart + chunk;
          auto vregType = dyn_cast<VRegType>(resultTypes[flatIndex]);
          if (!vregType)
            return rewriter.notifyMatchFailure(
                op, "block8 group_load result must be vreg");
          FailureOr<Value> allMask =
              createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
          if (failed(allMask))
            return rewriter.notifyMatchFailure(
                op, "failed to create block8 group_load mask");
          Value chunkOffset = createGroupChunkOffset(
              op.getLoc(), *offset, *rowStride, chunk * kGroupsPerBlock8Load,
              part * resultLayout.getBlockElems(), rewriter);
          Value chunkBase = makePtr(chunkOffset);
          results.push_back(rewriter
                                .create<VsldbOp>(op.getLoc(), vregType,
                                                 chunkBase, blockStride,
                                                 zeroI16, *allMask)
                                .getResult());
        }
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        resultVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op, "group_load requires num_groups to evenly divide lane count");
    if (failed(checkContiguousFullGroupChunks(op, resultVMIType, *groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter)))
      return failure();

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (static_cast<int64_t>(resultTypes.size()) != groupCount * chunksPerGroup)
      return rewriter.notifyMatchFailure(op, "group_load arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op,
                                           "group_load result must be vreg");
      int64_t group = index / chunksPerGroup;
      int64_t chunkInGroup = index % chunksPerGroup;
      Value chunkOffset =
          createGroupChunkOffset(op.getLoc(), *offset, *rowStride, group,
                                 chunkInGroup * lanesPerPart, rewriter);
      results.push_back(rewriter
                            .create<VldsOp>(op.getLoc(), resultType,
                                            /*updated_base=*/Type{}, *source,
                                            chunkOffset,
                                            /*dist=*/nullptr)
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIGroupSlotLoadOpPattern
    : OneToNOpConversionPattern<VMIGroupSlotLoadOp> {
  using OneToNOpConversionPattern<
      VMIGroupSlotLoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupSlotLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr layout = resultVMIType.getLayoutAttr();
    if (!layout || !layout.isGroupSlots() || layout.getSlots() <= 0)
      return rewriter.notifyMatchFailure(
          op, "group_slot_load requires explicit group_slots layout");

    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(),
        "group_slot_load source must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "group_slot_load offset must convert to one value", rewriter);
    FailureOr<Value> sourceGroupStride = getSingleValue(
        op, adaptor.getSourceGroupStride(),
        "group_slot_load source_group_stride must convert to one value",
        rewriter);
    if (failed(source) || failed(offset) || failed(sourceGroupStride))
      return failure();
    if (!isa<PtrType>((*source).getType()))
      return rewriter.notifyMatchFailure(
          op, "group_slot_load requires !pto.ptr source");

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    int64_t numGroups = op.getNumGroupsAttr().getInt();
    int64_t slots = layout.getSlots();
    int64_t expectedArity = ceilDivNonNegative(numGroups, slots);
    if (static_cast<int64_t>(resultTypes.size()) != expectedArity)
      return rewriter.notifyMatchFailure(op, "group_slot_load arity mismatch");

    auto makeI16 = [&](int64_t value) -> Value {
      return rewriter.create<arith::ConstantIntOp>(op.getLoc(), value, 16);
    };
    Value zeroI16 = makeI16(0);
    auto makePtr = [&](Value elementOffset) -> Value {
      return rewriter
          .create<AddPtrOp>(op.getLoc(), (*source).getType(), *source,
                            elementOffset)
          .getResult();
    };

    SmallVector<Value> results;
    results.reserve(resultTypes.size());

    if (slots == 8) {
      std::optional<int64_t> stride =
          getConstantIndexValue(op.getSourceGroupStride());
      if (!stride || *stride != 1)
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_slot_load requires constant unit stride");
      if (resultTypes.size() != 1)
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_slot_load expects one physical result");
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      if (!resultType)
        return rewriter.notifyMatchFailure(
            op, "group_slot_load result must be vreg");
      FailureOr<MaskType> maskType =
          getMaskTypeForVReg(resultType, rewriter.getContext());
      if (failed(maskType))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_slot_load mask");
      FailureOr<Value> oneBlockMask =
          createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
      if (failed(oneBlockMask))
        return rewriter.notifyMatchFailure(
            op, "failed to create group_slot_load mask");
      Value slotBase = makePtr(*offset);
      results.push_back(rewriter
                            .create<VsldbOp>(op.getLoc(), resultType, slotBase,
                                             zeroI16, zeroI16, *oneBlockMask)
                            .getResult());
      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    if (slots != 1)
      return rewriter.notifyMatchFailure(
          op, "group_slot_load supports only slots=8 or slots=1");
    unsigned elementBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    if (elementBits == 0 || 256 % elementBits != 0)
      return rewriter.notifyMatchFailure(
          op, "slots=1 group_slot_load requires supported element width");
    int64_t alignedStrideElems = 256 / elementBits;
    std::optional<int64_t> constantStride =
        getConstantIndexValue(op.getSourceGroupStride());
    if (!constantStride || *constantStride <= 0 ||
        *constantStride % alignedStrideElems != 0)
      return rewriter.notifyMatchFailure(
          op, Twine("slots=1 group_slot_load requires constant positive "
                    "source_group_stride divisible by ") +
                  Twine(alignedStrideElems) +
                  " elements for 32B lane-0 vsldb alignment");

    for (auto [group, resultType] : llvm::enumerate(resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(
            op, "group_slot_load result must be vreg");
      FailureOr<MaskType> maskType =
          getMaskTypeForVReg(vregType, rewriter.getContext());
      if (failed(maskType))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_slot_load mask");
      FailureOr<Value> oneBlockMask =
          createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
      if (failed(oneBlockMask))
        return rewriter.notifyMatchFailure(
            op, "failed to create group_slot_load mask");
      Value groupOffset = *offset;
      if (group != 0) {
        Value groupIndex =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), group);
        Value rowOffset = rewriter
                              .create<arith::MulIOp>(
                                  op.getLoc(), *sourceGroupStride, groupIndex)
                              .getResult();
        groupOffset =
            rewriter.create<arith::AddIOp>(op.getLoc(), groupOffset, rowOffset)
                .getResult();
      }
      Value slotBase = makePtr(groupOffset);
      results.push_back(rewriter
                            .create<VsldbOp>(op.getLoc(), vregType, slotBase,
                                             zeroI16, zeroI16, *oneBlockMask)
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIMaskedLoadOpPattern
    : OneToNOpConversionPattern<VMIMaskedLoadOp> {
  using OneToNOpConversionPattern<VMIMaskedLoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIMaskedLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "masked_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "masked_load offset must convert to one value",
        rewriter);
    if (failed(source) || failed(offset))
      return failure();

    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, (*source).getType(), *offset, rewriter);
    if (failed(lanesPerPart))
      return failure();

    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (maskParts.size() != passthruParts.size() ||
        passthruParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "masked_load physical arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [index, maskPassthruAndType] : llvm::enumerate(
             llvm::zip_equal(maskParts, passthruParts, resultTypes))) {
      auto [mask, passthru, resultType] = maskPassthruAndType;
      if (!isa<MaskType>(mask.getType()) || passthru.getType() != resultType ||
          !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "masked_load physical part type mismatch");

      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      Value loaded =
          rewriter
              .create<VldsOp>(op.getLoc(), resultType,
                              /*updated_base=*/Type{}, *source, chunkOffset,
                              /*dist=*/nullptr)
              .getResult();
      results.push_back(
          rewriter
              .create<VselOp>(op.getLoc(), resultType, loaded, passthru, mask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIGatherOpPattern : OneToNOpConversionPattern<VMIGatherOp> {
  using OneToNOpConversionPattern<VMIGatherOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGatherOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "gather source must convert to one value", rewriter);
    if (failed(source))
      return failure();

    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (indicesParts.size() != maskParts.size() ||
        indicesParts.size() != passthruParts.size() ||
        indicesParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "gather physical arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [indices, mask, passthru, resultType] :
         llvm::zip_equal(indicesParts, maskParts, passthruParts, resultTypes)) {
      if (!isa<VRegType>(indices.getType()) || !isa<MaskType>(mask.getType()) ||
          passthru.getType() != resultType || !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "gather physical part type mismatch");

      Value gathered = rewriter
                           .create<Vgather2BcOp>(op.getLoc(), resultType,
                                                 *source, indices, mask)
                           .getResult();
      results.push_back(
          rewriter
              .create<VselOp>(op.getLoc(), resultType, gathered, passthru, mask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIExpandLoadOpPattern
    : OneToNOpConversionPattern<VMIExpandLoadOp> {
  using OneToNOpConversionPattern<VMIExpandLoadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIExpandLoadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source = getSingleValue(
        op, adaptor.getSource(), "expand_load source must convert to one value",
        rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "expand_load offset must convert to one value",
        rewriter);
    if (failed(source) || failed(offset))
      return failure();

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (isStaticAllActiveMask(op.getMask(), resultVMIType.getElementCount())) {
      FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
          op, resultVMIType, (*source).getType(), *offset, rewriter);
      if (failed(lanesPerPart))
        return failure();

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
        if (!isa<VRegType>(resultType))
          return rewriter.notifyMatchFailure(op,
                                             "expand_load result must be vreg");
        Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                              index * *lanesPerPart, rewriter);
        results.push_back(rewriter
                              .create<VldsOp>(op.getLoc(), resultType,
                                              /*updated_base=*/Type{}, *source,
                                              chunkOffset,
                                              /*dist=*/nullptr)
                              .getResult());
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    ValueRange maskParts = adaptor.getMask();
    ValueRange passthruParts = adaptor.getPassthru();
    if (resultTypes.size() != 1 || maskParts.size() != 1 ||
        passthruParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load supports only one physical chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType ||
        passthruParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "runtime expand_load requires physical result/passthru/mask");

    auto baseType = dyn_cast<PtrType>((*source).getType());
    if (!baseType)
      return rewriter.notifyMatchFailure(op,
                                         "runtime expand_load requires ptr");
    Value gatherBase = rewriter
                           .create<AddPtrOp>(op.getLoc(), (*source).getType(),
                                             *source, *offset)
                           .getResult();
    auto indexType =
        VRegType::get(rewriter.getContext(), resultType.getElementCount(),
                      rewriter.getI32Type());
    FailureOr<Value> indexSeedMask =
        createAllTrueMaskForVReg(op.getLoc(), indexType, rewriter);
    if (failed(indexSeedMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create runtime expand_load index seed mask");
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    Value carrier =
        rewriter
            .create<VdupOp>(op.getLoc(), indexType, zero, *indexSeedMask,
                            /*position=*/nullptr)
            .getResult();
    Value indices =
        rewriter
            .create<VusqzOp>(op.getLoc(), indexType, carrier, maskParts.front())
            .getResult();
    Value gathered =
        rewriter
            .create<Vgather2BcOp>(op.getLoc(), resultType, gatherBase, indices,
                                  maskParts.front())
            .getResult();
    Value result = rewriter
                       .create<VselOp>(op.getLoc(), resultType, gathered,
                                       passthruParts.front(), maskParts.front())
                       .getResult();
    rewriter.replaceOp(op, SmallVector<Value>{result},
                       adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIStoreOpPattern : OneToNOpConversionPattern<VMIStoreOp> {
  using OneToNOpConversionPattern<VMIStoreOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "store requires known physical lanes per part");
    bool fullPhysicalChunks =
        succeeded(checkFullDataPhysicalChunks(valueVMIType, nullptr));
    FailureOr<Value> destination =
        getSingleValue(op, adaptor.getDestination(),
                       "store destination must convert to one value", rewriter);
    FailureOr<Value> offset =
        getSingleValue(op, adaptor.getOffset(),
                       "store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    VMILayoutSupport localSupports;
    FailureOr<VMIContiguousStoreSupport> storeSupport =
        localSupports.getContiguousStoreSupport(valueVMIType);
    if (succeeded(storeSupport) &&
        storeSupport->kind ==
            VMIContiguousStoreSupportKind::Deinterleaved2Vstsx2) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
      if (dist && !valueParts.empty() && valueParts.size() % 2 == 0) {
        int64_t groups = valueParts.size() / 2;
        for (int64_t group = 0; group < groups; ++group) {
          Value low = valueParts[group];
          Value high = valueParts[groups + group];
          if (low.getType() != high.getType())
            return rewriter.notifyMatchFailure(
                op, "vstsx2 requires matching low/high value types");
          auto vregType = dyn_cast<VRegType>(low.getType());
          if (!vregType)
            return rewriter.notifyMatchFailure(op, "store value must be vreg");
          FailureOr<Value> mask =
              createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "unsupported element type for store mask");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), *offset, group * 2 * *lanesPerPart, rewriter);
          rewriter.create<Vstsx2Op>(op.getLoc(), low, high, *destination,
                                    chunkOffset, rewriter.getStringAttr(*dist),
                                    *mask);
        }
        rewriter.eraseOp(op);
        return success();
      }
    }

    SmallVector<Type> contiguousTypes;
    contiguousTypes.reserve(valueParts.size());
    for (Value value : valueParts)
      contiguousTypes.push_back(value.getType());

    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousTypes, valueVMIType.getLayoutAttr(),
        VMILayoutAttr::getContiguous(rewriter.getContext()), rewriter);
    if (failed(storeParts))
      return failure();

    for (auto [index, value] : llvm::enumerate(*storeParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "store value must be vreg");
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> activeLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(activeLanes))
          return rewriter.notifyMatchFailure(
              op, "failed to compute store active lanes");
        if (*activeLanes == 0)
          continue;
      }
      FailureOr<Value> mask =
          fullPhysicalChunks
              ? createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter)
              : createContiguousStoreMask(op.getLoc(), valueVMIType, index,
                                          vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for store mask");
      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIGroupStoreOpPattern
    : OneToNOpConversionPattern<VMIGroupStoreOp> {
  using OneToNOpConversionPattern<VMIGroupStoreOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    VMILayoutAttr layout = valueVMIType.getLayoutAttr();

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "group_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(), "group_store offset must convert to one value",
        rewriter);
    FailureOr<Value> rowStride = getSingleValue(
        op, adaptor.getRowStride(),
        "group_store row_stride must convert to one value", rewriter);
    if (failed(destination) || failed(offset) || failed(rowStride))
      return failure();

    if (layout && layout.isGroupSlots() && layout.getSlots() == 1 &&
        layout.getNumGroups() == op.getNumGroupsAttr().getInt()) {
      ValueRange valueParts = adaptor.getValue();
      if (static_cast<int64_t>(valueParts.size()) != layout.getNumGroups())
        return rewriter.notifyMatchFailure(
            op, "slots=1 group_store arity mismatch");
      unsigned elementBits =
          pto::getPTOStorageElemBitWidth(valueVMIType.getElementType());
      if (elementBits == 0 || 256 % elementBits != 0)
        return rewriter.notifyMatchFailure(
            op, "slots=1 group_store requires supported element width");
      int64_t alignedStrideElems = 256 / elementBits;
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      if (!constantRowStride || *constantRowStride <= 0 ||
          *constantRowStride % alignedStrideElems != 0)
        return rewriter.notifyMatchFailure(
            op, Twine("slots=1 group_store requires constant positive "
                      "row_stride divisible by ") +
                    Twine(alignedStrideElems) +
                    " elements for 32B lane-0 vsts alignment");

      for (auto [group, value] : llvm::enumerate(valueParts)) {
        auto vregType = dyn_cast<VRegType>(value.getType());
        if (!vregType)
          return rewriter.notifyMatchFailure(op,
                                             "group_store value must be vreg");
        FailureOr<MaskType> maskType =
            getMaskTypeForVReg(vregType, rewriter.getContext());
        if (failed(maskType))
          return rewriter.notifyMatchFailure(
              op, "unsupported element type for group_store mask");
        FailureOr<Value> mask =
            createPrefixMask(op.getLoc(), *maskType, "PAT_VL1", rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create slots=1 group_store mask");
        Value groupOffset =
            createGroupChunkOffset(op.getLoc(), *offset, *rowStride, group,
                                   /*chunkLaneOffset=*/0, rewriter);
        rewriter.create<VstsOp>(op.getLoc(),
                                /*updated_base=*/Type{}, value, *destination,
                                groupOffset, /*dist=*/nullptr, *mask);
      }

      rewriter.eraseOp(op);
      return success();
    }

    if (layout && layout.isGroupSlots() && layout.getSlots() == 8 &&
        layout.getNumGroups() == op.getNumGroupsAttr().getInt()) {
      int64_t numGroups = layout.getNumGroups();
      std::optional<int64_t> constantRowStride =
          getConstantIndexValue(op.getRowStride());
      if (!constantRowStride || *constantRowStride != 1)
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_store requires constant unit row_stride");

      ValueRange valueParts = adaptor.getValue();
      if (static_cast<int64_t>(valueParts.size()) !=
          ceilDivNonNegative(numGroups, 8))
        return rewriter.notifyMatchFailure(
            op, "slots=8 group_store arity mismatch");

      for (auto [slotBlock, value] : llvm::enumerate(valueParts)) {
        auto vregType = dyn_cast<VRegType>(value.getType());
        if (!vregType)
          return rewriter.notifyMatchFailure(op,
                                             "group_store value must be vreg");
        FailureOr<MaskType> maskType =
            getMaskTypeForVReg(vregType, rewriter.getContext());
        if (failed(maskType))
          return rewriter.notifyMatchFailure(
              op, "unsupported element type for group_store mask");
        int64_t activeGroups = std::min<int64_t>(8, numGroups - slotBlock * 8);
        FailureOr<Value> mask = createPrefixMaskForActiveLanes(
            op.getLoc(), *maskType, activeGroups, rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create slots=8 group_store mask");
        Value groupOffset = createGroupChunkOffset(
            op.getLoc(), *offset, *rowStride, slotBlock * 8,
            /*chunkLaneOffset=*/0, rewriter);
        rewriter.create<VstsOp>(op.getLoc(),
                                /*updated_base=*/Type{}, value, *destination,
                                groupOffset, /*dist=*/nullptr, *mask);
      }

      rewriter.eraseOp(op);
      return success();
    }

    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    FailureOr<int64_t> groupSize =
        getGroupSizeFromNumGroups(valueVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op, "group_store requires num_groups to evenly divide lane count");
    if (failed(checkContiguousFullGroupChunks(op, valueVMIType, *groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter)))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    if (static_cast<int64_t>(valueParts.size()) != groupCount * chunksPerGroup)
      return rewriter.notifyMatchFailure(op, "group_store arity mismatch");

    for (auto [index, value] : llvm::enumerate(valueParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType)
        return rewriter.notifyMatchFailure(op,
                                           "group_store value must be vreg");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for group_store mask");
      int64_t group = index / chunksPerGroup;
      int64_t chunkInGroup = index % chunksPerGroup;
      Value chunkOffset =
          createGroupChunkOffset(op.getLoc(), *offset, *rowStride, group,
                                 chunkInGroup * lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIMaskedStoreOpPattern
    : OneToNOpConversionPattern<VMIMaskedStoreOp> {
  using OneToNOpConversionPattern<VMIMaskedStoreOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIMaskedStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "masked_store requires known physical lanes per part");

    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "masked_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "masked_store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != maskParts.size())
      return rewriter.notifyMatchFailure(
          op, "masked_store value/mask physical arity mismatch");

    SmallVector<Type> contiguousValueTypes;
    contiguousValueTypes.reserve(valueParts.size());
    for (Value value : valueParts)
      contiguousValueTypes.push_back(value.getType());
    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousValueTypes, valueVMIType.getLayoutAttr(),
        VMILayoutAttr::getContiguous(rewriter.getContext()), rewriter);
    if (failed(storeParts))
      return failure();

    auto maskVMIType = cast<VMIMaskType>(op.getMask().getType());
    SmallVector<Type> contiguousMaskTypes;
    contiguousMaskTypes.reserve(maskParts.size());
    for (Value mask : maskParts)
      contiguousMaskTypes.push_back(mask.getType());
    FailureOr<SmallVector<Value>> storeMasks = materializeMaskLayoutConversion(
        op, maskParts, contiguousMaskTypes, maskVMIType.getLayoutAttr(),
        VMILayoutAttr::getContiguous(rewriter.getContext()), rewriter);
    if (failed(storeMasks))
      return failure();

    if (storeParts->size() != storeMasks->size())
      return rewriter.notifyMatchFailure(
          op, "masked_store converted value/mask arity mismatch");

    for (auto [index, valueAndMask] :
         llvm::enumerate(llvm::zip_equal(*storeParts, *storeMasks))) {
      auto [value, mask] = valueAndMask;
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType || !isa<MaskType>(mask.getType()))
        return rewriter.notifyMatchFailure(
            op, "masked_store converted parts must be vreg/mask");
      FailureOr<int64_t> activeLanes =
          getContiguousActiveDataLanes(valueVMIType, index);
      if (failed(activeLanes))
        return rewriter.notifyMatchFailure(
            op, "failed to compute masked_store active lanes");
      if (*activeLanes == 0)
        continue;
      FailureOr<Value> storeMask = createMaskedStorePredicate(
          op.getLoc(), valueVMIType, index, mask, vregType, rewriter);
      if (failed(storeMask))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize masked_store predicate");
      Value chunkOffset = createChunkOffset(op.getLoc(), *offset,
                                            index * *lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *storeMask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIScatterOpPattern : OneToNOpConversionPattern<VMIScatterOp> {
  using OneToNOpConversionPattern<VMIScatterOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIScatterOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "scatter destination must convert to one value", rewriter);
    if (failed(destination))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange indicesParts = adaptor.getIndices();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != indicesParts.size() ||
        valueParts.size() != maskParts.size())
      return rewriter.notifyMatchFailure(op, "scatter physical arity mismatch");

    for (auto [value, indices, mask] :
         llvm::zip_equal(valueParts, indicesParts, maskParts)) {
      if (!isa<VRegType>(value.getType()) ||
          !isa<VRegType>(indices.getType()) || !isa<MaskType>(mask.getType()))
        return rewriter.notifyMatchFailure(
            op, "scatter physical part type mismatch");
      rewriter.create<VscatterOp>(op.getLoc(), value, *destination, indices,
                                  mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMITileReadOpPattern : OneToNOpConversionPattern<VMITileReadOp> {
  using OneToNOpConversionPattern<VMITileReadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMITileReadOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<Value> source =
        getSingleValue(op, adaptor.getSource(),
                       "tile_read source must convert to one value", rewriter);
    if (failed(source))
      return failure();

    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    FailureOr<int64_t> lanesPerPart = verifyFullOrSafeReadVRegChunks(
        op, resultVMIType, (*source).getType(), zero, rewriter);
    if (failed(lanesPerPart))
      return failure();

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (resultLayout && resultLayout.isDeinterleaved() &&
        resultLayout.getFactor() == 2) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(resultVMIType.getElementType(), "DINTLV");
      if (dist && !resultTypes.empty() && resultTypes.size() % 2 == 0) {
        int64_t groups = resultTypes.size() / 2;
        SmallVector<Value> lows;
        SmallVector<Value> highs;
        lows.reserve(groups);
        highs.reserve(groups);
        for (int64_t group = 0; group < groups; ++group) {
          Type lowType = resultTypes[group];
          Type highType = resultTypes[groups + group];
          if (lowType != highType)
            return rewriter.notifyMatchFailure(
                op, "vldsx2 requires matching low/high result types");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), zero, group * 2 * *lanesPerPart, rewriter);
          auto load = rewriter.create<Vldsx2Op>(op.getLoc(), lowType, highType,
                                                *source, chunkOffset,
                                                rewriter.getStringAttr(*dist));
          lows.push_back(load.getLow());
          highs.push_back(load.getHigh());
        }
        SmallVector<Value> results;
        results.reserve(resultTypes.size());
        results.append(lows);
        results.append(highs);
        rewriter.replaceOp(op, results, adaptor.getResultMapping());
        return success();
      }
    }

    SmallVector<Value> contiguousParts;
    contiguousParts.reserve(resultTypes.size());
    for (auto [index, resultType] : llvm::enumerate(resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "tile_read result must be vreg");
      Value chunkOffset =
          createChunkOffset(op.getLoc(), zero, index * *lanesPerPart, rewriter);
      contiguousParts.push_back(rewriter
                                    .create<VldsOp>(op.getLoc(), resultType,
                                                    /*updated_base=*/Type{},
                                                    *source, chunkOffset,
                                                    /*dist=*/nullptr)
                                    .getResult());
    }

    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, contiguousParts, resultTypes,
        VMILayoutAttr::getContiguous(rewriter.getContext()),
        resultVMIType.getLayoutAttr(), rewriter);
    if (failed(results))
      return failure();

    rewriter.replaceOp(op, *results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMITileWriteOpPattern : OneToNOpConversionPattern<VMITileWriteOp> {
  using OneToNOpConversionPattern<VMITileWriteOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMITileWriteOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto valueVMIType = cast<VMIVRegType>(op.getValue().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(valueVMIType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(
          op, "tile_write requires known physical lanes per part");
    bool fullPhysicalChunks =
        succeeded(checkFullDataPhysicalChunks(valueVMIType, nullptr));
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "tile_write destination must convert to one value", rewriter);
    if (failed(destination))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    VMILayoutSupport localSupports;
    FailureOr<VMIContiguousStoreSupport> storeSupport =
        localSupports.getContiguousStoreSupport(valueVMIType);
    if (succeeded(storeSupport) &&
        storeSupport->kind ==
            VMIContiguousStoreSupportKind::Deinterleaved2Vstsx2) {
      std::optional<std::string> dist =
          getX2MemoryDistToken(valueVMIType.getElementType(), "INTLV");
      if (dist && !valueParts.empty() && valueParts.size() % 2 == 0) {
        int64_t groups = valueParts.size() / 2;
        for (int64_t group = 0; group < groups; ++group) {
          Value low = valueParts[group];
          Value high = valueParts[groups + group];
          if (low.getType() != high.getType())
            return rewriter.notifyMatchFailure(
                op, "vstsx2 requires matching low/high value types");
          auto vregType = dyn_cast<VRegType>(low.getType());
          if (!vregType)
            return rewriter.notifyMatchFailure(op,
                                               "tile_write value must be vreg");
          FailureOr<Value> mask =
              createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
          if (failed(mask))
            return rewriter.notifyMatchFailure(
                op, "unsupported element type for tile_write mask");
          Value chunkOffset = createChunkOffset(
              op.getLoc(), zero, group * 2 * *lanesPerPart, rewriter);
          rewriter.create<Vstsx2Op>(op.getLoc(), low, high, *destination,
                                    chunkOffset, rewriter.getStringAttr(*dist),
                                    *mask);
        }
        rewriter.eraseOp(op);
        return success();
      }
    }

    SmallVector<Type> contiguousTypes;
    contiguousTypes.reserve(valueParts.size());
    for (Value value : valueParts)
      contiguousTypes.push_back(value.getType());

    FailureOr<SmallVector<Value>> storeParts = materializeDataLayoutConversion(
        op, valueParts, contiguousTypes, valueVMIType.getLayoutAttr(),
        VMILayoutAttr::getContiguous(rewriter.getContext()), rewriter);
    if (failed(storeParts))
      return failure();

    for (auto [index, value] : llvm::enumerate(*storeParts)) {
      auto vregType = dyn_cast<VRegType>(value.getType());
      if (!vregType)
        return rewriter.notifyMatchFailure(op, "tile_write value must be vreg");
      if (!fullPhysicalChunks) {
        FailureOr<int64_t> activeLanes =
            getContiguousActiveDataLanes(valueVMIType, index);
        if (failed(activeLanes))
          return rewriter.notifyMatchFailure(
              op, "failed to compute tile_write active lanes");
        if (*activeLanes == 0)
          continue;
      }
      FailureOr<Value> mask =
          fullPhysicalChunks
              ? createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter)
              : createContiguousStoreMask(op.getLoc(), valueVMIType, index,
                                          vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for tile_write mask");
      Value chunkOffset =
          createChunkOffset(op.getLoc(), zero, index * *lanesPerPart, rewriter);
      rewriter.create<VstsOp>(op.getLoc(),
                              /*updated_base=*/Type{}, value, *destination,
                              chunkOffset, /*dist=*/nullptr, *mask);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIBinaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical binary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType || lhs.getType() != resultType ||
          rhs.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "physical binary part type mismatch");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for all-true binary mask");
      results.push_back(
          rewriter.create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *mask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIFmaOpPattern : OneToNOpConversionPattern<VMIFmaOp> {
  using OneToNOpConversionPattern<VMIFmaOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIFmaOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    ValueRange accParts = adaptor.getAcc();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != accParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "fma physical arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, acc, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, accParts, resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType || lhs.getType() != resultType ||
          rhs.getType() != resultType || acc.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "fma requires matching physical vreg parts");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(op,
                                           "unsupported element type for fma");
      results.push_back(
          rewriter
              .create<VmulaOp>(op.getLoc(), resultType, acc, lhs, rhs, *mask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIUnaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical unary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [source, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType || source.getType() != resultType)
        return rewriter.notifyMatchFailure(op,
                                           "physical unary part type mismatch");
      FailureOr<Value> mask =
          createAllTrueMaskForVReg(op.getLoc(), vregType, rewriter);
      if (failed(mask))
        return rewriter.notifyMatchFailure(
            op, "unsupported element type for all-true unary mask");
      results.push_back(
          rewriter.create<TargetOp>(op.getLoc(), resultType, source, *mask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIMaskBinaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "physical mask binary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType || lhs.getType() != resultType ||
          rhs.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "physical mask binary part type mismatch");
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask binary seed");
      results.push_back(
          rewriter
              .create<TargetOp>(op.getLoc(), resultType, lhs, rhs, *seedMask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct OneToNVMIMaskUnaryOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op,
                                         "physical mask unary arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [source, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType || source.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "physical mask unary part type mismatch");
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true mask unary seed");
      results.push_back(
          rewriter.create<TargetOp>(op.getLoc(), resultType, source, *seedMask)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

template <typename SourceOp>
struct OneToNVMICmpOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    std::optional<StringRef> cmpMode = getVPTOCmpMode(op.getPredicate());
    if (!cmpMode)
      return op.emitOpError()
             << kVMIDiagUnsupportedPrefix << "compare predicate "
             << op.getPredicate()
             << " cannot be lowered to pto.vcmp; supported predicates are "
                "eq/ne/lt/le/gt/ge, ordered FP forms "
                "oeq/one/olt/ole/ogt/oge, and signed integer forms "
                "slt/sle/sgt/sge";

    ValueRange lhsParts = adaptor.getLhs();
    ValueRange rhsParts = adaptor.getRhs();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (lhsParts.size() != rhsParts.size() ||
        lhsParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical cmp arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [lhs, rhs, resultType] :
         llvm::zip_equal(lhsParts, rhsParts, resultTypes)) {
      auto maskType = dyn_cast<MaskType>(resultType);
      if (!maskType || lhs.getType() != rhs.getType() ||
          !isa<VRegType>(lhs.getType()))
        return rewriter.notifyMatchFailure(op,
                                           "physical cmp part type mismatch");
      FailureOr<Value> seedMask =
          createAllTrueMask(op.getLoc(), maskType, rewriter);
      if (failed(seedMask))
        return rewriter.notifyMatchFailure(
            op, "unsupported mask type for all-true cmp seed");
      results.push_back(rewriter
                            .create<VcmpOp>(op.getLoc(), resultType, lhs, rhs,
                                            *seedMask,
                                            rewriter.getStringAttr(*cmpMode))
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMISelectOpPattern : OneToNOpConversionPattern<VMISelectOp> {
  using OneToNOpConversionPattern<VMISelectOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMISelectOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    ValueRange trueParts = adaptor.getTrueValue();
    ValueRange falseParts = adaptor.getFalseValue();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (maskParts.size() != trueParts.size() ||
        trueParts.size() != falseParts.size() ||
        trueParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical select arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [mask, trueValue, falseValue, resultType] :
         llvm::zip_equal(maskParts, trueParts, falseParts, resultTypes)) {
      if (!isa<MaskType>(mask.getType()) || trueValue.getType() != resultType ||
          falseValue.getType() != resultType || !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "physical select part type mismatch");
      results.push_back(rewriter
                            .create<VselOp>(op.getLoc(), resultType, trueValue,
                                            falseValue, mask)
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIActivePrefixIndexOpPattern
    : OneToNOpConversionPattern<VMIActivePrefixIndexOp> {
  using OneToNOpConversionPattern<
      VMIActivePrefixIndexOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIActivePrefixIndexOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange maskParts = adaptor.getMask();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (maskParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index supports only one physical part");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index requires physical vreg/mask parts");

    auto intType = dyn_cast<IntegerType>(resultType.getElementType());
    if (!intType || !intType.isSignless())
      return rewriter.notifyMatchFailure(
          op, "active_prefix_index requires signless integer result part");

    FailureOr<Value> seedMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(seedMask))
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for active_prefix_index seed mask");

    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0,
                                                       intType.getWidth());
    Value carrier =
        rewriter
            .create<VdupOp>(op.getLoc(), resultType, zero, *seedMask,
                            /*position=*/nullptr)
            .getResult();
    Value result = rewriter
                       .create<VusqzOp>(op.getLoc(), resultType, carrier,
                                        maskParts.front())
                       .getResult();
    rewriter.replaceOp(op, SmallVector<Value>{result},
                       adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMICompressOpPattern : OneToNOpConversionPattern<VMICompressOp> {
  using OneToNOpConversionPattern<VMICompressOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICompressOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.size() != 1 || maskParts.size() != 1 ||
        resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "compress supports only one physical part");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!resultType || sourceParts.front().getType() != resultType ||
        !isa<MaskType>(maskParts.front().getType()))
      return rewriter.notifyMatchFailure(
          op, "compress requires physical source/mask/result parts");

    Value result = rewriter
                       .create<VsqzOp>(op.getLoc(), resultType,
                                       sourceParts.front(), maskParts.front())
                       .getResult();
    rewriter.replaceOp(op, SmallVector<Value>{result},
                       adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMICompressStoreOpPattern
    : OneToNOpConversionPattern<VMICompressStoreOp> {
  using OneToNOpConversionPattern<
      VMICompressStoreOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMICompressStoreOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<Value> destination = getSingleValue(
        op, adaptor.getDestination(),
        "compress_store destination must convert to one value", rewriter);
    FailureOr<Value> offset = getSingleValue(
        op, adaptor.getOffset(),
        "compress_store offset must convert to one value", rewriter);
    if (failed(destination) || failed(offset))
      return failure();

    ValueRange valueParts = adaptor.getValue();
    ValueRange maskParts = adaptor.getMask();
    if (valueParts.size() != 1 || maskParts.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "compress_store supports only one physical part");

    auto valueType = dyn_cast<VRegType>(valueParts.front().getType());
    if (!valueType || !isa<MaskType>(maskParts.front().getType()) ||
        !isa<PtrType>((*destination).getType()))
      return rewriter.notifyMatchFailure(
          op, "compress_store requires physical value/mask and ptr "
              "destination");

    Value storeBase =
        rewriter
            .create<AddPtrOp>(op.getLoc(), (*destination).getType(),
                              *destination, *offset)
            .getResult();
    Value squeezed = rewriter
                         .create<VsqzOp>(op.getLoc(), valueType,
                                         valueParts.front(), maskParts.front())
                         .getResult();
    auto align = rewriter.create<InitAlignOp>(
        op.getLoc(), AlignType::get(rewriter.getContext()));
    auto store = rewriter.create<VsturOp>(
        op.getLoc(), align.getResult().getType(), align.getResult(), squeezed,
        storeBase, rewriter.getStringAttr("POST_UPDATE"));
    rewriter.create<VstarOp>(op.getLoc(), store.getAlignOut(), storeBase);
    rewriter.eraseOp(op);
    return success();
  }
};

struct OneToNVMIReduceAddIOpPattern
    : OneToNOpConversionPattern<VMIReduceAddIOp> {
  using OneToNOpConversionPattern<VMIReduceAddIOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange initParts = adaptor.getInit();
    ValueRange maskParts = adaptor.getMask();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        initParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduce_addi requires matching source/mask chunks and one "
              "init/result chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType || initParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "reduce_addi requires matching physical source/init/result "
              "vregs and one mask");

    for (Value sourcePart : sourceParts)
      if (sourcePart.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addi requires every source chunk to match result "
                "vreg type");
    for (Value maskPart : maskParts)
      if (maskPart.getType() != maskType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addi requires every mask chunk to have the same "
                "predicate type");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create reduce_addi first-lane mask");

    Value accumulator = initParts.front();
    for (auto [sourcePart, maskPart] :
         llvm::zip_equal(sourceParts, maskParts)) {
      Value reduced =
          rewriter
              .create<VcaddOp>(op.getLoc(), resultType, sourcePart, maskPart)
              .getResult();
      accumulator = rewriter
                        .create<VaddOp>(op.getLoc(), resultType, reduced,
                                        accumulator, *firstLaneMask)
                        .getResult();
    }

    rewriter.replaceOp(op, SmallVector<Value>{accumulator},
                       adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIReduceAddFOpPattern
    : OneToNOpConversionPattern<VMIReduceAddFOp> {
  using OneToNOpConversionPattern<VMIReduceAddFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIReduceAddFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange initParts = adaptor.getInit();
    ValueRange maskParts = adaptor.getMask();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        initParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "reduce_addf requires matching source/mask chunks and one "
              "init/result chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType || initParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "reduce_addf requires matching physical source/init/result "
              "vregs and one mask");

    for (Value sourcePart : sourceParts)
      if (sourcePart.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addf requires every source chunk to match result "
                "vreg type");
    for (Value maskPart : maskParts)
      if (maskPart.getType() != maskType)
        return rewriter.notifyMatchFailure(
            op, "reduce_addf requires every mask chunk to have the same "
                "predicate type");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create reduce_addf first-lane mask");

    Value accumulator = initParts.front();
    for (auto [sourcePart, maskPart] :
         llvm::zip_equal(sourceParts, maskParts)) {
      Value reduced =
          rewriter
              .create<VcaddOp>(op.getLoc(), resultType, sourcePart, maskPart)
              .getResult();
      accumulator = rewriter
                        .create<VaddOp>(op.getLoc(), resultType, reduced,
                                        accumulator, *firstLaneMask)
                        .getResult();
    }

    rewriter.replaceOp(op, SmallVector<Value>{accumulator},
                       adaptor.getResultMapping());
    return success();
  }
};

template <typename OpTy, typename GroupReduceOpTy, typename CombineOpTy>
struct OneToNVMIGroupReduceOpPattern : OneToNOpConversionPattern<OpTy> {
  OneToNVMIGroupReduceOpPattern(TypeConverter &typeConverter,
                                MLIRContext *context,
                                const VMITargetCapabilityRegistry &capabilities)
      : OneToNOpConversionPattern<OpTy>(typeConverter, context),
        capabilities(capabilities) {}

  LogicalResult
  matchAndRewrite(OpTy op,
                  typename OneToNOpConversionPattern<OpTy>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);

    VMILayoutSupport supports;
    std::string supportReason;
    FailureOr<VMIGroupReduceAddFSupport> support =
        getSupport(supports, op, &supportReason);
    if (failed(support))
      return rewriter.notifyMatchFailure(
          op, Twine(op->getName().getStringRef()) +
                  " has no layout support: " + supportReason);

    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        sourceVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op, "group reduce requires num_groups to evenly divide lane count");

    if (support->kind == VMIGroupReduceAddFSupportKind::OneVLaneVcgadd) {
      if (sourceParts.size() != maskParts.size() ||
          sourceParts.size() != resultTypes.size() || sourceParts.empty())
        return rewriter.notifyMatchFailure(
            op, "vcgadd group_reduce_addf path requires matching physical "
                "arity");
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "vcgadd group_reduce_addf path requires physical vreg/mask");
      for (auto [sourcePart, maskPart, physicalResultType] :
           llvm::zip_equal(sourceParts, maskParts, resultTypes)) {
        if (sourcePart.getType() != resultType ||
            maskPart.getType() != maskType || physicalResultType != resultType)
          return rewriter.notifyMatchFailure(
              op, "vcgadd group_reduce_addf path requires uniform physical "
                  "chunk types");
      }

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (auto [sourceIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        results.push_back(rewriter
                              .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                       sourcePart,
                                                       maskParts[sourceIndex])
                              .getResult());
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    if (support->kind ==
        VMIGroupReduceAddFSupportKind::TwoVLaneDeinterleaved2VcgaddVadd) {
      int64_t resultPartCount = resultTypes.size();
      if (static_cast<int64_t>(sourceParts.size()) != resultPartCount * 2 ||
          maskParts.size() != sourceParts.size())
        return rewriter.notifyMatchFailure(
            op, "s16 block8 group_reduce_addf arity mismatch");

      SmallVector<Value> results;
      results.reserve(resultPartCount);
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "s16 block8 group_reduce_addf requires physical vreg/mask");
      int64_t numGroups = op.getNumGroupsAttr().getInt();

      for (int64_t resultIndex = 0; resultIndex < resultPartCount;
           ++resultIndex) {
        int64_t activeGroups =
            std::min<int64_t>(8, numGroups - resultIndex * 8);
        FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
            op.getLoc(), maskType, activeGroups, rewriter);
        if (failed(combineMask))
          return rewriter.notifyMatchFailure(
              op, "failed to create s16 block8 combine mask");
        Value loSource = sourceParts[resultIndex];
        Value hiSource = sourceParts[resultPartCount + resultIndex];
        Value loMask = maskParts[resultIndex];
        Value hiMask = maskParts[resultPartCount + resultIndex];
        Type physicalResultType = resultTypes[resultIndex];
        if (physicalResultType != resultType ||
            loSource.getType() != resultType ||
            hiSource.getType() != resultType || loMask.getType() != maskType ||
            hiMask.getType() != maskType)
          return rewriter.notifyMatchFailure(
              op, "s16 block8 group_reduce_addf requires uniform physical "
                  "types");
        Value lo = rewriter
                       .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                loSource, loMask)
                       .getResult();
        Value hi = rewriter
                       .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                                hiSource, hiMask)
                       .getResult();
        results.push_back(rewriter
                              .create<CombineOpTy>(op.getLoc(), resultType, lo,
                                                   hi, *combineMask)
                              .getResult());
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    if (support->kind ==
        VMIGroupReduceAddFSupportKind::FourVLaneDeinterleaved4VcgaddTree) {
      int64_t resultPartCount = resultTypes.size();
      if (static_cast<int64_t>(sourceParts.size()) != resultPartCount * 4 ||
          maskParts.size() != sourceParts.size())
        return rewriter.notifyMatchFailure(
            op, "s32 block8 group_reduce_addf arity mismatch");

      SmallVector<Value> results;
      results.reserve(resultPartCount);
      auto resultType = dyn_cast<VRegType>(resultTypes.front());
      auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
      if (!resultType || !maskType)
        return rewriter.notifyMatchFailure(
            op, "s32 block8 group_reduce_addf requires physical vreg/mask");
      int64_t numGroups = op.getNumGroupsAttr().getInt();

      for (int64_t resultIndex = 0; resultIndex < resultPartCount;
           ++resultIndex) {
        int64_t activeGroups =
            std::min<int64_t>(8, numGroups - resultIndex * 8);
        FailureOr<Value> combineMask = createPrefixMaskForActiveLanes(
            op.getLoc(), maskType, activeGroups, rewriter);
        if (failed(combineMask))
          return rewriter.notifyMatchFailure(
              op, "failed to create s32 block8 combine mask");
        SmallVector<Value, 4> partials;
        partials.reserve(4);
        for (int64_t part = 0; part < 4; ++part) {
          int64_t sourceIndex = part * resultPartCount + resultIndex;
          Value source = sourceParts[sourceIndex];
          Value mask = maskParts[sourceIndex];
          Type physicalResultType = resultTypes[resultIndex];
          if (physicalResultType != resultType ||
              source.getType() != resultType || mask.getType() != maskType)
            return rewriter.notifyMatchFailure(
                op, "s32 block8 group_reduce_addf requires uniform physical "
                    "types");
          partials.push_back(rewriter
                                 .create<GroupReduceOpTy>(
                                     op.getLoc(), resultType, source, mask)
                                 .getResult());
        }
        Value sum01 =
            rewriter
                .create<CombineOpTy>(op.getLoc(), resultType, partials[0],
                                     partials[1], *combineMask)
                .getResult();
        Value sum23 =
            rewriter
                .create<CombineOpTy>(op.getLoc(), resultType, partials[2],
                                     partials[3], *combineMask)
                .getResult();
        results.push_back(rewriter
                              .create<CombineOpTy>(op.getLoc(), resultType,
                                                   sum01, sum23, *combineMask)
                              .getResult());
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    if (support->kind != VMIGroupReduceAddFSupportKind::ContiguousVcaddRows)
      return rewriter.notifyMatchFailure(op,
                                         "unknown group_reduce_add support");

    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    int64_t chunksPerGroup = 0;
    if (failed(checkContiguousFullGroupChunks(op, sourceVMIType, *groupSize,
                                              &lanesPerPart, &groupCount,
                                              &chunksPerGroup, rewriter)))
      return failure();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    bool rowLocalSlots1Result = resultLayout && resultLayout.isGroupSlots() &&
                                resultLayout.getNumGroups() == groupCount &&
                                resultLayout.getSlots() == 1;
    int64_t expectedResultParts =
        rowLocalSlots1Result ? groupCount : groupCount * chunksPerGroup;
    if (sourceParts.size() != maskParts.size() ||
        static_cast<int64_t>(sourceParts.size()) !=
            groupCount * chunksPerGroup ||
        static_cast<int64_t>(resultTypes.size()) != expectedResultParts)
      return rewriter.notifyMatchFailure(
          op, "group_reduce_addf requires matching source/mask/result arity");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto vregType = dyn_cast<VRegType>(resultType);
      if (!vregType)
        return rewriter.notifyMatchFailure(
            op, "group_reduce_addf result must be vreg");
      FailureOr<Value> zero = createZeroVector(op.getLoc(), vregType, rewriter);
      if (failed(zero))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize group_reduce_addf zero result");
      results.push_back(*zero);
    }

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType)
      return rewriter.notifyMatchFailure(
          op, "group_reduce_addf requires physical vreg result and mask");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create group_reduce_addf masks");

    for (int64_t group = 0; group < groupCount; ++group) {
      Value accumulator;

      for (int64_t chunk = 0; chunk < chunksPerGroup; ++chunk) {
        int64_t index = group * chunksPerGroup + chunk;
        if (sourceParts[index].getType() != resultType ||
            maskParts[index].getType() != maskType)
          return rewriter.notifyMatchFailure(
              op, "group_reduce_addf requires uniform physical chunk types");
        Value reduced =
            rewriter
                .create<GroupReduceOpTy>(op.getLoc(), resultType,
                                         sourceParts[index], maskParts[index])
                .getResult();
        if (!accumulator) {
          accumulator = reduced;
          continue;
        }
        accumulator = rewriter
                          .create<CombineOpTy>(op.getLoc(), resultType, reduced,
                                               accumulator, *firstLaneMask)
                          .getResult();
      }

      int64_t destChunk = rowLocalSlots1Result ? group : group * chunksPerGroup;
      results[destChunk] =
          rewriter
              .create<VselOp>(op.getLoc(), resultType, accumulator,
                              results[destChunk], *firstLaneMask)
              .getResult();
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }

private:
  FailureOr<VMIGroupReduceAddFSupport> getSupport(VMILayoutSupport &supports,
                                                  VMIGroupReduceAddFOp op,
                                                  std::string *reason) const {
    return supports.getGroupReduceAddFSupport(capabilities, op, reason);
  }

  FailureOr<VMIGroupReduceAddFSupport> getSupport(VMILayoutSupport &supports,
                                                  VMIGroupReduceAddIOp op,
                                                  std::string *reason) const {
    return supports.getGroupReduceAddISupport(capabilities, op, reason);
  }

  FailureOr<VMIGroupReduceAddFSupport> getSupport(VMILayoutSupport &supports,
                                                  VMIGroupReduceMaxFOp op,
                                                  std::string *reason) const {
    return supports.getGroupReduceMaxFSupport(capabilities, op, reason);
  }

  const VMITargetCapabilityRegistry &capabilities;
};

struct OneToNVMIGroupBroadcastOpPattern
    : OneToNOpConversionPattern<VMIGroupBroadcastOp> {
  using OneToNOpConversionPattern<
      VMIGroupBroadcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIGroupBroadcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
        sourceVMIType, op.getNumGroupsAttr().getInt());
    if (failed(groupSize))
      return rewriter.notifyMatchFailure(
          op,
          "group_broadcast requires num_groups to evenly divide lane count");
    int64_t lanesPerPart = 0;
    int64_t groupCount = 0;
    if (failed(checkFullGroupSlotSourceShape(
            op, sourceVMIType, *groupSize, op.getNumGroupsAttr().getInt(),
            &lanesPerPart, &groupCount, rewriter)))
      return failure();
    int64_t resultLayoutFactor = 0;
    int64_t resultGroupCount = 0;
    if (failed(checkFullGroupBroadcastResultShape(
            op, resultVMIType, *groupSize, lanesPerPart, &resultLayoutFactor,
            &resultGroupCount, rewriter)))
      return failure();
    if (resultGroupCount != groupCount)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast requires matching source/result group slots");

    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.empty() || resultTypes.empty())
      return rewriter.notifyMatchFailure(op, "group_broadcast arity mismatch");

    auto firstSourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!firstSourceType)
      return rewriter.notifyMatchFailure(op,
                                         "group_broadcast source must be vreg");
    unsigned indexBits =
        pto::getPTOStorageElemBitWidth(firstSourceType.getElementType());
    if (indexBits != 8 && indexBits != 16 && indexBits != 32)
      return rewriter.notifyMatchFailure(
          op, "group_broadcast requires 8/16/32-bit index elements");
    auto indexElementType = IntegerType::get(rewriter.getContext(), indexBits);
    auto indexType =
        VRegType::get(rewriter.getContext(), firstSourceType.getElementCount(),
                      indexElementType);
    FailureOr<Value> allMask =
        createAllTrueMaskForVReg(op.getLoc(), firstSourceType, rewriter);
    if (failed(allMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create group_broadcast all mask");
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    int64_t selectionGroupSize = *groupSize;
    if (resultLayoutFactor != 1 && resultLayout &&
        resultLayout.isDeinterleaved() && resultLayout.getBlockElems() > 1 &&
        *groupSize < lanesPerPart)
      selectionGroupSize = resultLayout.getBlockElems();
    auto resolveLargeGroupSource = [&](int64_t group, int64_t chunksPerGroup,
                                       int64_t &sourceChunk,
                                       int64_t &baseGroupSlot) {
      int64_t slots = sourceLayout.getSlots();
      if (slots > 0) {
        sourceChunk = group / slots;
        baseGroupSlot = group % slots;
        return;
      }
      sourceChunk = group * chunksPerGroup;
      baseGroupSlot = 0;
    };

    SmallVector<Value> results;
    results.resize(resultTypes.size());
    for (auto [flatIndex, resultType] : llvm::enumerate(resultTypes)) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType || resultVRegType != firstSourceType)
        return rewriter.notifyMatchFailure(
            op, "group_broadcast requires uniform physical vreg types");
      int64_t sourceChunk = flatIndex;
      int64_t baseGroupSlot = 0;
      if (resultLayoutFactor == 1) {
        if (*groupSize >= lanesPerPart) {
          int64_t chunksPerGroup = *groupSize / lanesPerPart;
          int64_t group = flatIndex / chunksPerGroup;
          resolveLargeGroupSource(group, chunksPerGroup, sourceChunk,
                                  baseGroupSlot);
        } else {
          VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
          int64_t slots = sourceLayout.getSlots();
          if (slots <= 0) {
            if (sourceParts.empty() ||
                groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast small-group source requires explicit "
                      "group_slots slots or derivable legacy slot count");
            slots = groupCount / sourceParts.size();
          }
          int64_t groupsPerResultChunk = lanesPerPart / *groupSize;
          int64_t firstGroup = flatIndex * groupsPerResultChunk;
          sourceChunk = firstGroup / slots;
          baseGroupSlot = firstGroup % slots;
        }
      } else {
        bool blockFragmentSmallGroup =
            resultLayout && resultLayout.isDeinterleaved() &&
            resultLayout.getBlockElems() > 1 && *groupSize < lanesPerPart;
        if (blockFragmentSmallGroup) {
          int64_t runningFlatIndex = 0;
          bool found = false;
          for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
            FailureOr<int64_t> chunks =
                getDataChunksInPart(resultVMIType, part);
            if (failed(chunks))
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast failed to enumerate result chunks");
            for (int64_t chunk = 0; chunk < *chunks;
                 ++chunk, ++runningFlatIndex) {
              if (runningFlatIndex != static_cast<int64_t>(flatIndex))
                continue;
              int64_t groupsPerResultChunk =
                  lanesPerPart / resultLayout.getBlockElems();
              int64_t firstGroup = chunk * groupsPerResultChunk;
              int64_t slots = sourceLayout.getSlots();
              if (slots <= 0) {
                if (sourceParts.empty() ||
                    groupCount % static_cast<int64_t>(sourceParts.size()) != 0)
                  return rewriter.notifyMatchFailure(
                      op,
                      "group_broadcast block-fragment source requires explicit "
                      "group_slots slots or derivable legacy slot count");
                slots = groupCount / sourceParts.size();
              }
              sourceChunk = firstGroup / slots;
              baseGroupSlot = firstGroup % slots;
              found = true;
              break;
            }
          }
          if (!found)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast result chunk index is out of range");
        } else {
          int64_t runningFlatIndex = 0;
          bool found = false;
          for (int64_t part = 0; part < resultLayoutFactor && !found; ++part) {
            FailureOr<int64_t> chunks =
                getDataChunksInPart(resultVMIType, part);
            if (failed(chunks))
              return rewriter.notifyMatchFailure(
                  op, "group_broadcast failed to enumerate result chunks");
            for (int64_t chunk = 0; chunk < *chunks;
                 ++chunk, ++runningFlatIndex) {
              if (runningFlatIndex != static_cast<int64_t>(flatIndex))
                continue;
              FailureOr<int64_t> firstLogical =
                  mapPhysicalLaneToLogical(resultVMIType, part, chunk, 0);
              FailureOr<int64_t> lastLogical = mapPhysicalLaneToLogical(
                  resultVMIType, part, chunk, lanesPerPart - 1);
              if (failed(firstLogical) || failed(lastLogical))
                return rewriter.notifyMatchFailure(
                    op, "group_broadcast failed to map result chunk lanes");
              int64_t firstGroup = *firstLogical / *groupSize;
              int64_t lastGroup = *lastLogical / *groupSize;
              if (firstGroup != lastGroup)
                return rewriter.notifyMatchFailure(
                    op, "group_broadcast result chunk crosses logical groups");
              int64_t chunksPerGroup = *groupSize / lanesPerPart;
              resolveLargeGroupSource(firstGroup, chunksPerGroup, sourceChunk,
                                      baseGroupSlot);
              found = true;
              break;
            }
          }
          if (!found)
            return rewriter.notifyMatchFailure(
                op, "group_broadcast result chunk index is out of range");
        }
      }
      if (*groupSize >= lanesPerPart) {
        if (sourceChunk < 0 ||
            sourceChunk >= static_cast<int64_t>(sourceParts.size()))
          return rewriter.notifyMatchFailure(
              op, "group_broadcast source chunk is out of range");
        if (sourceLayout.getSlots() > 1) {
          FailureOr<Value> groupSlotIndex = createGroupSlotIndexVector(
              op.getLoc(), indexType, selectionGroupSize, baseGroupSlot,
              rewriter);
          if (failed(groupSlotIndex))
            return rewriter.notifyMatchFailure(
                op, "failed to create group_broadcast group-slot index vector");
          results[flatIndex] =
              rewriter
                  .create<VselrOp>(op.getLoc(), resultType,
                                   sourceParts[sourceChunk], *groupSlotIndex)
                  .getResult();
        } else {
          results[flatIndex] =
              rewriter
                  .create<VdupOp>(op.getLoc(), resultType,
                                  sourceParts[sourceChunk], *allMask,
                                  rewriter.getStringAttr("LOWEST"))
                  .getResult();
        }
      } else {
        bool blockFragmentSmallGroup = resultLayout &&
                                       resultLayout.isDeinterleaved() &&
                                       resultLayout.getBlockElems() > 1;
        if (resultLayoutFactor != 1 && !blockFragmentSmallGroup)
          return rewriter.notifyMatchFailure(
              op, "group_broadcast small-group deinterleaved result is not "
                  "supported");
        if (sourceChunk < 0 ||
            sourceChunk >= static_cast<int64_t>(sourceParts.size()))
          return rewriter.notifyMatchFailure(
              op, "group_broadcast source chunk is out of range");
        FailureOr<Value> groupSlotIndex = createGroupSlotIndexVector(
            op.getLoc(), indexType, selectionGroupSize, baseGroupSlot,
            rewriter);
        if (failed(groupSlotIndex))
          return rewriter.notifyMatchFailure(
              op, "failed to create group_broadcast group-slot index vector");
        results[flatIndex] =
            rewriter
                .create<VselrOp>(op.getLoc(), resultType,
                                 sourceParts[sourceChunk], *groupSlotIndex)
                .getResult();
      }
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIDhistOpPattern : OneToNOpConversionPattern<VMIDhistOp> {
  using OneToNOpConversionPattern<VMIDhistOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIDhistOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange accParts = adaptor.getAcc();
    ValueRange sourceParts = adaptor.getSource();
    ValueRange maskParts = adaptor.getMask();
    if (accParts.size() != 2 || sourceParts.empty() ||
        sourceParts.size() != maskParts.size())
      return rewriter.notifyMatchFailure(
          op, "expected two accumulator parts and matching source/mask chunks");

    auto loType = dyn_cast<VRegType>(accParts[0].getType());
    auto hiType = dyn_cast<VRegType>(accParts[1].getType());
    if (!loType || loType != hiType)
      return rewriter.notifyMatchFailure(op,
                                         "expected matching ui16 acc parts");
    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    FailureOr<int64_t> lanesPerPart =
        getDataLanesPerPart(sourceType.getElementType());
    if (failed(lanesPerPart))
      return rewriter.notifyMatchFailure(op, "failed to compute source lanes");

    Location loc = op.getLoc();
    Value bin0 = createI32Constant(loc, 0, rewriter);
    Value bin1 = createI32Constant(loc, 1, rewriter);
    Value lo = accParts[0];
    Value hi = accParts[1];

    for (size_t index = 0, e = sourceParts.size(); index < e; ++index) {
      Value source = sourceParts[index];
      Value userMask = maskParts[index];
      auto maskType = dyn_cast<MaskType>(userMask.getType());
      if (!maskType || !maskType.isB8())
        return rewriter.notifyMatchFailure(op, "expected b8 source mask");

      Value chunkMask = userMask;
      int64_t firstLane = static_cast<int64_t>(index) * *lanesPerPart;
      int64_t activeLanes = std::min<int64_t>(
          *lanesPerPart, sourceType.getElementCount() - firstLane);
      if (activeLanes < *lanesPerPart) {
        FailureOr<Value> validMask = createPrefixMaskForActiveLanes(
            loc, maskType, activeLanes, rewriter);
        FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
        if (failed(validMask) || failed(allMask))
          return rewriter.notifyMatchFailure(
              op, "failed to materialize tail-valid b8 mask");
        chunkMask =
            rewriter
                .create<PandOp>(loc, maskType, chunkMask, *validMask, *allMask)
                .getResult();
      }

      lo = rewriter.create<Dhistv2Op>(loc, loType, lo, source, chunkMask, bin0)
               .getResult();
      hi = rewriter.create<Dhistv2Op>(loc, hiType, hi, source, chunkMask, bin1)
               .getResult();
    }

    rewriter.replaceOp(op, SmallVector<Value>{lo, hi},
                       adaptor.getResultMapping());
    return success();
  }
};

template <typename SourceOp, typename ChunkReduceOp, typename CombineOp>
struct OneToNVMIReduceMinMaxFOpPattern : OneToNOpConversionPattern<SourceOp> {
  using OneToNOpConversionPattern<SourceOp>::OneToNOpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OneToNOpConversionPattern<SourceOp>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    ValueRange initParts = adaptor.getInit();
    ValueRange maskParts = adaptor.getMask();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.empty() || sourceParts.size() != maskParts.size() ||
        initParts.size() != 1 || resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "floating min/max reduction requires matching source/mask chunks "
              "and one init/result chunk");

    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    auto maskType = dyn_cast<MaskType>(maskParts.front().getType());
    if (!resultType || !maskType || initParts.front().getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "floating min/max reduction requires matching physical source/"
              "init/result vregs and one mask");

    for (Value sourcePart : sourceParts)
      if (sourcePart.getType() != resultType)
        return rewriter.notifyMatchFailure(
            op, "floating min/max reduction requires every source chunk to "
                "match result vreg type");
    for (Value maskPart : maskParts)
      if (maskPart.getType() != maskType)
        return rewriter.notifyMatchFailure(
            op, "floating min/max reduction requires every mask chunk to have "
                "the same predicate type");

    FailureOr<Value> firstLaneMask =
        createPrefixMask(op.getLoc(), maskType, "PAT_VL1", rewriter);
    if (failed(firstLaneMask))
      return rewriter.notifyMatchFailure(
          op, "failed to create floating min/max reduction first-lane mask");

    Value accumulator = initParts.front();
    for (auto [sourcePart, maskPart] :
         llvm::zip_equal(sourceParts, maskParts)) {
      Value reduced = rewriter
                          .create<ChunkReduceOp>(op.getLoc(), resultType,
                                                 sourcePart, maskPart)
                          .getResult();
      accumulator = rewriter
                        .create<CombineOp>(op.getLoc(), resultType, reduced,
                                           accumulator, *firstLaneMask)
                        .getResult();
    }

    rewriter.replaceOp(op, SmallVector<Value>{accumulator},
                       adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIExtFOpPattern : OneToNOpConversionPattern<VMIExtFOp> {
  using OneToNOpConversionPattern<VMIExtFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIExtFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.empty())
      return rewriter.notifyMatchFailure(
          op, "extf requires at least one physical source chunk");

    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(op, "expected physical extf source");
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType)
        return rewriter.notifyMatchFailure(
            op, "extf source physical parts must have matching type");
    }

    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType ||
          (resultVRegTypes.empty() ? !resultVRegType.getElementType().isF32()
                                   : resultVRegType != resultVRegTypes.front()))
        return rewriter.notifyMatchFailure(
            op, "unsupported physical extf result type");
      resultVRegTypes.push_back(resultVRegType);
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (sourceBits == 16 && resultTypes.size() == 2 * sourceParts.size()) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = 2;
    } else if (sourceBits == 8 &&
               resultTypes.size() == 4 * sourceParts.size()) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical extf source/result width relation");
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask))
      return rewriter.notifyMatchFailure(op, "failed to build extf seed mask");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask,
                                /*rnd=*/nullptr, /*sat=*/nullptr,
                                rewriter.getStringAttr(parts[partIndex]))
                .getResult());
      }
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMITruncFOpPattern : OneToNOpConversionPattern<VMITruncFOp> {
  using OneToNOpConversionPattern<VMITruncFOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMITruncFOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots()) {
      if (sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
          sourceLayout.getSlots() != 1 || resultLayout.getSlots() != 1 ||
          !sourceVMIType.getElementType().isF32() ||
          pto::getPTOStorageElemBitWidth(resultVMIType.getElementType()) !=
              16 ||
          sourceParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "unsupported group-slot truncf shape");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      StringAttr rnd = rewriter.getStringAttr("R");
      StringAttr sat = rewriter.getStringAttr("SAT");
      StringAttr even = rewriter.getStringAttr("EVEN");
      FailureOr<Value> lane0Mask = createPrefixMask(
          op.getLoc(), MaskType::get(rewriter.getContext(), "b32"), "PAT_VL1",
          rewriter);
      if (failed(lane0Mask))
        return rewriter.notifyMatchFailure(
            op, "failed to build group-slot truncf lane0 mask");
      for (auto [sourcePart, physicalResultType] :
           llvm::zip_equal(sourceParts, resultTypes)) {
        auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
        auto resultType = dyn_cast<VRegType>(physicalResultType);
        if (!sourceType || !sourceType.getElementType().isF32() ||
            !resultType ||
            pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 16)
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot truncf physical type");
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *lane0Mask, rnd, sat,
                                              even)
                              .getResult());
      }
      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    if ((sourceParts.size() != 2 && sourceParts.size() != 4) ||
        resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "only f32 deinterleaved=2/4 to 16/8-bit contiguous truncf is "
              "supported");

    auto sourceType0 = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!sourceType0 || !sourceType0.getElementType().isF32() || !resultType)
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result type");
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != sourceType0)
        return rewriter.notifyMatchFailure(
            op, "truncf source physical parts must have matching f32 type");
    }

    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    ArrayRef<StringRef> parts;
    if (sourceParts.size() == 2 && resultBits == 16) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
    } else if (sourceParts.size() == 4 && resultBits == 8) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical truncf source/result width relation");
    }

    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType0, rewriter);
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(sourceMask) || failed(resultMask))
      return rewriter.notifyMatchFailure(op, "failed to build truncf masks");

    StringAttr rnd = rewriter.getStringAttr("R");
    StringAttr sat = rewriter.getStringAttr("SAT");
    SmallVector<Value> partials;
    partials.reserve(parts.size());
    for (auto [sourcePart, part] : llvm::zip_equal(sourceParts, parts)) {
      partials.push_back(rewriter
                             .create<VcvtOp>(op.getLoc(), resultType,
                                             sourcePart, *sourceMask, rnd, sat,
                                             rewriter.getStringAttr(part))
                             .getResult());
    }

    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials))
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                  *resultMask)
                   .getResult();

    rewriter.replaceOp(op, merged, adaptor.getResultMapping());
    return success();
  }
};

template <typename OpT>
struct OneToNVMIExtIOpPattern : OneToNOpConversionPattern<OpT> {
  using OneToNOpConversionPattern<OpT>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op,
                  typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.empty())
      return rewriter.notifyMatchFailure(
          op, "integer extension requires at least one physical source chunk");

    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType)
      return rewriter.notifyMatchFailure(
          op, "expected physical integer extension source");
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType)
        return rewriter.notifyMatchFailure(
            op, "integer extension source physical parts must have matching "
                "type");
    }

    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType ||
          !isa<IntegerType>(resultVRegType.getElementType()) ||
          (resultVRegTypes.empty() ? pto::getPTOStorageElemBitWidth(
                                         resultVRegType.getElementType()) != 32
                                   : resultVRegType != resultVRegTypes.front()))
        return rewriter.notifyMatchFailure(
            op, "unsupported physical integer extension result type");
      resultVRegTypes.push_back(resultVRegType);
    }

    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceType.getElementType());
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (sourceBits == 16 && resultTypes.size() == 2 * sourceParts.size()) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = 2;
    } else if (sourceBits == 8 &&
               resultTypes.size() == 4 * sourceParts.size()) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = 4;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical integer extension source/result width "
              "relation");
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask))
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t partIndex = 0; partIndex < factor; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        results.push_back(
            rewriter
                .create<VcvtOp>(op.getLoc(), resultType, sourcePart, *mask,
                                /*rnd=*/nullptr, /*sat=*/nullptr,
                                rewriter.getStringAttr(parts[partIndex]))
                .getResult());
      }
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMITruncIOpPattern : OneToNOpConversionPattern<VMITruncIOp> {
  using OneToNOpConversionPattern<VMITruncIOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMITruncIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);

    VMILayoutAttr sourceLayout = sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = resultVMIType.getLayoutAttr();
    if (sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots()) {
      if (sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
          sourceLayout.getSlots() != 1 || resultLayout.getSlots() != 1 ||
          pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType()) !=
              32 ||
          pto::getPTOStorageElemBitWidth(resultVMIType.getElementType()) !=
              16 ||
          sourceParts.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(
            op, "unsupported group-slot trunci shape");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      StringAttr sat = rewriter.getStringAttr("SAT");
      StringAttr even = rewriter.getStringAttr("EVEN");
      FailureOr<Value> lane0Mask = createPrefixMask(
          op.getLoc(), MaskType::get(rewriter.getContext(), "b32"), "PAT_VL1",
          rewriter);
      if (failed(lane0Mask))
        return rewriter.notifyMatchFailure(
            op, "failed to build group-slot trunci lane0 mask");
      for (auto [sourcePart, physicalResultType] :
           llvm::zip_equal(sourceParts, resultTypes)) {
        auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
        auto resultType = dyn_cast<VRegType>(physicalResultType);
        if (!sourceType ||
            pto::getPTOStorageElemBitWidth(sourceType.getElementType()) != 32 ||
            !resultType ||
            pto::getPTOStorageElemBitWidth(resultType.getElementType()) != 16)
          return rewriter.notifyMatchFailure(
              op, "unsupported group-slot trunci physical type");
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType,
                                              sourcePart, *lane0Mask,
                                              /*rnd=*/nullptr, sat, even)
                              .getResult());
      }
      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    if ((sourceParts.size() != 2 && sourceParts.size() != 4) ||
        resultTypes.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "only 32-bit integer deinterleaved=2/4 to 16/8-bit contiguous "
              "trunci is supported");

    auto sourceType0 = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!sourceType0 || !isa<IntegerType>(sourceType0.getElementType()) ||
        !resultType || !isa<IntegerType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    for (Value sourcePart : sourceParts) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!sourceType || sourceType != sourceType0)
        return rewriter.notifyMatchFailure(
            op, "trunci source physical parts must have matching 32-bit "
                "integer type");
    }

    if (pto::getPTOStorageElemBitWidth(sourceType0.getElementType()) != 32)
      return rewriter.notifyMatchFailure(
          op, "trunci source physical element width must be 32-bit");
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    ArrayRef<StringRef> parts;
    if (sourceParts.size() == 2 && resultBits == 16) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
    } else if (sourceParts.size() == 4 && resultBits == 8) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result width relation");
    }

    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType0, rewriter);
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    if (failed(sourceMask) || failed(resultMask))
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");

    StringAttr sat = rewriter.getStringAttr("SAT");
    SmallVector<Value> partials;
    partials.reserve(parts.size());
    for (auto [sourcePart, part] : llvm::zip_equal(sourceParts, parts)) {
      partials.push_back(rewriter
                             .create<VcvtOp>(op.getLoc(), resultType,
                                             sourcePart, *sourceMask,
                                             /*rnd=*/nullptr, sat,
                                             rewriter.getStringAttr(part))
                             .getResult());
    }

    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials))
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultType, merged, partial,
                                  *resultMask)
                   .getResult();

    rewriter.replaceOp(op, merged, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIBitcastOpPattern : OneToNOpConversionPattern<VMIBitcastOp> {
  using OneToNOpConversionPattern<VMIBitcastOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIBitcastOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    if (sourceParts.size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "physical bitcast arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      if (!isa<VRegType>(sourcePart.getType()) || !isa<VRegType>(resultType))
        return rewriter.notifyMatchFailure(
            op, "physical bitcast part type mismatch");
      results.push_back(
          rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
              .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIChannelSplitOpPattern
    : OneToNOpConversionPattern<VMIChannelSplitOp> {
  using OneToNOpConversionPattern<VMIChannelSplitOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIChannelSplitOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    int64_t channels = op.getNumResults();
    if (channels != 2 && channels != 4)
      return rewriter.notifyMatchFailure(
          op, "channel_split only supports 2 or 4 channels");

    auto sourceType = cast<VMIVRegType>(op.getSource().getType());
    VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
    auto channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    if (!sourceLayout ||
        (!sourceLayout.isContiguous() && sourceLayout != channelLayout))
      return rewriter.notifyMatchFailure(
          op,
          "channel_split requires contiguous or matching deinterleaved source "
          "layout");
    for (Value result : op.getResults()) {
      auto resultType = cast<VMIVRegType>(result.getType());
      VMILayoutAttr resultLayout = resultType.getLayoutAttr();
      if (!resultLayout || !resultLayout.isContiguous())
        return rewriter.notifyMatchFailure(
            op, "channel_split requires contiguous result layouts");
    }

    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes();
    FailureOr<SmallVector<Value>> results =
        materializeDataLayoutConversion(op, adaptor.getSource(), resultTypes,
                                        sourceLayout, channelLayout, rewriter);
    if (failed(results))
      return failure();

    rewriter.replaceOp(op, *results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIChannelMergeOpPattern
    : OneToNOpConversionPattern<VMIChannelMergeOp> {
  using OneToNOpConversionPattern<VMIChannelMergeOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIChannelMergeOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    int64_t channels = op.getInputs().size();
    if (channels != 2 && channels != 4)
      return rewriter.notifyMatchFailure(
          op, "channel_merge only supports 2 or 4 channels");

    for (Value input : op.getInputs()) {
      auto inputType = cast<VMIVRegType>(input.getType());
      VMILayoutAttr inputLayout = inputType.getLayoutAttr();
      if (!inputLayout || !inputLayout.isContiguous())
        return rewriter.notifyMatchFailure(
            op, "channel_merge requires contiguous input layouts");
    }
    auto resultType = cast<VMIVRegType>(op.getResult().getType());
    VMILayoutAttr resultLayout = resultType.getLayoutAttr();
    auto channelLayout =
        VMILayoutAttr::getDeinterleaved(rewriter.getContext(), channels);
    if (!resultLayout ||
        (!resultLayout.isContiguous() && resultLayout != channelLayout))
      return rewriter.notifyMatchFailure(
          op,
          "channel_merge requires contiguous or matching deinterleaved result "
          "layout");

    FailureOr<SmallVector<Value>> results = materializeDataLayoutConversion(
        op, adaptor.getFlatOperands(),
        adaptor.getResultMapping().getConvertedTypes(0), channelLayout,
        resultLayout, rewriter);
    if (failed(results))
      return failure();

    rewriter.replaceOp(op, *results, adaptor.getResultMapping());
    return success();
  }
};

struct OneToNVMIShuffleOpPattern : OneToNOpConversionPattern<VMIShuffleOp> {
  using OneToNOpConversionPattern<VMIShuffleOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(VMIShuffleOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange sourceParts = adaptor.getSource();
    TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
    std::string reason;
    FailureOr<SmallVector<int64_t>> sourceFlatIndices =
        computeShuffleForwardingSourceParts(op, &reason);
    if (succeeded(sourceFlatIndices)) {
      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      for (int64_t sourceFlatIndex : *sourceFlatIndices) {
        if (sourceFlatIndex >= static_cast<int64_t>(sourceParts.size()))
          return rewriter.notifyMatchFailure(
              op, "shuffle forwarding source part range is out of bounds");
        results.push_back(sourceParts[sourceFlatIndex]);
      }

      if (failed(
              verifyIdentityPartForwarding(op, results, resultTypes, rewriter)))
        return failure();

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    std::string splatReason;
    FailureOr<int64_t> splatSource =
        computeShuffleLane0SplatSourcePart(op, &splatReason);
    if (succeeded(splatSource)) {
      if (*splatSource >= static_cast<int64_t>(sourceParts.size()))
        return rewriter.notifyMatchFailure(
            op, "shuffle lane0 splat source part range is out of bounds");

      SmallVector<Value> results;
      results.reserve(resultTypes.size());
      Value sourcePart = sourceParts[*splatSource];
      for (Type resultType : resultTypes) {
        auto sourceVRegType = dyn_cast<VRegType>(sourcePart.getType());
        auto resultVRegType = dyn_cast<VRegType>(resultType);
        if (!sourceVRegType || !resultVRegType ||
            sourceVRegType != resultVRegType)
          return rewriter.notifyMatchFailure(
              op, "shuffle lane0 splat requires matching physical vreg type");
        FailureOr<Value> mask =
            createAllTrueMaskForVReg(op.getLoc(), resultVRegType, rewriter);
        if (failed(mask))
          return rewriter.notifyMatchFailure(
              op, "failed to create shuffle lane0 splat mask");
        results.push_back(rewriter
                              .create<VdupOp>(op.getLoc(), resultType,
                                              sourcePart, *mask,
                                              rewriter.getStringAttr("LOWEST"))
                              .getResult());
      }

      rewriter.replaceOp(op, results, adaptor.getResultMapping());
      return success();
    }

    std::string vselrReason;
    FailureOr<SmallVector<ShuffleVselrPlan>> vselrPlans =
        computeShuffleVselrPlans(op, &vselrReason);
    if (failed(vselrPlans))
      return rewriter.notifyMatchFailure(op,
                                         Twine("shuffle vselr ") + vselrReason);

    if (vselrPlans->size() != resultTypes.size())
      return rewriter.notifyMatchFailure(op, "shuffle vselr arity mismatch");

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [plan, resultType] : llvm::zip_equal(*vselrPlans, resultTypes)) {
      if (plan.sourceFlatIndex >= static_cast<int64_t>(sourceParts.size()))
        return rewriter.notifyMatchFailure(
            op, "shuffle vselr source part range is out of bounds");

      auto sourceVRegType =
          dyn_cast<VRegType>(sourceParts[plan.sourceFlatIndex].getType());
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!sourceVRegType || !resultVRegType ||
          sourceVRegType.getElementCount() !=
              resultVRegType.getElementCount() ||
          sourceVRegType.getElementType() != resultVRegType.getElementType())
        return rewriter.notifyMatchFailure(
            op, "shuffle vselr source/result type mismatch");

      unsigned indexBits =
          pto::getPTOStorageElemBitWidth(sourceVRegType.getElementType());
      if (indexBits != 8 && indexBits != 16 && indexBits != 32)
        return rewriter.notifyMatchFailure(
            op, "shuffle vselr requires 8/16/32-bit index elements");

      auto indexElementType =
          IntegerType::get(rewriter.getContext(), indexBits);
      Type indexType =
          VRegType::get(rewriter.getContext(), sourceVRegType.getElementCount(),
                        indexElementType);
      FailureOr<Value> base = createScalarOffsetConstant(
          op.getLoc(), indexElementType, plan.baseLane, rewriter);
      if (failed(base))
        return rewriter.notifyMatchFailure(
            op, "failed to materialize shuffle vselr index base");
      StringAttr orderAttr =
          plan.descending ? rewriter.getStringAttr("DESC") : StringAttr{};
      Value indexVector =
          rewriter.create<VciOp>(op.getLoc(), indexType, *base, orderAttr)
              .getResult();
      results.push_back(rewriter
                            .create<VselrOp>(op.getLoc(), resultType,
                                             sourceParts[plan.sourceFlatIndex],
                                             indexVector)
                            .getResult());
    }

    rewriter.replaceOp(op, results, adaptor.getResultMapping());
    return success();
  }
};

Block *convertBranchDestBlock(Block *block, OneToNPatternRewriter &rewriter,
                              OneToNTypeConverter &typeConverter,
                              llvm::DenseMap<Block *, Block *> &converted) {
  auto [it, inserted] = converted.try_emplace(block, nullptr);
  if (!inserted)
    return it->second;

  OneToNTypeMapping argMapping(block->getArgumentTypes());
  if (failed(typeConverter.computeTypeMapping(block->getArgumentTypes(),
                                              argMapping)) ||
      !argMapping.hasNonIdentityConversion()) {
    it->second = block;
    return block;
  }

  Block *newBlock = rewriter.applySignatureConversion(block, argMapping);
  it->second = newBlock;
  return newBlock;
}

struct OneToNCFBranchOpPattern : OneToNOpConversionPattern<cf::BranchOp> {
  using OneToNOpConversionPattern<cf::BranchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *dest = convertBranchDestBlock(op.getDest(), rewriter, *converter,
                                         convertedBlocks);

    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        dest == op.getDest())
      return failure();

    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, dest,
                                              adaptor.getFlatOperands());
    return success();
  }
};

struct OneToNCFCondBranchOpPattern
    : OneToNOpConversionPattern<cf::CondBranchOp> {
  using OneToNOpConversionPattern<cf::CondBranchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *trueDest = convertBranchDestBlock(op.getTrueDest(), rewriter,
                                             *converter, convertedBlocks);
    Block *falseDest = convertBranchDestBlock(op.getFalseDest(), rewriter,
                                              *converter, convertedBlocks);

    if (!adaptor.getOperandMapping().hasNonIdentityConversion() &&
        trueDest == op.getTrueDest() && falseDest == op.getFalseDest())
      return failure();

    ValueRange condition = adaptor.getCondition();
    if (condition.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "condition converted to multiple values");

    SmallVector<Value> trueOperands;
    SmallVector<Value> falseOperands;
    ValueRange flatOperands = adaptor.getFlatOperands();
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();
    unsigned operandIndex = 1;
    for (unsigned i = 0, e = op.getNumTrueOperands(); i < e; ++i)
      llvm::append_range(trueOperands, operandMapping.getConvertedValues(
                                           flatOperands, operandIndex++));
    for (unsigned i = 0, e = op.getNumFalseOperands(); i < e; ++i)
      llvm::append_range(falseOperands, operandMapping.getConvertedValues(
                                            flatOperands, operandIndex++));

    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(op, condition.front(),
                                                  trueDest, trueOperands,
                                                  falseDest, falseOperands);
    return success();
  }
};

struct OneToNCFSwitchOpPattern : OneToNOpConversionPattern<cf::SwitchOp> {
  using OneToNOpConversionPattern<cf::SwitchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::SwitchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter<OneToNTypeConverter>();
    llvm::DenseMap<Block *, Block *> convertedBlocks;
    Block *defaultDest = convertBranchDestBlock(
        op.getDefaultDestination(), rewriter, *converter, convertedBlocks);

    SmallVector<Block *> caseDests;
    caseDests.reserve(op.getCaseDestinations().size());
    for (Block *dest : op.getCaseDestinations())
      caseDests.push_back(
          convertBranchDestBlock(dest, rewriter, *converter, convertedBlocks));

    bool changed = defaultDest != op.getDefaultDestination();
    for (auto [oldDest, newDest] :
         llvm::zip(op.getCaseDestinations(), caseDests))
      changed |= oldDest != newDest;
    changed |= adaptor.getOperandMapping().hasNonIdentityConversion();
    if (!changed)
      return failure();

    ValueRange flag = adaptor.getFlag();
    if (flag.size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "flag converted to multiple values");

    SmallVector<Value> defaultOperands;
    SmallVector<SmallVector<Value>> caseOperandStorage;
    SmallVector<ValueRange> caseOperands;
    ValueRange flatOperands = adaptor.getFlatOperands();
    const OneToNTypeMapping &operandMapping = adaptor.getOperandMapping();

    unsigned operandIndex = 1;
    for (unsigned i = 0, e = op.getDefaultOperands().size(); i < e; ++i)
      llvm::append_range(defaultOperands, operandMapping.getConvertedValues(
                                              flatOperands, operandIndex++));

    caseOperandStorage.reserve(op.getCaseOperandSegments().size());
    caseOperands.reserve(op.getCaseOperandSegments().size());
    for (int32_t segmentSize : op.getCaseOperandSegments()) {
      SmallVector<Value> operands;
      for (int32_t i = 0; i < segmentSize; ++i)
        llvm::append_range(operands, operandMapping.getConvertedValues(
                                         flatOperands, operandIndex++));
      caseOperandStorage.push_back(std::move(operands));
    }
    for (SmallVector<Value> &operands : caseOperandStorage)
      caseOperands.push_back(operands);

    rewriter.replaceOpWithNewOp<cf::SwitchOp>(
        op, flag.front(), defaultDest, defaultOperands, op.getCaseValuesAttr(),
        caseDests, caseOperands);
    return success();
  }
};

struct OneToNSCFExecuteRegionOpPattern
    : OneToNOpConversionPattern<scf::ExecuteRegionOp> {
  using OneToNOpConversionPattern<
      scf::ExecuteRegionOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ExecuteRegionOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
      llvm::append_range(resultTypes, resultMapping.getConvertedTypes(i));
    if (resultTypes == op->getResultTypes())
      return failure();

    auto newOp =
        rewriter.create<scf::ExecuteRegionOp>(op.getLoc(), resultTypes);
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());
    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

struct OneToNSCFIndexSwitchOpPattern
    : OneToNOpConversionPattern<scf::IndexSwitchOp> {
  using OneToNOpConversionPattern<
      scf::IndexSwitchOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    ValueRange arg = adaptor.getArg();
    if (arg.size() != 1)
      return rewriter.notifyMatchFailure(
          op, "index_switch selector converted to multiple values");

    SmallVector<Type> resultTypes;
    const OneToNTypeMapping &resultMapping = adaptor.getResultMapping();
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
      llvm::append_range(resultTypes, resultMapping.getConvertedTypes(i));
    if (resultTypes == op->getResultTypes())
      return failure();

    auto newOp = rewriter.create<scf::IndexSwitchOp>(
        op.getLoc(), resultTypes, arg.front(), op.getCases(), op.getNumCases());
    newOp->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op.getDefaultRegion(), newOp.getDefaultRegion(),
                                newOp.getDefaultRegion().end());
    for (auto [srcRegion, dstRegion] :
         llvm::zip(op.getCaseRegions(), newOp.getCaseRegions()))
      rewriter.inlineRegionBefore(srcRegion, dstRegion, dstRegion.end());
    rewriter.replaceOp(op, newOp->getResults(), resultMapping);
    return success();
  }
};

void populateVMIOneToNConversionPatterns(
    VMIToVPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
    const VMITargetCapabilityRegistry &capabilities) {
  populateFuncTypeConversionPatterns(typeConverter, patterns);
  scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
  patterns.add<OneToNCFBranchOpPattern, OneToNCFCondBranchOpPattern,
               OneToNCFSwitchOpPattern>(typeConverter, patterns.getContext());
  patterns.add<OneToNSCFExecuteRegionOpPattern, OneToNSCFIndexSwitchOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<OneToNVMIPackOpPattern, OneToNVMIUnpackOpPattern>(
      typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIEnsureLayoutOpPattern, OneToNVMIEnsureMaskLayoutOpPattern,
      OneToNVMIBroadcastOpPattern, OneToNVMIIotaOpPattern,
      OneToNVMIConstantOpPattern, OneToNVMIConstantMaskOpPattern,
      OneToNVMICreateMaskOpPattern, OneToNVMICreateGroupMaskOpPattern,
      OneToNVMIMaskBinaryOpPattern<VMIMaskAndOp, PandOp>,
      OneToNVMIMaskBinaryOpPattern<VMIMaskOrOp, PorOp>,
      OneToNVMIMaskBinaryOpPattern<VMIMaskXOrOp, PxorOp>,
      OneToNVMIMaskUnaryOpPattern<VMIMaskNotOp, PnotOp>, OneToNVMILoadOpPattern,
      OneToNVMIGroupLoadOpPattern, OneToNVMIGroupSlotLoadOpPattern,
      OneToNVMIMaskedLoadOpPattern, OneToNVMIGatherOpPattern,
      OneToNVMIExpandLoadOpPattern, OneToNVMIStoreOpPattern,
      OneToNVMIGroupStoreOpPattern, OneToNVMIMaskedStoreOpPattern,
      OneToNVMIScatterOpPattern, OneToNVMITileReadOpPattern,
      OneToNVMITileWriteOpPattern, OneToNVMIBinaryOpPattern<VMIAddFOp, VaddOp>,
      OneToNVMIBinaryOpPattern<VMIAddIOp, VaddOp>,
      OneToNVMIBinaryOpPattern<VMISubFOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMISubIOp, VsubOp>,
      OneToNVMIBinaryOpPattern<VMIMulFOp, VmulOp>,
      OneToNVMIBinaryOpPattern<VMIMulIOp, VmulOp>, OneToNVMIFmaOpPattern,
      OneToNVMIBinaryOpPattern<VMIDivFOp, VdivOp>,
      OneToNVMIBinaryOpPattern<VMIMinFOp, VminOp>,
      OneToNVMIBinaryOpPattern<VMIMaxFOp, VmaxOp>,
      OneToNVMIUnaryOpPattern<VMINegFOp, VnegOp>,
      OneToNVMIUnaryOpPattern<VMIAbsFOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMIAbsIOp, VabsOp>,
      OneToNVMIUnaryOpPattern<VMISqrtOp, VsqrtOp>,
      OneToNVMIUnaryOpPattern<VMIExpOp, VexpOp>,
      OneToNVMIUnaryOpPattern<VMILnOp, VlnOp>,
      OneToNVMIUnaryOpPattern<VMIReluOp, VreluOp>,
      OneToNVMIBinaryOpPattern<VMIAndIOp, VandOp>,
      OneToNVMIBinaryOpPattern<VMIOrIOp, VorOp>,
      OneToNVMIBinaryOpPattern<VMIXOrIOp, VxorOp>,
      OneToNVMIBinaryOpPattern<VMIShLIOp, VshlOp>,
      OneToNVMIBinaryOpPattern<VMIShRUIOp, VshrOp>,
      OneToNVMIUnaryOpPattern<VMINotOp, VnotOp>,
      OneToNVMICmpOpPattern<VMICmpFOp>, OneToNVMICmpOpPattern<VMICmpIOp>,
      OneToNVMISelectOpPattern, OneToNVMIActivePrefixIndexOpPattern,
      OneToNVMICompressOpPattern, OneToNVMICompressStoreOpPattern,
      OneToNVMIReduceAddIOpPattern, OneToNVMIReduceAddFOpPattern,
      OneToNVMIGroupBroadcastOpPattern, OneToNVMIDhistOpPattern,
      OneToNVMIReduceMinMaxFOpPattern<VMIReduceMaxFOp, VcmaxOp, VmaxOp>,
      OneToNVMIReduceMinMaxFOpPattern<VMIReduceMinFOp, VcminOp, VminOp>,
      OneToNVMIExtFOpPattern, OneToNVMITruncFOpPattern,
      OneToNVMIExtIOpPattern<VMIExtSIOp>, OneToNVMIExtIOpPattern<VMIExtUIOp>,
      OneToNVMITruncIOpPattern, OneToNVMIBitcastOpPattern,
      OneToNVMIChannelSplitOpPattern, OneToNVMIChannelMergeOpPattern,
      OneToNVMIShuffleOpPattern>(typeConverter, patterns.getContext());
  patterns.add<
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceAddFOp, VcgaddOp, VaddOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceAddIOp, VcgaddOp, VaddOp>,
      OneToNVMIGroupReduceOpPattern<VMIGroupReduceMaxFOp, VcgmaxOp, VmaxOp>>(
      typeConverter, patterns.getContext(), capabilities);
  patterns.add<OneToNVMIEnsureMaskGranularityOpPattern>(
      typeConverter, patterns.getContext(), capabilities);
}

LogicalResult verifyNoResidualVMIIR(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (isa<UnrealizedConversionCastOp>(op)) {
      op->emitError() << kVMIDiagResidualOpPrefix
                      << "unrealized conversion cast remains after vmi-to-vpto";
      return WalkResult::interrupt();
    }
    if (auto createMask = dyn_cast<VMICreateMaskOp>(op)) {
      if (!createMask.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
        createMask.emitError()
            << kVMIDiagUnsupportedPrefix
            << "dynamic pto.vmi.create_mask active_lanes could not be lowered "
               "by the current runtime predicate generation plan";
        return WalkResult::interrupt();
      }
    }
    if (auto constant = dyn_cast<VMIConstantOp>(op)) {
      auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (denseAttr && !denseAttr.isSplat()) {
        constant.emitError()
            << kVMIDiagUnsupportedPrefix
            << "non-splat pto.vmi.constant requires a vreg immediate or "
               "scratch materialization plan";
        return WalkResult::interrupt();
      }
    }
    if (isVMIOp(op) || hasVMIType(op)) {
      op->emitError() << kVMIDiagResidualOpPrefix
                      << "failed to convert all VMI ops/types to VPTO";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult checkSupportedExtFShape(VMIExtFOp op,
                                      std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtFSupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedTruncFShape(VMITruncFOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncFSupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedExtSIShape(VMIExtSIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtSISupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedExtUIShape(VMIExtUIOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getExtUISupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedTruncIShape(VMITruncIOp op,
                                        std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (failed(supports.getTruncISupport(op, reason)))
    return failure();
  return success();
}

LogicalResult checkSupportedBitcastShape(VMIBitcastOp op, std::string *reason) {
  VMILayoutSupport supports;
  if (failed(supports.getBitcastSupport(op, reason)))
    return failure();
  return success();
}

LogicalResult
checkSupportedChannelSplitShape(const VMITargetCapabilityRegistry &capabilities,
                                VMIChannelSplitOp op,
                                std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  int64_t channels = op.getNumResults();
  VMICapabilityResult channelCapability =
      capabilities.supportsChannelCount("pto.vmi.channel_split", channels);
  if (!channelCapability.isSupported())
    return fail(channelCapability.reason);

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  if (!sourceLayout)
    return fail("requires assigned source layout");
  auto expectedLayout =
      VMILayoutAttr::getDeinterleaved(op.getContext(), channels);
  if (!sourceLayout.isContiguous() && sourceLayout != expectedLayout)
    return fail("requires source layout to be contiguous or matching "
                "deinterleaved channel layout");

  for (Value result : op.getResults()) {
    VMILayoutAttr resultLayout =
        cast<VMIVRegType>(result.getType()).getLayoutAttr();
    if (!resultLayout || !resultLayout.isContiguous())
      return fail("requires every result layout to be contiguous");
  }

  auto channelType =
      VMIVRegType::get(op.getContext(), sourceType.getElementCount(),
                       sourceType.getElementType(), expectedLayout);
  std::string materializationReason;
  if (failed(checkSupportedLayoutMaterialization(
          capabilities, sourceType, channelType, sourceLayout, expectedLayout,
          &materializationReason)))
    return fail(Twine("cannot materialize source to channel layout; ") +
                materializationReason);

  FailureOr<int64_t> channelArity = getVMIPhysicalArity(channelType);
  int64_t resultArity = 0;
  for (Value result : op.getResults()) {
    FailureOr<int64_t> arity =
        getVMIPhysicalArity(cast<VMIVRegType>(result.getType()));
    if (failed(arity))
      return fail("requires computable result physical arity");
    resultArity += *arity;
  }
  if (failed(channelArity) || *channelArity != resultArity)
    return fail("requires channel physical arity to match all result parts");

  return success();
}

LogicalResult
checkSupportedChannelMergeShape(const VMITargetCapabilityRegistry &capabilities,
                                VMIChannelMergeOp op,
                                std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  int64_t channels = op.getInputs().size();
  VMICapabilityResult channelCapability =
      capabilities.supportsChannelCount("pto.vmi.channel_merge", channels);
  if (!channelCapability.isSupported())
    return fail(channelCapability.reason);

  int64_t inputArity = 0;
  for (Value input : op.getInputs()) {
    auto inputType = cast<VMIVRegType>(input.getType());
    VMILayoutAttr inputLayout = inputType.getLayoutAttr();
    if (!inputLayout || !inputLayout.isContiguous())
      return fail("requires every input layout to be contiguous");
    FailureOr<int64_t> arity = getVMIPhysicalArity(inputType);
    if (failed(arity))
      return fail("requires computable input physical arity");
    inputArity += *arity;
  }

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!resultLayout)
    return fail("requires assigned result layout");
  auto expectedLayout =
      VMILayoutAttr::getDeinterleaved(op.getContext(), channels);
  if (!resultLayout.isContiguous() && resultLayout != expectedLayout)
    return fail("requires result layout to be contiguous or matching "
                "deinterleaved channel layout");

  auto channelType =
      VMIVRegType::get(op.getContext(), resultType.getElementCount(),
                       resultType.getElementType(), expectedLayout);
  FailureOr<int64_t> channelArity = getVMIPhysicalArity(channelType);
  if (failed(channelArity) || *channelArity != inputArity)
    return fail("requires channel physical arity to match all input parts");

  std::string materializationReason;
  if (failed(checkSupportedLayoutMaterialization(
          capabilities, channelType, resultType, expectedLayout, resultLayout,
          &materializationReason)))
    return fail(Twine("cannot materialize channel layout to result; ") +
                materializationReason);

  return success();
}

LogicalResult
checkSupportedActivePrefixIndexShape(VMIActivePrefixIndexOp op,
                                     std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!maskLayout || !resultLayout)
    return fail("requires assigned mask and result layouts");
  if (!maskLayout.isContiguous() || !resultLayout.isContiguous())
    return fail("requires contiguous mask and result layouts");

  std::string resultFullReason;
  if (failed(checkFullDataPhysicalChunks(resultType, &resultFullReason)))
    return fail(Twine("requires full result physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                resultFullReason);

  std::string maskFullReason;
  if (failed(checkFullVMIPhysicalChunks(maskType, &maskFullReason)))
    return fail(Twine("requires full mask physical chunks so padding mask "
                      "lanes cannot affect the observable prefix; ") +
                maskFullReason);

  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(maskArity) || failed(resultArity))
    return fail("requires computable mask and result physical arity");
  if (*maskArity != 1 || *resultArity != 1)
    return fail("requires a single physical chunk; multi-chunk prefix needs "
                "cross-chunk carry");

  return success();
}

LogicalResult checkSupportedCompressShape(VMICompressOp op,
                                          std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !maskLayout || !resultLayout)
    return fail("requires assigned source, mask, and result layouts");
  if (!sourceLayout.isContiguous() || !maskLayout.isContiguous() ||
      !resultLayout.isContiguous())
    return fail("requires contiguous source, mask, and result layouts");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(sourceType, &fullChunkReason)))
    return fail(Twine("requires full source physical chunks so padding mask "
                      "lanes cannot be squeezed into the result; ") +
                fullChunkReason);

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(maskArity) || failed(resultArity))
    return fail("requires computable source, mask, and result physical arity");
  if (*sourceArity != 1 || *maskArity != 1 || *resultArity != 1)
    return fail("requires a single physical chunk; multi-chunk compress needs "
                "cross-chunk compaction");

  return success();
}

LogicalResult checkSupportedCompressStoreShape(
    const VMITargetCapabilityRegistry &capabilities, VMICompressStoreOp op,
    std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto valueType = cast<VMIVRegType>(op.getValue().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  VMILayoutAttr valueLayout = valueType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  if (!valueLayout || !maskLayout)
    return fail("requires assigned value and mask layouts");
  if (!valueLayout.isContiguous() || !maskLayout.isContiguous())
    return fail("requires contiguous value and mask layouts");

  VMICapabilityResult destinationCapability =
      capabilities.supportsUBPointerMemory(op.getDestination().getType(),
                                           "destination", "pto.vstur",
                                           "pto.vstur stores only to UB");
  if (!destinationCapability.isSupported())
    return fail(destinationCapability.reason);

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(valueType, &fullChunkReason)))
    return fail(Twine("requires full physical chunks so padding mask lanes "
                      "cannot be squeezed into memory; ") +
                fullChunkReason);

  FailureOr<int64_t> valueArity = getVMIPhysicalArity(valueType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  if (failed(valueArity) || failed(maskArity))
    return fail("requires computable value and mask physical arity");
  if (*valueArity != 1 || *maskArity != 1)
    return fail("requires a single physical chunk; multi-chunk "
                "compress_store needs cross-chunk compaction and SQZN "
                "state planning");

  return success();
}

template <typename OpTy>
LogicalResult
checkSupportedReduceShape(const VMITargetCapabilityRegistry &capabilities,
                          OpTy op, VMIReductionKind kind, bool requiresReassoc,
                          std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  if (requiresReassoc && !op->hasAttr("reassoc"))
    return fail("requires reassoc attr for pair-wise floating-point vcadd");

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto initType = cast<VMIVRegType>(op.getInit().getType());
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr initLayout = initType.getLayoutAttr();
  VMILayoutAttr maskLayout = maskType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !initLayout || !maskLayout || !resultLayout)
    return fail("requires assigned source, init, mask, and result layouts");
  if (!sourceLayout.isContiguous() || !initLayout.isContiguous() ||
      !maskLayout.isContiguous() || !resultLayout.isContiguous())
    return fail("requires contiguous source, init, mask, and result layouts");

  VMICapabilityResult elementCapability =
      capabilities.supportsReductionElementType(kind,
                                                sourceType.getElementType());
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(sourceType, &fullChunkReason)))
    return fail(Twine("requires full source physical chunks so padding lanes "
                      "do not participate in the reduction; ") +
                fullChunkReason);

  FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
  FailureOr<int64_t> initArity = getVMIPhysicalArity(initType);
  FailureOr<int64_t> maskArity = getVMIPhysicalArity(maskType);
  FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
  if (failed(sourceArity) || failed(initArity) || failed(maskArity) ||
      failed(resultArity))
    return fail("requires computable physical arity");
  if (*sourceArity < 1 || *maskArity != *sourceArity)
    return fail("requires source and mask physical arity to match and be "
                "non-empty");
  if (*initArity != 1 || *resultArity != 1)
    return fail("requires one init and result physical chunk");

  return success();
}

template <typename OpTy>
LogicalResult
checkSupportedGroupReduceShape(const VMITargetCapabilityRegistry &capabilities,
                               OpTy op, std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if constexpr (std::is_same_v<OpTy, VMIGroupReduceAddFOp>) {
    if (succeeded(supports.getGroupReduceAddFSupport(capabilities, op, reason)))
      return success();
  } else if constexpr (std::is_same_v<OpTy, VMIGroupReduceMaxFOp>) {
    if (succeeded(supports.getGroupReduceMaxFSupport(capabilities, op, reason)))
      return success();
  } else {
    if (succeeded(supports.getGroupReduceAddISupport(capabilities, op, reason)))
      return success();
  }
  return failure();
}

LogicalResult checkSupportedGroupBroadcastShape(
    const VMITargetCapabilityRegistry &capabilities, VMIGroupBroadcastOp op,
    std::string *reason = nullptr) {
  (void)capabilities;
  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getElementCount() != resultType.getElementCount()) {
    if (reason)
      *reason = "requires source/result shape and element type to match";
    return failure();
  }
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  VMILayoutAttr sourceLayout = sourceType.getLayoutAttr();
  VMILayoutAttr resultLayout = resultType.getLayoutAttr();
  if (!sourceLayout || !resultLayout)
    return fail("requires assigned source/result layouts");
  VMILayoutSupport supports;
  if (succeeded(supports.getGroupBroadcastSupport(capabilities, op, nullptr)))
    return success();
  if (!sourceLayout.isGroupSlots() ||
      sourceLayout.getNumGroups() != op.getNumGroupsAttr().getInt())
    return fail("requires matching num_groups source layout");
  if (resultLayout.isGroupSlots())
    return fail("requires dense result layout");

  if (sourceLayout.getSlots() > 0 && sourceLayout.getSlots() != 8 &&
      sourceLayout.getSlots() != 1)
    return fail("supports only slots=8 or slots=1 group_broadcast source "
                "layouts");

  std::string fullChunkReason;
  if (failed(checkFullDataPhysicalChunks(sourceType, &fullChunkReason)))
    return fail(Twine("requires full source physical chunks; ") +
                fullChunkReason);
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
  FailureOr<int64_t> groupSize = getGroupSizeFromNumGroups(
      sourceType, op.getNumGroupsAttr().getInt(), reason);
  if (failed(groupSize))
    return failure();
  if (*lanesPerPart % *groupSize != 0 && *groupSize % *lanesPerPart != 0)
    return fail("requires derived group size to divide or be a multiple of "
                "physical lanes per part");

  FailureOr<int64_t> resultFactor = getDataLayoutFactor(resultType);
  if (failed(resultFactor))
    return fail("requires known result layout factor");
  if (*resultFactor == 1)
    return success();
  bool blockFragmentSmallGroup =
      resultLayout.isDeinterleaved() && resultLayout.getBlockElems() > 1 &&
      *groupSize < *lanesPerPart &&
      *lanesPerPart % resultLayout.getBlockElems() == 0;
  if (blockFragmentSmallGroup)
    return success();
  int64_t logicalSpanPerResultChunk = *lanesPerPart * *resultFactor;
  if (*groupSize < *lanesPerPart || *groupSize % logicalSpanPerResultChunk != 0)
    return fail("deinterleaved result requires every physical result chunk to "
                "stay within one logical group");
  return success();
}

LogicalResult checkSupportedDhistShape(VMIDhistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getDhistSupport(op, reason)))
    return success();
  return failure();
}

LogicalResult checkSupportedChistShape(VMIChistOp op,
                                       std::string *reason = nullptr) {
  VMILayoutSupport supports;
  if (succeeded(supports.getChistSupport(op, reason)))
    return success();
  return failure();
}

LogicalResult
checkSupportedFmaShape(const VMITargetCapabilityRegistry &capabilities,
                       VMIFmaOp op, std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto lhsType = cast<VMIVRegType>(op.getLhs().getType());
  VMICapabilityResult elementCapability = capabilities.supportsElementType(
      lhsType.getElementType(), VMIElementPurpose::VMula);
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);

  FailureOr<int64_t> arity = getVMIPhysicalArity(lhsType);
  if (failed(arity) || *arity < 1)
    return fail("requires computable non-empty physical arity");

  return success();
}

LogicalResult
checkSupportedReluShape(const VMITargetCapabilityRegistry &capabilities,
                        VMIReluOp op, std::string *reason = nullptr) {
  auto fail = [&](const Twine &message) -> LogicalResult {
    if (reason)
      *reason = message.str();
    return failure();
  };

  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (failed(checkSupportedMaskableVReg(capabilities, resultType, reason)))
    return failure();

  VMICapabilityResult elementCapability = capabilities.supportsElementType(
      resultType.getElementType(), VMIElementPurpose::VRelu);
  if (!elementCapability.isSupported())
    return fail(elementCapability.reason);

  return success();
}

void emitEnsureLayoutMaterializationError(VMIEnsureLayoutOp ensure,
                                          VMIVRegType sourceType,
                                          VMIVRegType resultType,
                                          StringRef reason) {
  if (ensure.getResult().hasOneUse()) {
    OpOperand &use = *ensure.getResult().use_begin();
    Operation *requester = use.getOwner();
    InFlightDiagnostic diag =
        requester->emitError()
        << kVMIDiagUnsupportedPrefix << requester->getName() << " operand #"
        << use.getOperandNumber() << " has type " << sourceType
        << " but requires " << resultType
        << "; pto.vmi.ensure_layout cannot materialize this conversion";
    diag.attachNote(ensure.getLoc())
        << "failed helper conversion " << sourceType << " -> " << resultType
        << " (" << reason
        << "); partial/tail layout materialization requires an explicit "
           "packing plan";
    return;
  }

  ensure.emitError()
      << kVMIDiagUnsupportedPrefix
      << "pto.vmi.ensure_layout cannot materialize the requested data "
         "layout conversion ("
      << reason
      << "); partial/tail layout materialization requires an explicit "
         "packing plan";
}

LogicalResult
verifySupportedVMIToVPTOOps(ModuleOp module,
                            const VMITargetCapabilityRegistry &capabilities,
                            bool enableStableGatherMaskedLoad) {
  auto emitMemoryUnsupported =
      [&](Operation *op, StringRef opName, VMIVRegType type, Value source,
          std::optional<int64_t> constantOffset) -> WalkResult {
    std::string reason;
    if (succeeded(checkSupportedLoadShape(capabilities, type, source,
                                          source.getType(), constantOffset,
                                          &reason)))
      return WalkResult::advance();

    op->emitError()
        << kVMIDiagUnsupportedPrefix << opName
        << " requires full physical chunks without padding lanes or a "
           "statically safe full-read footprint ("
        << reason << ")";
    return WalkResult::interrupt();
  };

  auto emitMaskableUnsupported = [&](Operation *op, StringRef opName,
                                     VMIVRegType type) -> WalkResult {
    std::string reason;
    if (succeeded(checkSupportedMaskableVReg(capabilities, type, &reason)))
      return WalkResult::advance();

    op->emitError()
        << kVMIDiagUnsupportedPrefix << opName
        << " direct lowering requires physical vreg parts with b8/b16/b32 "
           "predicate masks ("
        << reason << ")";
    return WalkResult::interrupt();
  };

  auto emitTargetElementUnsupported =
      [&](Operation *op, StringRef opName, VMIVRegType type,
          VMIElementPurpose purpose, StringRef elementContract) -> WalkResult {
    std::string reason;
    if (succeeded(checkSupportedTargetElementVReg(capabilities, type, purpose,
                                                  elementContract, &reason)))
      return WalkResult::advance();

    op->emitError()
        << kVMIDiagUnsupportedPrefix << opName << " direct lowering requires "
        << elementContract
        << " and physical vreg parts with b8/b16/b32 predicate masks ("
        << reason << ")";
    return WalkResult::interrupt();
  };

  WalkResult result = module.walk([&](Operation *op) {
    if (auto constant = dyn_cast<VMIConstantOp>(op)) {
      auto denseAttr = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!denseAttr || !denseAttr.isSplat()) {
        constant.emitError()
            << kVMIDiagUnsupportedPrefix
            << "non-splat pto.vmi.constant requires a vreg immediate or "
               "scratch materialization plan";
        return WalkResult::interrupt();
      }
      return emitMaskableUnsupported(
          op, "pto.vmi.constant",
          cast<VMIVRegType>(constant.getResult().getType()));
    }

    if (auto broadcast = dyn_cast<VMIBroadcastOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.broadcast",
          cast<VMIVRegType>(broadcast.getResult().getType()));
    if (auto broadcast = dyn_cast<VMIGroupBroadcastOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupBroadcastShape(capabilities, broadcast,
                                                      &reason)))
        return WalkResult::advance();
      broadcast.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_broadcast requires full source chunks with "
             "#pto.vmi.layout<num_groups = G, slots = K>, a dense full result "
             "layout, "
             "and num_groups deriving a group size that divides or is a "
             "multiple of physical chunk lanes ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto hist = dyn_cast<VMIDhistOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedDhistShape(hist, &reason)))
        return WalkResult::advance();
      hist.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.dhist requires contiguous Nxui8 source, contiguous b8 "
             "mask, and contiguous 256xui16 acc/result ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto hist = dyn_cast<VMIChistOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedChistShape(hist, &reason)))
        return WalkResult::advance();
      hist.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.chist requires a verified CHISTv2 range semantics "
             "contract before lowering ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto load = dyn_cast<VMILoadOp>(op)) {
      return emitMemoryUnsupported(
          op, "pto.vmi.load", cast<VMIVRegType>(load.getResult().getType()),
          load.getSource(), getConstantIndexValue(load.getOffset()));
    }
    if (auto load = dyn_cast<VMIGroupLoadOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGroupLoadShape(capabilities, load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_load requires contiguous full result chunks, a "
             "supported UB source, and num_groups deriving a group size "
             "aligned to physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIGroupSlotLoadOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupSlotLoadShape(capabilities, load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_slot_load requires explicit group_slots result "
             "layout matching num_groups, a supported UB pointer source, "
             "and either slots=8 with constant unit source_group_stride or "
             "slots=1 row-local lowering ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIMaskedLoadOp>(op)) {
      if (enableStableGatherMaskedLoad) {
        load.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.masked_load stable VGATHER-based lowering is reserved "
               "for strict masked/tail loads but is not implemented yet";
        return WalkResult::interrupt();
      }
      std::string reason;
      if (succeeded(checkSupportedMaskedLoadShape(capabilities, load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.masked_load direct lowering requires a supported memory "
             "source, contiguous result/passthru/mask layouts, and either "
             "full physical chunks or a statically safe full-read footprint ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto gather = dyn_cast<VMIGatherOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedGatherShape(capabilities, gather, &reason)))
        return WalkResult::advance();
      gather.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.gather lowers through pto.vgather2_bc + pto.vsel only "
             "for UB pointer sources, contiguous full physical chunks, "
             "32-bit result elements, i32 indices, and b32 masks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto load = dyn_cast<VMIExpandLoadOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExpandLoadShape(capabilities, load, &reason)))
        return WalkResult::advance();
      load.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.expand_load direct lowering is currently supported for "
             "either a static all-active mask lowered as pto.vlds, or a "
             "one-full-chunk 32-bit UB runtime mask lowered through pto.vusqz "
             "+ pto.vgather2_bc + pto.vsel ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIStoreOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedStoreShape(
              capabilities, cast<VMIVRegType>(store.getValue().getType()),
              store.getDestination(), store.getDestination().getType(),
              &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.store requires an 8/16/32-bit predicate-maskable "
             "element type and either full physical chunks or contiguous "
             "tail-store layout, with UB-backed destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIGroupStoreOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupStoreShape(capabilities, store, &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_store requires contiguous full value chunks, a "
             "supported UB destination, and num_groups deriving a group size "
             "aligned to physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto store = dyn_cast<VMIMaskedStoreOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedMaskedStoreShape(
              capabilities, cast<VMIVRegType>(store.getValue().getType()),
              cast<VMIMaskType>(store.getMask().getType()),
              store.getDestination(), store.getDestination().getType(),
              &reason)))
        return WalkResult::advance();
      store.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.masked_store requires either full physical chunks or "
             "contiguous tail-store value/mask layout, with UB-backed "
             "destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto scatter = dyn_cast<VMIScatterOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedScatterShape(capabilities, scatter, &reason)))
        return WalkResult::advance();
      scatter.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.scatter lowers through pto.vscatter only with a UB "
             "pointer destination, contiguous full physical chunks, 32-bit "
             "value elements, i32 indices, and b32 masks ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto tileRead = dyn_cast<VMITileReadOp>(op))
      return emitMemoryUnsupported(
          op, "pto.vmi.tile_read",
          cast<VMIVRegType>(tileRead.getResult().getType()),
          tileRead.getSource(), 0);
    if (auto tileWrite = dyn_cast<VMITileWriteOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedStoreShape(
              capabilities, cast<VMIVRegType>(tileWrite.getValue().getType()),
              tileWrite.getDestination(), tileWrite.getDestination().getType(),
              &reason)))
        return WalkResult::advance();
      tileWrite.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.tile_write requires an 8/16/32-bit predicate-maskable "
             "element type and either full physical chunks or contiguous "
             "tail-store layout, with UB-backed destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto ensure = dyn_cast<VMIEnsureLayoutOp>(op)) {
      auto sourceType = cast<VMIVRegType>(ensure.getSource().getType());
      auto resultType = cast<VMIVRegType>(ensure.getResult().getType());
      std::string reason;
      if (succeeded(checkSupportedLayoutMaterialization(
              capabilities, sourceType, resultType, sourceType.getLayoutAttr(),
              resultType.getLayoutAttr(), &reason)))
        return WalkResult::advance();

      emitEnsureLayoutMaterializationError(ensure, sourceType, resultType,
                                           reason);
      return WalkResult::interrupt();
    }

    if (auto ensure = dyn_cast<VMIEnsureMaskLayoutOp>(op)) {
      auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
      auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
      std::string reason;
      if (succeeded(checkSupportedLayoutMaterialization(
              capabilities, sourceType, resultType, sourceType.getLayoutAttr(),
              resultType.getLayoutAttr(), &reason)))
        return WalkResult::advance();

      ensure.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.ensure_mask_layout cannot materialize the requested "
             "mask layout conversion ("
          << reason
          << "); partial/tail predicate layout materialization requires an "
             "explicit packing plan";
      return WalkResult::interrupt();
    }

    if (auto ensure = dyn_cast<VMIEnsureMaskGranularityOp>(op)) {
      auto sourceType = cast<VMIMaskType>(ensure.getSource().getType());
      auto resultType = cast<VMIMaskType>(ensure.getResult().getType());
      if (sourceType.getGranularity() == resultType.getGranularity())
        return WalkResult::advance();

      std::string reason;
      if (succeeded(checkSupportedMaskGranularityMaterialization(
              capabilities, sourceType, resultType, &reason)))
        return WalkResult::advance();

      ensure.emitError()
          << kVMIDiagUnsupportedPrefix
          << "non-identity mask granularity materialization requires concrete "
             "b8/b16/b32 masks with matching lane count and layout ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto addf = dyn_cast<VMIAddFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.addf", cast<VMIVRegType>(addf.getResult().getType()),
          VMIElementPurpose::F16BF16F32, "f16/bf16/f32 element type");
    if (auto addi = dyn_cast<VMIAddIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.addi", cast<VMIVRegType>(addi.getResult().getType()));
    if (auto subf = dyn_cast<VMISubFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.subf", cast<VMIVRegType>(subf.getResult().getType()),
          VMIElementPurpose::F16BF16F32, "f16/bf16/f32 element type");
    if (auto subi = dyn_cast<VMISubIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.subi", cast<VMIVRegType>(subi.getResult().getType()));
    if (auto mulf = dyn_cast<VMIMulFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.mulf", cast<VMIVRegType>(mulf.getResult().getType()),
          VMIElementPurpose::F16BF16F32, "f16/bf16/f32 element type");
    if (auto muli = dyn_cast<VMIMulIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.muli", cast<VMIVRegType>(muli.getResult().getType()));
    if (auto divf = dyn_cast<VMIDivFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.divf", cast<VMIVRegType>(divf.getResult().getType()),
          VMIElementPurpose::F16F32, "f16/f32 element type");
    if (auto minf = dyn_cast<VMIMinFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.minf", cast<VMIVRegType>(minf.getResult().getType()),
          VMIElementPurpose::F16BF16F32, "f16/bf16/f32 element type");
    if (auto maxf = dyn_cast<VMIMaxFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.maxf", cast<VMIVRegType>(maxf.getResult().getType()),
          VMIElementPurpose::F16BF16F32, "f16/bf16/f32 element type");
    if (auto negf = dyn_cast<VMINegFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.negf", cast<VMIVRegType>(negf.getResult().getType()),
          VMIElementPurpose::F16F32, "f16/f32 element type");
    if (auto absf = dyn_cast<VMIAbsFOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.absf", cast<VMIVRegType>(absf.getResult().getType()),
          VMIElementPurpose::F16F32, "f16/f32 element type");
    if (auto absi = dyn_cast<VMIAbsIOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.absi", cast<VMIVRegType>(absi.getResult().getType()),
          VMIElementPurpose::SignlessOrSignedI8I16I32,
          "signless/signed i8/i16/i32 element type");
    if (auto sqrt = dyn_cast<VMISqrtOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.sqrt", cast<VMIVRegType>(sqrt.getResult().getType()),
          VMIElementPurpose::F16F32, "f16/f32 element type");
    if (auto exp = dyn_cast<VMIExpOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.exp", cast<VMIVRegType>(exp.getResult().getType()),
          VMIElementPurpose::F16F32, "f16/f32 element type");
    if (auto ln = dyn_cast<VMILnOp>(op))
      return emitTargetElementUnsupported(
          op, "pto.vmi.ln", cast<VMIVRegType>(ln.getResult().getType()),
          VMIElementPurpose::F16F32, "f16/f32 element type");
    if (auto relu = dyn_cast<VMIReluOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReluShape(capabilities, relu, &reason)))
        return WalkResult::advance();
      relu.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.relu direct lowering requires physical vreg parts with "
             "b8/b16/b32 predicate masks and f16/f32 element type ("
          << reason << ")";
      return WalkResult::interrupt();
    }
    if (auto andi = dyn_cast<VMIAndIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.andi", cast<VMIVRegType>(andi.getResult().getType()));
    if (auto ori = dyn_cast<VMIOrIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.ori", cast<VMIVRegType>(ori.getResult().getType()));
    if (auto xori = dyn_cast<VMIXOrIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.xori", cast<VMIVRegType>(xori.getResult().getType()));
    if (auto shli = dyn_cast<VMIShLIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.shli", cast<VMIVRegType>(shli.getResult().getType()));
    if (auto shrui = dyn_cast<VMIShRUIOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.shrui", cast<VMIVRegType>(shrui.getResult().getType()));
    if (auto notOp = dyn_cast<VMINotOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.not", cast<VMIVRegType>(notOp.getResult().getType()));
    if (auto select = dyn_cast<VMISelectOp>(op))
      return emitMaskableUnsupported(
          op, "pto.vmi.select",
          cast<VMIVRegType>(select.getResult().getType()));

    if (auto cmpf = dyn_cast<VMICmpFOp>(op)) {
      WalkResult target = emitTargetElementUnsupported(
          op, "pto.vmi.cmpf", cast<VMIVRegType>(cmpf.getLhs().getType()),
          VMIElementPurpose::F16BF16F32, "f16/bf16/f32 element type");
      if (target.wasInterrupted())
        return target;
      if (succeeded(checkSupportedComparePredicate(op, cmpf.getPredicate())))
        return WalkResult::advance();
      return WalkResult::interrupt();
    }

    if (auto cmpi = dyn_cast<VMICmpIOp>(op)) {
      WalkResult target = emitTargetElementUnsupported(
          op, "pto.vmi.cmpi", cast<VMIVRegType>(cmpi.getLhs().getType()),
          VMIElementPurpose::AnyI8I16I32,
          "signless/signed/unsigned i8/i16/i32 element type");
      if (target.wasInterrupted())
        return target;
      if (succeeded(checkSupportedComparePredicate(op, cmpi.getPredicate())))
        return WalkResult::advance();
      return WalkResult::interrupt();
    }

    if (auto activePrefix = dyn_cast<VMIActivePrefixIndexOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedActivePrefixIndexShape(activePrefix, &reason)))
        return WalkResult::advance();
      activePrefix.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.active_prefix_index lowers through pto.vusqz only for "
             "one contiguous physical chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto compress = dyn_cast<VMICompressOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedCompressShape(compress, &reason)))
        return WalkResult::advance();
      compress.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.compress lowers through pto.vsqz only for one "
             "contiguous full physical chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto compressStore = dyn_cast<VMICompressStoreOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedCompressStoreShape(capabilities,
                                                     compressStore, &reason)))
        return WalkResult::advance();
      compressStore.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.compress_store lowers through pto.vsqz + pto.vstur "
             "only for one contiguous full physical chunk with a UB pointer "
             "destination ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceAddIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              capabilities, reduce, VMIReductionKind::AddI,
              /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_addi lowers through pto.vcadd only for "
             "contiguous full 32-bit integer source chunks with matching "
             "mask chunks and one init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceAddFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              capabilities, reduce, VMIReductionKind::AddF,
              /*requiresReassoc=*/true, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_addf lowers through pto.vcadd only with "
             "reassoc, f32 contiguous full source chunks, matching mask "
             "chunks, and one init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceAddFOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(capabilities, reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_addf lowers through pto.vcgadd for 32B "
             "VLane groups or through pto.vcadd with reassoc, contiguous full "
             "source/mask chunks, #pto.vmi.layout<num_groups = G, slots = K> "
             "result "
             "chunks, and num_groups deriving a group size aligned to "
             "physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceAddIOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(capabilities, reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_addi lowers through pto.vcgadd/vadd only "
             "for i32 accumulator values; i8/i16 storage must be cast to i32 "
             "before grouped reduction because narrow integer reductions "
             "widen their result ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIGroupReduceMaxFOp>(op)) {
      std::string reason;
      if (succeeded(
              checkSupportedGroupReduceShape(capabilities, reduce, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.group_reduce_maxf lowers through pto.vcgmax/vmax only "
             "for f16/f32 values, matching source/mask chunks, "
             "#pto.vmi.layout<num_groups = G, slots = K> result chunks, and "
             "num_groups deriving a group size aligned to physical chunks ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceMaxFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              capabilities, reduce, VMIReductionKind::MaxF,
              /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_maxf lowers through pto.vcmax only for f16/f32 "
             "contiguous full source chunks with matching mask chunks and one "
             "init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto reduce = dyn_cast<VMIReduceMinFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedReduceShape(
              capabilities, reduce, VMIReductionKind::MinF,
              /*requiresReassoc=*/false, &reason)))
        return WalkResult::advance();
      reduce.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.reduce_minf lowers through pto.vcmin only for f16/f32 "
             "contiguous full source chunks with matching mask chunks and one "
             "init/result chunk ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto fma = dyn_cast<VMIFmaOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedFmaShape(capabilities, fma, &reason)))
        return WalkResult::advance();
      fma.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.fma lowers through pto.vmula only for f16/bf16/f32 "
             "element types ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto extf = dyn_cast<VMIExtFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExtFShape(extf, &reason)))
        return WalkResult::advance();

      extf.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.extf supports contiguous 16-bit float-like or fp8-like "
             "physical source chunks to f32 deinterleaved=2/4 results; "
             "partial/tail is allowed only when source padding maps to result "
             "padding ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto truncf = dyn_cast<VMITruncFOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedTruncFShape(truncf, &reason)))
        return WalkResult::advance();

      truncf.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.truncf supports only f32 deinterleaved=2 source parts "
             "to one contiguous f16 result chunk or f32 deinterleaved=4 "
             "source parts to one contiguous fp8-like result chunk, or f32 "
             "group_slots(num_groups=G, slots=1) to f16 "
             "group_slots(num_groups=G, slots=1) ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto extsi = dyn_cast<VMIExtSIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExtSIShape(extsi, &reason)))
        return WalkResult::advance();

      extsi.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.extsi supports contiguous signed/signless 8-bit or "
             "16-bit integer physical source chunks to 32-bit integer "
             "deinterleaved=4/2 results ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto extui = dyn_cast<VMIExtUIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedExtUIShape(extui, &reason)))
        return WalkResult::advance();

      extui.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.extui supports contiguous unsigned 8-bit or 16-bit "
             "integer physical source chunks to unsigned 32-bit integer "
             "deinterleaved=4/2 results ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto trunci = dyn_cast<VMITruncIOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedTruncIShape(trunci, &reason)))
        return WalkResult::advance();

      trunci.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.trunci supports only 32-bit integer deinterleaved=2 "
             "source parts to one contiguous 16-bit integer result chunk, "
             "32-bit integer deinterleaved=4 source parts to one contiguous "
             "8-bit integer result chunk, or 32-bit integer "
             "group_slots(num_groups=G, slots=1) to 16-bit integer "
             "group_slots(num_groups=G, slots=1) ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto bitcast = dyn_cast<VMIBitcastOp>(op)) {
      std::string reason;
      if (succeeded(checkSupportedBitcastShape(bitcast, &reason)))
        return WalkResult::advance();

      bitcast.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.bitcast requires matching source/result layouts with "
             "identical physical arity and matching per-chunk logical bit "
             "footprints ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto split = dyn_cast<VMIChannelSplitOp>(op)) {
      int64_t channels = split.getNumResults();
      std::string reason;
      if (succeeded(
              checkSupportedChannelSplitShape(capabilities, split, &reason)))
        return WalkResult::advance();

      if (channels != 2 && channels != 4)
        split.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_split supports only 2 or 4 channels";
      else
        split.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_split requires source layout to be contiguous "
               "or matching deinterleaved channel layout, every result layout "
               "to be contiguous, and complete physical channel groups ("
            << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto merge = dyn_cast<VMIChannelMergeOp>(op)) {
      int64_t channels = merge.getInputs().size();
      std::string reason;
      if (succeeded(
              checkSupportedChannelMergeShape(capabilities, merge, &reason)))
        return WalkResult::advance();

      if (channels != 2 && channels != 4)
        merge.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_merge supports only 2 or 4 channels";
      else
        merge.emitError()
            << kVMIDiagUnsupportedPrefix
            << "pto.vmi.channel_merge requires every input layout to be "
               "contiguous and result layout to be contiguous or matching "
               "deinterleaved channel layout, with complete physical channel "
               "groups ("
            << reason << ")";
      return WalkResult::interrupt();
    }

    if (auto shuffle = dyn_cast<VMIShuffleOp>(op)) {
      std::string reason;
      if (succeeded(computeShuffleForwardingSourceParts(shuffle, &reason)))
        return WalkResult::advance();
      std::string splatReason;
      if (succeeded(computeShuffleLane0SplatSourcePart(shuffle, &splatReason)))
        return WalkResult::advance();
      std::string vselrReason;
      if (succeeded(computeShuffleVselrPlans(shuffle, &vselrReason)))
        return WalkResult::advance();

      shuffle.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.shuffle requires physical chunk forwarding or "
             "lane0 splat or vci-materializable vselr indices (forwarding: "
          << reason << "; lane0 splat: " << splatReason
          << "; vselr: " << vselrReason << ")";
      return WalkResult::interrupt();
    }

    if (auto constantMask = dyn_cast<VMIConstantMaskOp>(op)) {
      std::string reason;
      if (succeeded(computeConstantMaskMaterialization(constantMask, &reason)))
        return WalkResult::advance();

      constantMask.emitError()
          << kVMIDiagUnsupportedPrefix
          << "pto.vmi.constant_mask requires a dense bool constant with "
             "concrete layout and b8/b16/b32 granularity ("
          << reason << ")";
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

struct VMIToVPTOPass : public mlir::pto::impl::VMIToVPTOBase<VMIToVPTOPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIToVPTOPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyVMIToVPTOInputIR(module))) {
      signalPassFailure();
      return;
    }
    VMITargetCapabilityRegistry capabilities;
    if (failed(verifySupportedVMIToVPTOOps(module, capabilities,
                                           enableStableGatherMaskedLoad))) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    VMIToVPTOTypeConverter typeConverter;
    RewritePatternSet patterns(context);

    populateVMIOneToNConversionPatterns(typeConverter, patterns, capabilities);
    if (failed(applyPartialOneToNConversion(module, typeConverter,
                                            std::move(patterns)))) {
      module.emitError() << kVMIDiagResidualOpPrefix
                         << "failed to convert all VMI ops/types to VPTO";
      signalPassFailure();
      return;
    }
    if (failed(verifyNoResidualVMIIR(module))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIToVPTOPass() {
  return std::make_unique<VMIToVPTOPass>();
}
