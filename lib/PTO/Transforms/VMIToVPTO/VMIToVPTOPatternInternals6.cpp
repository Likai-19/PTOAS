// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals6.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

template <typename OpT>
struct OneToNVMIExtIOpPattern : OneToNOpConversionPattern<OpT> {
  using OneToNOpConversionPattern<OpT>::OneToNOpConversionPattern;

private:
  FailureOr<Value> buildFactorExtensionResult(
      OpT op, Value sourcePart, VRegType resultType, Value mask,
      StringRef part, OneToNPatternRewriter &rewriter) const {
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                        /*rnd=*/nullptr, /*sat=*/nullptr,
                        rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult emitFactorExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes,
      ArrayRef<StringRef> parts, int64_t liveParts, int64_t partStride,
      Value mask, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultVRegTypes.size());
    for (int64_t partIndex = 0; partIndex < liveParts; ++partIndex) {
      for (auto [chunkIndex, sourcePart] : llvm::enumerate(sourceParts)) {
        VRegType resultType =
            resultVRegTypes[partIndex * sourceParts.size() + chunkIndex];
        FailureOr<Value> result = buildFactorExtensionResult(
            op, sourcePart, resultType, mask,
            parts[partIndex * partStride], rewriter);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildDenseLaneExtensionResult(
      OpT op, Value sourcePart, VRegType resultType, Value mask,
      StringRef part, OneToNPatternRewriter &rewriter) const {
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, mask,
                        /*rnd=*/nullptr, /*sat=*/nullptr,
                        rewriter.getStringAttr(part))
        .getResult();
  }

  LogicalResult emitDenseLaneExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultTypes,
      VRegType sourceType, StringRef part,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildDenseLaneExtensionResult(
          op, sourcePart, resultType, *mask, part, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> getExtensionPartPlan(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      unsigned sourceBits, unsigned resultBits, int64_t sourceStride,
      OneToNPatternRewriter &rewriter) const {
    constexpr int64_t kEvenOddFactor = 2;
    constexpr int64_t kPacked4Factor = 4;
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    if (resultBits == sourceBits * kEvenOddFactor) {
      static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
      parts = kEvenOddParts;
      factor = kEvenOddFactor;
    } else if (resultBits == sourceBits * kPacked4Factor) {
      static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
      parts = kPacked4Parts;
      factor = kPacked4Factor;
    } else {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical integer extension source/result width "
              "relation");
    }
    if (sourceStride <= 0) {
      sourceStride = 1;
    }
    // The source carries one semantic element every `sourceStride` lanes, so
    // only every `sourceStride`-th conversion part is live; the rest are holes
    // that the result's deinterleave factor already names.  A unit source
    // stride keeps the full part set.
    if (factor % sourceStride != 0) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical integer extension source lane stride");
    }
    int64_t liveParts = factor / sourceStride;
    bool arityMismatch = resultTypes.size() !=
                         static_cast<size_t>(liveParts) * sourceParts.size();
    if (arityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical integer extension source/result arity "
              "relation");
    }
    return std::make_pair(parts, liveParts);
  }

  FailureOr<VRegType> validateDenseGroupSlotExtensionResult(
      OpT op, Type resultType, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    if (sourceBits == 0 || resultBits == 0) {
      return rewriter.notifyMatchFailure(
          op, "group-slot integer extension requires positive bit widths");
    }
    FailureOr<int64_t> sourceLanes =
        getDataLanesPerPart(sourceVMIType.getElementType());
    FailureOr<int64_t> resultLanes =
        getDataLanesPerPart(resultVMIType.getElementType());
    bool carrierShapeMismatch =
        failed(sourceLanes) || failed(resultLanes) ||
        *sourceLanes !=
            *resultLanes * static_cast<int64_t>(resultBits / sourceBits);
    auto physicalResultType = dyn_cast<VRegType>(resultType);
    bool invalidResultType =
        !physicalResultType || failed(resultLanes) ||
        physicalResultType.getElementCount() != *resultLanes ||
        pto::getPTOStorageElemBitWidth(physicalResultType.getElementType()) !=
            resultBits;
    if (carrierShapeMismatch || invalidResultType) {
      return rewriter.notifyMatchFailure(
          op, carrierShapeMismatch
                  ? "unsupported dense group-slot integer extension carrier shape"
                  : "unsupported dense group-slot integer extension result type");
    }
    return physicalResultType;
  }

  FailureOr<Value> buildDenseGroupSlotExtensionResult(
      OpT op, Value sourcePart, Type resultType, VMIVRegType sourceVMIType,
      VMIVRegType resultVMIType, IntegerType resultIntegerType,
      unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<VRegType> physicalResultType =
        validateDenseGroupSlotExtensionResult(
            op, resultType, sourceVMIType, resultVMIType, sourceBits,
            resultBits, rewriter);
    if (failed(physicalResultType)) {
      return failure();
    }

    Value current = sourcePart;
    unsigned currentBits = sourceBits;
    while (currentBits < resultBits) {
      FailureOr<Value> next = extendDenseGroupSlotCarrier(
          op, current, currentBits, resultIntegerType, rewriter);
      if (failed(next)) {
        return failure();
      }
      current = *next;
      currentBits *= mlir::pto::kValue2;
    }
    FailureOr<Value> result =
        bitcastVReg(op.getLoc(), current, *physicalResultType, rewriter);
    if (failed(result)) {
      return rewriter.notifyMatchFailure(
          op, "failed to materialize dense group-slot unpack result");
    }
    return *result;
  }

  FailureOr<Value> extendDenseGroupSlotCarrier(
      OpT op, Value current, unsigned currentBits,
      IntegerType resultIntegerType,
      OneToNPatternRewriter &rewriter) const {
    unsigned nextBits = currentBits * 2;
    auto nextElementType = IntegerType::get(
        rewriter.getContext(), nextBits, resultIntegerType.getSignedness());
    FailureOr<int64_t> nextLanes = getDataLanesPerPart(nextElementType);
    if (failed(nextLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive dense group-slot unpack result lanes");
    }
    auto nextType =
        VRegType::get(rewriter.getContext(), *nextLanes, nextElementType);
    auto currentType = dyn_cast<VRegType>(current.getType());
    bool unpackLaneMismatch =
        !currentType || currentType.getElementCount() != *nextLanes * 2;
    if (unpackLaneMismatch) {
      return rewriter.notifyMatchFailure(
          op, "dense group-slot unpack source/result lane mismatch");
    }
    Value part = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    if constexpr (std::is_same_v<OpT, VMIExtSIOp>) {
      return rewriter.create<VsunpackOp>(op.getLoc(), nextType, current, part)
          .getResult();
    }
    return rewriter.create<VzunpackOp>(op.getLoc(), nextType, current, part)
        .getResult();
  }

  LogicalResult lowerDenseGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      IntegerType resultIntegerType, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildDenseGroupSlotExtensionResult(
          op, sourcePart, resultType, sourceVMIType, resultVMIType,
          resultIntegerType, sourceBits, resultBits, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  FailureOr<Value> buildLegacyGroupSlotExtensionResult(
      OpT op, Value sourcePart, Type resultType, VRegType conversionSourceType,
      Value slotMask, StringAttr part, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    bool invalidResultType =
        !resultVRegType ||
        pto::getPTOStorageElemBitWidth(resultVRegType.getElementType()) !=
            resultBits;
    if (invalidResultType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot integer extension result type");
    }
    FailureOr<Value> conversionSource = bitcastVReg(
        op.getLoc(), sourcePart, conversionSourceType, rewriter);
    if (failed(conversionSource)) {
      return rewriter.notifyMatchFailure(
          op, "failed to expose group-slot extension source elements");
    }
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultVRegType, *conversionSource, slotMask,
                        /*rnd=*/nullptr, /*sat=*/nullptr, part)
        .getResult();
  }

  FailureOr<std::tuple<int64_t, VRegType, Value>> prepareLegacyGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        sourceBits == 0 || sourceBits >= resultBits ||
        resultBits % sourceBits != 0 ||
        (resultBits / sourceBits != 2 && resultBits / sourceBits != 4) ||
        (sourceLayout.getSlots() == 8 &&
         sourceLayout.getLaneStride() != resultBits / sourceBits) ||
        resultLayout.getLaneStride() != 1 ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot integer extension shape");
    }
    FailureOr<int64_t> sourceLanes =
        getDataLanesPerPart(sourceVMIType.getElementType());
    if (failed(sourceLanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group-slot integer extension source lanes");
    }
    auto conversionSourceType = VRegType::get(
        rewriter.getContext(), *sourceLanes, sourceVMIType.getElementType());
    FailureOr<MaskType> maskType =
        getMaskTypeForVReg(conversionSourceType, rewriter.getContext());
    if (failed(maskType)) {
      return rewriter.notifyMatchFailure(
          op, "failed to create group-slot integer extension mask type");
    }
    FailureOr<Value> slotMask = createPrefixMaskForActiveLanes(
        op.getLoc(), *maskType,
        sourceLayout.getSlots() * sourceLayout.getLaneStride(), rewriter);
    if (failed(slotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot integer extension mask");
    }
    return std::make_tuple(static_cast<int64_t>(resultBits / sourceBits), conversionSourceType,
                           *slotMask);
  }

  LogicalResult lowerLegacyGroupSlotExtension(
      OpT op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType sourceVMIType, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, unsigned sourceBits, unsigned resultBits,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<std::tuple<int64_t, VRegType, Value>> preparation =
        prepareLegacyGroupSlotExtension(
            op, sourceParts, resultTypes, sourceVMIType, sourceLayout,
            resultLayout, sourceBits, resultBits, rewriter);
    if (failed(preparation)) {
      return failure();
    }
    int64_t widenFactor = std::get<0>(*preparation);
    VRegType conversionSourceType = std::get<1>(*preparation);
    Value slotMask = std::get<2>(*preparation);
    StringAttr part =
        rewriter.getStringAttr(widenFactor == 2 ? "EVEN" : "P0");
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result = buildLegacyGroupSlotExtensionResult(
          op, sourcePart, resultType, conversionSourceType, slotMask, part,
          resultBits, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  LogicalResult lowerPhysicalExtension(
      OpT op, ValueRange sourceParts, ArrayRef<VRegType> resultVRegTypes,
      ArrayRef<Type> resultTypes, VRegType sourceType, unsigned sourceBits,
      unsigned resultBits, VMILayoutAttr sourceLayout,
      VMILayoutAttr resultLayout, OneToNPatternRewriter &rewriter) const {
    // The carrier view of each declared layout drives the plan, exactly as in
    // pto.vmi.extf: a single-carrier group-slot packet whose stride is the
    // widening factor, or a packet that keeps one group at lane 0 of every
    // part, selects the same source lanes as the dense lane-strided form and
    // forwards the physical part unchanged.
    VMILayoutAttr sourceView =
        getVMICastDenseCarrierView(sourceLayout, sourceType.getElementType());
    VMILayoutAttr resultView =
        resultVRegTypes.empty()
            ? VMILayoutAttr()
            : getVMICastDenseCarrierView(
                  resultLayout, resultVRegTypes.front().getElementType());
    unsigned pairWidth = static_cast<unsigned>(kPairWidth);
    unsigned quadWidth = static_cast<unsigned>(kQuadWidth);
    bool evenRadix = resultBits == sourceBits * pairWidth &&
                     sourceView.getLaneStride() == kPairWidth;
    bool quadRadix = resultBits == sourceBits * quadWidth &&
                     sourceView.getLaneStride() == kQuadWidth;
    bool sourceSelectsPartLanes =
        sourceView &&
        (evenRadix || quadRadix ||
         isVMISingleGroupPerPartPacket(sourceLayout));
    bool denseLaneExtension =
        sourceView && resultView && sourceView.isContiguous() &&
        resultView.isContiguous() && resultView.getLaneStride() == 1 &&
        sourceSelectsPartLanes && resultTypes.size() == sourceParts.size();
    if (denseLaneExtension) {
      StringRef part = resultBits == sourceBits * 2 ? StringRef("EVEN")
                                                    : StringRef("P0");
      return emitDenseLaneExtension(op, sourceParts, resultVRegTypes,
                                    sourceType, part, rewriter);
    }

    int64_t sourceStride =
        sourceView && sourceView.isContiguous() ? sourceView.getLaneStride() : 1;
    FailureOr<std::pair<ArrayRef<StringRef>, int64_t>> partPlan =
        getExtensionPartPlan(op, sourceParts, resultTypes, sourceBits,
                             resultBits, sourceStride, rewriter);
    if (failed(partPlan)) {
      return failure();
    }

    FailureOr<Value> mask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(mask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build integer extension seed mask");
    }

    return emitFactorExtension(op, sourceParts, resultVRegTypes, partPlan->first,
                               partPlan->second, sourceStride, *mask,
                               rewriter);
  }

  FailureOr<SmallVector<VRegType>> collectExtensionResultTypes(
      OpT op, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<VRegType> resultVRegTypes;
    resultVRegTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      bool invalidType =
          !resultVRegType || !isa<IntegerType>(resultVRegType.getElementType()) ||
          (!resultVRegTypes.empty() &&
           resultVRegType != resultVRegTypes.front());
      if (invalidType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported physical integer extension result type");
      }
      resultVRegTypes.push_back(resultVRegType);
    }
    return resultVRegTypes;
  }

  FailureOr<VRegType> getUniformExtensionSourceType(
      OpT op, ValueRange sourceParts,
      OneToNPatternRewriter &rewriter) const {
    if (sourceParts.empty()) {
      return rewriter.notifyMatchFailure(
          op, "integer extension requires at least one physical source chunk");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected physical integer extension source");
    }
    for (Value sourcePart : sourceParts) {
      auto currentSourceType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentSourceType || currentSourceType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "integer extension source physical parts must have matching "
                "type");
      }
    }
    return sourceType;
  }

  struct ExtensionLoweringInput {
    VMIVRegType sourceVMIType;
    VMIVRegType resultVMIType;
    ValueRange sourceParts;
    SmallVector<Type> resultTypes;
    VRegType sourceType;
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
  };

  FailureOr<ExtensionLoweringInput> getLoweringInput(
      OpT op, typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
      OneToNPatternRewriter &rewriter) const {
    auto sourceVMIType = cast<VMIVRegType>(op.getSource().getType());
    auto resultVMIType = cast<VMIVRegType>(op.getResult().getType());
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<SmallVector<Type>> resultTypes =
        getConvertedResultTypes(op, 0, *this->getTypeConverter());
    if (failed(resultTypes)) {
      return failure();
    }
    FailureOr<VRegType> sourceType =
        getUniformExtensionSourceType(op, sourceParts, rewriter);
    if (failed(sourceType)) {
      return failure();
    }
    return ExtensionLoweringInput{
        sourceVMIType, resultVMIType, sourceParts, std::move(*resultTypes),
        *sourceType, sourceVMIType.getLayoutAttr(), resultVMIType.getLayoutAttr()};
  }

  LogicalResult lowerGroupSlotByLayout(
      OpT op, const ExtensionLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input.sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(input.resultVMIType.getElementType());
    auto sourceIntegerType =
        dyn_cast<IntegerType>(input.sourceVMIType.getElementType());
    auto resultIntegerType =
        dyn_cast<IntegerType>(input.resultVMIType.getElementType());
    int64_t slots = input.sourceLayout.getSlots();
    bool denseGroupSlotExtension =
        sourceIntegerType && resultIntegerType &&
        input.sourceLayout.getNumGroups() == input.resultLayout.getNumGroups() &&
        input.sourceLayout.getLaneStride() == 1 &&
        input.resultLayout.getLaneStride() == 1 &&
        input.sourceLayout.getSlots() == input.resultLayout.getSlots() &&
        (slots == 2 || slots == 4 || slots == 8) && sourceBits > 0 &&
        resultBits > sourceBits && resultBits % sourceBits == 0 &&
        (resultBits / sourceBits == 2 || resultBits / sourceBits == 4) &&
        input.sourceParts.size() == input.resultTypes.size();
    if (denseGroupSlotExtension) {
      return lowerDenseGroupSlotExtension(
          op, input.sourceParts, input.resultTypes, input.sourceVMIType,
          input.resultVMIType, resultIntegerType, sourceBits, resultBits,
          rewriter);
    }
    return lowerLegacyGroupSlotExtension(
        op, input.sourceParts, input.resultTypes, input.sourceVMIType,
        input.sourceLayout, input.resultLayout, sourceBits, resultBits,
        rewriter);
  }

  LogicalResult lowerNonGroupSlotByLayout(
      OpT op, const ExtensionLoweringInput &input,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<SmallVector<VRegType>> resultVRegTypes =
        collectExtensionResultTypes(op, input.resultTypes, rewriter);
    if (failed(resultVRegTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(input.sourceType.getElementType());
    unsigned resultBits = pto::getPTOStorageElemBitWidth(
        resultVRegTypes->front().getElementType());
    return lowerPhysicalExtension(
        op, input.sourceParts, *resultVRegTypes, input.resultTypes,
        input.sourceType, sourceBits, resultBits, input.sourceLayout,
        input.resultLayout, rewriter);
  }

public:
  LogicalResult
  matchAndRewrite(OpT op,
                  typename OneToNOpConversionPattern<OpT>::OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<ExtensionLoweringInput> input =
        getLoweringInput(op, adaptor, rewriter);
    if (failed(input)) {
      return failure();
    }
    bool groupSlotLayouts =
        input->sourceLayout && input->resultLayout &&
        input->sourceLayout.isGroupSlots() && input->resultLayout.isGroupSlots();
    if (groupSlotLayouts) {
      return lowerGroupSlotByLayout(op, *input, rewriter);
    }
    return lowerNonGroupSlotByLayout(op, *input, rewriter);
  }
};

// TruncI lowering support matrix
//
// Keep this comment aligned with both:
//   1. verifySupportedVMIToVPTOOps() diagnostics below, and
//   2. the actual OneToN lowering implemented in this pattern.
//
// Dense logical layouts
//   - deinterleaved factor 2/4 -> contiguous
//     Example: 32 -> 16 or 32 -> 8.
//     Lowering shape: emit vcvt parts EVEN/ODD or P0/P1/P2/P3, then merge
//     physical results when multiple source chunks contribute to one result.
//     A result lane stride scales the required source arity: a result whose
//     logical elements sit `laneStride` apart inside a wider carrier already
//     consumes `laneStride` of the converted sub-lanes of one source lane, so
//     each result chunk is filled by `factor / laneStride` source chunks and
//     the k-th converted sub-lane is selected by part `parts[k * laneStride]`.
//     Example: 32 -> 8 with result lane_stride = 2 takes two source chunks and
//     selects parts P0 + P2 before the merge.
//   - deinterleaved factor 4 -> deinterleaved factor 2
//     Example: 32 -> 16.
//     Lowering shape: emit vcvt EVEN/ODD per source chunk pair.
//   - contiguous lane_stride = 1 -> contiguous lane_stride = 2/4
//     Example: 16 -> 8 lane_stride=2, 32 -> 8 lane_stride=4.
//     Lowering shape: NOSAT keeps/bitcasts the source carrier; SAT emits vcvt
//     into the logical-element vector whose live results occupy the requested
//     strided lanes.
//
// Group-slots logical layouts
//   - slots = 1 preserves the layout for 2x/4x narrowing.
//   - slots = 8 multiplies the source lane stride by the narrowing factor.
//   - 2x narrowing lowers with part = EVEN; 4x narrowing uses part = P0.
//   - 32-bit integer -> 8-bit integer, slots = 8, result lane_stride = 4
//     Lowering shape: no vcvt; keep/bitcast the 32-bit carrier and let the
//     later store consume it as PK4_B32.
struct OneToNVMITruncIOpPattern : OneToNOpConversionPattern<VMITruncIOp> {
  using OneToNOpConversionPattern<VMITruncIOp>::OneToNOpConversionPattern;

private:
  FailureOr<std::pair<VRegType, VRegType>> getUniformTruncTypes(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    bool emptyPhysicalParts = sourceParts.empty() || resultTypes.empty();
    if (emptyPhysicalParts) {
      return rewriter.notifyMatchFailure(
          op, "trunci requires non-empty physical source and result parts");
    }
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    bool invalidTypes =
        !sourceType || !isa<IntegerType>(sourceType.getElementType()) ||
        !resultType || !isa<IntegerType>(resultType.getElementType());
    if (invalidTypes) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    }
    for (Value sourcePart : sourceParts) {
      auto currentType = dyn_cast<VRegType>(sourcePart.getType());
      if (!currentType || currentType != sourceType) {
        return rewriter.notifyMatchFailure(
            op, "trunci source physical parts must have matching integer type");
      }
    }
    for (Type physicalResultType : resultTypes) {
      auto currentType = dyn_cast<VRegType>(physicalResultType);
      if (!currentType || currentType != resultType) {
        return rewriter.notifyMatchFailure(
            op, "trunci result physical parts must have matching integer type");
      }
    }
    return std::make_pair(sourceType, resultType);
  }

  void finalizeResults(VMITruncIOp op, SmallVectorImpl<Value> &results,
                       bool requiresResultBitcast, ArrayRef<Type> originalResultTypes,
                       OneToNPatternRewriter &rewriter) const {
    if (requiresResultBitcast) {
      for (auto &&[index, result] : llvm::enumerate(results)) {
        result = rewriter
                     .create<VbitcastOp>(op.getLoc(), originalResultTypes[index],
                                         result)
                     .getResult();
      }
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
  }

  FailureOr<Value> lowerGroupSlotTruncPart(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      VMIVRegType resultVMIType, VMILayoutAttr resultLayout,
      unsigned sourceLogicalBits, unsigned resultLogicalBits,
      Value activeSlotMask, StringAttr sat, OneToNPatternRewriter &rewriter) const {
    unsigned physicalResultBits =
        pto::getPTOStorageElemBitWidth(resultType.getElementType());
    bool directCarrier = resultLayout.hasLaneStride() &&
                         resultLayout.getLaneStride() == 4 &&
                         resultLogicalBits == 8 && physicalResultBits == 32;
    if (directCarrier) {
      return lowerGroupSlotDirectCarrier(op, sourcePart, resultType, rewriter);
    }
    bool wideCarrier = resultLayout.hasLaneStride() &&
                       resultLayout.getLaneStride() == 2 &&
                       resultLogicalBits == 16 && physicalResultBits == 32;
    if (wideCarrier) {
      return lowerGroupSlotWideCarrier(op, sourcePart, resultType,
                                       resultVMIType, activeSlotMask, sat,
                                       rewriter);
    }
    bool validNarrowResult = physicalResultBits == 16 || physicalResultBits == 8;
    if (!validNarrowResult) {
      return rewriter.notifyMatchFailure(
          op, "unsupported group-slot trunci physical type");
    }
    StringAttr part = rewriter.getStringAttr(
        sourceLogicalBits == 2 * resultLogicalBits ? "EVEN" : "P0");
    return rewriter
        .create<VcvtOp>(op.getLoc(), resultType, sourcePart, activeSlotMask,
                        nullptr, sat, part)
        .getResult();
  }

  FailureOr<Value> lowerGroupSlotDirectCarrier(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      OneToNPatternRewriter &rewriter) const {
    return sourcePart.getType() == resultType
               ? sourcePart
               : rewriter.create<VbitcastOp>(op.getLoc(), resultType, sourcePart)
                     .getResult();
  }

  FailureOr<Value> lowerGroupSlotWideCarrier(
      VMITruncIOp op, Value sourcePart, VRegType resultType,
      VMIVRegType resultVMIType, Value activeSlotMask, StringAttr sat,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<int64_t> lanes =
        getDataLanesPerPart(resultVMIType.getElementType());
    if (failed(lanes)) {
      return rewriter.notifyMatchFailure(
          op, "failed to derive group-slot trunci conversion lanes");
    }
    auto conversionType = VRegType::get(
        rewriter.getContext(), *lanes, resultVMIType.getElementType());
    Value converted = rewriter
                          .create<VcvtOp>(op.getLoc(), conversionType,
                                          sourcePart, activeSlotMask, nullptr,
                                          sat, rewriter.getStringAttr("EVEN"))
                          .getResult();
    FailureOr<Value> carrier =
        bitcastVReg(op.getLoc(), converted, resultType, rewriter);
    if (failed(carrier)) {
      return rewriter.notifyMatchFailure(
          op, "failed to expose group-slot trunci result carrier");
    }
    return *carrier;
  }

  FailureOr<std::pair<bool, bool>> getGroupSlotTruncModes(
      VMITruncIOp op, VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool supportsDirect =
        (sourceBits == 32 && (resultBits == 16 || resultBits == 8)) ||
        (sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 1);
    bool supportsPacked =
        sourceBits == 16 && resultBits == 8 && sourceLayout.getSlots() == 8 &&
        resultLayout.getSlots() == 8 && resultLayout.hasLaneStride() &&
        resultLayout.getLaneStride() ==
            sourceLayout.getLaneStride() * kPairWidth;
    bool invalidShape =
        sourceLayout.getNumGroups() != resultLayout.getNumGroups() ||
        sourceLayout.getSlots() != resultLayout.getSlots() ||
        (sourceLayout.getSlots() != 1 && sourceLayout.getSlots() != 8) ||
        (!supportsDirect && !supportsPacked) ||
        sourceParts.size() != resultTypes.size();
    if (invalidShape) {
      (void)rewriter.notifyMatchFailure(op,
                                        "unsupported group-slot trunci shape");
      return failure();
    }
    return std::make_pair(supportsDirect, supportsPacked);
  }

  FailureOr<SmallVector<Value>> lowerGroupSlotTruncParts(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMIVRegType resultVMIType, VMILayoutAttr resultLayout,
      unsigned sourceBits, unsigned resultBits, bool supportsPacked,
      Value activeSlotMask, StringAttr sat, OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, physicalResultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto sourceType = dyn_cast<VRegType>(sourcePart.getType());
      auto resultType = dyn_cast<VRegType>(physicalResultType);
      bool validPhysicalTypes =
          sourceType &&
          pto::getPTOStorageElemBitWidth(sourceType.getElementType()) ==
              sourceBits &&
          resultType;
      if (!validPhysicalTypes) {
        (void)rewriter.notifyMatchFailure(
            op, "unsupported group-slot trunci physical type");
        return failure();
      }
      if (supportsPacked) {
        results.push_back(rewriter
                              .create<VcvtOp>(op.getLoc(), resultType, sourcePart,
                                              activeSlotMask, nullptr, sat,
                                              rewriter.getStringAttr("EVEN"))
                              .getResult());
        continue;
      }
      FailureOr<Value> lowered = lowerGroupSlotTruncPart(
          op, sourcePart, resultType, resultVMIType, resultLayout, sourceBits,
          resultBits, activeSlotMask, sat, rewriter);
      if (failed(lowered)) {
        return failure();
      }
      results.push_back(*lowered);
    }
    return results;
  }

  LogicalResult lowerGroupSlotTrunc(
      VMITruncIOp op, OpAdaptor adaptor, OneToNPatternRewriter &rewriter,
      VMIVRegType sourceVMIType, VMIVRegType resultVMIType,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      ArrayRef<Type> resultTypes) const {
    ValueRange sourceParts = adaptor.getSource();
    FailureOr<std::pair<bool, bool>> modes = getGroupSlotTruncModes(
        op, sourceVMIType, resultVMIType, sourceLayout, resultLayout,
        sourceParts, resultTypes, rewriter);
    if (failed(modes)) {
      return failure();
    }
    unsigned sourceLogicalBits =
        pto::getPTOStorageElemBitWidth(sourceVMIType.getElementType());
    unsigned resultLogicalBits =
        pto::getPTOStorageElemBitWidth(resultVMIType.getElementType());
    bool supportsPacked = modes->second;

    StringAttr sat = op->getAttrOfType<StringAttr>("saturate");
    if (sourceLogicalBits == 16 && resultLogicalBits == 8 &&
        sat && sat.getValue() == "NOSAT") {
      // slots=8 doubles the source lane stride for low bytes; slots=1 has
      // only lane zero live. Neither view needs a conversion instruction.
      // In particular, A5 has no signed 16-to-signed-8 vcvt instruction.
      SmallVector<Value> results;
      for (auto [source, type] : llvm::zip_equal(sourceParts, resultTypes)) {
        auto result = bitcastVReg(op.getLoc(), source, cast<VRegType>(type),
                                 rewriter);
        if (failed(result)) {
          return failure();
        }
        results.push_back(*result);
      }
      finalizeResults(op, results, false, resultTypes, rewriter);
      return success();
    }
    // Native 16-bit sums occupy the even halfword lanes. Include the whole
    // strided packet so all eight low halves reach the narrowing conversion.
    int64_t activeLanes = sourceLayout.getSlots() * sourceLayout.getLaneStride();
    StringRef activeSlotGranularity = sourceLogicalBits == 16 ? "b16" : "b32";
    FailureOr<Value> activeSlotMask = createPrefixMaskForActiveLanes(
        op.getLoc(), MaskType::get(rewriter.getContext(), activeSlotGranularity),
        activeLanes, rewriter);
    if (failed(activeSlotMask)) {
      return rewriter.notifyMatchFailure(
          op, "failed to build group-slot trunci active slot mask");
    }
    FailureOr<SmallVector<Value>> results = lowerGroupSlotTruncParts(
        op, sourceParts, resultTypes, resultVMIType, resultLayout,
        sourceLogicalBits, resultLogicalBits, supportsPacked, *activeSlotMask,
        sat, rewriter);
    if (failed(results)) {
      return failure();
    }
    finalizeResults(op, *results, false, resultTypes, rewriter);
    return success();
  }

  LogicalResult lowerDenseLaneStrideTrunc(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      int64_t factor, StringAttr sat, bool requiresResultBitcast,
      ArrayRef<Type> originalResultTypes,
      OneToNPatternRewriter &rewriter) const {
    StringAttr part = rewriter.getStringAttr(factor == 2 ? "EVEN" : "P0");
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported dense trunci source type");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    if (failed(sourceMask)) {
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      auto resultVRegType = dyn_cast<VRegType>(resultType);
      if (!resultVRegType) {
        return rewriter.notifyMatchFailure(
            op, "unsupported dense trunci result type");
      }
      results.push_back(rewriter
                            .create<VcvtOp>(op.getLoc(), resultVRegType,
                                            sourcePart, *sourceMask,
                                            /*rnd=*/nullptr, sat, part)
                            .getResult());
    }
    finalizeResults(op, results, requiresResultBitcast, originalResultTypes, rewriter);
    return success();
  }

  LogicalResult lowerNoSatDenseCarrier(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      OneToNPatternRewriter &rewriter) const {
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [sourcePart, resultType] :
         llvm::zip_equal(sourceParts, resultTypes)) {
      FailureOr<Value> result =
          bitcastVReg(op.getLoc(), sourcePart, resultType, rewriter);
      if (failed(result)) {
        return rewriter.notifyMatchFailure(
            op, "failed to forward NOSAT trunci carrier");
      }
      results.push_back(*result);
    }
    replaceOpWithFlatConvertedValues(rewriter, op, results,
                                     *this->getTypeConverter());
    return success();
  }

  // Physical plan for the dense factor route of an integer narrowing.
  //
  // `factor` is the element-width ratio; `resultLaneStride` is the lane stride
  // of the declared result layout, taken from its carrier view.  A result whose
  // elements sit `resultLaneStride` apart in a wider carrier consumes
  // `resultLaneStride` converted sub-lanes of one source lane per result lane,
  // so one result chunk is filled by `factor / resultLaneStride` source chunks
  // rather than by `factor` of them, and the part that carries the k-th
  // converted sub-lane is `parts[k * resultLaneStride]`.
  //
  // This is the same shape the fp-to-int narrowing already uses; before this
  // existed the required source arity was `factor` chunks per result chunk,
  // which only ever matched layouts whose result carried no lane stride.
  struct TruncINarrowingPlan {
    ArrayRef<StringRef> parts;
    int64_t factor = 0;
    int64_t sourceFactor = 0;
    int64_t resultLaneStride = 0;
  };

  FailureOr<Value> buildFactorTruncResult(
      VMITruncIOp op, ValueRange sourceParts, Type resultType,
      int64_t resultIndex, const TruncINarrowingPlan &plan, StringAttr sat,
      Value sourceMask, Value resultMask,
      OneToNPatternRewriter &rewriter) const {
    bool invalidPlan =
        plan.factor <= 0 || plan.sourceFactor <= 0 ||
        plan.resultLaneStride <= 0 ||
        static_cast<size_t>(plan.factor) > plan.parts.size();
    if (invalidPlan) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci conversion factor");
    }
    auto resultVRegType = dyn_cast<VRegType>(resultType);
    if (!resultVRegType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci result type");
    }

    // One result chunk consumes its `sourceFactor` source chunks.  A unit
    // result lane stride takes them as a contiguous run, this route's
    // long-standing grouping.  A result that carries a lane stride with several
    // physical chunks instead takes them interleaved across the chunks, the same
    // grouping the narrow fp-to-int route uses; the contiguous run would index
    // past the physical arity there.
    SmallVector<Value> partials;
    partials.reserve(static_cast<size_t>(plan.sourceFactor));
    int64_t resultCount =
        static_cast<int64_t>(sourceParts.size()) / plan.sourceFactor;
    for (int64_t partIndex = 0; partIndex < plan.sourceFactor; ++partIndex) {
      int64_t sourceIndex = plan.resultLaneStride > 1
                                ? partIndex * resultCount + resultIndex
                                : resultIndex * plan.factor + partIndex;
      if (sourceIndex < 0 ||
          sourceIndex >= static_cast<int64_t>(sourceParts.size())) {
        return rewriter.notifyMatchFailure(
            op, "trunci source part index exceeds physical arity");
      }
      int64_t partNameIndex = partIndex * plan.resultLaneStride;
      if (partNameIndex < 0 ||
          partNameIndex >= static_cast<int64_t>(plan.parts.size())) {
        return rewriter.notifyMatchFailure(
            op, "trunci part index exceeds the conversion factor");
      }
      partials.push_back(
          rewriter
              .create<VcvtOp>(op.getLoc(), resultVRegType,
                              sourceParts[sourceIndex], sourceMask, nullptr, sat,
                              rewriter.getStringAttr(plan.parts[partNameIndex]))
              .getResult());
    }

    Value merged = partials.front();
    for (Value partial : llvm::drop_begin(partials)) {
      merged = rewriter
                   .create<VorOp>(op.getLoc(), resultVRegType, merged, partial,
                                  resultMask)
                   .getResult();
    }
    return merged;
  }

  LogicalResult lowerFactorTrunc(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      const TruncINarrowingPlan &plan, StringAttr sat, bool requiresResultBitcast,
      ArrayRef<Type> originalResultTypes,
      OneToNPatternRewriter &rewriter) const {
    auto sourceType = dyn_cast<VRegType>(sourceParts.front().getType());
    auto resultType = dyn_cast<VRegType>(resultTypes.front());
    if (!sourceType || !resultType) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result type");
    }
    FailureOr<Value> sourceMask =
        createAllTrueMaskForVReg(op.getLoc(), sourceType, rewriter);
    FailureOr<Value> resultMask =
        createAllTrueMaskForVReg(op.getLoc(), resultType, rewriter);
    bool failedMasks = failed(sourceMask) || failed(resultMask);
    if (failedMasks) {
      return rewriter.notifyMatchFailure(op, "failed to build trunci masks");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (int64_t resultIndex = 0;
         resultIndex < static_cast<int64_t>(resultTypes.size()); ++resultIndex) {
      FailureOr<Value> result = buildFactorTruncResult(
          op, sourceParts, resultTypes[resultIndex], resultIndex, plan, sat,
          *sourceMask, *resultMask, rewriter);
      if (failed(result)) {
        return failure();
      }
      results.push_back(*result);
    }
    finalizeResults(op, results, requiresResultBitcast, originalResultTypes, rewriter);
    return success();
  }

  struct TruncIPhysicalPlan {
    VRegType sourceType;
    VRegType resultType;
    int64_t factor;
    bool denseLaneStride;
    StringAttr saturate;
  };

  FailureOr<TruncIPhysicalPlan> buildPhysicalPlan(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<std::pair<VRegType, VRegType>> uniformTypes =
        getUniformTruncTypes(op, sourceParts, resultTypes, rewriter);
    if (failed(uniformTypes)) {
      return failure();
    }
    unsigned sourceBits =
        pto::getPTOStorageElemBitWidth(uniformTypes->first.getElementType());
    unsigned resultBits =
        pto::getPTOStorageElemBitWidth(uniformTypes->second.getElementType());
    bool invalidWidth = sourceBits == 0 || resultBits == 0 ||
                        sourceBits % resultBits != 0;
    if (invalidWidth) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result width relation");
    }
    int64_t factor = sourceBits / resultBits;
    bool denseLaneStride =
        sourceLayout && resultLayout && sourceLayout.isContiguous() &&
        sourceLayout.getLaneStride() == 1 && resultLayout.isContiguous() &&
        resultLayout.getLaneStride() == factor &&
        sourceParts.size() == resultTypes.size();
    bool unsupportedDenseFactor = denseLaneStride && factor != 2 && factor != 4;
    if (unsupportedDenseFactor) {
      return rewriter.notifyMatchFailure(
          op, "unsupported dense lane_stride trunci result layout");
    }
    return TruncIPhysicalPlan{uniformTypes->first, uniformTypes->second, factor,
                              denseLaneStride,
                              op->getAttrOfType<StringAttr>("saturate")};
  }

  struct TruncIAliasPlan {
    SmallVector<Value> sourceParts;
    SmallVector<Type> resultTypes;
    SmallVector<Type> originalResultTypes;
    bool requiresResultBitcast = false;
  };

  TruncIAliasPlan materializeSignedNarrowingAlias(
      VMITruncIOp op, ValueRange sourceParts, ArrayRef<Type> resultTypes,
      const TruncIPhysicalPlan &physicalPlan,
      OneToNPatternRewriter &rewriter) const {
    TruncIAliasPlan aliasPlan;
    auto sourceIntegerType =
        dyn_cast<IntegerType>(physicalPlan.sourceType.getElementType());
    auto resultIntegerType =
        dyn_cast<IntegerType>(physicalPlan.resultType.getElementType());
    bool signedNarrowing = sourceIntegerType && resultIntegerType &&
                           sourceIntegerType.isSigned() &&
                           resultIntegerType.isSigned() &&
                           sourceIntegerType.getWidth() >
                               resultIntegerType.getWidth();
    // A5 has no signed vcvt to an 8-bit destination (S16/S32 -> S8); the wider
    // signed destinations such as S32 -> S16 are native.  For the non-native
    // pairs a non-saturating truncation is bit-identical to the unsigned
    // conversion, which is the only form the hardware offers.
    bool noSat = !physicalPlan.saturate ||
                 physicalPlan.saturate.getValue() == "NOSAT";
    bool noNativeSignedNarrow = resultIntegerType.getWidth() == mlir::pto::kValue8;
    bool requiresAlias = signedNarrowing && noSat && noNativeSignedNarrow;
    if (!requiresAlias) {
      aliasPlan.sourceParts.assign(sourceParts.begin(), sourceParts.end());
      aliasPlan.resultTypes.assign(resultTypes.begin(), resultTypes.end());
      return aliasPlan;
    }
    auto uSourceElemTy =
        rewriter.getIntegerType(sourceIntegerType.getWidth(), /*isSigned=*/false);
    auto uResultElemTy =
        rewriter.getIntegerType(resultIntegerType.getWidth(), /*isSigned=*/false);
    aliasPlan.originalResultTypes.assign(resultTypes.begin(), resultTypes.end());
    aliasPlan.sourceParts.reserve(sourceParts.size());
    for (Value sourcePart : sourceParts) {
      auto sourcePartType = cast<VRegType>(sourcePart.getType());
      aliasPlan.sourceParts.push_back(
          rewriter.create<VbitcastOp>(
              op.getLoc(), VRegType::get(sourcePartType.getContext(),
                                          sourcePartType.getElementCount(),
                                          uSourceElemTy),
              sourcePart).getResult());
    }
    aliasPlan.resultTypes.reserve(resultTypes.size());
    for (Type resultType : resultTypes) {
      auto resultVRegType = cast<VRegType>(resultType);
      aliasPlan.resultTypes.push_back(VRegType::get(
          resultVRegType.getContext(), resultVRegType.getElementCount(),
          uResultElemTy));
    }
    aliasPlan.requiresResultBitcast = true;
    return aliasPlan;
  }

  FailureOr<TruncINarrowingPlan> buildTruncINarrowingPlan(
      VMITruncIOp op, const TruncIPhysicalPlan &physicalPlan,
      VMIVRegType resultVMIType, size_t sourcePartCount,
      size_t resultPartCount, OneToNPatternRewriter &rewriter) const {
    int64_t factor = physicalPlan.factor;
    bool unsupportedFactor =
        factor != mlir::pto::kValue2 && factor != mlir::pto::kValue4;
    if (unsupportedFactor) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci conversion factor");
    }
    VMILayoutAttr resultView = getVMICastDenseCarrierView(
        resultVMIType.getLayoutAttr(), resultVMIType.getElementType());
    int64_t resultLaneStride =
        resultView && resultView.isContiguous() ? resultView.getLaneStride() : 1;
    bool unsupportedLaneStride =
        resultLaneStride <= 0 || factor % resultLaneStride != 0;
    if (unsupportedLaneStride) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci result lane stride");
    }
    int64_t sourceFactor = factor / resultLaneStride;
    bool sourceArityMismatch =
        sourcePartCount != static_cast<size_t>(sourceFactor) * resultPartCount;
    if (sourceArityMismatch) {
      return rewriter.notifyMatchFailure(
          op, "unsupported physical trunci source/result arity relation");
    }
    static constexpr StringRef kEvenOddParts[] = {"EVEN", "ODD"};
    static constexpr StringRef kPacked4Parts[] = {"P0", "P1", "P2", "P3"};
    ArrayRef<StringRef> parts = factor == mlir::pto::kValue2
                                    ? ArrayRef<StringRef>(kEvenOddParts)
                                    : ArrayRef<StringRef>(kPacked4Parts);
    return TruncINarrowingPlan{parts, factor, sourceFactor, resultLaneStride};
  }

  LogicalResult lowerNonGroupSlotTrunc(
      VMITruncIOp op, ValueRange sourceParts, SmallVector<Type> resultTypes,
      VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout,
      OneToNPatternRewriter &rewriter) const {
    FailureOr<TruncIPhysicalPlan> physicalPlan = buildPhysicalPlan(
        op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter);
    if (failed(physicalPlan)) {
      return failure();
    }
    bool useNoSatCarrier =
        physicalPlan->denseLaneStride && physicalPlan->saturate &&
        physicalPlan->saturate.getValue() == "NOSAT";
    if (useNoSatCarrier) {
      return lowerNoSatDenseCarrier(op, sourceParts, resultTypes, rewriter);
    }
    TruncIAliasPlan aliasPlan = materializeSignedNarrowingAlias(
        op, sourceParts, resultTypes, *physicalPlan, rewriter);
    if (physicalPlan->denseLaneStride) {
      return lowerDenseLaneStrideTrunc(
          op, aliasPlan.sourceParts, aliasPlan.resultTypes,
          physicalPlan->factor, physicalPlan->saturate,
          aliasPlan.requiresResultBitcast, aliasPlan.originalResultTypes,
          rewriter);
    }
    FailureOr<TruncINarrowingPlan> narrowingPlan = buildTruncINarrowingPlan(
        op, *physicalPlan, cast<VMIVRegType>(op.getResult().getType()),
        aliasPlan.sourceParts.size(), aliasPlan.resultTypes.size(), rewriter);
    if (failed(narrowingPlan)) {
      return failure();
    }
    return lowerFactorTrunc(
        op, aliasPlan.sourceParts, aliasPlan.resultTypes, *narrowingPlan,
        physicalPlan->saturate, aliasPlan.requiresResultBitcast,
        aliasPlan.originalResultTypes, rewriter);
  }

public:

  LogicalResult
  matchAndRewrite(VMITruncIOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    FailureOr<VMIPhysicalConversionInput> input =
        getVMIPhysicalConversionInput(op, adaptor, *this->getTypeConverter());
    if (failed(input)) {
      return failure();
    }
    VMILayoutAttr sourceLayout = input->sourceVMIType.getLayoutAttr();
    VMILayoutAttr resultLayout = input->resultVMIType.getLayoutAttr();
    bool groupSlotLayouts =
        sourceLayout && resultLayout && sourceLayout.isGroupSlots() &&
        resultLayout.isGroupSlots();
    if (groupSlotLayouts) {
      return lowerGroupSlotTrunc(op, adaptor, rewriter, input->sourceVMIType,
                                 input->resultVMIType, sourceLayout,
                                 resultLayout, input->resultTypes);
    }

    return lowerNonGroupSlotTrunc(op, input->sourceParts,
                                  std::move(input->resultTypes), sourceLayout,
                                  resultLayout, rewriter);
  }
};
