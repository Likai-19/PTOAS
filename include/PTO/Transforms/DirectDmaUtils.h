// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Shared direct-view extraction and GM/UB burst lowering for tload, tstore,
// and tdump. Callers validate element types and tile shapes before emission.
#ifndef PTO_TRANSFORMS_DIRECTDMAUTILS_H
#define PTO_TRANSFORMS_DIRECTDMAUTILS_H
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <optional>
namespace mlir::pto::dma {
inline Value i64c(int64_t value, Location loc, OpBuilder& b) {
    return b.create<arith::ConstantOp>(loc, b.getI64IntegerAttr(value));
}
inline Value i64c0(Location loc, OpBuilder& b) { return i64c(0, loc, b); }
inline Value i64c1(Location loc, OpBuilder& b) { return i64c(1, loc, b); }
struct DmaViewInfo {
    Value gmPtr;
    Value linearOffset;
    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    SmallVector<Value> offsets;
};

inline FailureOr<DmaViewInfo> extractDirectDmaViewInfo(
    Value view, Operation* consumer, StringRef strideError = "A2/A3 DMA lowering requires a unit innermost stride",
    StringRef rankError = "A2/A3 DMA lowering requires rank-preserving nested partitions",
    StringRef partitionError = {}, StringRef rootError = {}) {
    SmallVector<pto::PartitionViewOp> partitions;
    Value root = view;
    while (auto partition = root.getDefiningOp<pto::PartitionViewOp>()) {
        partitions.push_back(partition);
        root = partition.getSource();
    }
    if (partitions.empty()) {
        if (!partitionError.empty()) {
            consumer->emitError(partitionError);
        }
        return failure();
    }

    auto makeView = root.getDefiningOp<pto::MakeTensorViewOp>();
    if (!makeView) {
        if (!rootError.empty()) {
            consumer->emitError(rootError);
        }
        return failure();
    }

    DmaViewInfo info;
    info.gmPtr = makeView.getPtr();
    info.sizes.assign(partitions.front().getSizes().begin(), partitions.front().getSizes().end());
    info.strides.assign(makeView.getStrides().begin(), makeView.getStrides().end());
    const bool unitStride = !info.strides.empty() &&
        matchPattern(info.strides.back(), m_One());
    if (!unitStride) {
        consumer->emitError(strideError);
        return failure();
    }

    info.offsets.assign(partitions.front().getOffsets().begin(), partitions.front().getOffsets().end());
    OpBuilder builder(consumer);
    for (pto::PartitionViewOp partition : llvm::drop_begin(partitions)) {
        const bool sameRank = partition.getOffsets().size() == info.offsets.size() &&
            partition.getSizes().size() == info.sizes.size();
        if (!sameRank) {
            consumer->emitError(rankError);
            return failure();
        }
        for (auto [index, offset] : llvm::enumerate(partition.getOffsets())) {
            info.offsets[index] = builder.create<arith::AddIOp>(consumer->getLoc(), info.offsets[index], offset);
        }
    }
    return info;
}

inline Value i64Cast(Location loc, OpBuilder& b, Value indexVal) {
    return b.create<arith::IndexCastOp>(loc, b.getI64Type(), indexVal).getResult();
}

// i64 byte-stride of a GM view dimension (index-typed stride x elemSize).
inline Value strideBytes(Location loc, OpBuilder& b, Value idxStride, unsigned elemSize) {
    return b.create<arith::MulIOp>(loc, i64Cast(loc, b, idxStride), i64c(elemSize, loc, b)).getResult();
}

// i64 byte-length of the innermost (contiguous) burst dimension.
inline Value burstBytes(Location loc, OpBuilder& b, Value lenBurstElts, unsigned elemSize) {
    return b.create<arith::MulIOp>(loc, i64Cast(loc, b, lenBurstElts), i64c(elemSize, loc, b)).getResult();
}

// Product (in elements) of all dims below `i`, used as the UB-side stride
// of loop level `i` (the UB tile is dense).
inline Value innerDimElems(Location loc, OpBuilder& b, const DmaViewInfo& viewInfo, int i) {
    Value innerElems = i64c1(loc, b);
    for (int j = i + 1; j < static_cast<int>(viewInfo.sizes.size()); ++j) {
        innerElems = b.create<arith::MulIOp>(loc, innerElems, i64Cast(loc, b, viewInfo.sizes[j])).getResult();
    }
    return innerElems;
}

// Shared burst-transfer geometry validated by both GM<->UB emitters:
// UB tile row span in bytes plus the view's rank, with the same shape and
// rank guards the emitters historically applied inline.
struct BurstGeometry {
    int64_t ubCols;
    unsigned elemSize;
    unsigned nd;
};

inline std::optional<BurstGeometry> getBurstGeometry(
    const DmaViewInfo& viewInfo, unsigned elemSize, ArrayRef<int64_t> tileShape) {
    const bool hasTileRows = tileShape.size() >= mlir::pto::kValue2;
    if (!hasTileRows) {
        return std::nullopt;
    }
    BurstGeometry g;
    g.ubCols = tileShape[1];
    g.elemSize = elemSize;
    g.nd = viewInfo.sizes.size();
    if (g.nd < mlir::pto::kValue2) {
        return std::nullopt;
    }
    return g;
}

// The innermost-burst operands shared by both GM<->UB directions: burst
// length and the dense UB row stride.
struct BurstPlan {
    Value lenBurst;
    Value ubRowStride;
};

inline BurstPlan buildBurstPlan(Location loc, OpBuilder& b, const DmaViewInfo& viewInfo, const BurstGeometry& g) {
    // The burst length covers one innermost view dim; the UB row stride is
    // the full UB tile row.
    Value lenBurst = burstBytes(loc, b, viewInfo.sizes[g.nd - 1], g.elemSize);
    Value ubRowStride = b.create<arith::MulIOp>(loc, i64c(g.ubCols, loc, b), i64c(g.elemSize, loc, b)).getResult();
    return {lenBurst, ubRowStride};
}

inline LogicalResult emitMteGmUb(
    Location loc, OpBuilder& b, Value gmPtr, Value ubPtr, const DmaViewInfo& viewInfo, unsigned elemSize,
    ArrayRef<int64_t> tileShape) {
    auto geometry = getBurstGeometry(viewInfo, elemSize, tileShape);
    if (!geometry) {
        return failure();
    }
    BurstPlan plan = buildBurstPlan(loc, b, viewInfo, *geometry);

    // GM -> UB: the burst reads a contiguous GM run and writes a dense UB row.
    pto::DmaLoopConfig nburst{
        i64Cast(loc, b, viewInfo.sizes[geometry->nd - 2]),
        strideBytes(loc, b, viewInfo.strides[geometry->nd - 2], geometry->elemSize), plan.ubRowStride};

    SmallVector<pto::DmaLoopConfig> loops;
    const int lastOuterDim = static_cast<int>(geometry->nd) - 3;
    for (int i = lastOuterDim; i >= 0; --i) {
        Value count = i64Cast(loc, b, viewInfo.sizes[i]);
        Value srcStride = strideBytes(loc, b, viewInfo.strides[i], geometry->elemSize);
        Value dstStride =
            b.create<arith::MulIOp>(loc, innerDimElems(loc, b, viewInfo, i), i64c(geometry->elemSize, loc, b))
                .getResult();
        loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteGmUbOp>(
        loc, gmPtr, ubPtr, i64c0(loc, b), plan.lenBurst, nburst, llvm::ArrayRef(loops), std::nullopt);
    return success();
}

inline LogicalResult emitMteUbGm(
    Location loc, OpBuilder& b, Value ubPtr, Value gmPtr, const DmaViewInfo& viewInfo, unsigned elemSize,
    ArrayRef<int64_t> tileShape) {
    auto geometry = getBurstGeometry(viewInfo, elemSize, tileShape);
    if (!geometry) {
        return failure();
    }
    BurstPlan plan = buildBurstPlan(loc, b, viewInfo, *geometry);

    // UB -> GM: the burst reads a dense UB row and writes a strided GM run.
    pto::DmaLoopConfig nburst{
        i64Cast(loc, b, viewInfo.sizes[geometry->nd - 2]), plan.ubRowStride,
        strideBytes(loc, b, viewInfo.strides[geometry->nd - 2], geometry->elemSize)};

    SmallVector<pto::DmaLoopConfig> loops;
    const int lastOuterDim = static_cast<int>(geometry->nd) - 3;
    for (int i = lastOuterDim; i >= 0; --i) {
        Value count = i64Cast(loc, b, viewInfo.sizes[i]);
        Value srcStride =
            b.create<arith::MulIOp>(loc, innerDimElems(loc, b, viewInfo, i), i64c(geometry->elemSize, loc, b))
                .getResult();
        Value dstStride = strideBytes(loc, b, viewInfo.strides[i], geometry->elemSize);
        loops.push_back({count, srcStride, dstStride});
    }
    b.create<pto::MteUbGmOp>(loc, ubPtr, gmPtr, plan.lenBurst, nburst, Value{}, llvm::ArrayRef(loops));
    return success();
}

} // namespace mlir::pto::dma
#endif // PTO_TRANSFORMS_DIRECTDMAUTILS_H
