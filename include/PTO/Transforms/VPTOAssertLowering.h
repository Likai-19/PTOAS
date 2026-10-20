// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOAssertLowering.h - shared pto.assert lowering pattern ----------===//
//===----------------------------------------------------------------------===//
//
// pto.assert traps via llvm.hivm.TRAP when its condition is false. The
// optional message attribute is ignored (the VPTO backend has no device
// printf channel).
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VPTOASSERTLOWERING_H
#define PTO_TRANSFORMS_VPTOASSERTLOWERING_H

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::pto {

template <typename LoweringState>
class LowerAssertOpPattern : public OpConversionPattern<pto::AssertOp> {
public:
  explicit LowerAssertOpPattern(const TypeConverter &typeConverter,
                                MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::AssertOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::AssertOp op, pto::AssertOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr StringLiteral calleeName = "llvm.hivm.TRAP";
    Location loc = op.getLoc();
    Value trueVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(true));
    Value notCond =
        rewriter.create<arith::XOrIOp>(loc, adaptor.getCondition(), trueVal);
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, notCond, /*withElseRegion=*/false);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    rewriter.create<func::CallOp>(loc, calleeName, TypeRange{}, ValueRange{});

    auto funcType = rewriter.getFunctionType({}, {});
    state.plannedDecls.push_back({calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VPTOASSERTLOWERING_H
