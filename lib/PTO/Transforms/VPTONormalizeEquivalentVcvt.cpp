// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTONORMALIZEEQUIVALENTVCVT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool isOddPart(StringRef part) {
  return part == "ODD" || part == "PART_ODD";
}

static bool isAllTrueMask(Value mask) {
  if (auto op = mask.getDefiningOp<PsetB8Op>())
    return op.getPattern() == "PAT_ALL";
  if (auto op = mask.getDefiningOp<PsetB16Op>())
    return op.getPattern() == "PAT_ALL";
  if (auto op = mask.getDefiningOp<PsetB32Op>())
    return op.getPattern() == "PAT_ALL";
  return false;
}

static bool isPairEquivalentLoadDist(StringRef dist) {
  return dist == "BRC_B8" || dist == "BRC_B16" || dist == "BRC_B32" ||
         dist == "US_B8" || dist == "US_B16" || dist == "E2B_B16" ||
         dist == "E2B_B32";
}

static bool hasEvenOddEquivalentLanes(Value value) {
  if (value.getDefiningOp<VbrOp>())
    return true;

  auto load = value.getDefiningOp<VldsOp>();
  if (!load || value != load.getResult())
    return false;

  std::optional<StringRef> dist = load.getDist();
  return dist && isPairEquivalentLoadDist(*dist);
}

static bool isNarrowToWideVcvt(VcvtOp op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType)
    return false;

  unsigned inputBits = getPTOStorageElemBitWidth(inputType.getElementType());
  unsigned resultBits = getPTOStorageElemBitWidth(resultType.getElementType());
  return inputBits != 0 && resultBits != 0 && inputBits < resultBits;
}

struct VPTONormalizeEquivalentVcvtPass
    : public pto::impl::VPTONormalizeEquivalentVcvtBase<
          VPTONormalizeEquivalentVcvtPass> {
  void runOnOperation() override {
    StringAttr even = StringAttr::get(&getContext(), "EVEN");

    getOperation().walk([&](VcvtOp op) {
      std::optional<StringRef> part = op.getPart();
      if (!part || !isOddPart(*part))
        return;
      if (!isNarrowToWideVcvt(op))
        return;
      if (!isAllTrueMask(op.getMask()))
        return;
      if (!hasEvenOddEquivalentLanes(op.getInput()))
        return;

      op.setPartAttr(even);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTONormalizeEquivalentVcvtPass() {
  return std::make_unique<VPTONormalizeEquivalentVcvtPass>();
}
