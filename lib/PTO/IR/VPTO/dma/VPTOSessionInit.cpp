// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOSessionInit.cpp - pto.session_init methods ---------------------===//
//===----------------------------------------------------------------------===//

#include "VPTODmaInternal.h"

using namespace mlir;
using namespace mlir::pto;

// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated verifier callback has a non-const signature.
LogicalResult SessionInitOp::verify()
{
    if (failed(verifyAsyncTransferPtr(getOperation(), getTemplateGm().getType(), "template")))
        return failure();

    // The fill reads GM with pto.ld_dev, which a SIMT scope does not provide, and
    // the session it fills is only usable by ops under the same restriction.
    if (isInsideSimtExecutionScope(getOperation()))
        return emitOpError() << "must be outside pto.simt_entry functions and pto.section.simt";
    auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
    if (!funcOp || !pto::isPTOEntryFunction(funcOp))
        return emitOpError() << "requires an enclosing ordinary AICore entry function";
    return success();
}

// NOLINTNEXTLINE(readability-make-member-function-const): ODS-generated side-effect callback has a non-const signature.
void SessionInitOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
    // The session is a result now, so the use-def edge to each post is what keeps
    // this ahead of them; there is no session operand to mark written. A session
    // nobody uses is genuinely dead and may be dropped.
    effects.emplace_back(MemoryEffects::Read::get(), &getTemplateGmMutable());
}
