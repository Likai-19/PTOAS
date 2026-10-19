// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_VPTOEXPANDASYNCCOMM_H
#define PTO_TRANSFORMS_VPTOEXPANDASYNCCOMM_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto::expand_async {

// Generation that owns the SQE / doorbell path for async GM↔GM posts.
enum class DmaArch { A2A3, A5 };

// Lower every async session in one ordered walk: materialize each
// `pto.session_init` into its backing struct, expand each `pto.sdma_gm_gm`
// against that struct, then erase the init ops. Must run ahead of the greedy
// wrapper patterns because a session changes type across this expansion.
LogicalResult lowerAsyncSessions(func::FuncOp func, DmaArch dmaArch);

} // namespace mlir::pto::expand_async

#endif // PTO_TRANSFORMS_VPTOEXPANDASYNCCOMM_H
