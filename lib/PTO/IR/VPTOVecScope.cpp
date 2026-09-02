// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOVecScope.cpp - VPTO vecscope verifiers --------------------------===//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <optional>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VPTOMemoryDist.h"
#include "PTO/Support/CodeConstants.h"


#include "VPTOInternal.h"

using namespace mlir;
using namespace mlir::pto;

LogicalResult VecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0) {
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();
  }

  if (Operation *boundaryOp = findForbiddenSyncInRegion(bodyRegion)) {
    return boundaryOp->emitOpError()
           << "must be outside 'pto.vecscope'; synchronization operations "
              "delimit vector scopes";
  }

  return success();
}

LogicalResult StrictVecScopeOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != getCaptures().size()) {
    return emitOpError() << "expects body block to have "
                         << getCaptures().size()
                         << " arguments to match explicit captures, got "
                         << body.getNumArguments();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(body.getArguments(), getCaptures()))) {
    BlockArgument blockArg = std::get<0>(pair);
    Value capture = std::get<1>(pair);
    if (blockArg.getType() != capture.getType()) {
      return emitOpError() << "expects body block argument #" << idx
                           << " to have type " << capture.getType()
                           << ", got " << blockArg.getType();
    }
  }

  if (Operation *boundaryOp = findForbiddenSyncInRegion(bodyRegion)) {
    return boundaryOp->emitOpError()
           << "must be outside 'pto.strict_vecscope'; synchronization "
              "operations delimit vector scopes";
  }
  return success();
}

