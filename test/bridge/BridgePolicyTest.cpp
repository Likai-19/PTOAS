// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/VPTOBridgeRegistry.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::pto;

namespace {
bool expectPolicy(llvm::StringRef yaml, bool valid) {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);
  auto policy = parseBridgeRoutePolicyFromBuffer(yaml, "test policy", stream);
  const bool parsedAsExpected = mlir::succeeded(policy) == valid;
  if (!parsedAsExpected) {
    llvm::errs() << "unexpected policy result: " << diagnostic << '\n';
    return false;
  }
  return true;
}
} // namespace

int main() {
  std::string diagnostic;
  llvm::raw_string_ostream stream(diagnostic);
  auto defaults = loadBridgeRoutePolicy("", stream);
  // Pipe and print are complete end to end, so the built-in default routes
  // them; a cube op keeps its ExpandTileOp path unless selected.
  const bool validDefaults = mlir::succeeded(defaults) &&
                             defaults->routesOp("pipe", "pto.tpush") &&
                             !defaults->routesOp("cube", "pto.tmatmul") &&
                             defaults->routesOp("print", "pto.print") &&
                             defaults->routesOp("print", "pto.tprint");
  if (!validDefaults) {
    llvm::errs() << "unexpected default routes: " << diagnostic << '\n';
    return 1;
  }
  constexpr llvm::StringLiteral valid = R"yaml(version: 1
families:
  pipe: { enabled: false }
  cube: { enabled_ops: [pto.tmatmul] }
  print: { enabled_ops: [pto.print, pto.tprint] }
)yaml";
  auto routed = parseBridgeRoutePolicyFromBuffer(valid, "test policy", stream);
  const bool validRoutes = mlir::succeeded(routed) &&
                           !routed->routesOp("pipe", "pto.tpush") &&
                           routed->routesOp("cube", "pto.tmatmul") &&
                           routed->routesOp("print", "pto.tprint") &&
                           !routed->routesOp("cube", "func.return") &&
                           !routed->routesOp("cube", "pto.print");
  if (!validRoutes) {
    llvm::errs() << "unexpected explicit routes: " << diagnostic << '\n';
    return 1;
  }
  const bool invalidPoliciesRejected =
      expectPolicy("version: 2\nfamilies: {}\n", false) &&
      expectPolicy("version: 1\nfamilies:\n  cube:\n    abi: i64\n", false) &&
      expectPolicy("version: 1\nfamilies:\n  cube:\n    enabled_ops: [pto.unknown]\n", false) &&
      expectPolicy("version: 1\nfamilies:\n  cube:\n    enabled_ops: [pto.tmatmul, pto.tmatmul]\n", false) &&
      expectPolicy("version: 1\nfamilies:\n  print:\n    enabled: true\n", false);
  if (!invalidPoliciesRejected) {
    return 1;
  }
  const BridgeFunctionDesc *matmul = findBridgeFunction(BridgeEntryId::CubeTMatmul);
  const bool validMatmul = matmul && matmul->arguments.size() == 3 &&
                           stringifyBridgeEntryId(matmul->id) == "cube.tmatmul";
  if (!validMatmul) {
    llvm::errs() << "unexpected cube registry ABI\n";
    return 1;
  }
  return 0;
}
