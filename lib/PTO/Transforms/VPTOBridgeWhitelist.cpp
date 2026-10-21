// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// The YAML file selects routes only. ABI and wrapper details belong to the registry.
#include "PTO/Transforms/VPTOBridgeWhitelist.h"
#include "PTO/IR/VPTOBridgeRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/YAMLTraits.h"

using namespace mlir::pto;
using mlir::FailureOr;

namespace llvm::yaml {
template <> struct MappingTraits<BridgeFamilyPolicy> {
  static void mapping(IO &io, BridgeFamilyPolicy &value) {
    io.mapOptional("enabled", value.enabled, false);
    io.mapOptional("enabled_ops", value.enabledOps);
  }
};

template <> struct MappingTraits<BridgePolicyFamilies> {
  static void mapping(IO &io, BridgePolicyFamilies &value) {
    io.mapOptional("pipe", value.pipe);
    io.mapOptional("cube", value.cube);
    io.mapOptional("print", value.print);
  }
};

template <> struct MappingTraits<BridgeRoutePolicy> {
  static void mapping(IO &io, BridgeRoutePolicy &value) {
    io.mapRequired("version", value.version);
    io.mapRequired("families", value.families);
  }
};
} // namespace llvm::yaml

namespace {
// Enabled by default: the families whose whole chain (typed lowering,
// renderer, wrapper compile and link) is complete and covered by ptoas-level
// tests. Print has no non-bridge VPTO lowering at all, so routing it is what
// makes printing work. Cube keeps its ExpandTileOp path by default; selecting
// a cube op moves it onto the bridge.
constexpr llvm::StringLiteral kDefaultPolicy = R"yaml(version: 1
families:
  pipe: { enabled: true }
  cube: { enabled_ops: [] }
  print: { enabled_ops: [pto.print, pto.tprint] }
)yaml";

bool validateOps(const BridgeFamilyPolicy &family, BridgeFamily kind,
                 llvm::raw_ostream &diagOS) {
  llvm::StringSet<> seen;
  for (const std::string &name : family.enabledOps) {
    const BridgeFunctionDesc *entry = findBridgeFunctionByOp(name);
    const bool knownCube = kind == BridgeFamily::Cube && entry &&
                           entry->family == BridgeFamily::Cube;
    const bool knownPrint = kind == BridgeFamily::Print &&
                            (name == "pto.print" || name == "pto.tprint");
    if (!seen.insert(name).second || (!knownCube && !knownPrint)) {
      diagOS << "VPTO bridge policy: invalid or duplicate "
             << stringifyBridgeFamily(kind) << " op '" << name << "'\n";
      return false;
    }
  }
  return true;
}
} // namespace

bool BridgeRoutePolicy::routesFamily(llvm::StringRef family) const {
  return family == "pipe" && families.pipe.enabled;
}

bool BridgeRoutePolicy::routesOp(llvm::StringRef family,
                                 llvm::StringRef opName) const {
  if (opName.empty()) {
    return false;
  }
  if (family == "pipe") {
    const BridgeFunctionDesc *entry = findBridgeFunctionByOp(opName);
    return families.pipe.enabled && entry && entry->family == BridgeFamily::Pipe;
  }
  if (family == "cube") {
    const BridgeFunctionDesc *entry = findBridgeFunctionByOp(opName);
    return entry && entry->family == BridgeFamily::Cube &&
           llvm::is_contained(families.cube.enabledOps, opName);
  }
  if (family == "print") {
    return llvm::is_contained(families.print.enabledOps, opName);
  }
  return false;
}

FailureOr<BridgeRoutePolicy> mlir::pto::parseBridgeRoutePolicyFromBuffer(
    llvm::StringRef content, llvm::StringRef sourceName,
    llvm::raw_ostream &diagOS) {
  BridgeRoutePolicy policy;
  llvm::yaml::Input input(content);
  input.setAllowUnknownKeys(false);
  input >> policy;
  if (input.error()) {
    diagOS << "VPTO bridge policy: cannot parse '" << sourceName << "'\n";
    return failure();
  }
  if (policy.version != 1) {
    diagOS << "VPTO bridge policy: unsupported version in '" << sourceName << "'\n";
    return failure();
  }
  const bool hasInvalidFamilySettings =
      !policy.families.pipe.enabledOps.empty() || policy.families.cube.enabled ||
      policy.families.print.enabled;
  if (hasInvalidFamilySettings) {
    diagOS << "VPTO bridge policy: pipe uses enabled; cube and print use enabled_ops\n";
    return failure();
  }
  const bool validCube =
      validateOps(policy.families.cube, BridgeFamily::Cube, diagOS);
  const bool validPrint =
      validateOps(policy.families.print, BridgeFamily::Print, diagOS);
  if (!validCube || !validPrint) {
    return failure();
  }
  return policy;
}

FailureOr<BridgeRoutePolicy> mlir::pto::loadBridgeRoutePolicy(
    llvm::StringRef optionValue, llvm::raw_ostream &diagOS,
    std::string *sourceName) {
  constexpr llvm::StringLiteral defaultName = "<built-in vpto bridge policy>";
  if (sourceName) {
    *sourceName = optionValue.empty() ? defaultName.str() : optionValue.str();
  }
  if (optionValue.empty()) {
    return parseBridgeRoutePolicyFromBuffer(kDefaultPolicy, defaultName, diagOS);
  }
  auto buffer = llvm::MemoryBuffer::getFile(optionValue);
  if (!buffer) {
    diagOS << "VPTO bridge policy: cannot read '" << optionValue << "'\n";
    return failure();
  }
  return parseBridgeRoutePolicyFromBuffer(buffer.get()->getBuffer(),
                                          optionValue, diagOS);
}
