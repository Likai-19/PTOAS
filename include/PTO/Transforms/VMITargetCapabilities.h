// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMITargetCapabilities.h - VMI target capability registry -*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMITARGETCAPABILITIES_H
#define PTO_TRANSFORMS_VMITARGETCAPABILITIES_H

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Twine.h"

#include <string>

namespace mlir::pto {

enum class VMICapabilityStatus {
  supported,
  unsupported_missing_capability,
  unsupported_disabled_by_option,
  unsupported_resource,
};

enum class VMIElementPurpose {
  PredicateMask,
  F16F32,
  F16BF16F32,
  SignlessOrSignedI8I16I32,
  AnyI8I16I32,
  VMula,
  VRelu,
};

enum class VMIReductionKind {
  AddI,
  AddF,
  GroupAddI,
  GroupAddF,
  GroupMaxF,
  MaxF,
  MinF,
};

enum class VMIFallbackResourceKind {
  ScratchMemory,
  GuardedControlFlow,
};

struct VMICapabilityResult {
  VMICapabilityStatus status = VMICapabilityStatus::supported;
  std::string reason;

  static VMICapabilityResult supported() { return {}; }

  static VMICapabilityResult missingCapability(const Twine &reason) {
    VMICapabilityResult result;
    result.status = VMICapabilityStatus::unsupported_missing_capability;
    result.reason = reason.str();
    return result;
  }

  bool isSupported() const { return status == VMICapabilityStatus::supported; }

  LogicalResult toLogicalResult(std::string *outReason = nullptr) const {
    if (isSupported())
      return success();
    if (outReason)
      *outReason = reason;
    return failure();
  }
};

class VMITargetCapabilityRegistry {
public:
  VMICapabilityResult supportsElementType(Type type,
                                          VMIElementPurpose purpose) const {
    switch (purpose) {
    case VMIElementPurpose::PredicateMask: {
      unsigned elementBits = pto::getPTOStorageElemBitWidth(type);
      if (elementBits == 8 || elementBits == 16 || elementBits == 32)
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "requires an 8/16/32-bit element type so VPTO b8/b16/b32 "
          "predicate masks can be materialized");
    }
    case VMIElementPurpose::F16F32:
      if (type.isF16() || type.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "requires f16/f32 element type for direct VPTO lowering");
    case VMIElementPurpose::F16BF16F32:
      if (type.isF16() || type.isBF16() || type.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "requires f16/bf16/f32 element type for direct VPTO lowering");
    case VMIElementPurpose::SignlessOrSignedI8I16I32:
      if (isSignlessOrSignedI8I16I32(type))
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "requires signless/signed i8/i16/i32 element type for direct VPTO "
          "lowering");
    case VMIElementPurpose::AnyI8I16I32:
      if (isAnyI8I16I32(type))
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "requires signless/signed/unsigned i8/i16/i32 element type for "
          "direct VPTO lowering");
    case VMIElementPurpose::VMula:
      if (type.isF16() || type.isBF16() || type.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "requires f16, bf16, or f32 element type for pto.vmula");
    case VMIElementPurpose::VRelu:
      if (type.isF16() || type.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "pto.vrelu direct lowering supports only f16/f32 VMI "
          "floating-point element types");
    }
    llvm_unreachable("unhandled VMI element purpose");
  }

  VMICapabilityResult supportsDirectMemory(Type type, StringRef role) const {
    switch (classifyDirectMemoryRole(type)) {
    case DirectMemoryRole::UB:
    case DirectMemoryRole::Unknown:
      return VMICapabilityResult::supported();
    case DirectMemoryRole::GM:
      return VMICapabilityResult::missingCapability(
          Twine(role) +
          " is GM-backed, but current direct VMI-to-VPTO memory lowering "
          "emits pto.vlds/pto.vsts and requires UB-backed memory");
    case DirectMemoryRole::Other:
      return VMICapabilityResult::missingCapability(
          Twine(role) +
          " is not UB-backed memory supported by pto.vlds/pto.vsts");
    }
    llvm_unreachable("unhandled direct memory role");
  }

  VMICapabilityResult supportsUBPointerMemory(Type type, StringRef role,
                                              StringRef physicalOp,
                                              StringRef ubReason) const {
    auto ptrType = dyn_cast<PtrType>(type);
    if (!ptrType)
      return VMICapabilityResult::missingCapability(
          Twine("requires a !pto.ptr ") + role + " because " + physicalOp +
          " is pointer-only");
    if (ptrType.getMemorySpace().getAddressSpace() != AddressSpace::VEC)
      return VMICapabilityResult::missingCapability(
          Twine("requires a UB ") + role + " because " + ubReason);
    return VMICapabilityResult::supported();
  }

  VMICapabilityResult supportsChannelCount(StringRef opName,
                                           int64_t channels) const {
    if (channels == 2 || channels == 4)
      return VMICapabilityResult::supported();
    return VMICapabilityResult::missingCapability(
        Twine(opName) + " supports only 2 or 4 channels");
  }

  VMICapabilityResult supportsLayoutConversion(VMILayoutAttr sourceLayout,
                                               VMILayoutAttr resultLayout,
                                               Type elementType) const {
    (void)elementType;
    if (!sourceLayout || !resultLayout)
      return VMICapabilityResult::missingCapability(
          "requires assigned source/result layouts");
    if (sourceLayout == resultLayout)
      return VMICapabilityResult::supported();
    if (sourceLayout.isContiguous() && resultLayout.isDeinterleaved() &&
        (resultLayout.getFactor() == 2 || resultLayout.getFactor() == 4))
      return VMICapabilityResult::supported();
    if (sourceLayout.isDeinterleaved() && resultLayout.isContiguous() &&
        (sourceLayout.getFactor() == 2 || sourceLayout.getFactor() == 4))
      return VMICapabilityResult::supported();
    return VMICapabilityResult::missingCapability(
        "unsupported source/result layout pair");
  }

  VMICapabilityResult
  supportsMaskGranularityConversion(StringRef sourceGranularity,
                                    StringRef resultGranularity) const {
    if (!VMIMaskType::isConcreteGranularity(sourceGranularity) ||
        !VMIMaskType::isConcreteGranularity(resultGranularity))
      return VMICapabilityResult::missingCapability(
          "requires concrete b8/b16/b32 source and result granularities");
    return VMICapabilityResult::supported();
  }

  VMICapabilityResult supportsTrueMaskedLoad(Type sourceType, Type resultType,
                                             Type maskType) const {
    (void)sourceType;
    (void)resultType;
    (void)maskType;
    return VMICapabilityResult::missingCapability(
        "target true masked/non-faulting load is unavailable because the "
        "current VPTO pto.vlds surface has no mask operand");
  }

  VMICapabilityResult
  supportsFallbackResource(VMIFallbackResourceKind kind) const {
    switch (kind) {
    case VMIFallbackResourceKind::ScratchMemory:
      return VMICapabilityResult::missingCapability(
          "scratch memory fallback resource allocation is not implemented");
    case VMIFallbackResourceKind::GuardedControlFlow:
      return VMICapabilityResult::missingCapability(
          "guarded memory fallback control-flow lowering is not implemented");
    }
    llvm_unreachable("unhandled VMI fallback resource kind");
  }

  VMICapabilityResult supportsReductionElementType(VMIReductionKind kind,
                                                   Type elementType) const {
    switch (kind) {
    case VMIReductionKind::AddI:
      if (pto::getPTOStorageElemBitWidth(elementType) == 32 &&
          isa<IntegerType>(elementType))
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "currently supports only 32-bit integer elements because narrow "
          "vcadd widens its result");
    case VMIReductionKind::AddF:
      if (elementType.isF16() || elementType.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "currently supports only f16/f32 elements for floating-point "
          "reduction");
    case VMIReductionKind::GroupAddI: {
      auto intType = dyn_cast<IntegerType>(elementType);
      if (intType && intType.getWidth() == 32)
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "grouped integer add reduction supports only i32 accumulator "
          "elements because narrow integer reductions widen their result; "
          "cast i8/i16 storage before grouped reduction");
    }
    case VMIReductionKind::GroupAddF:
    case VMIReductionKind::GroupMaxF:
      if (elementType.isF16() || elementType.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "grouped floating-point reduction supports f16/f32 accumulator "
          "elements");
    case VMIReductionKind::MaxF:
    case VMIReductionKind::MinF:
      if (elementType.isF16() || elementType.isF32())
        return VMICapabilityResult::supported();
      return VMICapabilityResult::missingCapability(
          "currently supports only f16/f32 elements because pto.vcmax/"
          "pto.vcmin support only those floating-point element types");
    }
    llvm_unreachable("unhandled VMI reduction kind");
  }

private:
  enum class DirectMemoryRole { Unknown, UB, GM, Other };

  DirectMemoryRole classifyDirectMemoryRole(Type type) const {
    if (auto ptrType = dyn_cast<PtrType>(type)) {
      switch (ptrType.getMemorySpace().getAddressSpace()) {
      case AddressSpace::GM:
      case AddressSpace::Zero:
        return DirectMemoryRole::GM;
      case AddressSpace::VEC:
        return DirectMemoryRole::UB;
      default:
        return DirectMemoryRole::Other;
      }
    }

    auto memrefType = dyn_cast<MemRefType>(type);
    if (!memrefType)
      return DirectMemoryRole::Other;

    Attribute memorySpace = memrefType.getMemorySpace();
    if (!memorySpace)
      return DirectMemoryRole::Unknown;

    if (auto addressSpace = dyn_cast<AddressSpaceAttr>(memorySpace)) {
      switch (addressSpace.getAddressSpace()) {
      case AddressSpace::GM:
      case AddressSpace::Zero:
        return DirectMemoryRole::GM;
      case AddressSpace::VEC:
        return DirectMemoryRole::UB;
      default:
        return DirectMemoryRole::Other;
      }
    }

    if (auto integerSpace = dyn_cast<IntegerAttr>(memorySpace)) {
      switch (integerSpace.getInt()) {
      case static_cast<int64_t>(AddressSpace::GM):
      case static_cast<int64_t>(AddressSpace::Zero):
        return DirectMemoryRole::GM;
      case static_cast<int64_t>(AddressSpace::VEC):
        return DirectMemoryRole::UB;
      default:
        return DirectMemoryRole::Other;
      }
    }

    return DirectMemoryRole::Other;
  }

  static bool isSignlessOrSignedI8I16I32(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType || intType.isUnsigned())
      return false;
    return intType.getWidth() == 8 || intType.getWidth() == 16 ||
           intType.getWidth() == 32;
  }

  static bool isAnyI8I16I32(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    if (!intType)
      return false;
    return intType.getWidth() == 8 || intType.getWidth() == 16 ||
           intType.getWidth() == 32;
  }
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMITARGETCAPABILITIES_H
