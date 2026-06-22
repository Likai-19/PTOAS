// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILocalRecipeRegistry.h - VMI local recipe queries ------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILOCALRECIPEREGISTRY_H
#define PTO_TRANSFORMS_VMILOCALRECIPEREGISTRY_H

#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir::pto {

class VMITargetCapabilityRegistry;

enum class VMIContiguousStoreRecipeKind {
  ContiguousVsts,
  Deinterleaved2Vstsx2,
  DeinterleavedMaterializeThenVsts,
};

struct VMIContiguousStoreRecipe {
  VMIContiguousStoreRecipeKind kind =
      VMIContiguousStoreRecipeKind::ContiguousVsts;
};

enum class VMILayoutMaterializationRecipeKind {
  Identity,
  ContiguousToDeinterleaved,
  DeinterleavedToContiguous,
};

struct VMILayoutMaterializationRecipe {
  VMILayoutMaterializationRecipeKind kind =
      VMILayoutMaterializationRecipeKind::Identity;
};

enum class VMIMaskGranularityMaterializationRecipeKind {
  Identity,
  PredicateCast,
};

struct VMIMaskGranularityMaterializationRecipe {
  VMIMaskGranularityMaterializationRecipeKind kind =
      VMIMaskGranularityMaterializationRecipeKind::Identity;
};

enum class VMIGroupSlotLoadRecipeKind {
  Slots8UnitStrideVsldb,
  Slots1AlignedLane0Vsldb,
};

struct VMIGroupSlotLoadRecipe {
  VMIGroupSlotLoadRecipeKind kind =
      VMIGroupSlotLoadRecipeKind::Slots8UnitStrideVsldb;
};

enum class VMIGroupLoadRecipeKind {
  S16Block8Vsldb,
  S32Block8Vsldb,
};

struct VMIGroupLoadRecipe {
  VMIGroupLoadRecipeKind kind = VMIGroupLoadRecipeKind::S16Block8Vsldb;
};

enum class VMIGroupSlotsStoreRecipeKind {
  Slots8UnitStrideVsts,
  Slots1AlignedLane0Vsts,
};

struct VMIGroupSlotsStoreRecipe {
  VMIGroupSlotsStoreRecipeKind kind =
      VMIGroupSlotsStoreRecipeKind::Slots8UnitStrideVsts;
};

enum class VMIGroupReduceAddFRecipeKind {
  S8Vcgadd,
  S16Deinterleaved2VcgaddVadd,
  S32Deinterleaved4VcgaddTree,
  S64ContiguousVcaddRows,
};

struct VMIGroupReduceAddFRecipe {
  VMIGroupReduceAddFRecipeKind kind = VMIGroupReduceAddFRecipeKind::S8Vcgadd;
};

enum class VMIGroupBroadcastRecipeKind {
  GroupSlotsVselr,
};

struct VMIGroupBroadcastRecipe {
  VMIGroupBroadcastRecipeKind kind =
      VMIGroupBroadcastRecipeKind::GroupSlotsVselr;
};

enum class VMITruncFRecipeKind {
  Deinterleaved2F32ToContiguousF16,
  Deinterleaved4F32ToContiguousF8,
  GroupSlots1F32ToF16,
};

struct VMITruncFRecipe {
  VMITruncFRecipeKind kind =
      VMITruncFRecipeKind::Deinterleaved2F32ToContiguousF16;
};

enum class VMIExtFRecipeKind {
  ContiguousF16ToDeinterleaved2F32,
  ContiguousF8ToDeinterleaved4F32,
};

struct VMIExtFRecipe {
  VMIExtFRecipeKind kind =
      VMIExtFRecipeKind::ContiguousF16ToDeinterleaved2F32;
};

enum class VMIBitcastRecipeKind {
  PerPartVbitcast,
};

struct VMIBitcastRecipe {
  VMIBitcastRecipeKind kind = VMIBitcastRecipeKind::PerPartVbitcast;
};

class VMILocalRecipeRegistry {
public:
  FailureOr<VMIContiguousStoreRecipe>
  getContiguousStoreRecipe(VMIVRegType valueType,
                           std::string *reason = nullptr) const;

  LogicalResult canFoldContiguousStoreMaterialization(
      VMIVRegType sourceType, VMIVRegType resultType,
      std::string *reason = nullptr) const;

  FailureOr<VMILayoutMaterializationRecipe>
  getDataLayoutMaterializationRecipe(VMIVRegType sourceType,
                                     VMIVRegType resultType,
                                     std::string *reason = nullptr) const;

  FailureOr<VMILayoutMaterializationRecipe>
  getMaskLayoutMaterializationRecipe(VMIMaskType sourceType,
                                     VMIMaskType resultType,
                                     std::string *reason = nullptr) const;

  FailureOr<VMIMaskGranularityMaterializationRecipe>
  getMaskGranularityMaterializationRecipe(VMIMaskType sourceType,
                                          VMIMaskType resultType,
                                          std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLoadRecipe>
  getGroupSlotLoadRecipe(const VMITargetCapabilityRegistry &capabilities,
                         VMIGroupSlotLoadOp op,
                         std::string *reason = nullptr) const;

  FailureOr<VMIGroupLoadRecipe>
  getGroupLoadRecipe(const VMITargetCapabilityRegistry &capabilities,
                     VMIGroupLoadOp op,
                     std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotsStoreRecipe>
  getGroupSlotsStoreRecipe(const VMITargetCapabilityRegistry &capabilities,
                           VMIGroupStoreOp op,
                           std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceAddFRecipe>
  getGroupReduceAddFRecipe(const VMITargetCapabilityRegistry &capabilities,
                           VMIGroupReduceAddFOp op,
                           std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastRecipe>
  getGroupBroadcastRecipe(const VMITargetCapabilityRegistry &capabilities,
                          VMIGroupBroadcastOp op,
                          std::string *reason = nullptr) const;

  FailureOr<VMITruncFRecipe>
  getTruncFRecipe(VMITruncFOp op, std::string *reason = nullptr) const;

  FailureOr<VMIExtFRecipe>
  getExtFRecipe(VMIExtFOp op, std::string *reason = nullptr) const;

  FailureOr<VMIBitcastRecipe>
  getBitcastRecipe(VMIBitcastOp op, std::string *reason = nullptr) const;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILOCALRECIPEREGISTRY_H
