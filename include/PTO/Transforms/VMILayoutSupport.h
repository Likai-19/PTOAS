// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutSupport.h - VMI layout support queries ------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTSUPPORT_H
#define PTO_TRANSFORMS_VMILAYOUTSUPPORT_H

#include "PTO/IR/PTO.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir::pto {

class VMITargetCapabilityRegistry;

enum class VMIContiguousStoreSupportKind {
  ContiguousVsts,
  Deinterleaved2Vstsx2,
  DeinterleavedMaterializeThenVsts,
};

struct VMIContiguousStoreSupport {
  VMIContiguousStoreSupportKind kind =
      VMIContiguousStoreSupportKind::ContiguousVsts;
};

enum class VMILayoutMaterializationSupportKind {
  Identity,
  ContiguousToDeinterleaved,
  DeinterleavedToContiguous,
};

struct VMILayoutMaterializationSupport {
  VMILayoutMaterializationSupportKind kind =
      VMILayoutMaterializationSupportKind::Identity;
};

enum class VMIMaskGranularityMaterializationSupportKind {
  Identity,
  PredicateCast,
};

struct VMIMaskGranularityMaterializationSupport {
  VMIMaskGranularityMaterializationSupportKind kind =
      VMIMaskGranularityMaterializationSupportKind::Identity;
};

enum class VMICastLayoutKind {
  Widen2x,
  Widen4x,
  Narrow2x,
  Narrow4x,
};

struct VMICastLayoutFact {
  VMICastLayoutKind kind = VMICastLayoutKind::Widen2x;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t sourceBits = 0;
  int64_t resultBits = 0;
  int64_t factor = 1;
};

enum class VMIGroupSlotLoadSupportKind {
  Slots8UnitStrideVsldb,
  Slots1AlignedLane0Vsldb,
};

struct VMIGroupSlotLoadSupport {
  VMIGroupSlotLoadSupportKind kind =
      VMIGroupSlotLoadSupportKind::Slots8UnitStrideVsldb;
};

enum class VMIGroupLoadSupportKind {
  S16Block8Vsldb,
  S32Block8Vsldb,
};

struct VMIGroupLoadSupport {
  VMIGroupLoadSupportKind kind = VMIGroupLoadSupportKind::S16Block8Vsldb;
};

enum class VMIGroupSlotsStoreSupportKind {
  Slots8UnitStrideVsts,
  Slots1AlignedLane0Vsts,
};

struct VMIGroupSlotsStoreSupport {
  VMIGroupSlotsStoreSupportKind kind =
      VMIGroupSlotsStoreSupportKind::Slots8UnitStrideVsts;
};

enum class VMIGroupReduceLayoutKind {
  OneVLane,
  TwoVLane,
  FourVLane,
  RowLocal,
};

struct VMIGroupReduceLayoutFact {
  VMIGroupReduceLayoutKind kind = VMIGroupReduceLayoutKind::OneVLane;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  int64_t groupSize = 0;
  int64_t lanesPerPart = 0;
  int64_t vlaneElems = 0;
};

enum class VMIGroupReduceAddFSupportKind {
  OneVLaneVcgadd,
  TwoVLaneDeinterleaved2VcgaddVadd,
  FourVLaneDeinterleaved4VcgaddTree,
  ContiguousVcaddRows,
};

struct VMIGroupReduceAddFSupport {
  VMIGroupReduceAddFSupportKind kind =
      VMIGroupReduceAddFSupportKind::OneVLaneVcgadd;
};

enum class VMIGroupBroadcastSupportKind {
  GroupSlotsVselr,
};

struct VMIGroupBroadcastSupport {
  VMIGroupBroadcastSupportKind kind =
      VMIGroupBroadcastSupportKind::GroupSlotsVselr;
};

enum class VMITruncFSupportKind {
  Deinterleaved2F32ToContiguousF16,
  Deinterleaved4F32ToContiguousF8,
  GroupSlots1F32ToF16,
};

struct VMITruncFSupport {
  VMITruncFSupportKind kind =
      VMITruncFSupportKind::Deinterleaved2F32ToContiguousF16;
};

enum class VMIExtFSupportKind {
  ContiguousF16ToDeinterleaved2F32,
  ContiguousF8ToDeinterleaved4F32,
};

struct VMIExtFSupport {
  VMIExtFSupportKind kind =
      VMIExtFSupportKind::ContiguousF16ToDeinterleaved2F32;
};

enum class VMITruncISupportKind {
  Deinterleaved2I32ToContiguousI16,
  Deinterleaved4I32ToContiguousI8,
  GroupSlots1I32ToI16,
};

struct VMITruncISupport {
  VMITruncISupportKind kind =
      VMITruncISupportKind::Deinterleaved2I32ToContiguousI16;
};

enum class VMIExtISupportKind {
  ContiguousI16ToDeinterleaved2I32,
  ContiguousI8ToDeinterleaved4I32,
};

struct VMIExtISupport {
  VMIExtISupportKind kind =
      VMIExtISupportKind::ContiguousI16ToDeinterleaved2I32;
};

enum class VMIBitcastSupportKind {
  PerPartVbitcast,
};

struct VMIBitcastSupport {
  VMIBitcastSupportKind kind = VMIBitcastSupportKind::PerPartVbitcast;
};

enum class VMIHistogramSupportKind {
  Full256BinDhist,
};

struct VMIHistogramSupport {
  VMIHistogramSupportKind kind = VMIHistogramSupportKind::Full256BinDhist;
};

class VMILayoutSupport {
public:
  FailureOr<VMIContiguousStoreSupport>
  getContiguousStoreSupport(VMIVRegType valueType,
                            std::string *reason = nullptr) const;

  LogicalResult
  canFoldContiguousStoreMaterialization(VMIVRegType sourceType,
                                        VMIVRegType resultType,
                                        std::string *reason = nullptr) const;

  FailureOr<VMILayoutMaterializationSupport>
  getDataLayoutMaterializationSupport(VMIVRegType sourceType,
                                      VMIVRegType resultType,
                                      std::string *reason = nullptr) const;

  LogicalResult canMaterializeDataLayout(VMIVRegType sourceType,
                                         VMIVRegType resultType,
                                         std::string *reason = nullptr) const;

  FailureOr<VMILayoutMaterializationSupport>
  getMaskLayoutMaterializationSupport(VMIMaskType sourceType,
                                      VMIMaskType resultType,
                                      std::string *reason = nullptr) const;

  LogicalResult canMaterializeMaskLayout(VMIMaskType sourceType,
                                         VMIMaskType resultType,
                                         std::string *reason = nullptr) const;

  FailureOr<VMIMaskGranularityMaterializationSupport>
  getMaskGranularityMaterializationSupport(VMIMaskType sourceType,
                                           VMIMaskType resultType,
                                           std::string *reason = nullptr) const;

  LogicalResult
  canMaterializeMaskGranularity(VMIMaskType sourceType, VMIMaskType resultType,
                                std::string *reason = nullptr) const;

  FailureOr<VMICastLayoutFact>
  getPreferredCastLayoutFact(VMIVRegType sourceType, VMIVRegType resultType,
                             std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotLoadSupport>
  getGroupSlotLoadSupport(const VMITargetCapabilityRegistry &capabilities,
                          VMIGroupSlotLoadOp op,
                          std::string *reason = nullptr) const;

  FailureOr<VMIGroupLoadSupport>
  getGroupLoadSupport(const VMITargetCapabilityRegistry &capabilities,
                      VMIGroupLoadOp op, std::string *reason = nullptr) const;

  FailureOr<VMIGroupSlotsStoreSupport>
  getGroupSlotsStoreSupport(const VMITargetCapabilityRegistry &capabilities,
                            VMIGroupStoreOp op,
                            std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceLayoutFact>
  getPreferredGroupReduceLayoutFact(VMIVRegType sourceType, int64_t numGroups,
                                    std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceAddFSupport>
  getGroupReduceAddFSupport(const VMITargetCapabilityRegistry &capabilities,
                            VMIGroupReduceAddFOp op,
                            std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceAddFSupport>
  getGroupReduceMaxFSupport(const VMITargetCapabilityRegistry &capabilities,
                            VMIGroupReduceMaxFOp op,
                            std::string *reason = nullptr) const;

  FailureOr<VMIGroupReduceAddFSupport>
  getGroupReduceAddISupport(const VMITargetCapabilityRegistry &capabilities,
                            VMIGroupReduceAddIOp op,
                            std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastSupport>
  getGroupBroadcastSupport(const VMITargetCapabilityRegistry &capabilities,
                           VMIGroupBroadcastOp op,
                           std::string *reason = nullptr) const;

  FailureOr<VMIGroupBroadcastSupport>
  getGroupBroadcastSupport(const VMITargetCapabilityRegistry &capabilities,
                           VMIVRegType sourceType, VMIVRegType resultType,
                           int64_t numGroups,
                           std::string *reason = nullptr) const;

  FailureOr<VMITruncFSupport>
  getTruncFSupport(VMITruncFOp op, std::string *reason = nullptr) const;

  FailureOr<VMIExtFSupport> getExtFSupport(VMIExtFOp op,
                                           std::string *reason = nullptr) const;

  FailureOr<VMIExtISupport>
  getExtSISupport(VMIExtSIOp op, std::string *reason = nullptr) const;

  FailureOr<VMIExtISupport>
  getExtUISupport(VMIExtUIOp op, std::string *reason = nullptr) const;

  FailureOr<VMITruncISupport>
  getTruncISupport(VMITruncIOp op, std::string *reason = nullptr) const;

  FailureOr<VMIBitcastSupport>
  getBitcastSupport(VMIBitcastOp op, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramSupport>
  getDhistSupport(VMIDhistOp op, std::string *reason = nullptr) const;

  FailureOr<VMIHistogramSupport>
  getChistSupport(VMIChistOp op, std::string *reason = nullptr) const;
};

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTSUPPORT_H
