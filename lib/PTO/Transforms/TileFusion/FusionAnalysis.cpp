// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {

namespace {

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Type type) {
  if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
    return SmallVector<int64_t, 4>(tileType.getValidShape().begin(),
                                   tileType.getValidShape().end());
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return SmallVector<int64_t, 4>(shapedType.getShape().begin(),
                                   shapedType.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  SmallVector<int64_t, 4> validShape = getValidShapeVec(value.getType());
  if (auto alloc = value.getDefiningOp<pto::AllocTileOp>()) {
    if (validShape.size() >= 1 && alloc.getValidRow())
      validShape[0] = getConstantIndexOrDynamic(alloc.getValidRow());
    if (validShape.size() >= 2 && alloc.getValidCol())
      validShape[1] = getConstantIndexOrDynamic(alloc.getValidCol());
  } else if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (validShape.size() >= 1 && bind.getValidRow())
      validShape[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (validShape.size() >= 2 && bind.getValidCol())
      validShape[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return validShape;
}

struct Rank2IterationSpace {
  int64_t rows = ShapedType::kDynamic;
  int64_t cols = ShapedType::kDynamic;
};

static std::optional<Rank2IterationSpace> getRank2IterationSpace(Value value) {
  SmallVector<int64_t, 4> validShape = getValidShapeVec(value);
  if (validShape.size() < 2)
    return std::nullopt;
  return Rank2IterationSpace{validShape[0], validShape[1]};
}

static void mergeIterationDim(int64_t &mergedDim, int64_t dim,
                              IterationDomainInfo &info) {
  if (mergedDim == ShapedType::kDynamic || dim == ShapedType::kDynamic) {
    mergedDim = ShapedType::kDynamic;
    if (info.unprovenReason == IterationDomainUnprovenReason::None)
      info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
    return;
  }

  if (mergedDim != dim) {
    mergedDim = ShapedType::kDynamic;
    info.unprovenReason = IterationDomainUnprovenReason::InconsistentShape;
  }
}

static IterationDomainInfo
inferConsensusIterationDomain(ArrayRef<Value> anchorValues) {
  IterationDomainInfo info;
  info.unprovenReason = IterationDomainUnprovenReason::None;

  if (anchorValues.empty())
    return info;

  std::optional<Rank2IterationSpace> firstSpace =
      getRank2IterationSpace(anchorValues.front());
  if (!firstSpace)
    return info;

  info.vRow = firstSpace->rows;
  info.vCol = firstSpace->cols;

  if (info.vRow == ShapedType::kDynamic || info.vCol == ShapedType::kDynamic)
    info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;

  for (Value value : ArrayRef<Value>(anchorValues).drop_front()) {
    std::optional<Rank2IterationSpace> space = getRank2IterationSpace(value);
    if (!space) {
      info.vRow = ShapedType::kDynamic;
      info.vCol = ShapedType::kDynamic;
      info.unprovenReason = IterationDomainUnprovenReason::MissingTileDomain;
      return info;
    }
    mergeIterationDim(info.vRow, space->rows, info);
    mergeIterationDim(info.vCol, space->cols, info);
  }

  if (info.unprovenReason == IterationDomainUnprovenReason::None &&
      info.vRow != ShapedType::kDynamic && info.vCol != ShapedType::kDynamic) {
    info.proof = IterationDomainProof::Proven;
    return info;
  }

  if (info.unprovenReason == IterationDomainUnprovenReason::None)
    info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
  return info;
}

static IterationDomainInfo
inferIterationDomainInfo(const FusionOpSemantics &semantics) {
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise: {
    SmallVector<Value, 6> anchors;
    anchors.append(semantics.tileInputs.begin(), semantics.tileInputs.end());
    anchors.append(semantics.tileOutputs.begin(), semantics.tileOutputs.end());
    return inferConsensusIterationDomain(anchors);
  }
  case FusionComputeFamily::ScalarExpand:
  case FusionComputeFamily::RowBroadcastBinary:
    return inferConsensusIterationDomain(semantics.tileOutputs);
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol:
    return inferConsensusIterationDomain(semantics.tileInputs);
  case FusionComputeFamily::Unknown:
    return IterationDomainInfo();
  }
  return IterationDomainInfo();
}

static unsigned assignIterationDomainClass(
    SmallVectorImpl<IterationDomainClass> &classes,
    DenseMap<std::pair<int64_t, int64_t>, unsigned> &provenClassByKey,
    const IterationDomainInfo &info, unsigned nodeId) {
  if (info.proof == IterationDomainProof::Proven) {
    std::pair<int64_t, int64_t> key{info.vRow, info.vCol};
    auto it = provenClassByKey.find(key);
    if (it != provenClassByKey.end()) {
      classes[it->second].members.push_back(nodeId);
      return it->second;
    }

    unsigned classId = classes.size();
    IterationDomainClass klass;
    klass.id = classId;
    klass.info = info;
    klass.members.push_back(nodeId);
    classes.push_back(std::move(klass));
    provenClassByKey.try_emplace(key, classId);
    return classId;
  }

  unsigned classId = classes.size();
  IterationDomainClass klass;
  klass.id = classId;
  klass.info = info;
  klass.members.push_back(nodeId);
  classes.push_back(std::move(klass));
  return classId;
}

struct MutableLiveness {
  FusionValueLiveness live;
};

struct MutableWriteInstance {
  FusionWriteInstanceLiveness live;
  unsigned producerBlockOrder = 0;
};

static FusionWriteInstanceEscapeClass classifyEscapeClass(
    const FusionWriteInstanceLiveness &live) {
  if (live.hasExternalUsers || live.escapesBlock ||
      live.hasLocalHardBoundaryUsers) {
    return FusionWriteInstanceEscapeClass::HardExternal;
  }
  if (live.hasLocalBoundaryUsers)
    return FusionWriteInstanceEscapeClass::LocalBoundaryExternal;
  return FusionWriteInstanceEscapeClass::Internal;
}

static Value getWriteInstanceStorageValue(Operation *op, unsigned outputIndex,
                                          Value output) {
  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    unsigned tileOutputIndex = 0;
    for (Value init : dpsIface.getDpsInits()) {
      if (!isa<pto::TileBufType>(init.getType()))
        continue;
      if (tileOutputIndex == outputIndex)
        return init;
      ++tileOutputIndex;
    }
  }
  return output;
}

static unsigned getOrCreateLivenessSlot(DenseMap<Value, unsigned> &slotByValue,
                                        SmallVectorImpl<MutableLiveness> &slots,
                                        Value value) {
  auto [it, inserted] = slotByValue.try_emplace(value, slots.size());
  if (inserted) {
    MutableLiveness state;
    state.live.value = value;
    slots.push_back(std::move(state));
  }
  return it->second;
}

static void appendUniqueNode(SmallVectorImpl<unsigned> &nodes, unsigned nodeId) {
  if (!llvm::is_contained(nodes, nodeId))
    nodes.push_back(nodeId);
}

static void recordLastLocalConsumer(std::optional<unsigned> &lastLocalConsumer,
                                    unsigned consumerId) {
  if (!lastLocalConsumer || consumerId > *lastLocalConsumer)
    lastLocalConsumer = consumerId;
}

static void finalizeBlockLiveness(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    SmallVectorImpl<MutableLiveness> &mutableLiveness) {
  for (MutableLiveness &state : mutableLiveness) {
    for (OpOperand &use : state.live.value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getBlock() != &block) {
        state.live.hasExternalUsers = true;
        state.live.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end())
        continue;

      if (user->hasTrait<OpTrait::IsTerminator>())
        state.live.escapesBlock = true;

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end())
          continue;
        unsigned consumerId = nodeIt->second;
        appendUniqueNode(state.live.consumerNodes, consumerId);
        recordLastLocalConsumer(state.live.lastLocalConsumer, consumerId);
        break;
      }
      case FusionOpKind::LocalBoundary:
        state.live.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        state.live.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }
}

static std::optional<unsigned> findReachingWriteInstance(
    ArrayRef<unsigned> writeInstanceIds,
    ArrayRef<MutableWriteInstance> mutableWriteInstances,
    std::optional<unsigned> userBlockOrder) {
  if (writeInstanceIds.empty())
    return std::nullopt;

  if (!userBlockOrder)
    return writeInstanceIds.back();

  for (unsigned writeInstanceId : llvm::reverse(writeInstanceIds)) {
    if (mutableWriteInstances[writeInstanceId].producerBlockOrder <
        *userBlockOrder)
      return writeInstanceId;
  }
  return std::nullopt;
}

static bool isDpsInitOperandUse(OpOperand &use) {
  auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(use.getOwner());
  if (!dpsIface)
    return false;

  for (OpOperand &dpsInit : dpsIface.getDpsInitsMutable())
    if (&dpsInit == &use)
      return true;
  return false;
}

static void finalizeWriteInstances(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    DenseMap<Operation *, unsigned> &blockOrderByOp,
    ArrayRef<MutableLiveness> mutableLiveness,
    SmallVectorImpl<MutableWriteInstance> &mutableWriteInstances) {
  for (const MutableLiveness &storageState : mutableLiveness) {
    if (storageState.live.writeInstances.empty())
      continue;

    for (OpOperand &use : storageState.live.value.getUses()) {
      if (isDpsInitOperandUse(use))
        continue;

      Operation *user = use.getOwner();
      bool isInBlock = user->getBlock() == &block;
      std::optional<unsigned> userBlockOrder;
      if (isInBlock) {
        auto orderIt = blockOrderByOp.find(user);
        if (orderIt != blockOrderByOp.end())
          userBlockOrder = orderIt->second;
      }

      std::optional<unsigned> writeInstanceId = findReachingWriteInstance(
          storageState.live.writeInstances, mutableWriteInstances,
          userBlockOrder);
      if (!writeInstanceId)
        continue;

      FusionWriteInstanceLiveness &writeLive =
          mutableWriteInstances[*writeInstanceId].live;

      if (!isInBlock) {
        writeLive.hasExternalUsers = true;
        writeLive.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end())
        continue;

      if (user->hasTrait<OpTrait::IsTerminator>())
        writeLive.escapesBlock = true;

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end())
          continue;
        unsigned consumerId = nodeIt->second;
        appendUniqueNode(writeLive.consumerNodes, consumerId);
        recordLastLocalConsumer(writeLive.lastLocalConsumer, consumerId);
        break;
      }
      case FusionOpKind::LocalBoundary:
        writeLive.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        writeLive.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }

  for (MutableWriteInstance &state : mutableWriteInstances)
    state.live.escapeClass = classifyEscapeClass(state.live);
}

static FailureOr<FusionBlockAnalysis> analyzeBlock(Block &block) {
  FusionBlockAnalysis analysis;
  analysis.block = &block;

  DenseMap<Value, unsigned> producerByValue;
  DenseMap<Value, unsigned> livenessSlotByValue;
  SmallVector<MutableLiveness, 8> mutableLiveness;
  SmallVector<MutableWriteInstance, 8> mutableWriteInstances;
  DenseMap<Operation *, FusionOpKind> kindByOp;
  DenseMap<Operation *, unsigned> computeNodeByOp;
  DenseMap<Operation *, unsigned> blockOrderByOp;
  DenseMap<std::pair<int64_t, int64_t>, unsigned> provenClassByKey;

  unsigned blockOrder = 0;
  for (Operation &op : block) {
    FailureOr<FusionOpSemantics> semanticsOr = getFusionOpSemantics(&op);
    if (failed(semanticsOr)) {
      op.emitError("failed to normalize fusion op semantics");
      return failure();
    }
    blockOrderByOp[&op] = blockOrder;
    kindByOp[&op] = semanticsOr->kind;

    if (semanticsOr->kind == FusionOpKind::LocalBoundary) {
      for (Value input : semanticsOr->tileInputs)
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      for (Value output : semanticsOr->tileOutputs)
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      ++blockOrder;
      continue;
    }

    if (semanticsOr->kind != FusionOpKind::Compute) {
      ++blockOrder;
      continue;
    }

    FusionComputeNode node;
    node.id = analysis.computeNodes.size();
    node.blockOrder = blockOrder;
    node.op = &op;
    node.semantics = *semanticsOr;
    computeNodeByOp[&op] = node.id;

    IterationDomainInfo domainInfo = inferIterationDomainInfo(node.semantics);
    node.iterationDomainClass = assignIterationDomainClass(
        analysis.iterationDomainClasses, provenClassByKey, domainInfo, node.id);

    for (auto [outputIdx, output] : llvm::enumerate(node.semantics.tileOutputs)) {
      producerByValue[output] = node.id;
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      mutableLiveness[liveSlot].live.producerNode = node.id;

      MutableWriteInstance writeInstance;
      writeInstance.live.id = mutableWriteInstances.size();
      writeInstance.live.value = output;
      writeInstance.live.storageValue =
          getWriteInstanceStorageValue(&op, outputIdx, output);
      writeInstance.live.producerNode = node.id;
      writeInstance.producerBlockOrder = blockOrder;
      mutableLiveness[liveSlot].live.writeInstances.push_back(
          writeInstance.live.id);
      mutableWriteInstances.push_back(std::move(writeInstance));
    }

    for (Value input : node.semantics.tileInputs) {
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      appendUniqueNode(mutableLiveness[liveSlot].live.consumerNodes, node.id);
      recordLastLocalConsumer(mutableLiveness[liveSlot].live.lastLocalConsumer,
                              node.id);

      auto producerIt = producerByValue.find(input);
      if (producerIt == producerByValue.end())
        continue;

      FusionDFGEdge edge;
      edge.producerNode = producerIt->second;
      edge.consumerNode = node.id;
      edge.value = input;

      unsigned edgeId = analysis.edges.size();
      analysis.edges.push_back(edge);
      node.incomingEdges.push_back(edgeId);
      if (edge.producerNode < analysis.computeNodes.size())
        analysis.computeNodes[edge.producerNode].outgoingEdges.push_back(edgeId);
    }

    analysis.computeNodes.push_back(std::move(node));
    ++blockOrder;
  }

  finalizeBlockLiveness(block, kindByOp, computeNodeByOp, mutableLiveness);
  finalizeWriteInstances(block, kindByOp, computeNodeByOp, blockOrderByOp,
                         mutableLiveness, mutableWriteInstances);

  analysis.liveness.reserve(mutableLiveness.size());
  for (MutableLiveness &state : mutableLiveness)
    analysis.liveness.push_back(std::move(state.live));
  analysis.writeInstances.reserve(mutableWriteInstances.size());
  for (MutableWriteInstance &state : mutableWriteInstances)
    analysis.writeInstances.push_back(std::move(state.live));

  return std::move(analysis);
}

static LogicalResult analyzeRegion(Region &region,
                                   SmallVectorImpl<FusionBlockAnalysis> &blocks) {
  for (Block &block : region.getBlocks()) {
    FailureOr<FusionBlockAnalysis> blockAnalysis = analyzeBlock(block);
    if (failed(blockAnalysis))
      return failure();
    blocks.push_back(std::move(*blockAnalysis));
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        if (failed(analyzeRegion(nested, blocks)))
          return failure();
  }
  return success();
}

} // namespace

FailureOr<PreFusionAnalysisResult> buildPreFusionAnalysis(func::FuncOp func) {
  PreFusionAnalysisResult result;
  if (failed(analyzeRegion(func.getRegion(), result.blocks)))
    return failure();
  return std::move(result);
}

} // namespace pto
} // namespace mlir
