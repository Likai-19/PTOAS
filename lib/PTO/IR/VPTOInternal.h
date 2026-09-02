// Internal shared helpers for VPTO split files.
// This header is internal to lib/PTO/IR and not installed.

#ifndef PTO_IR_VPTO_INTERNAL_H
#define PTO_IR_VPTO_INTERNAL_H

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


using namespace mlir;
using namespace mlir::pto;

extern llvm::cl::opt<bool> disableVPTOAlignChainVerification;

LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                  StringRef roleDescription);
LogicalResult verifyStoreAlignChain(Value align, Operation *user,
                                    StringRef roleDescription);
LogicalResult verifyLoadAlignChain(Value align, Operation *user,
                                   StringRef roleDescription);


enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

[[maybe_unused]] static MemoryRole classifyMemoryRole(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      switch (ptrType.getMemorySpace().getAddressSpace()) {
      case pto::AddressSpace::GM:
      case pto::AddressSpace::Zero:
        return MemoryRole::GM;
      case pto::AddressSpace::VEC:
        return MemoryRole::UB;
      default:
        return MemoryRole::Other;
      }
    }
    return MemoryRole::Other;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (!memorySpace) {
    return MemoryRole::Unknown;
  }

  if (auto addrSpace = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return MemoryRole::GM;
    case pto::AddressSpace::VEC:
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(pto::AddressSpace::GM):
    case static_cast<int64_t>(pto::AddressSpace::Zero):
      return MemoryRole::GM;
    case static_cast<int64_t>(pto::AddressSpace::VEC):
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  return MemoryRole::Other;
}

[[maybe_unused]] static bool isBufferLike(Type type) {
  return isa<BaseMemRefType, pto::PtrType>(type);
}

[[maybe_unused]] static bool isForbiddenSynchronizationInsideVecScope(Operation *op) {
  // High-level synchronization is forbidden before and after lowering.
  if (isa<RecordEventOp, WaitEventOp, BarrierSyncOp>(op)) {
    return true;
  }

  // Intra-core pipeline and buffer-id synchronization executes outside the
  // vector interval that it orders.
  if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, GetBufOp,
          GetBufDynOp, RlsBufOp, RlsBufDynOp, BarrierOp>(op)) {
    return true;
  }

  // Intra-block, cross-core, system, cache, and SIMT synchronization likewise
  // delimit vector intervals. MemBarOp is intentionally not in this list.
  return isa<SyncSetOp, SyncWaitOp, CmoCacheInvalidOp, FenceBarrierAllOp,
             TSyncOp, SyncAllOp, DsbOp, DcciOp, SyncthreadsOp, ThreadfenceOp,
             ThreadfenceBlockOp>(op);
}

[[maybe_unused]] static Operation *findForbiddenSyncInRegion(Region &body) {
  Operation *boundaryOp = nullptr;
  body.walk([&](Operation *op) {
    if (!isForbiddenSynchronizationInsideVecScope(op)) {
      return WalkResult::advance();
    }
    boundaryOp = op;
    return WalkResult::interrupt();
  });
  return boundaryOp;
}


#endif // PTO_IR_VPTO_INTERNAL_H
