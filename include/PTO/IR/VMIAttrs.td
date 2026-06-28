// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIAttrs.td - PTO VMI attributes definitions --------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VMIATTRS
#define MLIR_DIALECT_PTO_IR_VMIATTRS

def VMILayoutAttr : PTO_Attr<"VMILayout", "vmi.layout"> {
  let summary = "VMI logical vector register layout";
  let parameters = (ins
    StringRefParameter<"layout kind">:$kind,
    "int64_t":$factor,
    "int64_t":$blockElems,
    "int64_t":$slots
  );
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static VMILayoutAttr getContiguous(::mlir::MLIRContext *context);
    static VMILayoutAttr getDeinterleaved(::mlir::MLIRContext *context,
                                          int64_t factor,
                                          int64_t blockElems = 1);
    static VMILayoutAttr getGroupSlots(::mlir::MLIRContext *context,
                                       int64_t numGroups,
                                       int64_t slots = 0,
                                       int64_t laneStride = 1);

    bool isContiguous() const { return getKind() == "contiguous"; }
    bool isDeinterleaved() const { return getKind() == "deinterleaved"; }
    bool isGroupSlots() const { return getKind() == "num_groups"; }
    int64_t getNumGroups() const { return getFactor(); }
    bool hasLaneStride() const {
      return isGroupSlots() && getBlockElems() != 1;
    }
    int64_t getLaneStride() const {
      return isGroupSlots() ? getBlockElems() : 1;
    }
  }];
}

#endif // MLIR_DIALECT_PTO_IR_VMIATTRS
