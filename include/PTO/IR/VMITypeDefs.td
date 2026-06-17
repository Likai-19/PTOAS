// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMITypeDefs.td - PTO VMI type definitions -----------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VMITYPEDEFS
#define MLIR_DIALECT_PTO_IR_VMITYPEDEFS

include "PTO/IR/PTODialect.td"
include "PTO/IR/PTOAttrs.td"

def VMIVRegType : TypeDef<PTO_Dialect, "VMIVReg"> {
  let mnemonic = "vmi.vreg";
  let summary = "A VMI logical vector register value";

  let parameters = (ins
    "int64_t":$elementCount,
    "Type":$elementType,
    "mlir::Attribute":$layout
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    bool hasLayout() const { return static_cast<bool>(getLayout()); }
    VMILayoutAttr getLayoutAttr() const {
      return ::llvm::dyn_cast_or_null<VMILayoutAttr>(getLayout());
    }
  }];
}

def VMIMaskType : TypeDef<PTO_Dialect, "VMIMask"> {
  let mnemonic = "vmi.mask";
  let summary = "A VMI logical predicate mask value";

  let parameters = (ins
    "int64_t":$elementCount,
    StringRefParameter<"mask granularity view">:$granularity,
    "mlir::Attribute":$layout
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static bool isSupportedGranularity(::llvm::StringRef granularity);
    static bool isConcreteGranularity(::llvm::StringRef granularity);

    bool hasLayout() const { return static_cast<bool>(getLayout()); }
    bool isPred() const { return getGranularity() == "pred"; }
    bool isB8() const { return getGranularity() == "b8"; }
    bool isB16() const { return getGranularity() == "b16"; }
    bool isB32() const { return getGranularity() == "b32"; }
    VMILayoutAttr getLayoutAttr() const {
      return ::llvm::dyn_cast_or_null<VMILayoutAttr>(getLayout());
    }
  }];
}

#endif // MLIR_DIALECT_PTO_IR_VMITYPEDEFS
