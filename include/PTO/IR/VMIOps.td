// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIOps.td - PTO VMI semantic operations -------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_VMIOPS
#define MLIR_DIALECT_PTO_IR_VMIOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VMI_VRegTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VMIVRegType>($_self)">,
  "VMI logical vector register type">;

def VMI_MaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VMIMaskType>($_self)">,
  "VMI logical mask type">;

def VMI_ValueTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VMIVRegType, ::mlir::pto::VMIMaskType>($_self)">,
  "VMI logical vector or mask type">;

def PTO_PhysicalVRegTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::VRegType>($_self)">,
  "PTO physical vector register type">;

def PTO_PhysicalMaskTypeConstraint : Type<
  CPred<"::llvm::isa<::mlir::pto::MaskType>($_self)">,
  "PTO physical mask type">;

def PTO_PhysicalVMIPartTypeConstraint : AnyTypeOf<
  [PTO_PhysicalVRegTypeConstraint, PTO_PhysicalMaskTypeConstraint],
  "PTO physical vector register or mask type">;

class VMI_Op<string mnemonic, list<Trait> traits = []>
    : PTO_Op<"vmi." # mnemonic, traits>;

def VMIConstantOp : VMI_Op<"constant"> {
  let summary = "VMI logical vector constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIBroadcastOp : VMI_Op<"broadcast"> {
  let summary = "Broadcast one scalar or rank-0 VMI vector to a VMI logical vector";
  let arguments = (ins AnyType:$value);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($value) `->` type($result)";
}

def VMIIotaOp : VMI_Op<"iota"> {
  let summary = "Create a VMI logical index vector from a scalar base";
  let arguments = (ins
    AnyTypeOf<[AnyInteger, AnyFloat], "integer/float scalar">:$base,
    OptionalAttr<StrAttr>:$order
  );
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$base attr-dict `:` type($base) `->` type($result)";
}

def VMICreateMaskOp : VMI_Op<"create_mask"> {
  let summary = "Create a VMI logical prefix predicate mask";
  let arguments = (ins Index:$active_lanes);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$active_lanes attr-dict `:` type($active_lanes) `->` type($result)";
}

def VMICreateGroupMaskOp : VMI_Op<"create_group_mask"> {
  let summary = "Create a VMI logical grouped predicate mask";
  let description = [{
    Creates a mask where lane i is active iff
    `(i % group_size) < active_elems_per_group`.
  }];
  let arguments = (ins
    Index:$active_elems_per_group,
    I64Attr:$num_groups,
    I64Attr:$group_size
  );
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$active_elems_per_group attr-dict `:` type($active_elems_per_group) `->` type($result)";
}

def VMIConstantMaskOp : VMI_Op<"constant_mask"> {
  let summary = "VMI logical predicate mask constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIMaskAndOp : VMI_Op<"mask_and"> {
  let summary = "VMI logical predicate mask and";
  let arguments = (ins VMI_MaskTypeConstraint:$lhs, VMI_MaskTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaskOrOp : VMI_Op<"mask_or"> {
  let summary = "VMI logical predicate mask or";
  let arguments = (ins VMI_MaskTypeConstraint:$lhs, VMI_MaskTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaskXOrOp : VMI_Op<"mask_xor"> {
  let summary = "VMI logical predicate mask xor";
  let arguments = (ins VMI_MaskTypeConstraint:$lhs, VMI_MaskTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaskNotOp : VMI_Op<"mask_not"> {
  let summary = "VMI logical predicate mask not";
  let arguments = (ins VMI_MaskTypeConstraint:$source);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAddFOp : VMI_Op<"addf"> {
  let summary = "VMI floating-point elementwise add";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIAddIOp : VMI_Op<"addi"> {
  let summary = "VMI integer elementwise add";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMISubFOp : VMI_Op<"subf"> {
  let summary = "VMI floating-point elementwise subtract";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMISubIOp : VMI_Op<"subi"> {
  let summary = "VMI integer elementwise subtract";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMulFOp : VMI_Op<"mulf"> {
  let summary = "VMI floating-point elementwise multiply";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMulIOp : VMI_Op<"muli"> {
  let summary = "VMI integer elementwise multiply";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIFmaOp : VMI_Op<"fma"> {
  let summary = "VMI fused floating-point multiply-add";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs,
                       VMI_VRegTypeConstraint:$acc);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs `,` $acc attr-dict `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)";
}

def VMIDivFOp : VMI_Op<"divf"> {
  let summary = "VMI floating-point elementwise divide";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMinFOp : VMI_Op<"minf"> {
  let summary = "VMI floating-point elementwise minimum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIMaxFOp : VMI_Op<"maxf"> {
  let summary = "VMI floating-point elementwise maximum";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMINegFOp : VMI_Op<"negf"> {
  let summary = "VMI floating-point elementwise negate";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAbsFOp : VMI_Op<"absf"> {
  let summary = "VMI floating-point elementwise absolute value";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAbsIOp : VMI_Op<"absi"> {
  let summary = "VMI integer elementwise absolute value";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMISqrtOp : VMI_Op<"sqrt"> {
  let summary = "VMI floating-point elementwise square root";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIExpOp : VMI_Op<"exp"> {
  let summary = "VMI floating-point elementwise exponential";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMILnOp : VMI_Op<"ln"> {
  let summary = "VMI floating-point elementwise natural logarithm";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIReluOp : VMI_Op<"relu"> {
  let summary = "VMI floating-point elementwise ReLU";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIAndIOp : VMI_Op<"andi"> {
  let summary = "VMI integer elementwise bitwise and";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIOrIOp : VMI_Op<"ori"> {
  let summary = "VMI integer elementwise bitwise or";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIXOrIOp : VMI_Op<"xori"> {
  let summary = "VMI integer elementwise bitwise xor";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIShLIOp : VMI_Op<"shli"> {
  let summary = "VMI integer elementwise left shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMIShRUIOp : VMI_Op<"shrui"> {
  let summary = "VMI unsigned integer elementwise right shift";
  let arguments = (ins VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMINotOp : VMI_Op<"not"> {
  let summary = "VMI integer elementwise bitwise not";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMICmpFOp : VMI_Op<"cmpf"> {
  let summary = "VMI floating-point elementwise compare";
  let arguments = (ins StrAttr:$predicate, VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMICmpIOp : VMI_Op<"cmpi"> {
  let summary = "VMI integer elementwise compare";
  let arguments = (ins StrAttr:$predicate, VMI_VRegTypeConstraint:$lhs, VMI_VRegTypeConstraint:$rhs);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def VMISelectOp : VMI_Op<"select"> {
  let summary = "VMI elementwise select";
  let arguments = (ins VMI_MaskTypeConstraint:$mask, VMI_VRegTypeConstraint:$true_value,
                       VMI_VRegTypeConstraint:$false_value);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$mask `,` $true_value `,` $false_value attr-dict `:` type($mask) `,` type($true_value) `,` type($false_value) `->` type($result)";
}

def VMIActivePrefixIndexOp : VMI_Op<"active_prefix_index"> {
  let summary = "VMI per-lane active-prefix index from a predicate mask";
  let arguments = (ins VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$mask attr-dict `:` type($mask) `->` type($result)";
}

def VMICompressOp : VMI_Op<"compress"> {
  let summary = "VMI compact active source lanes according to a predicate mask";
  let arguments = (ins VMI_VRegTypeConstraint:$source, VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMICompressStoreOp : VMI_Op<"compress_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI store active source lanes contiguously according to a predicate mask";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)";
}

def VMIReduceAddIOp : VMI_Op<"reduce_addi"> {
  let summary = "VMI masked integer add reduction with a rank-0 vector init";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_VRegTypeConstraint:$init,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $init `,` $mask attr-dict `:` type($source) `,` type($init) `,` type($mask) `->` type($result)";
}

def VMIReduceAddFOp : VMI_Op<"reduce_addf"> {
  let summary = "VMI masked floating-point add reduction with explicit reassociation permission";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_VRegTypeConstraint:$init,
                       VMI_MaskTypeConstraint:$mask,
                       OptionalAttr<UnitAttr>:$reassoc);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $init `,` $mask attr-dict `:` type($source) `,` type($init) `,` type($mask) `->` type($result)";
}

def VMIReduceMaxFOp : VMI_Op<"reduce_maxf"> {
  let summary = "VMI masked floating-point maximum reduction with a rank-0 vector init";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_VRegTypeConstraint:$init,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $init `,` $mask attr-dict `:` type($source) `,` type($init) `,` type($mask) `->` type($result)";
}

def VMIReduceMinFOp : VMI_Op<"reduce_minf"> {
  let summary = "VMI masked floating-point minimum reduction with a rank-0 vector init";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_VRegTypeConstraint:$init,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $init `,` $mask attr-dict `:` type($source) `,` type($init) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceAddFOp : VMI_Op<"group_reduce_addf"> {
  let summary = "VMI masked floating-point add reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups,
                       OptionalAttr<UnitAttr>:$reassoc);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceMaxFOp : VMI_Op<"group_reduce_maxf"> {
  let summary = "VMI masked floating-point maximum reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupReduceAddIOp : VMI_Op<"group_reduce_addi"> {
  let summary = "VMI masked integer add reduction within fixed logical groups";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `,` $mask attr-dict `:` type($source) `,` type($mask) `->` type($result)";
}

def VMIGroupBroadcastOp : VMI_Op<"group_broadcast"> {
  let summary = "VMI broadcast group-slot values back to each logical group";
  let arguments = (ins VMI_VRegTypeConstraint:$source,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

class VMIHistogramOp<string mnemonic, string summaryText>
    : VMI_Op<mnemonic> {
  let summary = summaryText;
  let arguments = (ins VMI_VRegTypeConstraint:$acc,
                       VMI_VRegTypeConstraint:$source,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$acc `,` $source `,` $mask attr-dict `:` type($acc) `,` type($source) `,` type($mask) `->` type($result)";
}

def VMIDhistOp : VMIHistogramOp<"dhist",
  "VMI full 256-bin distribution histogram over unsigned 8-bit source lanes">;

def VMIChistOp : VMIHistogramOp<"chist",
  "VMI full 256-bin cumulative histogram over unsigned 8-bit source lanes">;

def VMIExtFOp : VMI_Op<"extf"> {
  let summary = "VMI floating-point elementwise extension";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMITruncFOp : VMI_Op<"truncf"> {
  let summary = "VMI floating-point elementwise truncation";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIExtSIOp : VMI_Op<"extsi"> {
  let summary = "VMI signed integer elementwise extension";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIExtUIOp : VMI_Op<"extui"> {
  let summary = "VMI unsigned integer elementwise extension";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMITruncIOp : VMI_Op<"trunci"> {
  let summary = "VMI saturating integer elementwise truncation";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIBitcastOp : VMI_Op<"bitcast"> {
  let summary = "VMI bitwise vector reinterpretation";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMILoadOp : VMI_Op<"load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical vector load";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` attr-dict `:` type($source) `->` type($result)";
}

def VMIGroupLoadOp : VMI_Op<"group_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical grouped vector load with a row stride between groups";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset, Index:$row_stride,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $row_stride attr-dict `:` type($source) `->` type($result)";
}

def VMIGroupSlotLoadOp : VMI_Op<"group_slot_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI load one scalar value per logical group into group slots";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset, Index:$source_group_stride,
                       I64Attr:$num_groups);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $source_group_stride attr-dict `:` type($source) `->` type($result)";
}

def VMIMaskedLoadOp : VMI_Op<"masked_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked vector load with passthrough lanes";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset,
                       VMI_MaskTypeConstraint:$mask,
                       VMI_VRegTypeConstraint:$passthru);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $mask `,` $passthru attr-dict `:` type($source) `,` type($mask) `,` type($passthru) `->` type($result)";
}

def VMIGatherOp : VMI_Op<"gather", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked indexed gather with passthrough lanes";
  let arguments = (ins PtrOrMemRef:$source,
                       VMI_VRegTypeConstraint:$indices,
                       VMI_MaskTypeConstraint:$mask,
                       VMI_VRegTypeConstraint:$passthru);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $indices `]` `,` $mask `,` $passthru attr-dict `:` type($source) `,` type($indices) `,` type($mask) `,` type($passthru) `->` type($result)";
}

def VMIExpandLoadOp : VMI_Op<"expand_load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI load a dense active-lane stream into masked logical lanes";
  let arguments = (ins PtrOrMemRef:$source, Index:$offset,
                       VMI_MaskTypeConstraint:$mask,
                       VMI_VRegTypeConstraint:$passthru);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $offset `]` `,` $mask `,` $passthru attr-dict `:` type($source) `,` type($mask) `,` type($passthru) `->` type($result)";
}

def VMIStoreOp : VMI_Op<"store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical vector store";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination, Index:$offset);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` attr-dict `:` type($value) `,` type($destination)";
}

def VMIGroupStoreOp : VMI_Op<"group_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical grouped vector store with a row stride between groups";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, Index:$row_stride, I64Attr:$num_groups);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $row_stride attr-dict `:` type($value) `,` type($destination)";
}

def VMIMaskedStoreOp : VMI_Op<"masked_store", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked vector store";
  let arguments = (ins VMI_VRegTypeConstraint:$value, PtrOrMemRef:$destination,
                       Index:$offset, VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $offset `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($mask)";
}

def VMIScatterOp : VMI_Op<"scatter", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical masked indexed scatter";
  let arguments = (ins VMI_VRegTypeConstraint:$value,
                       PtrOrMemRef:$destination,
                       VMI_VRegTypeConstraint:$indices,
                       VMI_MaskTypeConstraint:$mask);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination `[` $indices `]` `,` $mask attr-dict `:` type($value) `,` type($destination) `,` type($indices) `,` type($mask)";
}

def VMITileReadOp : VMI_Op<"tile_read", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical tile read";
  let arguments = (ins AnyType:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMITileWriteOp : VMI_Op<"tile_write", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "VMI logical tile write";
  let arguments = (ins VMI_VRegTypeConstraint:$value, AnyType:$destination);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = "$value `,` $destination attr-dict `:` type($value) `,` type($destination)";
}

def VMIShuffleOp : VMI_Op<"shuffle"> {
  let summary = "VMI static lane shuffle";
  let arguments = (ins VMI_VRegTypeConstraint:$source, DenseI64ArrayAttr:$indices);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
}

def VMIChannelSplitOp : VMI_Op<"channel_split"> {
  let summary = "VMI split interleaved logical channels";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs Variadic<VMI_VRegTypeConstraint>:$results);
  let hasVerifier = 1;
}

def VMIChannelMergeOp : VMI_Op<"channel_merge"> {
  let summary = "VMI merge logical channels by interleaving";
  let arguments = (ins Variadic<VMI_VRegTypeConstraint>:$inputs);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
}

def VMIEnsureLayoutOp : VMI_Op<"ensure_layout"> {
  let summary = "Internal VMI data layout materialization helper";
  let arguments = (ins VMI_VRegTypeConstraint:$source);
  let results = (outs VMI_VRegTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIEnsureMaskLayoutOp : VMI_Op<"ensure_mask_layout"> {
  let summary = "Internal VMI mask layout materialization helper";
  let arguments = (ins VMI_MaskTypeConstraint:$source);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIEnsureMaskGranularityOp : VMI_Op<"ensure_mask_granularity"> {
  let summary = "Internal VMI mask granularity materialization helper";
  let arguments = (ins VMI_MaskTypeConstraint:$source);
  let results = (outs VMI_MaskTypeConstraint:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($result)";
}

def VMIUnpackOp : VMI_Op<"unpack"> {
  let summary = "Internal VMI value projection to physical parts";
  let arguments = (ins VMI_ValueTypeConstraint:$source);
  let results = (outs Variadic<PTO_PhysicalVMIPartTypeConstraint>:$parts);
  let hasVerifier = 1;
}

def VMIPackOp : VMI_Op<"pack"> {
  let summary = "Internal physical parts materialized as one VMI value";
  let arguments = (ins Variadic<PTO_PhysicalVMIPartTypeConstraint>:$parts);
  let results = (outs VMI_ValueTypeConstraint:$result);
  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_PTO_IR_VMIOPS
