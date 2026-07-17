// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Provide the missing RTTI symbol expected by the ptoas shared module when it
// is loaded through ctypes.
asm(
    ".section .data.rel.ro\n"
    ".globl _ZTIN4llvm2cl6OptionE\n"
    ".type _ZTIN4llvm2cl6OptionE, @object\n"
    ".size _ZTIN4llvm2cl6OptionE, 16\n"
    "_ZTIN4llvm2cl6OptionE:\n"
    "  .quad _ZTVN10__cxxabiv117__class_type_infoE+16\n"
    "  .quad .Lllvm_cl_option_name\n"
    ".section .rodata\n"
    ".Lllvm_cl_option_name:\n"
    "  .asciz \"N4llvm2cl6OptionE\"\n"
    ".section .data.rel.ro\n"
    ".globl _ZTIN4llvm2cl18GenericOptionValueE\n"
    ".type _ZTIN4llvm2cl18GenericOptionValueE, @object\n"
    ".size _ZTIN4llvm2cl18GenericOptionValueE, 16\n"
    "_ZTIN4llvm2cl18GenericOptionValueE:\n"
    "  .quad _ZTVN10__cxxabiv117__class_type_infoE+16\n"
    "  .quad .Lllvm_cl_generic_option_value_name\n"
    ".section .rodata\n"
    ".Lllvm_cl_generic_option_value_name:\n"
    "  .asciz \"N4llvm2cl18GenericOptionValueE\"\n"
    ".section .data.rel.ro\n"
    ".globl _ZTIN4llvm2cl19generic_parser_baseE\n"
    ".type _ZTIN4llvm2cl19generic_parser_baseE, @object\n"
    ".size _ZTIN4llvm2cl19generic_parser_baseE, 16\n"
    "_ZTIN4llvm2cl19generic_parser_baseE:\n"
    "  .quad _ZTVN10__cxxabiv117__class_type_infoE+16\n"
    "  .quad .Lllvm_cl_generic_parser_base_name\n"
    ".section .rodata\n"
    ".Lllvm_cl_generic_parser_base_name:\n"
    "  .asciz \"N4llvm2cl19generic_parser_baseE\"\n"
    ".section .data.rel.ro\n"
    ".globl _ZTIN4mlir4PassE\n"
    ".type _ZTIN4mlir4PassE, @object\n"
    ".size _ZTIN4mlir4PassE, 16\n"
    "_ZTIN4mlir4PassE:\n"
    "  .quad _ZTVN10__cxxabiv117__class_type_infoE+16\n"
    "  .quad .Lmlir_pass_name\n"
    ".section .rodata\n"
    ".Lmlir_pass_name:\n"
    "  .asciz \"N4mlir4PassE\"\n");
