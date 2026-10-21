// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PrintEncoding.h - pto.print format string analysis -------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// The conversion in a pto.print format string selects the bridge entry and,
// for integers, whether the value is widened signed or unsigned. The host
// DebugTunnel decoder reads the same format string, so the operand type and
// the conversion have to agree.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {

/// Conversion kind of the single specifier in a print format string.
enum class PrintConversionKind : uint8_t {
  Float,       // %f, %F, %e, %E, %g, %G, %a, %A
  SignedInt,   // %d, %i
  UnsignedInt, // %u, %x, %X, %o
};

struct PrintFormatInfo {
  PrintConversionKind conversion;
  /// True when the specifier carries a 64-bit length modifier (`ll`, `l` or
  /// `z`). A 64-bit integer value needs one, or the host decoder reads the
  /// wrong width out of the record.
  bool hasWideLengthModifier = false;
};

/// True when \p c may appear in a specifier's flag field.
inline bool isPrintFormatFlag(char c) {
  return c == '+' || c == '-' || c == '0' || c == '#' || c == ' ';
}

inline bool isPrintFormatDigit(char c) { return c >= '0' && c <= '9'; }

/// Advances past the flags, width and precision of a specifier, leaving \p pos
/// on its length modifier or conversion character.
inline void skipPrintSpecifierPrefix(llvm::StringRef format, size_t &pos) {
  auto atFlag = [&format, &pos] {
    const bool inRange = pos < format.size();
    return inRange && isPrintFormatFlag(format[pos]);
  };
  auto atDigit = [&format, &pos] {
    const bool inRange = pos < format.size();
    return inRange && isPrintFormatDigit(format[pos]);
  };
  while (atFlag()) {
    ++pos;
  }
  while (atDigit()) {
    ++pos;
  }
  const bool hasPrecision = pos < format.size() && format[pos] == '.';
  if (!hasPrecision) {
    return;
  }
  ++pos;
  while (atDigit()) {
    ++pos;
  }
}

/// Consumes a length modifier, including doubled forms such as `ll` in %lld.
/// Returns whether it widens the value to 64 bits.
inline bool consumePrintLengthModifier(llvm::StringRef format, size_t &pos) {
  const bool hasModifier =
      pos < format.size() &&
      (format[pos] == 'l' || format[pos] == 'h' || format[pos] == 'z');
  if (!hasModifier) {
    return false;
  }
  const char modifier = format[pos++];
  const bool doubled = pos < format.size() && format[pos] == modifier;
  if (doubled) {
    ++pos;
  }
  return modifier == 'z' || modifier == 'l';
}

/// Classifies a conversion character, or returns nullopt when it is not one
/// the print bridge supports.
inline std::optional<PrintConversionKind> classifyPrintConversion(char conv) {
  switch (conv) {
  case 'f':
  case 'F':
  case 'e':
  case 'E':
  case 'g':
  case 'G':
  case 'a':
  case 'A':
    return PrintConversionKind::Float;
  case 'd':
  case 'i':
    return PrintConversionKind::SignedInt;
  case 'u':
  case 'x':
  case 'X':
  case 'o':
    return PrintConversionKind::UnsignedInt;
  default:
    return std::nullopt;
  }
}

/// Parses a print format string that must hold exactly one conversion
/// specifier; `%%` is a literal percent and does not count. Returns failure
/// without a diagnostic, leaving the message to the caller that has the op.
inline FailureOr<PrintFormatInfo> analyzePrintFormat(llvm::StringRef format) {
  if (format.empty()) {
    return failure();
  }
  PrintFormatInfo info{};
  bool hasConversion = false;
  size_t pos = 0;
  while (pos < format.size()) {
    if (format[pos] != '%') {
      ++pos;
      continue;
    }
    ++pos;
    if (pos >= format.size()) {
      return failure(); // trailing '%'
    }
    if (format[pos] == '%') {
      ++pos;
      continue;
    }
    if (hasConversion) {
      return failure(); // more than one conversion
    }
    skipPrintSpecifierPrefix(format, pos);
    info.hasWideLengthModifier = consumePrintLengthModifier(format, pos);
    if (pos >= format.size()) {
      return failure(); // incomplete specifier
    }
    std::optional<PrintConversionKind> conversion =
        classifyPrintConversion(format[pos++]);
    if (!conversion) {
      return failure();
    }
    info.conversion = *conversion;
    hasConversion = true;
    // Keep scanning so a second conversion is still rejected.
  }
  if (!hasConversion) {
    return failure();
  }
  return info;
}

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H
