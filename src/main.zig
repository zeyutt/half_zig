//! Half precision and quantized floating point library for Zig
//!
//! This library provides support for:
//! - bfloat16 (Brain Floating Point 16-bit)
//! - Native f16 utilities
//! - Quantized integer types (qint8, qint4, etc.)

const std = @import("std");

// Re-export all modules
pub const bf16 = @import("bfloat16.zig").bf16;
pub const f16_utils = @import("f16_utils.zig");
pub const quantized = @import("quantized.zig");

// Re-export quantized types for convenience
pub const qint8 = quantized.qint8;
pub const quint8 = quantized.quint8;
pub const qint4 = quantized.qint4;
pub const quint4 = quantized.quint4;
pub const PackedInt4Array = quantized.PackedInt4Array;
pub const BlockQuantizedInt4 = quantized.BlockQuantizedInt4;

test {
    // Import all tests
    _ = @import("bfloat16.zig");
    _ = @import("f16_utils.zig");
    _ = @import("quantized.zig");
}

test "simple test" {
    var x: f16 = undefined;
    x = 1.0; // Assign a value to x
    // More tests will be added here
    try std.testing.expect(true);
}
