//! Basic usage example of the half_zig library

const std = @import("std");
const half_zig = @import("half_zig");

pub fn main() !void {
    std.debug.print("\x1b[35m==================================\x1b[0m\n", .{});
    std.debug.print("\x1b[35mHalf Precision Floating Point Demo\x1b[0m\n", .{});
    std.debug.print("\x1b[35m==================================\x1b[0m\n\n", .{});

    // Native f16 usage
    std.debug.print("\x1b[38;5;219mNative f16:\x1b[0m\n", .{});
    const f16_val: f16 = 3.14159;
    const f32_val: f32 = 3.14159;
    std.debug.print("f16 value: {}\n", .{f16_val});
    std.debug.print("f32 value: {}\n", .{f32_val});
    std.debug.print("f16 as f32: {}\n\n", .{@as(f32, f16_val)});

    // bfloat16 usage
    std.debug.print("\x1b[38;5;219mbfloat16 (bf16):\x1b[0m\n", .{});
    const bf16_val = half_zig.bf16.fromF32(3.14159);
    std.debug.print("bf16 from f32: {}\n", .{bf16_val});
    std.debug.print("bf16 to f32: {}\n", .{bf16_val.toF32()});
    std.debug.print("bf16 bits: 0x{X}\n\n", .{bf16_val.toBits()});

    // Arithmetic operations
    std.debug.print("\x1b[38;5;219mbf16 Arithmetic:\x1b[0m\n", .{});
    const a = half_zig.bf16.fromF32(2.5);
    const b = half_zig.bf16.fromF32(1.5);

    std.debug.print("a = {}, b = {}\n", .{ a, b });
    std.debug.print("a + b = {}\n", .{a.add(b)});
    std.debug.print("a - b = {}\n", .{a.sub(b)});
    std.debug.print("a * b = {}\n", .{a.mul(b)});
    std.debug.print("a / b = {}\n", .{a.div(b)});

    // Constants
    std.debug.print("\n\x1b[38;5;219mbf16 Constants:\x1b[0m\n", .{});
    std.debug.print("PI = {}\n", .{half_zig.bf16.PI});
    std.debug.print("E = {}\n", .{half_zig.bf16.E});
    std.debug.print("SQRT_2 = {}\n", .{half_zig.bf16.SQRT_2});

    // Special values
    std.debug.print("\n\x1b[38;5;219mSpecial values:\x1b[0m\n", .{});
    std.debug.print("INFINITY.isInfinite() = {}\n", .{half_zig.bf16.INFINITY.isInfinite()});
    std.debug.print("NAN.isNan() = {}\n", .{half_zig.bf16.NAN.isNan()});
    std.debug.print("ONE.isNormal() = {}\n", .{half_zig.bf16.ONE.isNormal()});
}
