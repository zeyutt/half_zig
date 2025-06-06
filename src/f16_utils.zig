//! Utilities for working with native f16 type

const std = @import("std");

/// Convert array of f32 to f16
pub fn f32ArrayToF16(allocator: std.mem.Allocator, f32_array: []const f32) ![]f16 {
    var f16_array = try allocator.alloc(f16, f32_array.len);
    for (f32_array, 0..) |val, i| {
        f16_array[i] = @as(f16, @floatCast(val));
    }
    return f16_array;
}

/// Convert array of f16 to f32
pub fn f16ArrayToF32(allocator: std.mem.Allocator, f16_array: []const f16) ![]f32 {
    var f32_array = try allocator.alloc(f32, f16_array.len);
    for (f16_array, 0..) |val, i| {
        f32_array[i] = @as(f32, @floatCast(val));
    }
    return f32_array;
}

/// Check if f16 value is approximately equal
pub fn approxEqF16(a: f16, b: f16, tolerance: f16) bool {
    const diff = if (a > b) a - b else b - a;
    return diff <= tolerance;
}

test "f16 array conversions" {
    std.debug.print("\n\x1b[36m ▶ Test: f16 array conversions\x1b[0m\n", .{});
    const testing = std.testing;
    const allocator = testing.allocator;

    const f32_values = [_]f32{ 1.0, 2.5, -3.7, 0.0 };

    const f16_array = try f32ArrayToF16(allocator, &f32_values);
    defer allocator.free(f16_array);

    const f32_array_back = try f16ArrayToF32(allocator, f16_array);
    defer allocator.free(f32_array_back);

    for (f32_values, 0..) |original, i| {
        try testing.expectApproxEqAbs(original, f32_array_back[i], 1e-3);
    }
}

test "f16 approximate equality" {
    std.debug.print("\n\x1b[36m ▶ Test: f16 approximate equality\x1b[0m\n", .{});
    const testing = std.testing;

    const a: f16 = 1.0;
    const b: f16 = 1.001;
    const c: f16 = 1.1;

    try testing.expect(approxEqF16(a, b, 0.01));
    try testing.expect(!approxEqF16(a, c, 0.01));
}
