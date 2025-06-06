//! Example demonstrating slice operations and batch conversions for half precision types

const std = @import("std");
const print = std.debug.print;
const half_zig = @import("half_zig");

const ARRAY_SIZE_SMALL: usize = 8;
const ARRAY_SIZE_MEDIUM: usize = 1024;
const ARRAY_SIZE_LARGE: usize = 100_000;

pub fn main() !void {
    print("\x1b[35m=====================================\x1b[0m\n", .{});
    print("\x1b[35mHalf Precision Slice Operations Demo\x1b[0m\n", .{});
    print("\x1b[35m=====================================\x1b[0m\n\n", .{});

    var allocator = std.heap.page_allocator;

    // Example 1: Basic slice conversion (f32 <-> bf16)
    print("\x1b[38;5;219m1. Basic Slice Conversion (f32 <-> bf16)\x1b[0m\n", .{});
    print("\x1b[38;5;219m--------------------------------------\x1b[0m\n", .{});

    // Create test data
    const test_data = [_]f32{ 1.5, -2.3, 3.14159, -0.5, 42.0, 0.0, -7.25, 12.375 };

    // Allocate bf16 array
    var bf16_array = try allocator.alloc(half_zig.bf16, test_data.len);
    defer allocator.free(bf16_array);

    // Convert f32 -> bf16
    for (test_data, 0..) |val, i| {
        bf16_array[i] = half_zig.bf16.fromF32(val);
    }

    print("Original f32 values: ", .{});
    for (test_data) |val| {
        print("{d:.5} ", .{val});
    }
    print("\n", .{});

    print("Converted to bf16:   ", .{});
    for (bf16_array) |val| {
        print("{} ", .{val});
    }
    print("\n", .{});

    print("Converted back to f32: ", .{});
    for (bf16_array) |val| {
        print("{d:.5} ", .{val.toF32()});
    }
    print("\n\n", .{});

    // Example 2: Memory reinterpretation between u16 and bf16
    print("\x1b[38;5;219m2. Memory Reinterpretation (u16 <-> bf16)\x1b[0m\n", .{});
    print("\x1b[38;5;219m---------------------------------------\x1b[0m\n", .{});

    // Create u16 array with known bf16 bit patterns
    const u16_data = [_]u16{
        0x3F80, // 1.0
        0xBF80, // -1.0
        0x4049, // ~3.14
        0x0000, // 0.0
        0x7F80, // infinity
        0xFF80, // -infinity
        0x7FC0, // NaN
        0x4000, // 2.0
    };

    // Reinterpret u16 array as bf16 array
    //  no write access, just view
    const bf16_view = @as([*]const half_zig.bf16, @ptrCast(&u16_data[0]))[0..u16_data.len];
    // if need alter the data, use:
    // const bf16_view = @as([*]half_zig.bf16, @ptrCast(@constCast(&u16_data[0])))[0..u16_data.len];
    print("Original u16 values (hex): ", .{});
    for (u16_data) |val| {
        print("0x{X:0>4} ", .{val});
    }
    print("\n", .{});

    print("Viewed as bf16 values: ", .{});
    for (bf16_view) |val| {
        print("{} ", .{val});
    }
    print("\n", .{});

    print("Converted to f32: ", .{});
    for (bf16_view) |val| {
        print("{d:.5} ", .{val.toF32()});
    }
    print("\n\n", .{});

    // Example 3: Native f16 slice operations
    print("\x1b[38;5;219m3. Native f16 Slice Operations\x1b[0m\n", .{});
    print("\x1b[38;5;219m----------------------------\x1b[0m\n", .{});

    // Create native f16 array
    var f16_array = try allocator.alloc(f16, test_data.len);
    defer allocator.free(f16_array);

    // Convert f32 -> f16
    for (test_data, 0..) |val, i| {
        f16_array[i] = @floatCast(val);
    }

    print("Original f32 values: ", .{});
    for (test_data) |val| {
        print("{d:.5} ", .{val});
    }
    print("\n", .{});

    print("Converted to f16:   ", .{});
    for (f16_array) |val| {
        print("{} ", .{val});
    }
    print("\n", .{});

    print("Converted back to f32: ", .{});
    for (f16_array) |val| {
        print("{d:.5} ", .{@as(f32, val)});
    }
    print("\n\n", .{});

    // Example 4: Batch conversion performance (f32 -> bf16)
    print("\x1b[38;5;219m4. Batch Conversion Performance\x1b[0m\n", .{});
    print("\x1b[38;5;219m------------------------------\x1b[0m\n", .{});

    // Create large test arrays
    const large_f32_array = try allocator.alloc(f32, ARRAY_SIZE_LARGE);
    defer allocator.free(large_f32_array);

    // Fill with test pattern
    for (large_f32_array, 0..) |*val, i| {
        val.* = @sin(@as(f32, @floatFromInt(i)) * 0.01) * 10.0;
    }

    var large_bf16_array = try allocator.alloc(half_zig.bf16, ARRAY_SIZE_LARGE);
    defer allocator.free(large_bf16_array);

    // Measure conversion time
    var timer = try std.time.Timer.start();
    const start = timer.lap();

    // Convert f32 -> bf16 in batch
    batchConvertF32ToBf16(large_f32_array, large_bf16_array);

    const end = timer.read();
    const elapsed_ns = end - start;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    print("Converted {d} f32 values to bf16 in {d:.2} ms\n", .{ ARRAY_SIZE_LARGE, elapsed_ms });
    print("Throughput: {d:.2} million conversions/second\n", .{@as(f64, @floatFromInt(ARRAY_SIZE_LARGE)) / (elapsed_ms / 1000.0) / 1_000_000.0});

    // Verify a few values
    print("\nSample values verification:\n", .{});
    for (0..5) |i| {
        print("  f32[{d}] = {d:.5} -> bf16 -> {d:.5}\n", .{ i, large_f32_array[i], large_bf16_array[i].toF32() });
    }
    print("  ...\n", .{});
    for (ARRAY_SIZE_LARGE - 5..ARRAY_SIZE_LARGE) |i| {
        print("  f32[{d}] = {d:.5} -> bf16 -> {d:.5}\n", .{ i, large_f32_array[i], large_bf16_array[i].toF32() });
    }
    print("\n", .{});

    // Example 5: Memory usage comparison
    print("\x1b[38;5;219m5. Memory Usage Comparison\x1b[0m\n", .{});
    print("\x1b[38;5;219m-------------------------\x1b[0m\n", .{});

    // Calculate sizes
    const f32_size = ARRAY_SIZE_MEDIUM * @sizeOf(f32);
    const f16_size = ARRAY_SIZE_MEDIUM * @sizeOf(f16);
    const bf16_size = ARRAY_SIZE_MEDIUM * @sizeOf(half_zig.bf16);
    const qint8_size = ARRAY_SIZE_MEDIUM * @sizeOf(half_zig.qint8);
    const qint4_packed_size = (ARRAY_SIZE_MEDIUM + 1) / 2; // Each byte holds 2 int4 values

    print("Memory required for {d} values:\n", .{ARRAY_SIZE_MEDIUM});
    print("  f32:      {d} bytes (baseline)\n", .{f32_size});
    print("  f16:      {d} bytes ({d:.1}% of f32)\n", .{ f16_size, @as(f64, @floatFromInt(f16_size)) / @as(f64, @floatFromInt(f32_size)) * 100.0 });
    print("  bf16:     {d} bytes ({d:.1}% of f32)\n", .{ bf16_size, @as(f64, @floatFromInt(bf16_size)) / @as(f64, @floatFromInt(f32_size)) * 100.0 });
    print("  qint8:    {d} bytes ({d:.1}% of f32)\n", .{ qint8_size, @as(f64, @floatFromInt(qint8_size)) / @as(f64, @floatFromInt(f32_size)) * 100.0 });
    print("  qint4:    {d} bytes ({d:.1}% of f32)\n", .{ qint4_packed_size, @as(f64, @floatFromInt(qint4_packed_size)) / @as(f64, @floatFromInt(f32_size)) * 100.0 });

    print("\n\x1b[35mSlice operations demo completed!\x1b[0m\n", .{});
}

// Batch conversion from f32 slice to bf16 slice
fn batchConvertF32ToBf16(src: []const f32, dst: []half_zig.bf16) void {
    std.debug.assert(src.len == dst.len);
    for (src, 0..) |value, i| {
        dst[i] = half_zig.bf16.fromF32(value);
    }
}

// Batch conversion from bf16 slice to f32 slice
fn batchConvertBf16ToF32(src: []const half_zig.bf16, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, 0..) |value, i| {
        dst[i] = value.toF32();
    }
}

// Batch conversion from f32 slice to quantized int8 slice
fn batchConvertF32ToQint8(src: []const f32, dst: []half_zig.qint8, scale: f32, zero_point: i8) void {
    std.debug.assert(src.len == dst.len);
    for (src, 0..) |value, i| {
        dst[i] = half_zig.qint8.fromF32(value, scale, zero_point);
    }
}

// Batch conversion from quantized int8 slice to f32 slice
fn batchConvertQint8ToF32(src: []const half_zig.qint8, dst: []f32, scale: f32, zero_point: i8) void {
    std.debug.assert(src.len == dst.len);
    for (src, 0..) |value, i| {
        dst[i] = value.toF32(scale, zero_point);
    }
}
