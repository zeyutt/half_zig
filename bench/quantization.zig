//! Benchmark tests for quantized integer types

const std = @import("std");
const print = std.debug.print;
const Timer = std.time.Timer;
const half_zig = @import("half_zig");

const BENCH_ITERATIONS: u32 = 1_000_000;
const ARRAY_SIZE: usize = 1024;

fn simpleBenchmarkFunction(
    comptime name: []const u8,
    comptime func: anytype,
    args: anytype,
    iterations: u32,
) !void {
    var timer = try Timer.start();

    const start = timer.lap();

    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(@call(.auto, func, args));
    }

    const end = timer.read();
    const elapsed_ns = end - start;
    const ns_per_iter = elapsed_ns / iterations;

    print("{s}: {d:.2} ns/iter ({d} iterations)\n", .{ name, @as(f64, @floatFromInt(ns_per_iter)), iterations });
}

fn benchQuantizationConversions() !void {
    print("\n\x1b[38;5;24m=== Quantization Conversions ===\x1b[0m\n", .{});

    const test_val: f32 = 1.5;
    const scale: f32 = 0.1;
    const zero_point_i8: i8 = 0;
    const zero_point_u8: u8 = 128;
    const zero_point_u4: u4 = 8;

    // qint8 conversions
    try simpleBenchmarkFunction("qint8::fromF32", half_zig.qint8.fromF32, .{ test_val, scale, zero_point_i8 }, BENCH_ITERATIONS);

    const q8_val = half_zig.qint8.fromF32(test_val, scale, zero_point_i8);
    try simpleBenchmarkFunction("qint8::toF32", half_zig.qint8.toF32, .{ q8_val, scale, zero_point_i8 }, BENCH_ITERATIONS);

    // quint8 conversions
    try simpleBenchmarkFunction("quint8::fromF32", half_zig.quint8.fromF32, .{ test_val, scale, zero_point_u8 }, BENCH_ITERATIONS);

    const qu8_val = half_zig.quint8.fromF32(test_val, scale, zero_point_u8);
    try simpleBenchmarkFunction("quint8::toF32", half_zig.quint8.toF32, .{ qu8_val, scale, zero_point_u8 }, BENCH_ITERATIONS);

    // qint4 conversions
    try simpleBenchmarkFunction("qint4::fromF32", half_zig.qint4.fromF32, .{ test_val, scale }, BENCH_ITERATIONS);

    const q4_val = half_zig.qint4.fromF32(test_val, scale);
    try simpleBenchmarkFunction("qint4::toF32", half_zig.qint4.toF32, .{ q4_val, scale }, BENCH_ITERATIONS);

    // quint4 conversions
    try simpleBenchmarkFunction("quint4::fromF32", half_zig.quint4.fromF32, .{ test_val, scale, zero_point_u4 }, BENCH_ITERATIONS);

    const qu4_val = half_zig.quint4.fromF32(test_val, scale, zero_point_u4);
    try simpleBenchmarkFunction("quint4::toF32", half_zig.quint4.toF32, .{ qu4_val, scale, zero_point_u4 }, BENCH_ITERATIONS);
}

fn benchArithmeticOperations() !void {
    print("\n\x1b[38;5;24m=== Arithmetic Operations ===\x1b[0m\n", .{});

    const q8_a = half_zig.qint8.fromRaw(42);
    const q8_b = half_zig.qint8.fromRaw(17);

    try simpleBenchmarkFunction("qint8::add", half_zig.qint8.add, .{ q8_a, q8_b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("qint8::sub", half_zig.qint8.sub, .{ q8_a, q8_b }, BENCH_ITERATIONS);
}

fn benchArrayOperations() !void {
    print("\n\x1b[38;5;24m=== Array Operations ===\x1b[0m\n", .{});

    var allocator = std.heap.page_allocator;

    // Prepare test data
    const test_data = try allocator.alloc(f32, ARRAY_SIZE);
    defer allocator.free(test_data);

    for (test_data, 0..) |*val, i| {
        val.* = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 2.0;
    }

    // PackedInt4Array benchmark
    const scale: f32 = 0.5;

    // 为每个操作使用单独的计时器实例
    {
        var timer = try Timer.start();
        const start = timer.read();
        var packed_four = try half_zig.PackedInt4Array.fromF32Array(allocator, test_data, scale);
        defer packed_four.deinit();
        const end = timer.read();
        const elapsed_ns = end - start;
        print("PackedInt4Array::fromF32Array ({} elements): {d:.2} ms\n", .{ ARRAY_SIZE, @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0 });

        // 使用新的计时器测量转换回F32的时间
        timer = try Timer.start();
        const start2 = timer.read();
        const reconstructed = try packed_four.toF32Array(allocator, scale);
        defer allocator.free(reconstructed);
        const end2 = timer.read();
        const elapsed_ns2 = end2 - start2;
        print("PackedInt4Array::toF32Array ({} elements): {d:.2} ms\n", .{ ARRAY_SIZE, @as(f64, @floatFromInt(elapsed_ns2)) / 1_000_000.0 });
    }

    // BlockQuantizedInt4 benchmark
    {
        var timer = try Timer.start();
        const start = timer.read();
        var block_quantized = try half_zig.BlockQuantizedInt4.fromF32Array(allocator, test_data);
        defer block_quantized.deinit();
        const end = timer.read();
        const elapsed_ns = end - start;
        print("BlockQuantizedInt4::fromF32Array ({} elements): {d:.2} ms\n", .{ ARRAY_SIZE, @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0 });

        // 使用新的计时器测量转换回F32的时间
        timer = try Timer.start();
        const start2 = timer.read();
        const block_reconstructed = try block_quantized.toF32Array(allocator);
        defer allocator.free(block_reconstructed);
        const end2 = timer.read();
        const elapsed_ns2 = end2 - start2;
        print("BlockQuantizedInt4::toF32Array ({} elements): {d:.2} ms\n", .{ ARRAY_SIZE, @as(f64, @floatFromInt(elapsed_ns2)) / 1_000_000.0 });
    }
}

fn benchMemoryEfficiency() !void {
    print("\n\x1b[38;5;24m=== Memory Efficiency Analysis ===\x1b[0m\n", .{});

    var allocator = std.heap.page_allocator;
    const test_size: usize = 1024;

    // Original f32 array
    const f32_array = try allocator.alloc(f32, test_size);
    defer allocator.free(f32_array);

    for (f32_array, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i)) * 0.01;
    }

    print("Original f32 array: {d} bytes\n", .{f32_array.len * @sizeOf(f32)});

    // bf16 array
    var bf16_array = try allocator.alloc(half_zig.bf16, test_size);
    defer allocator.free(bf16_array);
    for (f32_array, 0..) |val, i| {
        bf16_array[i] = half_zig.bf16.fromF32(val);
    }
    print("bf16 array: {} bytes ({d:.1}x smaller)\n", .{ bf16_array.len * @sizeOf(half_zig.bf16), @as(f64, @floatFromInt(f32_array.len * @sizeOf(f32))) / @as(f64, @floatFromInt(bf16_array.len * @sizeOf(half_zig.bf16))) });

    // qint8 array
    var qint8_array = try allocator.alloc(half_zig.qint8, test_size);
    defer allocator.free(qint8_array);
    for (f32_array, 0..) |val, i| {
        qint8_array[i] = half_zig.qint8.fromF32(val, 0.01, 0);
    }
    print("qint8 array: {} bytes ({d:.1}x smaller)\n", .{ qint8_array.len * @sizeOf(half_zig.qint8), @as(f64, @floatFromInt(f32_array.len * @sizeOf(f32))) / @as(f64, @floatFromInt(qint8_array.len * @sizeOf(half_zig.qint8))) });

    // PackedInt4Array
    var packed_four = try half_zig.PackedInt4Array.fromF32Array(allocator, f32_array, 0.1);
    defer packed_four.deinit();
    print("PackedInt4Array: {} bytes ({d:.1}x smaller)\n", .{ packed_four.data.len, @as(f64, @floatFromInt(f32_array.len * @sizeOf(f32))) / @as(f64, @floatFromInt(packed_four.data.len)) });
}

pub fn main() !void {
    print("\x1b[38;5;33m======================================\x1b[0m\n", .{});
    print("\x1b[38;5;39mNeural Network Quantization Benchmarks\x1b[0m\n", .{});
    print("\x1b[38;5;33m======================================\x1b[0m\n", .{});

    try benchQuantizationConversions();
    try benchArithmeticOperations();
    try benchArrayOperations();
    try benchMemoryEfficiency();

    print("\n\x1b[38;5;39mQuantization benchmarks completed!\x1b[0m\n", .{});
}
