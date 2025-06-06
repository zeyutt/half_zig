//! Benchmark tests for half precision floating point conversions
//! Ported from half-rs benchmarks

const std = @import("std");
const print = std.debug.print;
const Timer = std.time.Timer;
const half_zig = @import("half_zig");

const SIMD_LARGE_BENCH_SLICE_LEN: usize = 1024;
const BENCH_ITERATIONS: u32 = 1_000_000;

// Benchmark runner utility - 修改为接受运行时字符串
fn benchmarkFunction(
    name: []const u8, // 移除 comptime
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

// 简化版本的基准测试函数，不使用动态字符串分配
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

// bf16 conversion benchmarks
fn benchBf16FromF32() !void {
    print("\n\x1b[38;5;24m=== bf16 From f32 Benchmarks ===\x1b[0m\n", .{});

    // 使用编译时已知的名称，避免动态分配
    try simpleBenchmarkFunction("bf16::fromF32(0.0)", half_zig.bf16.fromF32, .{0.0}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(-0.0)", half_zig.bf16.fromF32, .{-0.0}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(1.0)", half_zig.bf16.fromF32, .{1.0}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(floatMin)", half_zig.bf16.fromF32, .{std.math.floatMin(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(floatMax)", half_zig.bf16.fromF32, .{std.math.floatMax(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(floatTrueMin)", half_zig.bf16.fromF32, .{std.math.floatTrueMin(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(-inf)", half_zig.bf16.fromF32, .{-std.math.inf(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(inf)", half_zig.bf16.fromF32, .{std.math.inf(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(nan)", half_zig.bf16.fromF32, .{std.math.nan(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(e)", half_zig.bf16.fromF32, .{std.math.e}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF32(pi)", half_zig.bf16.fromF32, .{std.math.pi}, BENCH_ITERATIONS);
}

fn benchBf16FromF64() !void {
    print("\n\x1b[38;5;24m=== bf16 From f64 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("bf16::fromF64(0.0)", half_zig.bf16.fromF64, .{@as(f64, 0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(-0.0)", half_zig.bf16.fromF64, .{@as(f64, -0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(1.0)", half_zig.bf16.fromF64, .{@as(f64, 1.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(floatMin)", half_zig.bf16.fromF64, .{std.math.floatMin(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(floatMax)", half_zig.bf16.fromF64, .{std.math.floatMax(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(floatTrueMin)", half_zig.bf16.fromF64, .{std.math.floatTrueMin(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(-inf)", half_zig.bf16.fromF64, .{-std.math.inf(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(inf)", half_zig.bf16.fromF64, .{std.math.inf(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(nan)", half_zig.bf16.fromF64, .{std.math.nan(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(e)", half_zig.bf16.fromF64, .{@as(f64, std.math.e)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::fromF64(pi)", half_zig.bf16.fromF64, .{@as(f64, std.math.pi)}, BENCH_ITERATIONS);
}

fn benchBf16ToF32() !void {
    print("\n\x1b[38;5;24m=== bf16 to f32 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("bf16::toF32(ZERO)", half_zig.bf16.toF32, .{half_zig.bf16.ZERO}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(NEG_ZERO)", half_zig.bf16.toF32, .{half_zig.bf16.NEG_ZERO}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(ONE)", half_zig.bf16.toF32, .{half_zig.bf16.ONE}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(floatMin)", half_zig.bf16.toF32, .{half_zig.bf16.fromF32(std.math.floatMin(f32))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(floatMax)", half_zig.bf16.toF32, .{half_zig.bf16.fromF32(std.math.floatMax(f32))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(floatTrueMin)", half_zig.bf16.toF32, .{half_zig.bf16.fromF32(std.math.floatTrueMin(f32))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(NEG_INFINITY)", half_zig.bf16.toF32, .{half_zig.bf16.NEG_INFINITY}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(INFINITY)", half_zig.bf16.toF32, .{half_zig.bf16.INFINITY}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(NAN)", half_zig.bf16.toF32, .{half_zig.bf16.NAN}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(E)", half_zig.bf16.toF32, .{half_zig.bf16.E}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF32(PI)", half_zig.bf16.toF32, .{half_zig.bf16.PI}, BENCH_ITERATIONS);
}

fn benchBf16ToF64() !void {
    print("\n\x1b[38;5;24m=== bf16 to f64 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("bf16::toF64(ZERO)", half_zig.bf16.toF64, .{half_zig.bf16.ZERO}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(NEG_ZERO)", half_zig.bf16.toF64, .{half_zig.bf16.NEG_ZERO}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(ONE)", half_zig.bf16.toF64, .{half_zig.bf16.ONE}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(floatMin)", half_zig.bf16.toF64, .{half_zig.bf16.fromF32(std.math.floatMin(f32))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(floatMax)", half_zig.bf16.toF64, .{half_zig.bf16.fromF32(std.math.floatMax(f32))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(floatTrueMin)", half_zig.bf16.toF64, .{half_zig.bf16.fromF32(std.math.floatTrueMin(f32))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(NEG_INFINITY)", half_zig.bf16.toF64, .{half_zig.bf16.NEG_INFINITY}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(INFINITY)", half_zig.bf16.toF64, .{half_zig.bf16.INFINITY}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(NAN)", half_zig.bf16.toF64, .{half_zig.bf16.NAN}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(E)", half_zig.bf16.toF64, .{half_zig.bf16.E}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::toF64(PI)", half_zig.bf16.toF64, .{half_zig.bf16.PI}, BENCH_ITERATIONS);
}

// Native f16 conversion benchmarks
fn benchF16FromF32() !void {
    print("\n\x1b[38;5;24m=== f16 From f32 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("f16::fromF32(0.0)", castToF16, .{0.0}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(-0.0)", castToF16, .{-0.0}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(1.0)", castToF16, .{1.0}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(floatMin)", castToF16, .{std.math.floatMin(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(floatMax)", castToF16, .{std.math.floatMax(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(floatTrueMin)", castToF16, .{std.math.floatTrueMin(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(-inf)", castToF16, .{-std.math.inf(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(inf)", castToF16, .{std.math.inf(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(nan)", castToF16, .{std.math.nan(f32)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(e)", castToF16, .{std.math.e}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF32(pi)", castToF16, .{std.math.pi}, BENCH_ITERATIONS);
}

fn benchF16FromF64() !void {
    print("\n\x1b[38;5;24m=== f16 From f64 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("f16::fromF64(0.0)", castToF16FromF64, .{@as(f64, 0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(-0.0)", castToF16FromF64, .{@as(f64, -0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(1.0)", castToF16FromF64, .{@as(f64, 1.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(floatMin)", castToF16FromF64, .{std.math.floatMin(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(floatMax)", castToF16FromF64, .{std.math.floatMax(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(floatTrueMin)", castToF16FromF64, .{std.math.floatTrueMin(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(-inf)", castToF16FromF64, .{-std.math.inf(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(inf)", castToF16FromF64, .{std.math.inf(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(nan)", castToF16FromF64, .{std.math.nan(f64)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(e)", castToF16FromF64, .{@as(f64, std.math.e)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::fromF64(pi)", castToF16FromF64, .{@as(f64, std.math.pi)}, BENCH_ITERATIONS);
}

fn benchF16ToF32() !void {
    print("\n\x1b[38;5;24m=== f16 to f32 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("f16::toF32(0.0)", castToF32, .{@as(f16, 0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(-0.0)", castToF32, .{@as(f16, -0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(1.0)", castToF32, .{@as(f16, 1.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(floatMin)", castToF32, .{std.math.floatMin(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(floatMax)", castToF32, .{std.math.floatMax(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(floatTrueMin)", castToF32, .{std.math.floatTrueMin(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(-inf)", castToF32, .{-std.math.inf(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(inf)", castToF32, .{std.math.inf(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(nan)", castToF32, .{std.math.nan(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(e)", castToF32, .{@as(f16, @floatCast(std.math.e))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF32(pi)", castToF32, .{@as(f16, @floatCast(std.math.pi))}, BENCH_ITERATIONS);
}

fn benchF16ToF64() !void {
    print("\n\x1b[38;5;24m=== f16 to f64 Benchmarks ===\x1b[0m\n", .{});

    try simpleBenchmarkFunction("f16::toF64(0.0)", castToF64, .{@as(f16, 0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(-0.0)", castToF64, .{@as(f16, -0.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(1.0)", castToF64, .{@as(f16, 1.0)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(floatMin)", castToF64, .{std.math.floatMin(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(floatMax)", castToF64, .{std.math.floatMax(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(floatTrueMin)", castToF64, .{std.math.floatTrueMin(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(-inf)", castToF64, .{-std.math.inf(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(inf)", castToF64, .{std.math.inf(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(nan)", castToF64, .{std.math.nan(f16)}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(e)", castToF64, .{@as(f16, @floatCast(std.math.e))}, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::toF64(pi)", castToF64, .{@as(f16, @floatCast(std.math.pi))}, BENCH_ITERATIONS);
}

// Helper functions for native f16 conversions
fn castToF16(val: f32) f16 {
    return @as(f16, @floatCast(val));
}

fn castToF16FromF64(val: f64) f16 {
    return @as(f16, @floatCast(val));
}

fn castToF32(val: f16) f32 {
    return @as(f32, @floatCast(val));
}

fn castToF64(val: f16) f64 {
    return @as(f64, @floatCast(val));
}

// Slice/array conversion benchmarks
fn benchSliceF32ToBf16() !void {
    print("\n\x1b[38;5;24m=== Slice f32 to bf16 Benchmarks ===\x1b[0m\n", .{});

    var allocator = std.heap.page_allocator;

    // Constants benchmark
    const constants = [_]f32{
        0.0,
        -0.0,
        1.0,
        std.math.floatMin(f32),
        std.math.floatMax(f32),
        std.math.floatTrueMin(f32),
        -std.math.inf(f32),
        std.math.inf(f32),
        std.math.nan(f32),
        std.math.e,
        std.math.pi,
    };

    const constant_buffer = try allocator.alloc(half_zig.bf16, constants.len);
    defer allocator.free(constant_buffer);

    try simpleBenchmarkFunction("slice_f32_to_bf16/constants", convertF32ArrayToBf16, .{ &constants, constant_buffer }, 10000);

    // Large array benchmark
    const large_f32 = try allocator.alloc(f32, SIMD_LARGE_BENCH_SLICE_LEN);
    defer allocator.free(large_f32);

    for (large_f32, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt(i));
    }

    const large_bf16_buffer = try allocator.alloc(half_zig.bf16, SIMD_LARGE_BENCH_SLICE_LEN);
    defer allocator.free(large_bf16_buffer);

    try simpleBenchmarkFunction("slice_f32_to_bf16/large", convertF32ArrayToBf16, .{ large_f32, large_bf16_buffer }, 1000);
}

fn benchSliceBf16ToF32() !void {
    print("\n\x1b[38;5;24m=== Slice bf16 to f32 Benchmarks ===\x1b[0m\n", .{});

    var allocator = std.heap.page_allocator;

    // Constants benchmark
    const constants = [_]half_zig.bf16{
        half_zig.bf16.ZERO,
        half_zig.bf16.NEG_ZERO,
        half_zig.bf16.ONE,
        half_zig.bf16.fromF32(std.math.floatMin(f32)),
        half_zig.bf16.fromF32(std.math.floatMax(f32)),
        half_zig.bf16.fromF32(std.math.floatTrueMin(f32)),
        half_zig.bf16.NEG_INFINITY,
        half_zig.bf16.INFINITY,
        half_zig.bf16.NAN,
        half_zig.bf16.E,
        half_zig.bf16.PI,
    };

    const constant_buffer = try allocator.alloc(f32, constants.len);
    defer allocator.free(constant_buffer);

    try simpleBenchmarkFunction("slice_bf16_to_f32/constants", convertBf16ArrayToF32, .{ &constants, constant_buffer }, 10000);

    // Large array benchmark
    const large_bf16 = try allocator.alloc(half_zig.bf16, SIMD_LARGE_BENCH_SLICE_LEN);
    defer allocator.free(large_bf16);

    for (large_bf16, 0..) |*val, i| {
        val.* = half_zig.bf16.fromF32(@as(f32, @floatFromInt(i)));
    }

    const large_f32_buffer = try allocator.alloc(f32, SIMD_LARGE_BENCH_SLICE_LEN);
    defer allocator.free(large_f32_buffer);

    try simpleBenchmarkFunction("slice_bf16_to_f32/large", convertBf16ArrayToF32, .{ large_bf16, large_f32_buffer }, 1000);
}

fn benchSliceF16ToF32() !void {
    print("\n\x1b[38;5;24m=== Slice f16 to f32 Benchmarks ===\x1b[0m\n", .{});

    var allocator = std.heap.page_allocator;

    // Constants benchmark
    const constants = [_]f16{
        0.0,
        -0.0,
        1.0,
        std.math.floatMin(f16),
        std.math.floatMax(f16),
        std.math.floatTrueMin(f16),
        -std.math.inf(f16),
        std.math.inf(f16),
        std.math.nan(f16),
        @as(f16, @floatCast(std.math.e)),
        @as(f16, @floatCast(std.math.pi)),
    };

    const constant_buffer = try allocator.alloc(f32, constants.len);
    defer allocator.free(constant_buffer);

    try simpleBenchmarkFunction("slice_f16_to_f32/constants", convertF16ArrayToF32, .{ &constants, constant_buffer }, 10000);

    // Large array benchmark
    const large_f16 = try allocator.alloc(f16, SIMD_LARGE_BENCH_SLICE_LEN);
    defer allocator.free(large_f16);

    for (large_f16, 0..) |*val, i| {
        val.* = @as(f16, @floatCast(@as(f32, @floatFromInt(i))));
    }

    const large_f32_buffer = try allocator.alloc(f32, SIMD_LARGE_BENCH_SLICE_LEN);
    defer allocator.free(large_f32_buffer);

    try simpleBenchmarkFunction("slice_f16_to_f32/large", convertF16ArrayToF32, .{ large_f16, large_f32_buffer }, 1000);
}

// Helper functions for slice conversions
fn convertF32ArrayToBf16(input: []const f32, output: []half_zig.bf16) void {
    for (input, 0..) |val, i| {
        output[i] = half_zig.bf16.fromF32(val);
    }
}

fn convertBf16ArrayToF32(input: []const half_zig.bf16, output: []f32) void {
    for (input, 0..) |val, i| {
        output[i] = val.toF32();
    }
}

fn convertF16ArrayToF32(input: []const f16, output: []f32) void {
    for (input, 0..) |val, i| {
        output[i] = @as(f32, @floatCast(val));
    }
}

// Arithmetic operation benchmarks
fn benchBf16Arithmetic() !void {
    print("\n\x1b[38;5;24m=== bf16 Arithmetic Benchmarks ===\x1b[0m\n", .{});

    const a = half_zig.bf16.fromF32(2.5);
    const b = half_zig.bf16.fromF32(1.5);

    try simpleBenchmarkFunction("bf16::add", half_zig.bf16.add, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::sub", half_zig.bf16.sub, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::mul", half_zig.bf16.mul, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::div", half_zig.bf16.div, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("bf16::rem", half_zig.bf16.rem, .{ a, b }, BENCH_ITERATIONS);
}

fn benchF16Arithmetic() !void {
    print("\n\x1b[38;5;24m=== f16 Arithmetic Benchmarks ===\x1b[0m\n", .{});

    const a: f16 = 2.5;
    const b: f16 = 1.5;

    try simpleBenchmarkFunction("f16::add", addF16, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::sub", subF16, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::mul", mulF16, .{ a, b }, BENCH_ITERATIONS);
    try simpleBenchmarkFunction("f16::div", divF16, .{ a, b }, BENCH_ITERATIONS);
}

fn addF16(a: f16, b: f16) f16 {
    return a + b;
}

fn subF16(a: f16, b: f16) f16 {
    return a - b;
}

fn mulF16(a: f16, b: f16) f16 {
    return a * b;
}

fn divF16(a: f16, b: f16) f16 {
    return a / b;
}

pub fn main() !void {
    print("\x1b[38;5;33m========================================\x1b[0m\n", .{});
    print("\x1b[38;5;39mHalf Precision Floating Point Benchmarks\x1b[0m\n", .{});
    print("\x1b[38;5;33m========================================\x1b[0m\n", .{});

    // Single value conversion benchmarks
    try benchBf16FromF32();
    try benchBf16FromF64();
    try benchBf16ToF32();
    try benchBf16ToF64();

    try benchF16FromF32();
    try benchF16FromF64();
    try benchF16ToF32();
    try benchF16ToF64();

    // Slice conversion benchmarks
    try benchSliceF32ToBf16();
    try benchSliceBf16ToF32();
    try benchSliceF16ToF32();

    // Arithmetic benchmarks
    try benchBf16Arithmetic();
    try benchF16Arithmetic();

    print("\n\x1b[38;5;39mBenchmarks completed!\x1b[0m\n", .{});
}
