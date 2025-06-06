//! Example demonstrating quantized integer types for neural networks

const std = @import("std");
const print = std.debug.print;
const half_zig = @import("half_zig");

pub fn main() !void {
    print("\x1b[35m===================================\x1b[0m\n", .{});
    print("\x1b[35mNeural Network Quantization Examples\x1b[0m\n", .{});
    print("\x1b[35m===================================\x1b[0m\n\n", .{});

    // Example 1: Basic qint8 operations
    print("\x1b[38;5;219m1. qint8 (8-bit signed quantization)\x1b[0m\n", .{});
    print("\x1b[38;5;219m-------------------------------------\x1b[0m\n", .{});

    const scale: f32 = 0.1;
    const zero_point: i8 = 0;

    const weights = [_]f32{ 1.5, -2.3, 0.8, -0.1, 3.7 };
    print("Original weights: ", .{});
    for (weights) |w| {
        print("{d:.2} ", .{w});
    }
    print("\n", .{});

    print("Quantized weights (qint8): ", .{});
    var quantized_weights: [weights.len]half_zig.qint8 = undefined;
    for (weights, 0..) |w, i| {
        quantized_weights[i] = half_zig.qint8.fromF32(w, scale, zero_point);
        print("{} ", .{quantized_weights[i]});
    }
    print("\n", .{});

    print("Dequantized weights: ", .{});
    for (quantized_weights) |qw| {
        const dequantized = qw.toF32(scale, zero_point);
        print("{d:.2} ", .{dequantized});
    }
    print("\n\n", .{});

    // Example 2: qint4 for extreme compression
    print("\x1b[38;5;219m2. qint4 (4-bit signed quantization)\x1b[0m\n", .{});
    print("\x1b[38;5;219m-------------------------------------\x1b[0m\n", .{});

    const activations = [_]f32{ 0.5, -1.2, 2.1, -0.8, 1.9, 0.0, -2.5, 1.3 };
    const q4_scale: f32 = 0.4;

    print("Original activations: ", .{});
    for (activations) |a| {
        print("{d:.2} ", .{a});
    }
    print("\n", .{});

    print("Quantized activations (qint4): ", .{});
    var q4_activations: [activations.len]half_zig.qint4 = undefined;
    for (activations, 0..) |a, i| {
        q4_activations[i] = half_zig.qint4.fromF32(a, q4_scale);
        print("{} ", .{q4_activations[i]});
    }
    print("\n", .{});

    print("Dequantized activations: ", .{});
    for (q4_activations) |qa| {
        const dequantized = qa.toF32(q4_scale);
        print("{d:.2} ", .{dequantized});
    }
    print("\n\n", .{});

    // Example 3: Packed int4 array for memory efficiency
    print("\x1b[38;5;219m3. PackedInt4Array (memory-efficient storage)\x1b[0m\n", .{});
    print("\x1b[38;5;219m----------------------------------------------\x1b[0m\n", .{});

    var allocator = std.heap.page_allocator;

    const large_weights = [_]f32{ 1.2, -0.8, 2.3, -1.1, 0.7, -2.0, 1.8, -0.3, 0.9, -1.5, 2.7, -0.6, 1.4, -1.9, 0.2, -2.1 };

    var packed_four = try half_zig.PackedInt4Array.fromF32Array(allocator, &large_weights, 0.5);
    defer packed_four.deinit();

    print("Original array length: {} elements\n", .{large_weights.len});
    print("Packed storage: {} bytes ({d:.1}x compression)\n", .{ packed_four.data.len, @as(f32, @floatFromInt(large_weights.len * 4)) / @as(f32, @floatFromInt(packed_four.data.len)) });

    const reconstructed = try packed_four.toF32Array(allocator, 0.5);
    defer allocator.free(reconstructed);

    print("Reconstruction accuracy:\n", .{});
    var total_error: f32 = 0.0;
    for (large_weights, 0..) |original, i| {
        const abs_error = @abs(original - reconstructed[i]);
        total_error += abs_error;
        if (i < 8) { // Show first 8 values
            print("  {d:.2} -> {d:.2} (error: {d:.3})\n", .{ original, reconstructed[i], abs_error });
        }
    }
    print("Average error: {d:.3}\n\n", .{total_error / @as(f32, @floatFromInt(large_weights.len))});

    // Example 4: Block-wise quantization for better accuracy
    print("\x1b[38;5;219m4. BlockQuantizedInt4 (adaptive scaling)\x1b[0m\n", .{});
    print("\x1b[38;5;219m-----------------------------------------\x1b[0m\n", .{});

    const mixed_range_data = [_]f32{
        // Small values
        0.1, -0.05, 0.12, -0.08, 0.09, -0.11, 0.07, -0.06,
        // Medium values
        1.2, -0.8,  1.5,  -1.1,  0.9,  -1.3,  1.1,  -0.7,
        // Large values
        5.2, -4.8,  6.1,  -5.5,  4.9,  -5.8,  5.7,  -4.3,
        // Mixed range
        0.2, -2.1,  0.8,  -4.5,  1.2,  -0.3,  3.7,  -1.8,
    };

    var block_quantized = try half_zig.BlockQuantizedInt4.fromF32Array(allocator, &mixed_range_data);
    defer block_quantized.deinit();

    print("Block count: {}\n", .{block_quantized.block_count});
    print("Scales per block: ", .{});
    for (block_quantized.scales) |scale_val| {
        print("{d:.3} ", .{scale_val});
    }
    print("\n", .{});

    const block_reconstructed = try block_quantized.toF32Array(allocator);
    defer allocator.free(block_reconstructed);

    print("Block quantization accuracy:\n", .{});
    var block_total_error: f32 = 0.0;
    for (mixed_range_data, 0..) |original, i| {
        const abs_error = @abs(original - block_reconstructed[i]);
        block_total_error += abs_error;
        if (i % 8 == 0) { // Show first value of each block
            print("  Block {}: {d:.2} -> {d:.2} (error: {d:.3})\n", .{ i / 8, original, block_reconstructed[i], abs_error });
        }
    }
    print("Average error: {d:.3}\n\n", .{block_total_error / @as(f32, @floatFromInt(mixed_range_data.len))});

    // Example 5: Comparison of different quantization methods
    print("\x1b[38;5;219m5. Quantization Method Comparison\x1b[0m\n", .{});
    print("\x1b[38;5;219m---------------------------------\x1b[0m\n", .{});

    const test_data = [_]f32{ 2.7, -1.8, 0.5, -3.2, 1.1, -0.7, 2.9, -2.1 };

    // bf16 (reference)
    print("bf16 (reference):     ", .{});
    for (test_data) |val| {
        const bf16_val = half_zig.bf16.fromF32(val);
        const reconstructed_val = bf16_val.toF32();
        print("{d:.3} ", .{reconstructed_val});
    }
    print("\n", .{});

    // qint8
    print("qint8 (scale=0.1):    ", .{});
    for (test_data) |val| {
        const q8_val = half_zig.qint8.fromF32(val, 0.1, 0);
        const reconstructed_val = q8_val.toF32(0.1, 0);
        print("{d:.3} ", .{reconstructed_val});
    }
    print("\n", .{});

    // qint4
    print("qint4 (scale=0.5):    ", .{});
    for (test_data) |val| {
        const q4_val = half_zig.qint4.fromF32(val, 0.5);
        const reconstructed_val = q4_val.toF32(0.5);
        print("{d:.3} ", .{reconstructed_val});
    }
    print("\n", .{});

    print("\n\x1b[35mQuantization complete!\x1b[0m\n", .{});
}
