# half_zig

A comprehensive half-precision floating point and neural network quantization library for Zig.

## Features

### Floating Point Types
- **bfloat16**: Brain floating point 16-bit format used in machine learning
- **f16 utilities**: Enhanced utilities for native IEEE 754 half precision

### Quantized Integer Types
- **qint8/quint8**: 8-bit signed/unsigned quantized integers
- **qint4/quint4**: 4-bit signed/unsigned quantized integers  
- **PackedInt4Array**: Memory-efficient packed storage for int4 arrays
- **BlockQuantizedInt4**: Block-wise adaptive quantization for better accuracy

## Usage

### Basic Half Precision

```zig
const half_zig = @import("half_zig");

// bfloat16 operations
const a = half_zig.bf16.fromF32(3.14);
const b = half_zig.bf16.fromF32(2.71);
const result = a.add(b);
print("Result: {}\n", .{result.toF32()}); // 5.85

// Native f16 with utilities
const f16_array = [_]f16{ 1.0, -2.5, 3.7, -0.8 };
const f32_result = try half_zig.f16_utils.f16ArrayToF32(allocator, &f16_array);
```

### Neural Network Quantization

```zig
// 8-bit quantization
const weights = [_]f32{ 1.5, -2.3, 0.8, -0.1, 3.7 };
const scale: f32 = 0.1;
const zero_point: i8 = 0;

var quantized: [weights.len]half_zig.qint8 = undefined;
for (weights, 0..) |w, i| {
    quantized[i] = half_zig.qint8.fromF32(w, scale, zero_point);
}

// 4-bit packed arrays for extreme compression
var packed = try half_zig.PackedInt4Array.fromF32Array(allocator, &weights, 0.5);
defer packed.deinit();

// Block-wise quantization for mixed ranges
var block_quantized = try half_zig.BlockQuantizedInt4.fromF32Array(allocator, &weights);
defer block_quantized.deinit();
```

## Building

```bash
# Run all tests
zig build test

# Build all examples
zig build example

# Run examples
zig build run-basic      # Basic floating point operations
zig build run-quant      # Quantization demonstrations

# Run benchmarks
zig build bench          # All benchmarks
zig build bench-fp       # Floating point benchmarks only
zig build bench-quant    # Quantization benchmarks only
```

## Performance

The library is optimized for neural network workloads:

- **Memory efficiency**: int4 provides 8x compression vs f32
- **Cache performance**: Packed storage reduces memory bandwidth
- **Quantization accuracy**: Block-wise scaling minimizes error
- **Hardware friendly**: Formats compatible with modern AI accelerators

## Quantization Formats

| Type | Bits | Range | Use Case |
|------|------|--------|----------|
| qint8 | 8 | -128 to 127 | General quantization |
| quint8 | 8 | 0 to 255 | Activations |
| qint4 | 4 | -8 to 7 | Extreme compression |
| quint4 | 4 | 0 to 15 | Sparse networks |

## Examples

See the `examples/` directory for:
- Basic floating point operations
- Quantization techniques
- Memory efficiency analysis
- Performance comparisons

## License

MIT License - see LICENSE file for details.