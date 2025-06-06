# Benchmarks

This directory contains performance benchmarks for the half_zig library.

## Running Benchmarks

```bash
# Build and run all benchmarks
zig build bench

# Or build and run manually
zig build benchmark
./zig-out/bin/benchmark
```

## Benchmark Categories

### Single Value Conversions
- `bf16` ↔ `f32`/`f64` conversions
- Native `f16` ↔ `f32`/`f64` conversions

### Array/Slice Conversions
- Bulk conversion operations for arrays
- Both constants and large arrays (1024 elements)

### Arithmetic Operations
- Basic arithmetic operations for both `bf16` and native `f16`
- Addition, subtraction, multiplication, division, remainder

## Benchmark Structure

The benchmarks are designed to be similar to the Rust `half-rs` crate benchmarks:
- Same test values and edge cases
- Similar array sizes for SIMD comparisons
- Consistent iteration counts for meaningful results

## Performance Notes

- Benchmarks are compiled with `-O ReleaseFast` for optimal performance
- Each operation is repeated many times to get stable timing measurements
- Results are reported in nanoseconds per iteration