const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library
    const lib = b.addStaticLibrary(.{
        .name = "half_zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // Floating point conversion benchmarks
    const fp_benchmark = b.addExecutable(.{
        .name = "fp_benchmark",
        .root_source_file = b.path("bench/convert.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    fp_benchmark.root_module.addImport("half_zig", lib.root_module);

    const install_fp_benchmark = b.addInstallArtifact(fp_benchmark, .{});
    const fp_benchmark_step = b.step("benchmark-fp", "Build floating point benchmarks");
    fp_benchmark_step.dependOn(&install_fp_benchmark.step);

    const run_fp_benchmark = b.addRunArtifact(fp_benchmark);
    const run_fp_benchmark_step = b.step("bench-fp", "Run floating point benchmarks");
    run_fp_benchmark_step.dependOn(&run_fp_benchmark.step);

    // Quantization benchmarks
    const quant_benchmark = b.addExecutable(.{
        .name = "quant_benchmark",
        .root_source_file = b.path("bench/quantization.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    quant_benchmark.root_module.addImport("half_zig", lib.root_module);

    const install_quant_benchmark = b.addInstallArtifact(quant_benchmark, .{});
    const quant_benchmark_step = b.step("benchmark-quant", "Build quantization benchmarks");
    quant_benchmark_step.dependOn(&install_quant_benchmark.step);

    const run_quant_benchmark = b.addRunArtifact(quant_benchmark);
    const run_quant_benchmark_step = b.step("bench-quant", "Run quantization benchmarks");
    run_quant_benchmark_step.dependOn(&run_quant_benchmark.step);

    // All benchmarks
    const all_benchmark_step = b.step("benchmark", "Build all benchmarks");
    all_benchmark_step.dependOn(&install_fp_benchmark.step);
    all_benchmark_step.dependOn(&install_quant_benchmark.step);

    const run_all_benchmark_step = b.step("bench", "Run all benchmarks");
    run_all_benchmark_step.dependOn(&run_fp_benchmark.step);
    run_all_benchmark_step.dependOn(&run_quant_benchmark.step);

    // Basic example
    const basic_example = b.addExecutable(.{
        .name = "basic_example",
        .root_source_file = b.path("examples/basic.zig"),
        .target = target,
        .optimize = optimize,
    });
    basic_example.root_module.addImport("half_zig", lib.root_module);

    const install_basic_example = b.addInstallArtifact(basic_example, .{});
    const basic_example_step = b.step("example-basic", "Build basic example");
    basic_example_step.dependOn(&install_basic_example.step);

    const run_basic_example = b.addRunArtifact(basic_example);
    const run_basic_example_step = b.step("run-basic", "Run basic example");
    run_basic_example_step.dependOn(&run_basic_example.step);

    // Quantization example
    const quant_example = b.addExecutable(.{
        .name = "quantization_example",
        .root_source_file = b.path("examples/quantization.zig"),
        .target = target,
        .optimize = optimize,
    });
    quant_example.root_module.addImport("half_zig", lib.root_module);

    const install_quant_example = b.addInstallArtifact(quant_example, .{});
    const quant_example_step = b.step("example-quant", "Build quantization example");
    quant_example_step.dependOn(&install_quant_example.step);

    const run_quant_example = b.addRunArtifact(quant_example);
    const run_quant_example_step = b.step("run-quant", "Run quantization example");
    run_quant_example_step.dependOn(&run_quant_example.step);

    // All examples
    const example_step = b.step("example", "Build all examples");
    example_step.dependOn(&install_basic_example.step);
    example_step.dependOn(&install_quant_example.step);

    const run_example_step = b.step("run-examples", "Run all examples");
    run_example_step.dependOn(&run_basic_example.step);
    run_example_step.dependOn(&run_quant_example.step);
}
