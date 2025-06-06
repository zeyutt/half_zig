# half_zig

half_zig 是一个全面的 Zig 语言半精度浮点数和神经网络量化库。

## 特性

### 浮点类型
- **bfloat16**: 机器学习中使用的大脑浮点16位格式
- **f16工具**: 为原生 IEEE 754 半精度提供增强工具

### 量化整数类型
- **qint8/quint8**: 8位有符号/无符号量化整数
- **qint4/quint4**: 4位有符号/无符号量化整数
- **PackedInt4Array**: 内存高效的int4数组压缩存储
- **BlockQuantizedInt4**: 分块自适应量化，提供更高精度

## 使用方法

### 基本半精度

```zig
const half_zig = @import("half_zig");

// bfloat16 操作
const a = half_zig.bf16.fromF32(3.14);
const b = half_zig.bf16.fromF32(2.71);
const result = a.add(b);
print("结果: {}\n", .{result.toF32()}); // 5.85

// 原生f16及工具
const f16_array = [_]f16{ 1.0, -2.5, 3.7, -0.8 };
const f32_result = try half_zig.f16_utils.f16ArrayToF32(allocator, &f16_array);
```

### 神经网络量化

```zig
// 8位量化
const weights = [_]f32{ 1.5, -2.3, 0.8, -0.1, 3.7 };
const scale: f32 = 0.1;
const zero_point: i8 = 0;

var quantized: [weights.len]half_zig.qint8 = undefined;
for (weights, 0..) |w, i| {
    quantized[i] = half_zig.qint8.fromF32(w, scale, zero_point);
}

// 4位压缩数组用于极限压缩
var packed = try half_zig.PackedInt4Array.fromF32Array(allocator, &weights, 0.5);
defer packed.deinit();

// 分块量化处理混合范围
var block_quantized = try half_zig.BlockQuantizedInt4.fromF32Array(allocator, &weights);
defer block_quantized.deinit();
```

## 构建

```bash
# 运行所有测试
zig build test

# 构建所有示例
zig build example

# 运行示例
zig build run-basic      # 基本浮点操作
zig build run-quant      # 量化演示

# 运行基准测试
zig build bench          # 所有基准测试
zig build bench-fp       # 仅浮点基准测试
zig build bench-quant    # 仅量化基准测试
```

## 性能

该库针对神经网络工作负载进行了优化:

- **内存效率**: int4 相比 f32 提供8倍压缩率
- **缓存性能**: 压缩存储减少内存带宽
- **量化精度**: 分块缩放最小化误差
- **硬件友好**: 格式兼容现代AI加速器

## 量化格式

| 类型 | 位数 | 范围 | 用例 |
|------|------|--------|----------|
| qint8 | 8 | -128 到 127 | 通用量化 |
| quint8 | 8 | 0 到 255 | 激活函数 |
| qint4 | 4 | -8 到 7 | 极限压缩 |
| quint4 | 4 | 0 到 15 | 稀疏网络 |

## 示例

参见 `examples/` 目录获取:
- 基本浮点运算
- 量化技术
- 内存效率分析
- 性能比较

## 许可证

本库中的所有文件均采用双重许可，按照以下任一许可条款分发:

* [MIT 许可证](LICENSE-MIT)
  ([http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* [Apache 许可证, 版本 2.0](LICENSE-APACHE)
  ([http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

由您选择。

### 贡献

除非您明确声明，否则您有意提交包含在作品中的任何贡献，按照Apache-2.0许可证中的定义，均应按上述方式双重许可，无任何其他条款或条件。