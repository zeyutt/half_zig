//! Brain Floating Point 16-bit (bfloat16) implementation
//!
//! bfloat16 is a 16-bit floating point format with:
//! - 1 bit sign
//! - 8 bits exponent (same as f32)
//! - 7 bits mantissa
//!
//! This format is widely used in machine learning due to its larger
//! dynamic range compared to IEEE 754 f16.

const std = @import("std");
const math = std.math;

/// Brain Floating Point 16-bit type
pub const bf16 = packed struct {
    /// Raw 16-bit representation
    bits: u16,

    // Constants
    pub const SIGN_MASK: u16 = 0x8000;
    pub const EXPONENT_MASK: u16 = 0x7F80;
    pub const MANTISSA_MASK: u16 = 0x007F;
    pub const EXPONENT_BIAS: i32 = 127;
    pub const MANTISSA_BITS: u8 = 7;
    pub const EXPONENT_BITS: u8 = 8;

    // Special values
    pub const ZERO: bf16 = bf16{ .bits = 0x0000 };
    pub const NEG_ZERO: bf16 = bf16{ .bits = 0x8000 };
    pub const ONE: bf16 = bf16{ .bits = 0x3F80 };
    pub const NEG_ONE: bf16 = bf16{ .bits = 0xBF80 };
    pub const INFINITY: bf16 = bf16{ .bits = 0x7F80 };
    pub const NEG_INFINITY: bf16 = bf16{ .bits = 0xFF80 };
    pub const NAN: bf16 = bf16{ .bits = 0x7FC0 };

    // Mathematical constants (approximated to bf16 precision)
    pub const E: bf16 = bf16{ .bits = 0x402E }; // ≈ 2.71828
    pub const PI: bf16 = bf16{ .bits = 0x4049 }; // ≈ 3.14159
    pub const FRAC_1_PI: bf16 = bf16{ .bits = 0x3EA3 }; // ≈ 1/π
    pub const FRAC_2_PI: bf16 = bf16{ .bits = 0x3F23 }; // ≈ 2/π
    pub const FRAC_PI_2: bf16 = bf16{ .bits = 0x3FC9 }; // ≈ π/2
    pub const FRAC_PI_4: bf16 = bf16{ .bits = 0x3F49 }; // ≈ π/4
    pub const SQRT_2: bf16 = bf16{ .bits = 0x3FB5 }; // ≈ √2
    pub const FRAC_1_SQRT_2: bf16 = bf16{ .bits = 0x3F35 }; // ≈ 1/√2

    /// Create bf16 from raw bits
    pub fn fromBits(bits: u16) bf16 {
        return bf16{ .bits = bits };
    }

    /// Get raw bits
    pub fn toBits(self: bf16) u16 {
        return self.bits;
    }

    /// Convert f32 to bf16
    pub fn fromF32(val: f32) bf16 {
        const f32_bits = @as(u32, @bitCast(val));

        // Handle special cases
        if (math.isNan(val)) {
            return NAN;
        }
        if (math.isInf(val)) {
            if (math.isPositiveInf(val)) return INFINITY;
            return NEG_INFINITY;
        }

        // Simple truncation - take upper 16 bits of f32
        // This is the most common bfloat16 conversion method
        const bf16_bits = @as(u16, @truncate(f32_bits >> 16));
        return bf16{ .bits = bf16_bits };
    }

    /// Convert bf16 to f32
    pub fn toF32(self: bf16) f32 {
        // Extend bf16 to f32 by adding 16 zero bits to the right
        const f32_bits = (@as(u32, self.bits) << 16);
        return @as(f32, @bitCast(f32_bits));
    }

    /// Convert f64 to bf16
    pub fn fromF64(val: f64) bf16 {
        // Convert through f32 for simplicity
        return fromF32(@as(f32, @floatCast(val)));
    }

    /// Convert bf16 to f64
    pub fn toF64(self: bf16) f64 {
        return @as(f64, @floatCast(self.toF32()));
    }

    /// Check if value is NaN
    pub fn isNan(self: bf16) bool {
        const exp = (self.bits & EXPONENT_MASK) >> MANTISSA_BITS;
        const mantissa = self.bits & MANTISSA_MASK;
        return exp == 0xFF and mantissa != 0;
    }

    /// Check if value is infinite
    pub fn isInfinite(self: bf16) bool {
        const exp = (self.bits & EXPONENT_MASK) >> MANTISSA_BITS;
        const mantissa = self.bits & MANTISSA_MASK;
        return exp == 0xFF and mantissa == 0;
    }

    /// Check if value is finite
    pub fn isFinite(self: bf16) bool {
        return !self.isNan() and !self.isInfinite();
    }

    /// Check if value is normal (not zero, subnormal, infinite, or NaN)
    pub fn isNormal(self: bf16) bool {
        const exp = (self.bits & EXPONENT_MASK) >> MANTISSA_BITS;
        return exp != 0 and exp != 0xFF;
    }

    /// Check if sign is negative
    pub fn isSignNegative(self: bf16) bool {
        return (self.bits & SIGN_MASK) != 0;
    }

    /// Check if sign is positive
    pub fn isSignPositive(self: bf16) bool {
        return (self.bits & SIGN_MASK) == 0;
    }

    /// Absolute value
    pub fn abs(self: bf16) bf16 {
        return bf16{ .bits = self.bits & ~SIGN_MASK };
    }

    /// Copy sign from another bf16
    pub fn copySign(self: bf16, sign: bf16) bf16 {
        return bf16{ .bits = (self.bits & ~SIGN_MASK) | (sign.bits & SIGN_MASK) };
    }

    /// Negate
    pub fn neg(self: bf16) bf16 {
        return bf16{ .bits = self.bits ^ SIGN_MASK };
    }

    /// Maximum of two values
    pub fn max(self: bf16, other: bf16) bf16 {
        if (self.isNan()) return other;
        if (other.isNan()) return self;
        return if (self.toF32() > other.toF32()) self else other;
    }

    /// Minimum of two values
    pub fn min(self: bf16, other: bf16) bf16 {
        if (self.isNan()) return other;
        if (other.isNan()) return self;
        return if (self.toF32() < other.toF32()) self else other;
    }

    /// Addition
    pub fn add(self: bf16, other: bf16) bf16 {
        return fromF32(self.toF32() + other.toF32());
    }

    /// Subtraction
    pub fn sub(self: bf16, other: bf16) bf16 {
        return fromF32(self.toF32() - other.toF32());
    }

    /// Multiplication
    pub fn mul(self: bf16, other: bf16) bf16 {
        return fromF32(self.toF32() * other.toF32());
    }

    /// Division
    pub fn div(self: bf16, other: bf16) bf16 {
        return fromF32(self.toF32() / other.toF32());
    }

    /// Remainder
    pub fn rem(self: bf16, other: bf16) bf16 {
        return fromF32(@rem(self.toF32(), other.toF32()));
    }

    /// Format for printing
    pub fn format(
        self: bf16,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        // 使用可用的公开API格式化浮点数
        // 将bf16转换为f32，然后进行格式化
        const f32_val = self.toF32();

        // 直接使用标准格式化方式
        try std.fmt.formatType(f32_val, fmt, options, writer, 0);
    }
};

// Tests
test "bf16 basic constants" {
    std.debug.print("\n\x1b[36m ▶ Test: bf16 basic constants\n", .{});
    const testing = std.testing;

    try testing.expect(bf16.ZERO.toBits() == 0x0000);
    try testing.expect(bf16.ONE.toBits() == 0x3F80);
    try testing.expect(bf16.NEG_ONE.toBits() == 0xBF80);
    try testing.expect(bf16.INFINITY.toBits() == 0x7F80);
    try testing.expect(bf16.NEG_INFINITY.toBits() == 0xFF80);
}

test "bf16 f32 conversion" {
    std.debug.print("\n\x1b[36m ▶ Test: bf16 f32 conversion\n", .{});
    const testing = std.testing;

    // Test basic values
    try testing.expectEqual(@as(f32, 0.0), bf16.fromF32(0.0).toF32());
    try testing.expectEqual(@as(f32, 1.0), bf16.fromF32(1.0).toF32());
    try testing.expectEqual(@as(f32, -1.0), bf16.fromF32(-1.0).toF32());

    // Test special values
    const inf_bf16 = bf16.fromF32(std.math.inf(f32));
    try testing.expect(inf_bf16.isInfinite());
    try testing.expect(inf_bf16.isSignPositive());

    const neg_inf_bf16 = bf16.fromF32(-std.math.inf(f32));
    try testing.expect(neg_inf_bf16.isInfinite());
    try testing.expect(neg_inf_bf16.isSignNegative());

    const nan_bf16 = bf16.fromF32(std.math.nan(f32));
    try testing.expect(nan_bf16.isNan());
}

test "bf16 arithmetic operations" {
    std.debug.print("\n\x1b[36m ▶ Test: bf16 arithmetic operations\x1b[0m\n", .{});
    const testing = std.testing;

    const a = bf16.fromF32(2.0);
    const b = bf16.fromF32(3.0);

    try testing.expectApproxEqAbs(@as(f32, 5.0), a.add(b).toF32(), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), a.sub(b).toF32(), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 6.0), a.mul(b).toF32(), 1e-6);

    // bf16 has limited precision (7 bits mantissa), so we need larger tolerance for division
    // 2.0 / 3.0 = 0.6666667 in exact arithmetic
    // In bf16: 0.6640625 (due to truncation of mantissa)
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), a.div(b).toF32(), 0.01);
}

test "bf16 special value checks" {
    std.debug.print("\n\x1b[36m ▶ Test: bf16 special value checks\x1b[0m\n", .{});
    const testing = std.testing;

    try testing.expect(bf16.NAN.isNan());
    try testing.expect(!bf16.ONE.isNan());

    try testing.expect(bf16.INFINITY.isInfinite());
    try testing.expect(!bf16.ONE.isInfinite());

    try testing.expect(bf16.ONE.isFinite());
    try testing.expect(!bf16.INFINITY.isFinite());

    try testing.expect(bf16.ONE.isNormal());
    try testing.expect(!bf16.ZERO.isNormal());

    try testing.expect(bf16.NEG_ONE.isSignNegative());
    try testing.expect(bf16.ONE.isSignPositive());
}

test "bf16 min/max with NaN" {
    std.debug.print("\n\x1b[36m ▶ Test: bf16 min/max with NaN\x1b[0m\n", .{});
    const testing = std.testing;

    const a = bf16.fromF32(1.0);
    const b = bf16.fromF32(2.0);
    const nan = bf16.NAN;

    try testing.expectEqual(b, a.max(b));
    try testing.expectEqual(a, a.min(b));

    // NaN handling
    try testing.expectEqual(a, nan.max(a));
    try testing.expectEqual(a, a.max(nan));
    try testing.expectEqual(a, nan.min(a));
    try testing.expectEqual(a, a.min(nan));
}

test "bf16 precision analysis" {
    std.debug.print("\n\x1b[36m ▶ Test: bf16 precision analysis\x1b[0m\n", .{});
    const testing = std.testing;

    // Let's understand what precision we actually get with bf16
    const two_thirds_f32: f32 = 2.0 / 3.0;
    const two_thirds_bf16 = bf16.fromF32(two_thirds_f32);
    const back_to_f32 = two_thirds_bf16.toF32();

    // Print values for debugging
    std.debug.print("  \nPrecision analysis:\n", .{});
    std.debug.print("  Original f32: {}\n", .{two_thirds_f32});
    std.debug.print("  bf16 bits: 0x{X}\n", .{two_thirds_bf16.toBits()});
    std.debug.print("  Back to f32: {}\n", .{back_to_f32});
    std.debug.print("  Difference: {}\n", .{@abs(two_thirds_f32 - back_to_f32)});

    // The difference should be within bf16's precision limits
    // bf16 has 7 bits of mantissa, so the precision is roughly 2^-7 ≈ 0.0078
    try testing.expectApproxEqAbs(two_thirds_f32, back_to_f32, 0.01);
}
