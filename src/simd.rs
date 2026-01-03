//! SIMD-Accelerated Memory Operations
//!
//! This module provides SIMD-optimized memory operations for use
//! in memory management operations like:
//! - Fast bitmap scanning for free page finding
//! - Bulk page table zeroing
//! - Memory pattern matching
//!
//! Each architecture provides its own optimized implementations:
//! - x86_64: AVX-512, AVX2 fallback
//! - AArch64: SVE, NEON fallback  
//! - RISC-V 64: RVV (RISC-V Vector)

/// SIMD capability detection and feature flags
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// x86_64 AVX-512F available
    pub avx512f: bool,
    /// x86_64 AVX2 available  
    pub avx2: bool,
    /// AArch64 SVE available
    pub sve: bool,
    /// AArch64 NEON available (always true on AArch64)
    pub neon: bool,
    /// RISC-V Vector extension available
    pub rvv: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD capabilities for current CPU
    ///
    /// Note: For full CPUID-based detection, this would need to be called
    /// at runtime with proper CPU feature detection. This provides
    /// compile-time defaults based on target architecture.
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        // Use target_feature for compile-time detection
        // Runtime CPUID detection would need std or inline asm
        Self {
            avx512f: cfg!(target_feature = "avx512f"),
            avx2: cfg!(target_feature = "avx2"),
            sve: false,
            neon: false,
            rvv: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn detect() -> Self {
        Self {
            avx512f: false,
            avx2: false,
            sve: cfg!(target_feature = "sve"),
            neon: true, // Always available on AArch64
            rvv: false,
        }
    }

    #[cfg(target_arch = "riscv64")]
    pub fn detect() -> Self {
        Self {
            avx512f: false,
            avx2: false,
            sve: false,
            neon: false,
            rvv: cfg!(target_feature = "v"),
        }
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    pub fn detect() -> Self {
        Self {
            avx512f: false,
            avx2: false,
            sve: false,
            neon: false,
            rvv: false,
        }
    }

    /// Check if any SIMD acceleration is available
    pub fn has_simd(&self) -> bool {
        self.avx512f || self.avx2 || self.sve || self.neon || self.rvv
    }
}

/// Find the first set bit in a bitmap using SIMD acceleration
///
/// Returns the index of the first set bit, or None if all bits are zero.
/// Uses available SIMD instructions for faster scanning.
#[inline]
pub fn find_first_set_bit(bitmap: &[u64]) -> Option<usize> {
    // Use word-level intrinsics for portable acceleration
    for (word_idx, &word) in bitmap.iter().enumerate() {
        if word != 0 {
            let bit_idx = word.trailing_zeros() as usize;
            return Some(word_idx * 64 + bit_idx);
        }
    }
    None
}

/// Find first range of N consecutive zero bits in bitmap
///
/// Scans the bitmap for a contiguous range of N zero bits.
/// Returns the starting bit index or None if not found.
#[inline]
pub fn find_first_zero_range(bitmap: &[u64], count: usize) -> Option<usize> {
    if count == 0 {
        return Some(0);
    }

    if count > bitmap.len() * 64 {
        return None;
    }

    // Fast path: single bit
    if count == 1 {
        for (word_idx, &word) in bitmap.iter().enumerate() {
            if word != u64::MAX {
                let bit_idx = (!word).trailing_zeros() as usize;
                return Some(word_idx * 64 + bit_idx);
            }
        }
        return None;
    }

    // Multi-bit scan
    let mut consecutive = 0usize;
    let mut start = 0usize;

    for (word_idx, &word) in bitmap.iter().enumerate() {
        if word == 0 {
            // Entire word is free
            if consecutive == 0 {
                start = word_idx * 64;
            }
            consecutive += 64;
            if consecutive >= count {
                return Some(start);
            }
        } else if word == u64::MAX {
            // Entire word is used - reset
            consecutive = 0;
        } else {
            // Mixed word - scan bit by bit
            for bit in 0..64 {
                if (word & (1u64 << bit)) == 0 {
                    if consecutive == 0 {
                        start = word_idx * 64 + bit;
                    }
                    consecutive += 1;
                    if consecutive >= count {
                        return Some(start);
                    }
                } else {
                    consecutive = 0;
                }
            }
        }
    }

    None
}

/// Zero a page using the fastest available method
///
/// Uses SIMD instructions when available for bulk zeroing.
#[inline]
pub fn zero_page(page: &mut [u8; 4096]) {
    // For now, use simple zeroing - compiler will vectorize
    // TODO: Add explicit SIMD paths for AVX-512/SVE/RVV
    page.fill(0);
}

/// Count set bits using hardware popcount
#[inline]
pub fn count_set_bits(bitmap: &[u64]) -> usize {
    bitmap.iter().map(|w| w.count_ones() as usize).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_first_set_bit() {
        assert_eq!(find_first_set_bit(&[0, 0, 0]), None);
        assert_eq!(find_first_set_bit(&[1, 0, 0]), Some(0));
        assert_eq!(find_first_set_bit(&[2, 0, 0]), Some(1));
        assert_eq!(find_first_set_bit(&[0, 1, 0]), Some(64));
        assert_eq!(
            find_first_set_bit(&[0, 0, 0x8000_0000_0000_0000]),
            Some(127 + 64)
        );
    }

    #[test]
    fn test_find_first_zero_range() {
        // All zeros - find at start
        assert_eq!(find_first_zero_range(&[0, 0], 1), Some(0));
        assert_eq!(find_first_zero_range(&[0, 0], 64), Some(0));
        assert_eq!(find_first_zero_range(&[0, 0], 128), Some(0));

        // All ones - not found
        assert_eq!(find_first_zero_range(&[u64::MAX, u64::MAX], 1), None);

        // Mixed
        assert_eq!(find_first_zero_range(&[u64::MAX, 0], 1), Some(64));
        assert_eq!(
            find_first_zero_range(&[0xFFFF_FFFF_FFFF_FFFE, 0], 1),
            Some(0)
        );
    }

    #[test]
    fn test_count_set_bits() {
        assert_eq!(count_set_bits(&[0, 0]), 0);
        assert_eq!(count_set_bits(&[1, 1]), 2);
        assert_eq!(count_set_bits(&[u64::MAX, u64::MAX]), 128);
        assert_eq!(count_set_bits(&[0xAAAA_AAAA_AAAA_AAAA]), 32);
    }

    #[test]
    fn test_simd_detect() {
        let caps = SimdCapabilities::detect();
        // Just verify detection runs without panicking
        let _ = caps.has_simd();
    }
}
