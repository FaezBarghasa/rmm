//! Memory Personalities for ABI Compatibility
//!
//! This module provides memory layout abstractions for different operating
//! system ABIs, enabling Redox to run foreign binaries with appropriate
//! memory layouts.
//!
//! # Personalities
//! - **Windows**: NT-compatible with PE/COFF memory layout
//! - **Linux**: ELF-compatible with standard Linux memory regions  
//! - **Android**: Linux-based with Bionic-specific adjustments
//! - **Native**: Default Redox memory layout

use crate::{PhysicalAddress, VirtualAddress};

/// Memory zone types for physical memory classification
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryZone {
    /// DMA zone: 0-16MB, for legacy ISA devices (x86)
    Dma,
    /// DMA32 zone: 0-4GB, for 32-bit DMA-capable devices
    Dma32,
    /// Normal zone: Regular memory for kernel/userspace
    Normal,
    /// HighMem zone: Memory above kernel direct mapping (32-bit only)
    HighMem,
    /// Device zone: Memory-mapped I/O regions
    Device,
}

impl MemoryZone {
    /// Get the zone for a given physical address (x86_64)
    pub fn from_address_x86_64(addr: PhysicalAddress) -> Self {
        let a = addr.data();
        if a < 16 * 1024 * 1024 {
            Self::Dma
        } else if a < 4 * 1024 * 1024 * 1024 {
            Self::Dma32
        } else {
            Self::Normal
        }
    }

    /// Get the zone for a given physical address (AArch64)
    pub fn from_address_aarch64(addr: PhysicalAddress) -> Self {
        let a = addr.data();
        // AArch64 typically has no ISA DMA zone, just normal and device
        if a < 4 * 1024 * 1024 * 1024 {
            Self::Dma32
        } else {
            Self::Normal
        }
    }
}

/// Memory region descriptor for personality-specific layouts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRegion {
    /// Virtual address start
    pub virt_start: VirtualAddress,
    /// Virtual address end (exclusive)
    pub virt_end: VirtualAddress,
    /// Region type
    pub kind: RegionKind,
    /// Region permissions
    pub permissions: Permissions,
}

/// Types of memory regions in a process address space
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    /// Executable code (.text)
    Code,
    /// Read-only data (.rodata)
    ReadOnlyData,
    /// Initialized data (.data)
    Data,
    /// Uninitialized data (.bss)
    Bss,
    /// Heap (grows up)
    Heap,
    /// Stack (grows down)
    Stack,
    /// Memory-mapped files/anonymous regions
    Mmap,
    /// Thread-local storage
    Tls,
    /// Kernel space (not accessible from userspace)
    Kernel,
    /// Reserved/protected region
    Reserved,
}

/// Memory permissions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    pub execute: bool,
    pub user: bool,
}

impl Permissions {
    pub const RO: Self = Self {
        read: true,
        write: false,
        execute: false,
        user: true,
    };
    pub const RW: Self = Self {
        read: true,
        write: true,
        execute: false,
        user: true,
    };
    pub const RX: Self = Self {
        read: true,
        write: false,
        execute: true,
        user: true,
    };
    pub const RWX: Self = Self {
        read: true,
        write: true,
        execute: true,
        user: true,
    };
    pub const KERNEL: Self = Self {
        read: true,
        write: true,
        execute: true,
        user: false,
    };
}

/// Trait for memory personality implementations
pub trait MemoryPersonality: Sync + Send {
    /// Name of the personality (e.g., "windows", "linux", "android")
    fn name(&self) -> &'static str;

    /// Default base address for executable loading
    fn default_load_base(&self) -> VirtualAddress;

    /// Default heap start address
    fn heap_start(&self) -> VirtualAddress;

    /// Default stack top address (grows down)
    fn stack_top(&self) -> VirtualAddress;

    /// Default stack size in bytes
    fn default_stack_size(&self) -> usize;

    /// Kernel/user split address
    fn kernel_base(&self) -> VirtualAddress;

    /// Whether address space layout randomization (ASLR) is enabled by default
    fn default_aslr(&self) -> bool;

    /// Minimum alignment for mmap allocations
    fn mmap_alignment(&self) -> usize;

    /// Thread-local storage implementation variant
    fn tls_variant(&self) -> TlsVariant;
}

/// TLS implementation variants across ABIs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TlsVariant {
    /// Variant I: TLS block before TCB (e.g., ARM, RISC-V)
    VariantI,
    /// Variant II: TLS block after TCB (e.g., x86_64)
    VariantII,
}

/// Windows NT memory personality
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowsPersonality;

impl MemoryPersonality for WindowsPersonality {
    fn name(&self) -> &'static str {
        "windows"
    }

    fn default_load_base(&self) -> VirtualAddress {
        // Windows default image base for 64-bit
        VirtualAddress::new(0x0000_0001_4000_0000)
    }

    fn heap_start(&self) -> VirtualAddress {
        VirtualAddress::new(0x0000_0002_0000_0000)
    }

    fn stack_top(&self) -> VirtualAddress {
        // Windows stack in lower part of address space
        VirtualAddress::new(0x0000_0000_8000_0000)
    }

    fn default_stack_size(&self) -> usize {
        1024 * 1024 // 1MB default on Windows
    }

    fn kernel_base(&self) -> VirtualAddress {
        // Windows kernel at 0xFFFF8000_00000000+
        VirtualAddress::new(0xFFFF_8000_0000_0000)
    }

    fn default_aslr(&self) -> bool {
        true
    }

    fn mmap_alignment(&self) -> usize {
        64 * 1024
    } // 64KB allocation granularity

    fn tls_variant(&self) -> TlsVariant {
        TlsVariant::VariantII
    }
}

/// Linux memory personality
#[derive(Clone, Copy, Debug, Default)]
pub struct LinuxPersonality;

impl MemoryPersonality for LinuxPersonality {
    fn name(&self) -> &'static str {
        "linux"
    }

    fn default_load_base(&self) -> VirtualAddress {
        // Linux PIE default base (variable due to ASLR)
        VirtualAddress::new(0x0000_5555_5555_0000)
    }

    fn heap_start(&self) -> VirtualAddress {
        // Usually after .bss, managed by brk()
        VirtualAddress::new(0x0000_5555_6000_0000)
    }

    fn stack_top(&self) -> VirtualAddress {
        // Linux stack near top of user space
        VirtualAddress::new(0x0000_7FFF_FFFF_F000)
    }

    fn default_stack_size(&self) -> usize {
        8 * 1024 * 1024 // 8MB default on Linux
    }

    fn kernel_base(&self) -> VirtualAddress {
        VirtualAddress::new(0xFFFF_8000_0000_0000)
    }

    fn default_aslr(&self) -> bool {
        true
    }

    fn mmap_alignment(&self) -> usize {
        4096
    } // Page-aligned

    fn tls_variant(&self) -> TlsVariant {
        TlsVariant::VariantII
    }
}

/// Android memory personality (Linux-based with Bionic differences)
#[derive(Clone, Copy, Debug, Default)]
pub struct AndroidPersonality;

impl MemoryPersonality for AndroidPersonality {
    fn name(&self) -> &'static str {
        "android"
    }

    fn default_load_base(&self) -> VirtualAddress {
        // Android uses different base for 64-bit
        VirtualAddress::new(0x0000_7000_0000_0000)
    }

    fn heap_start(&self) -> VirtualAddress {
        VirtualAddress::new(0x0000_7100_0000_0000)
    }

    fn stack_top(&self) -> VirtualAddress {
        VirtualAddress::new(0x0000_7FFF_FF00_0000)
    }

    fn default_stack_size(&self) -> usize {
        8 * 1024 * 1024 // 8MB like Linux
    }

    fn kernel_base(&self) -> VirtualAddress {
        VirtualAddress::new(0xFFFF_8000_0000_0000)
    }

    fn default_aslr(&self) -> bool {
        true
    }

    fn mmap_alignment(&self) -> usize {
        4096
    }

    fn tls_variant(&self) -> TlsVariant {
        TlsVariant::VariantI
    } // Bionic uses Variant I
}

/// Native Redox memory personality
#[derive(Clone, Copy, Debug, Default)]
pub struct RedoxPersonality;

impl MemoryPersonality for RedoxPersonality {
    fn name(&self) -> &'static str {
        "redox"
    }

    fn default_load_base(&self) -> VirtualAddress {
        VirtualAddress::new(0x0000_0001_0000_0000)
    }

    fn heap_start(&self) -> VirtualAddress {
        VirtualAddress::new(0x0000_0002_0000_0000)
    }

    fn stack_top(&self) -> VirtualAddress {
        VirtualAddress::new(0x0000_7FFF_8000_0000)
    }

    fn default_stack_size(&self) -> usize {
        2 * 1024 * 1024 // 2MB default for Redox
    }

    fn kernel_base(&self) -> VirtualAddress {
        VirtualAddress::new(0xFFFF_8000_0000_0000)
    }

    fn default_aslr(&self) -> bool {
        true
    }

    fn mmap_alignment(&self) -> usize {
        4096
    }

    fn tls_variant(&self) -> TlsVariant {
        TlsVariant::VariantII
    }
}

/// Zone allocator for managing memory in different zones
pub struct ZoneAllocator {
    /// Boundaries for DMA zone (0-16MB typically)
    pub dma_end: PhysicalAddress,
    /// Boundaries for DMA32 zone (0-4GB)
    pub dma32_end: PhysicalAddress,
    /// Preferred zone for allocations
    pub preferred_zone: MemoryZone,
}

impl Default for ZoneAllocator {
    fn default() -> Self {
        Self {
            dma_end: PhysicalAddress::new(16 * 1024 * 1024),
            dma32_end: PhysicalAddress::new(4 * 1024 * 1024 * 1024),
            preferred_zone: MemoryZone::Normal,
        }
    }
}

impl ZoneAllocator {
    /// Check if an address is in the specified zone
    pub fn is_in_zone(&self, addr: PhysicalAddress, zone: MemoryZone) -> bool {
        let a = addr.data();
        match zone {
            MemoryZone::Dma => a < self.dma_end.data(),
            MemoryZone::Dma32 => a < self.dma32_end.data(),
            MemoryZone::Normal => a >= self.dma32_end.data(),
            MemoryZone::HighMem => false, // Not used on 64-bit
            MemoryZone::Device => false,  // Determined by ACPI/device tree
        }
    }

    /// Get the zone for a physical address
    pub fn zone_for(&self, addr: PhysicalAddress) -> MemoryZone {
        let a = addr.data();
        if a < self.dma_end.data() {
            MemoryZone::Dma
        } else if a < self.dma32_end.data() {
            MemoryZone::Dma32
        } else {
            MemoryZone::Normal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_zones() {
        let zone = MemoryZone::from_address_x86_64(PhysicalAddress::new(0x1000));
        assert_eq!(zone, MemoryZone::Dma);

        let zone = MemoryZone::from_address_x86_64(PhysicalAddress::new(0x2000_0000));
        assert_eq!(zone, MemoryZone::Dma32);
        // 0x1_0000_0000 = 4GB is exactly at boundary, which is >= 4GB so it's Normal
        let zone = MemoryZone::from_address_x86_64(PhysicalAddress::new(0x1_0000_0000));
        assert_eq!(zone, MemoryZone::Normal);

        // 0xFFFF_FFFF = 4GB - 1 is in DMA32 zone
        let zone = MemoryZone::from_address_x86_64(PhysicalAddress::new(0xFFFF_FFFF));
        assert_eq!(zone, MemoryZone::Dma32);
    }

    #[test]
    fn test_personalities() {
        let linux = LinuxPersonality;
        assert_eq!(linux.name(), "linux");
        assert!(linux.default_aslr());
        assert_eq!(linux.mmap_alignment(), 4096);

        let windows = WindowsPersonality;
        assert_eq!(windows.name(), "windows");
        assert_eq!(windows.mmap_alignment(), 64 * 1024);
        assert_eq!(windows.tls_variant(), TlsVariant::VariantII);

        let android = AndroidPersonality;
        assert_eq!(android.tls_variant(), TlsVariant::VariantI);
    }

    #[test]
    fn test_zone_allocator() {
        let allocator = ZoneAllocator::default();

        assert!(allocator.is_in_zone(PhysicalAddress::new(0x1000), MemoryZone::Dma));
        assert!(!allocator.is_in_zone(PhysicalAddress::new(0x2000_0000), MemoryZone::Dma));
        // 0x1_0000_0000 = 4GB is exactly at boundary, >= 4GB so Normal
        assert_eq!(
            allocator.zone_for(PhysicalAddress::new(0x1_0000_0000)),
            MemoryZone::Normal
        );

        // 3.5GB (0xE000_0000) is in DMA32
        assert_eq!(
            allocator.zone_for(PhysicalAddress::new(0xE000_0000)),
            MemoryZone::Dma32
        );
    }
}
