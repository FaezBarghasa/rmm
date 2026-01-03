use core::arch::asm;

use crate::{Arch, MemoryArea, PhysicalAddress, TableKind, VirtualAddress};

/// x86_64 PML5 (5-Level Paging) Implementation
///
/// PML5 extends x86_64 virtual addressing from 48-bit to 57-bit:
/// - PML5: Root table (512 entries, each covers 128PB)
/// - PML4: 256TB per entry
/// - PDP: 512GB per entry (can be huge page - 1GB)
/// - PD: 1GB per entry (can be huge page - 2MB)
/// - PT: 2MB per entry (leaf - 4KB)
///
/// This enables addressing of up to 128 petabytes of virtual memory,
/// suitable for supercomputer and datacenter workloads with enormous
/// memory pools.
///
/// PML5 requires Intel Ice Lake+ or AMD Zen 4+ CPUs with LA57 support.
#[derive(Clone, Copy, Debug)]
pub struct X8664Pml5Arch;

impl Arch for X8664Pml5Arch {
    const PAGE_SHIFT: usize = 12; // 4096 bytes per page
    const PAGE_ENTRY_SHIFT: usize = 9; // 512 entries per table (8 bytes each)
    const PAGE_LEVELS: usize = 5; // PML5, PML4, PDP, PD, PT

    const ENTRY_ADDRESS_WIDTH: usize = 40; // Same as PML4 (physical addr bits)
    const ENTRY_FLAG_DEFAULT_PAGE: usize = Self::ENTRY_FLAG_PRESENT;
    const ENTRY_FLAG_DEFAULT_TABLE: usize = Self::ENTRY_FLAG_PRESENT | Self::ENTRY_FLAG_READWRITE;
    const ENTRY_FLAG_PRESENT: usize = 1 << 0;
    const ENTRY_FLAG_READONLY: usize = 0;
    const ENTRY_FLAG_READWRITE: usize = 1 << 1;
    const ENTRY_FLAG_PAGE_USER: usize = 1 << 2;
    const ENTRY_FLAG_GLOBAL: usize = 1 << 8;
    const ENTRY_FLAG_NO_GLOBAL: usize = 0;
    const ENTRY_FLAG_NO_EXEC: usize = 1 << 63;
    const ENTRY_FLAG_EXEC: usize = 0;
    const ENTRY_FLAG_WRITE_COMBINING: usize = 1 << 7; // PAT bit for WC
    const ENTRY_FLAG_HUGE: usize = 1 << 7; // PS bit for huge pages

    /// Physical offset for direct mapping region in PML5
    /// With 57-bit VA, we use PML5 slot 256+ for kernel (high half)
    const PHYS_OFFSET: usize = Self::PAGE_NEGATIVE_MASK + (Self::PAGE_ADDRESS_SIZE >> 1) as usize;

    unsafe fn init() -> &'static [MemoryArea] {
        // Memory initialization is handled by the kernel or bootloader.
        // PML5 requires bootloader to detect LA57 capability and set up initial tables.
        &[]
    }

    #[inline(always)]
    unsafe fn invalidate(address: VirtualAddress) {
        unsafe {
            asm!("invlpg [{0}]", in(reg) address.data());
        }
    }

    #[inline(always)]
    unsafe fn invalidate_all() {
        unsafe {
            // Reload CR3 to flush entire TLB (less efficient but always works)
            let cr3: usize;
            asm!("mov {0}, cr3", out(reg) cr3);
            asm!("mov cr3, {0}", in(reg) cr3);
        }
    }

    #[inline(always)]
    unsafe fn table(_table_kind: TableKind) -> PhysicalAddress {
        unsafe {
            let address: usize;
            asm!("mov {0}, cr3", out(reg) address);
            // CR3 contains PML5 base (or PML4 if LA57 disabled, but we're PML5)
            PhysicalAddress::new(address & 0x000F_FFFF_FFFF_F000)
        }
    }

    #[inline(always)]
    unsafe fn set_table(_table_kind: TableKind, address: PhysicalAddress) {
        unsafe {
            // CR3 write automatically flushes TLB
            asm!("mov cr3, {0}", in(reg) address.data());
        }
    }

    fn virt_is_valid(address: VirtualAddress) -> bool {
        // PML5 uses 57-bit sign-extended addresses
        // Bit 56 is the sign bit; bits 63:57 must all equal bit 56
        let mask = !((Self::PAGE_ADDRESS_SIZE as usize - 1) >> 1);
        let masked = address.data() & mask;

        // Either all top bits are 0 (user), or all are 1 (kernel)
        masked == 0 || masked == mask
    }
}

#[cfg(test)]
mod tests {
    use super::X8664Pml5Arch;
    use crate::{Arch, VirtualAddress};

    #[test]
    fn constants() {
        assert_eq!(X8664Pml5Arch::PAGE_SIZE, 4096);
        assert_eq!(X8664Pml5Arch::PAGE_OFFSET_MASK, 0xFFF);
        assert_eq!(X8664Pml5Arch::PAGE_ADDRESS_SHIFT, 57);
        assert_eq!(X8664Pml5Arch::PAGE_ADDRESS_SIZE, 0x0200_0000_0000_0000u64);
        assert_eq!(X8664Pml5Arch::PAGE_ADDRESS_MASK, 0x01FF_FFFF_FFFF_F000);
        assert_eq!(X8664Pml5Arch::PAGE_ENTRY_SIZE, 8);
        assert_eq!(X8664Pml5Arch::PAGE_ENTRIES, 512);
        assert_eq!(X8664Pml5Arch::PAGE_ENTRY_MASK, 0x1FF);
        assert_eq!(X8664Pml5Arch::PAGE_NEGATIVE_MASK, 0xFE00_0000_0000_0000);

        assert_eq!(X8664Pml5Arch::ENTRY_ADDRESS_SIZE, 0x0000_0100_0000_0000);
        assert_eq!(X8664Pml5Arch::ENTRY_ADDRESS_MASK, 0x0000_00FF_FFFF_FFFF);

        // PHYS_OFFSET should be in kernel half of 57-bit space
        assert!(X8664Pml5Arch::PHYS_OFFSET >= 0xFF00_0000_0000_0000);
    }

    #[test]
    fn is_canonical() {
        #[track_caller]
        fn yes(addr: usize) {
            assert!(
                X8664Pml5Arch::virt_is_valid(VirtualAddress::new(addr)),
                "Expected {:#x} to be valid",
                addr
            );
        }

        #[track_caller]
        fn no(addr: usize) {
            assert!(
                !X8664Pml5Arch::virt_is_valid(VirtualAddress::new(addr)),
                "Expected {:#x} to be invalid",
                addr
            );
        }

        // Valid kernel addresses (all bits in 0xFF00_0000_0000_0000 mask set)
        yes(0xFFFF_FFFF_FFFF_FFFF);
        yes(0xFF00_0000_0000_0000);
        yes(0xFF80_1337_DEAD_BEEF);

        // Valid user addresses (all mask bits clear)
        yes(0x0000_0000_0000_0000);
        yes(0x0000_0000_0000_0042);
        yes(0x00FF_FFFF_FFFF_FFFF); // Max user address

        // Invalid (non-canonical)
        no(0xFE00_0000_0000_0000); // Some but not all mask bits
        no(0x0100_0000_0000_0000); // Bit 56 set, rest clear
        no(0x8000_0000_0000_0000); // Only bit 63
        no(0x1337_0000_0000_0000); // Random pattern
    }
}
