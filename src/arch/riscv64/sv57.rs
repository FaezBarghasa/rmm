use core::arch::asm;

use crate::{Arch, MemoryArea, PhysicalAddress, TableKind, VirtualAddress};

/// RISC-V 64-bit SV57 Paging Implementation
///
/// SV57 provides 57-bit virtual address space using 5-level page tables:
/// - L4: Root table (512 entries, each covers 128PB)
/// - L3: 512GB per entry
/// - L2: 1GB per entry (can be huge page)
/// - L1: 2MB per entry (can be huge page)
/// - L0: 4KB per entry (leaf)
///
/// This enables addressing of up to 128 petabytes of virtual memory,
/// suitable for supercomputer and datacenter workloads.
#[derive(Clone, Copy)]
pub struct RiscV64Sv57Arch;

impl Arch for RiscV64Sv57Arch {
    const PAGE_SHIFT: usize = 12; // 4096 bytes per page
    const PAGE_ENTRY_SHIFT: usize = 9; // 512 entries per table (8 bytes each)
    const PAGE_LEVELS: usize = 5; // L4, L3, L2, L1, L0

    const ENTRY_ADDRESS_WIDTH: usize = 44;
    const ENTRY_ADDRESS_SHIFT: usize = 10;

    const ENTRY_FLAG_DEFAULT_PAGE: usize = Self::ENTRY_FLAG_PRESENT | Self::ENTRY_FLAG_READONLY;
    const ENTRY_FLAG_DEFAULT_TABLE: usize = Self::ENTRY_FLAG_PRESENT;
    const ENTRY_FLAG_PRESENT: usize = 1 << 0; // Valid bit
    const ENTRY_FLAG_READONLY: usize = 1 << 1; // Read-only (R bit)
    const ENTRY_FLAG_READWRITE: usize = 3 << 1; // R + W bits
    const ENTRY_FLAG_PAGE_USER: usize = 1 << 4; // User mode accessible
    const ENTRY_FLAG_TABLE_USER: usize = 0;
    const ENTRY_FLAG_NO_EXEC: usize = 0; // RISC-V doesn't have no-exec, only exec
    const ENTRY_FLAG_EXEC: usize = 1 << 3; // Execute bit
    const ENTRY_FLAG_GLOBAL: usize = 1 << 5; // Global mapping
    const ENTRY_FLAG_NO_GLOBAL: usize = 0;
    const ENTRY_FLAG_WRITE_COMBINING: usize = 0;
    // Use bit 8 (RSW - Reserved for Software) as huge page marker
    const ENTRY_FLAG_HUGE: usize = 1 << 8;

    /// Physical offset for direct mapping region
    /// SV57 uses the upper half of the 57-bit address space for kernel
    const PHYS_OFFSET: usize = 0xFF00_0000_0000_0000;

    unsafe fn init() -> &'static [MemoryArea] {
        // Memory initialization is handled by the kernel or bootloader.
        // SV57 requires firmware/bootloader to set up initial page tables.
        &[]
    }

    #[inline(always)]
    unsafe fn invalidate(address: VirtualAddress) {
        unsafe {
            // SFENCE.VMA with rs1 = address (specific page invalidation)
            asm!("sfence.vma {}", in(reg) address.data());
        }
    }

    #[inline(always)]
    unsafe fn invalidate_all() {
        unsafe {
            // SFENCE.VMA with no operands (full TLB flush)
            asm!("sfence.vma");
        }
    }

    #[inline(always)]
    unsafe fn table(_table_kind: TableKind) -> PhysicalAddress {
        unsafe {
            let satp: usize;
            asm!("csrr {0}, satp", out(reg) satp);
            // Extract PPN from SATP (bits 0-43) and convert to physical address
            PhysicalAddress::new((satp & Self::ENTRY_ADDRESS_MASK) << Self::PAGE_SHIFT)
        }
    }

    #[inline(always)]
    unsafe fn set_table(_table_kind: TableKind, address: PhysicalAddress) {
        unsafe {
            // SATP format for SV57: MODE[63:60] = 10, ASID[59:44], PPN[43:0]
            // MODE = 10 indicates SV57 paging mode
            let ppn = address.data() >> Self::PAGE_SHIFT;

            // Verify PPN alignment (must be page-aligned)
            debug_assert_eq!(
                address.data() & Self::PAGE_OFFSET_MASK,
                0,
                "Page table address must be page-aligned"
            );

            let satp = (10usize << 60) | ppn;
            asm!("csrw satp, {0}", in(reg) satp);
            Self::invalidate_all();
        }
    }

    fn virt_is_valid(address: VirtualAddress) -> bool {
        // SV57 uses 57-bit sign-extended addresses
        // Bit 56 is the "sign bit" for the 57-bit address
        // Bits [63:56] must all equal bit 56 for a canonical address
        let mask = !((Self::PAGE_ADDRESS_SIZE as usize - 1) >> 1);
        let masked = address.data() & mask;

        // Either all top bits are 0, or all are 1
        masked == mask || masked == 0
    }
}

#[cfg(test)]
mod tests {
    use super::RiscV64Sv57Arch;
    use crate::Arch;

    #[test]
    fn constants() {
        assert_eq!(RiscV64Sv57Arch::PAGE_SIZE, 4096);
        assert_eq!(RiscV64Sv57Arch::PAGE_OFFSET_MASK, 0xFFF);
        assert_eq!(RiscV64Sv57Arch::PAGE_ADDRESS_SHIFT, 57);
        assert_eq!(RiscV64Sv57Arch::PAGE_ADDRESS_SIZE, 0x0200_0000_0000_0000u64);
        assert_eq!(RiscV64Sv57Arch::PAGE_ADDRESS_MASK, 0x01FF_FFFF_FFFF_F000);
        assert_eq!(RiscV64Sv57Arch::PAGE_ENTRY_SIZE, 8);
        assert_eq!(RiscV64Sv57Arch::PAGE_ENTRIES, 512);
        assert_eq!(RiscV64Sv57Arch::PAGE_ENTRY_MASK, 0x1FF);
        assert_eq!(RiscV64Sv57Arch::PAGE_NEGATIVE_MASK, 0xFE00_0000_0000_0000);

        assert_eq!(RiscV64Sv57Arch::ENTRY_ADDRESS_SIZE, 0x0000_1000_0000_0000);
        assert_eq!(RiscV64Sv57Arch::ENTRY_ADDRESS_MASK, 0x0000_0FFF_FFFF_FFFF);

        assert_eq!(RiscV64Sv57Arch::PHYS_OFFSET, 0xFF00_0000_0000_0000);
    }

    #[test]
    fn is_canonical() {
        use super::VirtualAddress;

        #[track_caller]
        fn yes(addr: usize) {
            assert!(
                RiscV64Sv57Arch::virt_is_valid(VirtualAddress::new(addr)),
                "Expected {:#x} to be valid",
                addr
            );
        }

        #[track_caller]
        fn no(addr: usize) {
            assert!(
                !RiscV64Sv57Arch::virt_is_valid(VirtualAddress::new(addr)),
                "Expected {:#x} to be invalid",
                addr
            );
        }

        // Valid kernel addresses (all top bits in 0xFF00_0000_0000_0000 must be 1)
        yes(0xFFFF_FFFF_FFFF_FFFF);
        yes(0xFF00_0000_0000_0000);
        yes(0xFF80_0000_1337_1337); // Kernel with some lower bits

        // Valid user addresses (all top bits in mask must be 0)
        yes(0x0000_0000_0000_0000);
        yes(0x0000_0000_0000_0042);
        yes(0x00FF_FFFF_FFFF_FFFF); // Max user: bit 56 = 0

        // Invalid (non-canonical) addresses: some but not all mask bits set
        no(0xFE00_0000_0000_0000); // Only some top bits set
        no(0x0100_0000_0000_0000); // Bit 56 set but rest of mask clear
        no(0x0200_0000_0000_0000); // Bit 57 set
        no(0x1337_0000_0000_0000); // Random high bits
        no(0x8000_0000_0000_0000); // Only bit 63 set
        no(0xFD00_0000_0000_0000); // Missing some kernel bits
    }
}
