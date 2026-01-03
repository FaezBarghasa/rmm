use core::arch::asm;

use crate::{Arch, MemoryArea, PhysicalAddress, TableKind, VirtualAddress};

/// AArch64 with LPA2 (Large Physical Address 2) - 52-bit Physical Addressing
///
/// LPA2 (ARMv8.7) extends physical addressing from 48-bit to 52-bit:
/// - Supports up to 4PB of physical memory
/// - Requires FEAT_LPA2 support (ARMv8.7+)
/// - Uses the same 4-level page table structure
/// - Extended descriptor format with additional address bits
///
/// Physical address bits [51:48] are stored in descriptor bits [9:8] and [49:48].
#[derive(Clone, Copy, Debug)]
pub struct AArch64Lpa2Arch;

impl Arch for AArch64Lpa2Arch {
    const PAGE_SHIFT: usize = 12; // 4096 bytes
    const PAGE_ENTRY_SHIFT: usize = 9; // 512 entries, 8 bytes each
    const PAGE_LEVELS: usize = 4; // L0, L1, L2, L3

    // LPA2 extends physical address width to 52 bits
    const ENTRY_ADDRESS_WIDTH: usize = 52;
    const ENTRY_ADDRESS_SHIFT: usize = 12; // PA bits start at bit 12 (was 10 for LPA1)

    const ENTRY_FLAG_DEFAULT_PAGE: usize = Self::ENTRY_FLAG_PRESENT
        | 1 << 1  // Page flag (vs Block)
        | 1 << 10 // Access flag
        | Self::ENTRY_FLAG_NO_GLOBAL;
    const ENTRY_FLAG_DEFAULT_TABLE: usize = Self::ENTRY_FLAG_PRESENT
        | Self::ENTRY_FLAG_READWRITE
        | 1 << 1  // Table flag
        | 1 << 10; // Access flag

    const ENTRY_FLAG_PRESENT: usize = 1 << 0;
    const ENTRY_FLAG_READONLY: usize = 1 << 7; // AP[2]
    const ENTRY_FLAG_READWRITE: usize = 0;
    const ENTRY_FLAG_PAGE_USER: usize = 1 << 6; // AP[1]
    const ENTRY_FLAG_NO_EXEC: usize = 0b11 << 53; // UXN + PXN
    const ENTRY_FLAG_EXEC: usize = 0;
    const ENTRY_FLAG_GLOBAL: usize = 0;
    const ENTRY_FLAG_NO_GLOBAL: usize = 1 << 11; // nG bit
    const ENTRY_FLAG_WRITE_COMBINING: usize = 0;
    const ENTRY_FLAG_HUGE: usize = 1 << 55; // Software marker

    // 52-bit VA with 4 levels = same kernel half split
    const PHYS_OFFSET: usize = 0xFFFF_8000_0000_0000;

    fn flags_to_huge(flags: usize) -> usize {
        // Clear bit 1 (Page) to indicate Block (Huge Page)
        (flags | Self::ENTRY_FLAG_HUGE) & !(1 << 1)
    }

    unsafe fn init() -> &'static [MemoryArea] {
        // LPA2 initialization requires TCR_EL1.DS=1 to be set by bootloader
        &[]
    }

    #[inline(always)]
    unsafe fn invalidate(address: VirtualAddress) {
        unsafe {
            // Use TLBI with range for LPA2 (ARMv8.4+)
            asm!(
                "dsb ishst",
                "tlbi vaae1is, {0}",
                "dsb ish",
                "isb",
                in(reg) (address.data() >> Self::PAGE_SHIFT)
            );
        }
    }

    #[inline(always)]
    unsafe fn invalidate_all() {
        unsafe {
            asm!("dsb ishst", "tlbi vmalle1is", "dsb ish", "isb");
        }
    }

    #[inline(always)]
    unsafe fn table(table_kind: TableKind) -> PhysicalAddress {
        unsafe {
            let address: usize;
            match table_kind {
                TableKind::User => {
                    asm!("mrs {0}, ttbr0_el1", out(reg) address);
                }
                TableKind::Kernel => {
                    asm!("mrs {0}, ttbr1_el1", out(reg) address);
                }
            }
            // TTBR contains BADDR which is already the full PA with LPA2
            PhysicalAddress::new(address & 0x000F_FFFF_FFFF_F000)
        }
    }

    #[inline(always)]
    unsafe fn set_table(table_kind: TableKind, address: PhysicalAddress) {
        unsafe {
            debug_assert_eq!(
                address.data() & Self::PAGE_OFFSET_MASK,
                0,
                "Page table address must be page-aligned"
            );

            match table_kind {
                TableKind::User => {
                    asm!("msr ttbr0_el1, {0}", in(reg) address.data());
                }
                TableKind::Kernel => {
                    asm!("msr ttbr1_el1, {0}", in(reg) address.data());
                }
            }
            Self::invalidate_all();
        }
    }

    fn virt_is_valid(address: VirtualAddress) -> bool {
        // AArch64 48-bit VA with sign extension to 64-bit
        // Top 16 bits must be all 0 (user) or all 1 (kernel)
        let mask = !((Self::PAGE_ADDRESS_SIZE as usize - 1) >> 1);
        let masked = address.data() & mask;
        masked == 0 || masked == mask
    }
}

#[cfg(test)]
mod tests {
    use super::AArch64Lpa2Arch;
    use crate::{Arch, VirtualAddress};

    #[test]
    fn constants() {
        assert_eq!(AArch64Lpa2Arch::PAGE_SIZE, 4096);
        assert_eq!(AArch64Lpa2Arch::PAGE_OFFSET_MASK, 0xFFF);
        assert_eq!(AArch64Lpa2Arch::PAGE_ADDRESS_SHIFT, 48);
        assert_eq!(AArch64Lpa2Arch::PAGE_LEVELS, 4);

        // 52-bit physical address support
        assert_eq!(AArch64Lpa2Arch::ENTRY_ADDRESS_WIDTH, 52);
        assert_eq!(AArch64Lpa2Arch::ENTRY_ADDRESS_SIZE, 0x0010_0000_0000_0000); // 4PB

        assert_eq!(AArch64Lpa2Arch::PHYS_OFFSET, 0xFFFF_8000_0000_0000);
    }

    #[test]
    fn is_canonical() {
        #[track_caller]
        fn yes(addr: usize) {
            assert!(
                AArch64Lpa2Arch::virt_is_valid(VirtualAddress::new(addr)),
                "Expected {:#x} to be valid",
                addr
            );
        }

        #[track_caller]
        fn no(addr: usize) {
            assert!(
                !AArch64Lpa2Arch::virt_is_valid(VirtualAddress::new(addr)),
                "Expected {:#x} to be invalid",
                addr
            );
        }

        // Valid kernel addresses (bits 63:47 must all be 1)
        yes(0xFFFF_FFFF_FFFF_FFFF);
        yes(0xFFFF_8000_0000_0000);
        yes(0xFFFF_8000_1337_BEEF); // Bit 47 must be set for kernel

        // Valid user addresses
        yes(0x0000_0000_0000_0000);
        yes(0x0000_0000_0000_0042);
        yes(0x0000_7FFF_FFFF_FFFF);

        // Invalid (non-canonical)
        no(0x0000_8000_0000_0000); // Bit 47 set but not sign-extended
        no(0x1337_0000_0000_0000);
        no(0x8000_0000_0000_0000);
    }
}
