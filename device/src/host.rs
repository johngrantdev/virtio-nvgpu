//! What this host's NVIDIA driver looks like, as the guest needs to be told.
//!
//! The guest driver needs a version string and a GPU table, and both are facts
//! about the host: which driver is loaded and which cards it owns. The kernel
//! publishes all of it under `/proc/driver/nvidia`, so this reads it there
//! rather than inventing it or taking it on a command line.
//!
//! Everything is parameterised by a root directory so it can be tested against
//! a fixture tree. On a real host that root is [`PROC_NVIDIA`].

use protocol::messages::{DescriptorGpu, PCI_ADDR_LEN};
use std::path::Path;

/// Where a loaded NVIDIA kernel module publishes itself.
pub const PROC_NVIDIA: &str = "/proc/driver/nvidia";

/// The loaded driver's version, e.g. `615.71.09`.
///
/// Returns `None` when no NVIDIA module is loaded, which is a normal state for
/// a host that has not yet had one inserted -- not an error to propagate.
pub fn driver_version(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join("version")).ok()?;
    // "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09 ..."
    // Matched by shape rather than by position: the words around it differ
    // between the open and proprietary modules, and have changed before.
    text.split_whitespace()
        .find(|w| {
            let mut parts = w.split('.');
            let ok = |p: Option<&str>| {
                p.is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            };
            ok(parts.next()) && ok(parts.next()) && ok(parts.next()) && parts.next().is_none()
        })
        .map(str::to_string)
}

/// Every GPU the host driver owns, in PCI address order.
///
/// The directory name under `gpus/` is the PCI address, and the minor number
/// is the `N` in `/dev/nvidiaN`. Ordering is by address so the table a guest
/// sees is stable across reboots; readdir order is not.
pub fn gpu_slots(root: &Path) -> Vec<DescriptorGpu> {
    let Ok(entries) = std::fs::read_dir(root.join("gpus")) else {
        return Vec::new();
    };
    let mut addrs: Vec<_> = entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    addrs.sort();

    addrs
        .iter()
        .enumerate()
        .map(|(minor, addr)| {
            let mut gpu = DescriptorGpu::default();
            // Leave one byte spare for the NUL the driver writes last, so
            // that write never truncates the address.
            let bytes = addr.as_bytes();
            let n = bytes.len().min(PCI_ADDR_LEN - 1);
            gpu.pci_addr[..n].copy_from_slice(&bytes[..n]);
            // The index is the minor only because the kernel numbers
            // /dev/nvidiaN in the same PCI order it lists these directories in.
            // A host where that stops being true needs the minor read from the
            // device node instead, and this is the line that would change.
            gpu.minor = minor as u32;
            gpu
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(files: &[(&str, &str)]) -> Self {
            let base = std::env::temp_dir().join(format!(
                "nvgpu-host-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&base);
            std::fs::create_dir_all(&base).expect("mkdir");
            for (path, body) in files {
                let p = base.join(path);
                std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
                std::fs::write(&p, body).expect("write");
            }
            Self(base)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The exact text an open 615.71.09 module publishes.
    const OPEN_615: &str = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build  (root@)  \nGCC version:  gcc version 16.2.1 20260810 (GCC)\n";

    #[test]
    fn reads_the_version_from_a_real_proc_line() {
        let f = Fixture::new(&[("version", OPEN_615)]);
        assert_eq!(driver_version(f.path()).as_deref(), Some("615.71.09"));
    }

    /// The proprietary module words the same line differently. Matching on
    /// field position rather than shape would read "for" or "x86_64" here.
    #[test]
    fn reads_the_version_from_the_proprietary_wording_too() {
        let f = Fixture::new(&[(
            "version",
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul  7 12:18:12 UTC 2026\n",
        )]);
        assert_eq!(driver_version(f.path()).as_deref(), Some("580.178.04"));
    }

    #[test]
    fn no_driver_loaded_is_none_not_an_error() {
        let f = Fixture::new(&[]);
        assert_eq!(driver_version(f.path()), None);
        assert!(gpu_slots(f.path()).is_empty());
    }

    #[test]
    fn a_single_gpu_becomes_a_slot_the_descriptor_can_carry() {
        let f = Fixture::new(&[("gpus/0000:01:00.0/information", "Model: X\n")]);
        let slots = gpu_slots(f.path());
        assert_eq!(slots.len(), 1);
        assert_eq!(&slots[0].pci_addr[..12], b"0000:01:00.0");
        let minor = slots[0].minor;
        assert_eq!(minor, 0);
    }

    /// readdir order is not sorted, and a guest that sees GPUs in a different
    /// order across reboots will address the wrong card.
    #[test]
    fn gpus_are_ordered_by_pci_address() {
        let f = Fixture::new(&[
            ("gpus/0000:41:00.0/information", "Model: C\n"),
            ("gpus/0000:01:00.0/information", "Model: A\n"),
            ("gpus/0000:21:00.0/information", "Model: B\n"),
        ]);
        let slots = gpu_slots(f.path());
        let addrs: Vec<_> = slots
            .iter()
            .map(|s| String::from_utf8_lossy(&s.pci_addr[..12]).into_owned())
            .collect();
        assert_eq!(addrs, ["0000:01:00.0", "0000:21:00.0", "0000:41:00.0"]);
        let minors: Vec<u32> = slots.iter().map(|s| s.minor).collect();
        assert_eq!(minors, [0, 1, 2]);
    }

    /// The guest driver NUL-terminates `pci_addr[15]` itself, so an address
    /// that filled all 16 bytes would lose its last character there.
    #[test]
    fn an_over_long_pci_address_keeps_room_for_its_terminator() {
        let f = Fixture::new(&[("gpus/0000:01:00.0:extra/information", "")]);
        let slots = gpu_slots(f.path());
        assert_eq!(slots[0].pci_addr[PCI_ADDR_LEN - 1], 0);
        assert_eq!(&slots[0].pci_addr[..15], b"0000:01:00.0:ex");
    }
}
