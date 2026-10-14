//! What the guest driver expects to find on the bus.
//!
//! Every constant and layout here is a contract with `driver/virtio_gpu_nv.c`,
//! and each one has been wrong at least once. A mismatch is not a build error
//! in either half -- the device compiles, the driver compiles, and the guest
//! simply fails to probe -- so the agreement is asserted by tests that mirror
//! the driver's own `static_assert`s.

/// The virtio device ID the guest driver probes for.
///
/// Must match `VIRTIO_ID_GPU_NV` in `driver/virtio_gpu_nv.c`. This said 0x8042
/// while the driver bound 45, so a device advertising it would never have been
/// probed by its own guest driver.
pub const VIRTIO_ID_GPU_NV: u32 = 45;

/// Virtqueue count.
///
/// Two, not one: `nvgpu_probe()` calls `virtio_find_vqs(vdev, 2, ...)` for a
/// control queue and an event queue, and returns the error from that call. A
/// device offering one queue fails to probe before it reads a byte of config.
pub const NUM_QUEUES: usize = 2;

/// Guest requests, device replies.
pub const CONTROL_QUEUE: usize = 0;
/// Device-initiated notifications to the guest.
pub const EVENT_QUEUE: usize = 1;

/// Recommended virtqueue size.
pub const QUEUE_SIZE: u16 = 256;

pub use protocol::messages::{MAX_FD_TRANSLATIONS, MAX_GPUS, PCI_ADDR_LEN};
/// Driver version string length in config space, including its NUL.
pub const DRIVER_VERSION_LEN: usize = 32;

use abi::ioctl::{
    NV_ESC_ALLOC_OS_EVENT, NV_ESC_FREE_OS_EVENT, NV_ESC_REGISTER_FD, NV_ESC_RM_ALLOC_MEMORY,
    NV_ESC_RM_MAP_MEMORY,
};

/// The ioctls that carry a file descriptor, and the byte offset it sits at.
///
/// Offsets are into the top-level parameter struct:
///
///   * `NV_ESC_REGISTER_FD` -- `nv_ioctl_register_fd_t` is the descriptor
///     alone, so offset 0.
///   * `NV_ESC_ALLOC_OS_EVENT` / `NV_ESC_FREE_OS_EVENT` --
///     `hClient(4) + hDevice(4)` precede it.
///   * `NV_ESC_RM_ALLOC_MEMORY` -- offset 48.
///   * `NV_ESC_RM_MAP_MEMORY` -- `NVOS33` carries the descriptor the mapping
///     is made on, at offset 48. Leaving it out cost the power readings:
///     nvidia-smi reported "GPU access blocked by the operating system" for
///     draw while every power *limit* came back correctly, which reads like a
///     permissions problem rather than an untranslated descriptor.
pub const FD_CARRYING_IOCTLS: &[(u32, u32)] = &[
    (NV_ESC_REGISTER_FD, 0),
    (NV_ESC_ALLOC_OS_EVENT, 8),
    (NV_ESC_FREE_OS_EVENT, 8),
    (NV_ESC_RM_ALLOC_MEMORY, 48),
    (NV_ESC_RM_MAP_MEMORY, 48),
];

/// This backend reads a deep block that holds several segments, one per
/// pointer inside an RM control's parameters, announced by
/// `deep_ptr_offset == protocol::segments::SEGMENTED`.
///
/// A guest that sends one to a backend without this bit has every described
/// control refused: a v0.1 backend reads the marker as a real offset and finds
/// it outside the parameter block. So the guest sends segments only when it
/// sees this.
pub const FEATURE_RMCTRL_SEGMENTS: u32 = 1 << 0;

/// Device configuration space.
///
/// Mirrors `struct virtio_gpu_nv_config`, 48 bytes. The struct carries only
/// what the driver wants before the control queue is up: the version it names
/// the device by, the caps it is served, and this backend's feature bits.
/// The GPU and fd-translation tables travel in a `MsgType::GetDescriptor`
/// response.
///
/// `vram_limit_mib` is not read by the guest. It is announced so the VMM can
/// refuse to start a guest whose backend is not enforcing the limit the VMM
/// was configured with. A limit is a flag on a process the VMM did not
/// start, and a missing flag would otherwise be a guest with the whole card.
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct VirtioGpuNvConfig {
    /// Host driver version, NUL-terminated, e.g. `615.71.09`.
    pub driver_version: [u8; DRIVER_VERSION_LEN],
    /// What the guest is served, as `crate::caps` bits. Never zero from this
    /// backend: zero is what a backend from before capabilities sends, and a
    /// guest driver treats it as "everything, as before".
    pub caps: u32,
    /// What this backend can do beyond v0.1, as [`FEATURE_RMCTRL_SEGMENTS`]
    /// and the like. This was padding; a backend from before it sent zero,
    /// which reads as "none of these", so the guest needs no version to read
    /// it and the struct's layout does not change.
    pub features: u32,
    /// The video memory limit this backend enforces, in MiB; 0 for none.
    pub vram_limit_mib: u64,
}

impl Default for VirtioGpuNvConfig {
    fn default() -> Self {
        Self {
            driver_version: [0; DRIVER_VERSION_LEN],
            caps: 0,
            features: 0,
            vram_limit_mib: 0,
        }
    }
}

impl VirtioGpuNvConfig {
    /// Build config space for a host driver.
    pub fn new(driver_version: &str, caps: crate::caps::Caps, vram_limit_mib: u64) -> Self {
        let mut cfg = Self::default();
        cfg.caps = caps.bits();
        cfg.vram_limit_mib = vram_limit_mib;
        cfg.features = FEATURE_RMCTRL_SEGMENTS;
        let v = driver_version.as_bytes();
        let n = v.len().min(DRIVER_VERSION_LEN - 1);
        cfg.driver_version[..n].copy_from_slice(&v[..n]);
        cfg
    }

    /// Config space as the bytes a guest reads.
    pub fn as_bytes(&self) -> &[u8] {
        // Safe: `repr(C, packed)` with no padding and no pointers, so every
        // byte of the struct is initialised and meaningful.
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    /// Serve a config read as exactly `size` bytes, zero-filled past the
    /// struct.
    ///
    /// A vhost-user frontend reads a shorter reply as an error. The frontend
    /// chooses the offset and the size, so a window past the struct answers
    /// zeros.
    pub fn read(&self, offset: u32, size: u32) -> Vec<u8> {
        let bytes = self.as_bytes();
        let start = (offset as usize).min(bytes.len());
        let end = start.saturating_add(size as usize).min(bytes.len());
        let mut out = vec![0u8; size as usize];
        out[..end - start].copy_from_slice(&bytes[start..end]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The driver asserts these exact numbers. If either side moves, a guest
    /// stops probing and nothing else says why.
    #[test]
    fn layout_matches_the_guest_driver() {
        assert_eq!(size_of::<VirtioGpuNvConfig>(), 48, "config size mismatch");
        assert_eq!(offset_of!(VirtioGpuNvConfig, driver_version), 0);
        assert_eq!(offset_of!(VirtioGpuNvConfig, caps), 32);
        assert_eq!(offset_of!(VirtioGpuNvConfig, features), 36);
        assert_eq!(offset_of!(VirtioGpuNvConfig, vram_limit_mib), 40);
    }

    /// The most one vhost-user config message carries is 256 bytes, so a
    /// config at or under that reaches a guest through any frontend.
    #[test]
    fn config_fits_one_config_message() {
        assert!(size_of::<VirtioGpuNvConfig>() <= 256);
    }

    #[test]
    fn version_caps_and_features_are_where_the_driver_reads_them() {
        let cfg = VirtioGpuNvConfig::new("615.71.09", crate::caps::Caps::DEFAULT, 0);
        let bytes = cfg.as_bytes();
        assert_eq!(&bytes[0..9], b"615.71.09");
        assert_eq!(
            u32::from_le_bytes(bytes[32..36].try_into().unwrap()),
            crate::caps::Caps::DEFAULT.bits(),
            "caps at offset 32"
        );
        assert_eq!(
            u32::from_le_bytes(bytes[36..40].try_into().unwrap()),
            FEATURE_RMCTRL_SEGMENTS,
            "features at offset 36"
        );
    }

    #[test]
    fn a_read_past_the_end_is_zero_filled_to_the_size_asked_for() {
        let cfg = VirtioGpuNvConfig::default();
        assert_eq!(cfg.read(40, 64).len(), 64);
        assert_eq!(cfg.read(99_999, 16).len(), 16);
        assert!(cfg.read(99_999, 16).iter().all(|b| *b == 0));
        assert_eq!(cfg.read(0, 48).len(), 48);
    }
}
