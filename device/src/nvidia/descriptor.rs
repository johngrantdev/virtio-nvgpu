//! GET_DESCRIPTOR: the GPU and fd-translation tables, served on the control
//! queue.

use super::*;

impl NvidiaBackend {
    /// Answer `MsgType::GetDescriptor` with a [`Descriptor`].
    ///
    /// The response is a bare struct, with no `MsgHeader`, like the file
    /// streams. The guest driver reads the buffer from its first byte, and
    /// the version field tells it whether the layout is known.
    pub(super) fn handle_descriptor(&self, resp_buf: &mut [u8]) -> usize {
        use crate::virtio::FD_CARRYING_IOCTLS;
        use protocol::messages::{
            DESCRIPTOR_VERSION, Descriptor, FdTranslation, MAX_FD_TRANSLATIONS, MAX_GPUS,
        };

        let mut d = Descriptor {
            version: DESCRIPTOR_VERSION,
            ..Default::default()
        };

        let n = self.gpu_slots.len().min(MAX_GPUS);
        for (slot, out) in self.gpu_slots.iter().take(n).zip(d.gpus.iter_mut()) {
            *out = *slot;
        }
        d.num_gpus = n as u32;

        let n = FD_CARRYING_IOCTLS.len().min(MAX_FD_TRANSLATIONS);
        for ((nr, offset), out) in FD_CARRYING_IOCTLS.iter().take(n).zip(d.fd_translations.iter_mut())
        {
            *out = FdTranslation {
                nr: *nr,
                payload_offset: *offset,
            };
        }
        d.num_fd_translations = n as u32;

        if resp_buf.len() < size_of::<Descriptor>() {
            log::warn!(
                "GET_DESCRIPTOR: response buffer holds {} of {} bytes",
                resp_buf.len(),
                size_of::<Descriptor>()
            );
            return 0;
        }
        write_struct(resp_buf, &d)
    }
}
