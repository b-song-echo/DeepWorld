Unsolved Problems
(Codex should not care about these at the moment.)


The FSDP accelerator preparation breaks due to device mismatch, two possible workaround (they are not the real solutions):
- Manual `model.to(accelerator.device)`
- Set `sync_module_states=False` and `cpu_ram_efficient_loading=False` when creating FSDP plugin


FSDP does not seem to yield any gain at all:
- Exclude vae, vggt, vit -> GPU mem 43GB
- Exclude vae, vggt, vit, dit -> GPU mem 37GB
Very wierd, more testing required.


Data preprocessing need further improvent, in particular, the heavy resizing and cropping still happen on CPU. This is not urgent because `batch_size` is 1, and `num_workers` is also 1. Furthermore, data batching still requires optimizations.


Data loading crashes due to system out of shared shm memory, even though `batch_size` is already 1.
- This can be be solved by setting `torch.multiprocessing.set_sharing_strategy` to either `file_system` or `file_descriptor`.
- The error goes away if `num_workers` is 0, but this slows down data loading significatly, and is really not feasible.
- The error goes away by setting `prefetch_factor` to 1, which defaults to 2 when `num_workers > 0`. This is the current solution.
This must some way to prevent this, otherwise, larger batch size becomes an issue.