## Model

Please check the implementation of DeepWorldHY model. It is built on top of HunyuanVideo1.5 480p_i2v variant, extending its two-stream MMDiT to triple-stream, with one extra stream dedicated to VGGT geometry features. It also incorporates MRoPE to its multimodal tokens during joint attention. It has some other adjustments, but please carefully analyze the official hunyuan implementation, to make absolutely sure that the model is correctly implemented.

The reason I doubt this implementation is because two issues I have noticed at a glance, which are marked by TODO comments. First, there is a `deterministic` property that is never used. Second, the core transformer block logic seems to be flawed. They are so obvious, so I suspect that there might be ton of hidden issues.

Additionally, add generation logic for evaluation and testing.


## Training

Because there are custom layers in DeepWorldHY, make sure these training techniques work: fully sharded data parallel, sequence parallelism, and gradient checkpointing.

Then, add evaluation mechanism. Evaluation is simply generating video given prompt and ref images, then save all to disk. There will be a `eval_every` config along side `log_every` and `save_every`. There is a `eval_manifest_path`, the total evaluation samples in the dataset is capped by `eval_num_sample`. The evaluation dataloader can use the the same `per_device_batch_size`, `sequence_parallel_size`, and `data_parallel_replicate` as training.

- Logs at stored in `output_dir/log.jsonl`
- Evaluation results are stored under `output_dir/evaluation/step_xxxxxxxx/`
- Checkpoints are saved under `output_dir/checkpoints/step_xxxxxxxx/`


## Renaming

There are renaming TODOs. In a nutshell, I wish to keep names consistent across classes, functions, methods, variables, docs, scripts, etc.. For example, I prefer using `txt`, `vis`, `geo` to represent text/textual, vision/visual, geometry/geometric respectively, because they have the same length and makes the code beautiful.