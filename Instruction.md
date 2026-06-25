## Overview

I’m working on `DeepWorldHY`. It is a world model built on top of HunyuanVideo-1.5, to be trained on synthesized data to perform scene-grounded video generation.

`DeepWorldHY` takes a detailed prompt and any number of reference images as input, then outputs a 10-sec clip. Technically, it is not a I2V image, because the initial frame may or may not be present in the reference set. Additionally, there is no temporal order among reference images. Fine-grained camera camera motions are specified in the prompt as text. Scene visuals are inferred from the input reference images. The user simply upload a couple of images of a scene, and naturally describe what to be generated, then the model will generate the world as the user prompts.

We make two major architectural changes:
- We use frozen VGGT as the new geometry encoder, and add a third geometry stream alongside the existing two streams to MMDiT blocks. The three streams interact at joint self-attention. The motivation is to incorporate a strong geometry prior for the MMDiT to draw upon for better scene alignment.
- We extend the 3D RoPE to other modalities (VAE features, VGGT features, SigLIP features, Qwen2.5-VL text features, etc.). The motivation is to enhance spatial reasoning and achieve image-index awareness, without destroying the pretrained video-only RoPE behavior.

Additionally, we may use adapters, projectors, embeddings, and other tricks to make the whole thing work. Beyond that, we don’t wish diverge too much from the pretrained model. Otherwise, we will lose the destroy world knowledge and have trouble converging the giant model with relatively limited resources.

For distributed training, we will train the model on 8 NVIDIA A100-80G GPUs. Use optional FSDP2, sequence parallelism, and gradient checkpointing. Local batch size is no larger than 1. For evaluation, we generate a couple of samples from the eval set and save the, to disk.


## The previous tasks

These are the prompted to Codex in another session. Codex had already implemented them.

### Config toggles

1. Remove the `use_txt_tokens` config. Text encoder is always loaded, and text tokens are always in the input sequence.
2. For `use_vae_tokens`, when it is set to `False`, the reference images are not encoded by the VAE, and `img_stream` only handles video tokens.
3. For `use_vis_tokens`, when it is set to `False`, the vision encoder is not loaded, the reference images are not encoded by it, and `txt_stream` only handles text tokens.
4. For `use_geo_tokens`, when set to `False`, the geometry encoder (VGGT) is not loaded, so are the entire `geo_stream` modules. The reference images are not encoded by it, MMDiT remains two-stream and does not handle geometry features.
5. Add a config toggle `use_mrope` to control whether to extend 3D RoPE beyond video tokens to other modalities. If set to `False`, only video tokens are applied 3D RoPE, other tokens are only weakly distinguished via modality embeddings.

If `use_vae_tokens`, `use_vis_tokens`, `use_geo_tokens`, and `use_mrope` are all set to `False`, and `transformer_version` is set to `480p_t2v`, then `DeepWorldHY` basically falls back to the pretrained backbone. In this case, the model should behave exactly as the pretrained T2V behavior. These “ablative” designs ensure that I can make incremental architectural Changs, and see which change fails or improves.

### Mixed-task training and condition dropout

First, I need you to reach out to `synthesize.py` to perform a minor modification. The `VideoCaptioningStage` result `video_caption` should be inserted into sample manifest, rather saved to disk as intermediate result. Then the final saved sample manifest contain `video_caption`, `synthesized_prompt`, and `distilled_prompts`.

Then, add a config `mixing_t2v_prob` that controls the proportion of pure text-to-video task, ranging from 0.0 to 1.0, defaults to 0.2. When a sample is assigned as such task, no reference images are used, and the text is `video_caption` rather than `synthesized_prompt`. Otherwise, everything else behave the same as now. This ensure that the model can function as a plain T2V model.

To support classifier-free-guidance, Use fine-grained condition dropout. Specifically, there is a global toggle `condition_dropout_prob` that controls whether all condition tokens should be dropped in both tasks. Next, there are individual toggles: `drop_vae_tokens_prob`, `drop_vis_tokens_prob`, `drop_geo_tokens_prob`, and `drop_txt_tokens_prob`. They controls modality-specific token dropping for non pure-t2v task. For token dropping itself, do not simply zero out them because they still affect attention. Instead, literally drop these tokens or even skip the corresponding encoding paths entirely, because they are not fed into the MMDiT.


### Training trick: KV-gates

When geometry stream is enabled, it makes the attention scores diverge from pretrained behavior significantly, which may destroy the pretrained weights. I need to mitigate this by using block-wise K/V gates.

Specifically, in each MMDiT block, use learnable per-modality KV gates to let new condition streams enter the pretrained Hunyuan backbone gradually: initialize scalar gates for `vae`, `geo`, `vis`, and `txt` K/V tensors near zero, multiply only their keys and values before joint attention, and leave video K/V unchanged. Whether to enable gates and their initial strengths are configurable for each modality. This preserves the pretrained attention behavior at initialization while allowing training to learn how strongly video tokens should attend to each added modality.


### Additional implementation

1. Add classifier-free-guidance support generation process. Because it currently does not support batched generation, conditional and unconditional velocity prediction can happen sequentially. Adopt `rescale_noise_cfg`, matching the pretrained Hunyuan’s default path.
2. Add a config toggle `video_vae_sample`, default to `False`. If set to `True`, then sample from posterior for video rather than taking modes. Note that for reference images, their VAE features are never sampled.
3. Optimizer states do not need to be saved, because I have no intention of resuming training. But I do need to load from a saved model weights checkpoint, if it is specified in the config. The current config class has this property but the training script has not implemented it.