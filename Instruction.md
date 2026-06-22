## Overview

I’m working on `DeepWorldHY`. It is a world model built on top of HunyuanVideo-1.5, to be trained on synthesized data to perform scene-grounded video generation.

`DeepWorldHY` takes a detailed prompt and any number of reference images as input, then outputs a 10-sec clip. Technically, it is not a I2V image, because the initial frame may or may not be present in the reference set. Additionally, there is no temporal order among reference images. Fine-grained camera camera motions are specified in the prompt as text. Scene visuals are inferred from the input reference images. The user simply upload a couple of images of a scene, and naturally describe what to be generated, then the model will generate the world as the user prompts.

We make two major architectural changes:
- We use frozen VGGT as the new geometry encoder, and add a third geometry stream alongside the existing two streams to MMDiT blocks. The three streams interact at joint self-attention. The motivation is to incorporate a strong geometry prior for the MMDiT to draw upon for better scene alignment.
- We extend the 3D RoPE to other modalities (VAE features, VGGT features, SigLIP features, Qwen2.5-VL text features, etc.). The motivation is to enhance spatial reasoning and achieve image-index awareness, without destroying the pretrained video-only RoPE behavior.

Additionally, we may use adapters, projectors, embeddings, and other tricks to make the whole thing work. Beyond that, we don’t wish diverge too much from the pretrained model. Otherwise, we will lose the destroy world knowledge and have trouble converging the giant model with relatively limited resources.

For distributed training, we will train the model on 8 NVIDIA A100-80G GPUs. Use optional FSDP2, sequence parallelism, and gradient checkpointing. Local batch size is no larger than 1. For evaluation, we generate a couple of samples from the eval set and save the, to disk.


## Your task

Please inspect the current implementation of `DeepWorldHY`. Use your own knowledge and intuition, together with the HunyuanVideo source code under `hyvideo` folder, ensure my implementation behaves as expected and stay faithful to the pretrained backbone.

When requirements are already satisfied, do nothing.