I have put two generated samples under `_generated` folder. Note that I didn't run and generate using this machine, but copied the partial content of two committed samples to this source code repo for you to examine, upon which you can improve the existing data synthesis pipeline. The sample examples do not contain video, images, and poses.

I have spotted a few noticeable issues:

- Motion caption:
  - It is not quantative enough, there are barely any concrete numbers, the downstream world model requires fine-grained control over camera motion.
  - Some tiny momement they are not well-distinguished from dominant ones. It should put more emphasis on major motion, ignore tiny movements.
- Wired caption:
  - It sometimes lacks lacks explicit start frame Designation. What's more, there are barely any cross-references in it.
- Distilled captions:
  - They should not omit crucial facts when the detailed prompt supports it, such as start frame designation. It is the trivial details (the model can infer or imagine) that can be discarded. 

Please refine the prompts, try to mitigate these issues.

Additionally, change the implementation so that the clip length should not tied to 5 second, but is determined by `clip_seconds`. The duration should not be hard coded into the prompt templates. Make sure the pipeline works with any reasonable `clip_seconds` such as 10.0, 30.0. When `motion_digesting_unit_seconds` and `video_captioning_fps` are specified, the longer `clip_seconds` is, the more compute LLM/VLM require because there are more frames and units.