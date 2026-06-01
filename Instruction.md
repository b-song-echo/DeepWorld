After inspecting some commited samples, I noticed some issues, and want to make some improvements to the pipeline. All stages work synergistically, changing one is mostly likely to affect some other stages.

When you make changes in implementation, make sure you also consider the documentation and YAML config files.

### Data Sampling Stage
- DSLR reference image selection. Spatial transform should be taken into account. The purpose is to increase their relavence to the video clip. Otherwise, there might be a lot of images whose content do not appear in the sampled video clip at all.  It is okay to have some irrelevant images, there is no need to be very strict, try to keep this clever and minimal. Note that the poses of DSLR images and iPhone video frames should be properly aligned to obtain correct results.
- Use pyiqa brisque metric to compute visual quality score for candidate DSLR images to filter out blurry, over-exposed, or noisy images.

### Motion Extraction Stage
- I found that most extracted video clips are shaky, and some are too stationary. Apply more moton quality filters in this stage.

### Motion Digesting Stage
- Some suggestions on prompt template. 1) The `Input field meanings` section and `Numeric interpretation hints` section in the prompt template have overlapping roles, perhaps there should be clearer separation. 2) The units for numeric values should be explicitly stated in the prompt template, such as in meters or degrees.
- I found out that even though the motion captions accurately captured motion types, yet most of them lacked quantatives (largely resorting to vague adverbs such as "slightly" and "sharply") and temporal grounding (not mentioning temporal locations in the video). Therefore, I want the generated caption to be more quantative and precise. Intuitively, I wish to be able to imagine exactly how the camera moves spatially with only this caption, which I find extremely hard with the current version.

### Video Captioning Stage
- Like in motion digesting stage, the captions should be more quantative and have more temporal grounding. The VLM backbone must be aware of the temporal location of each frame. This way, the VLM can temporally link video with motion caption rather than hallucinating, and produce more accurate and precise description of camera motions in the final video captions.

### Image Captioning Stage
- I found out that in one rare occasion, the image index in the produced caption was wrong (the fourth caption said "the first image"), such behavior causes confusion in the caption wiring stage and yields unreliable result. Such violation can be checked in the critic judging stage.

### Caption Wiring Stage
- Because an object in the video caption is very likely to appear in more than one reference images, its "wired reference image" can be any one or more of them. For example, if the sofa appears in the first two images, then all of the following expressions are reasonable "the sofa in the first image", "the sofa in the second image", or "the sofa in the first and the second images".
- Wiring is conpulsory in this stage. If the LLM is certain that something appears in reference images, then it must explicitly cross-refer to them. This may make the wired caption unreadable or super-verbose, this is okay, because it is the DistillationStage's job to produce downstream prompts. This wired caption serve as the "ceiling" version that contains everything that downstream stages need.

### Critic Judging Stage
- Currently, the checks do not comprehensively cover essential failure modes and quality rubrics. Some of them are even not appropriate. Based on the refined version of all stages, synergistically design checks.
- Importantly, bacause a single quality score is a kind of average over all entries, you should make sure that the number of checks for a group should be proportional to the "importance" of that group. For example, if caption wiring is very likely to be invalid, then it should have more entries in the fatal check section; if camera motion quality really matters, then it should have more entries in the quality check section.

### Distillation Stage
- As mentioned earlier, the original prompt is super-detailed and may be not be natural. This stage will rewrite it into two coarser prompts, optionally ommitting some details such as motion, visuals, references, etc.



