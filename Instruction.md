I got this error:

```text
(main) [hadoop-intelligence-studio@set-zw04-mlp-codelab-pc344 scripts]$ bash synthesize_light.sh --restart_fresh
[worker 0] loading VLM ...
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 750/750 [00:01<00:00, 435.76it/s]
[worker 0] loading LLM ...
[transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|███████████████████████████████████████████████████████████████████████| 427/427 [00:02<00:00, 148.88it/s]
True /home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/tuzihao/data/scannetpp_hf/8890d0a267/iphone/rgb.mkv
qwen-vl-utils using torchvision to read video.
[worker 0] committed 8890d0a267__f0001467 (1/100000)
False /home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/tuzihao/data/scannetpp_hf/d807fb583b/iphone/rgb.mkv
Traceback (most recent call last):
  File "/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py", line 1664, in <module>
    main()
  File "/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py", line 1646, in main
    run_worker(args, worker_index=0)
  File "/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py", line 1625, in run_worker
    run_stages(stages, ctx)
  File "/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py", line 1557, in run_stages
    stages[index](ctx)
  File "/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py", line 744, in __call__
    self._probe_video(ctx)
  File "/home/hadoop-intelligence-studio/dolphinfs_ssd_hadoop-intelligence-studio/songbaijun/DeepWorld/code/synthesize.py", line 447, in _probe_video
    probe = ffmpeg.probe(str(path), select_streams="v:0")
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hadoop-intelligence-studio/.local/lib/python3.12/site-packages/ffmpeg/_probe.py", line 23, in probe
    raise Error('ffprobe', out, err)
```

It sample was successfully committed, but the probing throws a runtime error at the second sample.