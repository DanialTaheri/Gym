# SpatialClaw PivotRL

This resource applies NeMo Gym's single-step PivotRL pattern to exact
SpatialClaw main-policy states. A state contains the original multimodal system
and user prompt plus every prior main-agent action and tool observation. The
planner, reflection, finalizer, and `vlm.*` sessions are deliberately excluded
because they are independent model sessions rather than turns in the trainable
main conversation.

SpatialClaw emits Python code instead of Responses function-call objects. The
verifier parses code with Python's AST and compares the ordered operation names
by default. This is the same coarse functional-reward principle used for SWE
pivots in the PivotRL paper: equivalent arguments and formatting can still
receive credit when the policy selected the correct next operations. Set
`python_call_comparison: canonical` to compare parsed arguments as well.
Terminal actions must contain exactly one `ReturnAnswer` call and are scored by
the existing SpatialClaw MCQA, exact-match, or token-F1 normalization.

## Data flow

1. Run the normal SpatialClaw agent with `keep_workspaces=true` and a persistent
   workspace root. Each workspace receives `rl_capture.json`; inline media is
   materialized beside it and the unmodified verifier result is recorded after
   the trajectory is scored.
2. Run `scripts/extract_spatialclaw_pivots.py` with the expert rollout JSONL.
   The rollout JSONL is optional when captures contain their verification
   result. Only trajectories meeting `--minimum-reward` are retained, and one
   row is emitted per supported main-agent decision.
3. Profile candidates with repeated frozen-policy rollouts through
   `spatialclaw_pivot_agent`, then run `ng_reward_profile`.
4. Run `scripts/filter_profiled_pivots.py`. It keeps rows with nonzero reward
   variance and mean reward below `--difficulty-threshold`, matching the
   paper's pivot filter. It also removes the per-repeat request seed added by
   `ng_collect_rollouts +num_repeats_add_seed=true`. That seed belongs only to
   frozen-policy profiling; retaining it in training would make every online
   generation in a GRPO group identical.

For corrected-v2 Energon SFT data, use
`scripts/convert_spatialclaw_sft_pivots.py`. It emits every supported
assistant boundary while preserving the full prior history. The source SFT
loader treats only the first user turn's 256 JPEGs as one preframed video;
later tool-observation images are ordinary images. Converted requests therefore
carry:

```json
{
  "video_as_images_frame_counts": [256, 1],
  "video_as_images_group_types": ["video", "image"]
}
```

The second entries appear only after a tool observation has introduced an
image. Profiling and training must use the accompanying vLLM frame-group patch
and the matching NeMo RL local preprocessor. Treating all JPEGs as independent
images or all groups as video is not equivalent to the SFT checkpoint.

The extracted rows preserve historical assistant prompt IDs, generation IDs,
and generation log probabilities. Extraction fails if any assistant history
lacks that metadata rather than silently retokenizing it. Media URLs point to
the persistent expert workspace, so those directories must remain available
for profiling and training.

During RL, the `tool_simulation_agent` performs exactly one policy call and
passes the sampled action to this verifier. No SpatialClaw tool is executed
online at this stage. All pivot history is prompt context and is masked by NeMo
RL; only the newly sampled action receives loss.
