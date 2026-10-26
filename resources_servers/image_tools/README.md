# Image Tools Pivot Verifier

PivotRL verifier for VLM image-tool trajectories. Scores a single rollout tool call
against the action demonstrated at that turn in an SFT trajectory.

## What a pivot is

One row = one decision point. The trajectory prefix is replayed verbatim
(teacher-forced) and the model makes **one** decision: one image-tool call, or,
on a terminal row, the final answer. No tool is executed. The reward is "did you
choose the action the demonstration chose"; on a terminal row that means
answering instead of calling a tool, with the right answer.

Pair it with `tool_simulation_agent` (`max_steps: 1`): one model call, straight to
`/verify`.

## Verification flow

```
 rollout tool call  +  expected (demonstrated) tool call
            |
            v
   Level 0: tool name       same tool?            -> no: reward 0
            |
            v
   Level 1: image target    same img_idx /        -> no: reward 0
            |               img_idx_a+b /
            |               img_indices?
            v
   Level 2: primary arg     per tool family       -> below threshold: reward 0
            |
            v
        reward 1.0   (binary)  or  score  (continuous)
```

`label` and `factor` are **never** compared: a pivot is about which region of which
image, not the caption or the magnification.

### zoom and crop are one decision

`image_zoom_in_tool` and `image_crop_tool` are treated as the **same** tool at Level 0.
The task system prompt defines crop as *"Like image_zoom_in_tool but without
magnification"* — i.e. crop is zoom at factor 1 — and `factor` is already ignored, so
failing a rollout for picking one over the other while calling magnification irrelevant
is self-contradictory, and it punishes a model that localised the right region.

This matters in practice: 88.5% of pivots in the bvstyle dataset expect one of these two
tools, and tool-name mismatch was the single largest failure bucket before the change.

Only this pair is unified. Do **not** collapse whole `tool_family()` groups — those exist
to dispatch argument comparison, not to assert action equivalence.
`find_color_tool` vs `count_objects_tool` (locate vs count) and `image_rotate_tool` vs
`image_flip_tool` (different transforms) are genuinely different decisions and stay
distinct. Set `unify_region_inspect_tools: false` to require exact tool names.

Each verify response records `rollout_tool_name` and `expected_tool_name`, so tool
selection errors can be read as a confusion matrix rather than one aggregate count.

## Terminal (answer) pivots

A row whose expected action is the sentinel `__answer__` marks a turn where the demonstration
answered instead of calling another tool:

```json
"expected_action": {"name": "__answer__", "arguments": {"answer": "3"}}
```

| rollout | reward | `failure_reason` |
| ------- | ------ | ---------------- |
| contains an image-tool call (even with the right answer in the text) | 0 | `tool_call_when_answer_expected` |
| no answer found after the last `</think>` | 0 | `answer_missing` |
| wrong answer | 0 | `answer_incorrect` |
| right answer | 1.0, or `string_match` partial credit (0.98 for a number within about 1%) | `none` |
| gold missing, `null`, empty, boolean, list/dict, NaN or inf | 0 | `expected_action_invalid` |

Reasoning that is still open after the last `</think>`, or a generation cut off at the token
limit whose reasoning never closed, counts as no answer (`answer_missing`).

Extraction and grading reuse `string_match`. Two optional row fields:

| field | default | meaning |
| ----- | ------- | ------- |
| `extraction_mode` | `final_answer` | `boxed` (last `\boxed{}`), `final_answer` (last `Answer:` line, else last `\boxed{}`), `last_line`, or `full_response` |
| `case_sensitive` | `false` | `true` compares the stripped strings exactly, with no normalization or partial credit |

With the default `final_answer`, a bare reply such as `3` scores 0 as `answer_missing`. Terminal
rows should ask for `\boxed{}` and set `extraction_mode: boxed`; see
[Building terminal rows](#building-terminal-rows).

## Tool families

| family   | tools                                            | Level 2 comparison                     |
| -------- | ------------------------------------------------ | -------------------------------------- |
| `bbox`   | `image_zoom_in_tool`, `image_crop_tool`          | IoU of `bbox_2d`                       |
| `point`  | `color_at_tool`                                  | normalized distance on `point_2d`      |
| `color`  | `find_color_tool`, `count_objects_tool`          | channel-wise closeness of `color`      |
| `pair`   | `image_diff_tool`, `image_overlay_tool`          | indices (Level 1) + `alpha`            |
| `multi`  | `image_side_by_side_tool`                        | `img_indices`, order-insensitive       |
| `scalar` | `image_rotate_tool`, `image_flip_tool`           | `degrees` **mod 360**, or `axis`       |

Rotation is compared modulo 360, so `350` and `-10` are the same action.

The parser and IoU are imported from `resources_servers.image_tools`
(`parse_image_tool_calls`, `bbox_iou`) — the same code the image-tools agent uses at
rollout time, so the verifier cannot disagree with the agent about what a tool call is.

## Config

| option                      | default  | meaning                                          |
| --------------------------- | -------- | ------------------------------------------------ |
| `reward_mode`               | `binary` | `binary` (1.0 iff score >= threshold) or `continuous` (score itself) |
| `argument_threshold`        | `0.5`    | IoU / closeness needed for credit                |
| `enable_target_match`       | `true`   | Level 1                                          |
| `numeric_tolerance`         | `1e-6`   | scalar/alpha comparison slack                    |
| `penalize_extra_tool_calls` | `true`   | reward 0 if the rollout emits more than one call |
| `unify_region_inspect_tools`| `true`   | zoom and crop count as the same Level 0 decision |

`reward_mode: continuous` is the softer option: it gives partial credit for a
near-miss box instead of a cliff at IoU 0.5.

## Data

Built from ShareGPT-style SFT trajectories by
`tools/convert_sft_to_image_pivot.py` (in the NeMo-RL repo — conversion scripts do
not belong in Gym). Each row:

```json
{
  "agent_ref": {"type": "responses_api_agents", "name": "image_tools_pivot_agent"},
  "responses_create_params": {
    "input": [ ...trajectory prefix, images as input_image parts... ],
    "metadata": {"extra_body": "{\"stop\": [\"</tool_call>\"], \"include_stop_str_in_output\": true}"}
  },
  "expected_action": {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [...], "img_idx": 0}},
  "metadata": {"source_id": "...", "turn_index": 4, "tool_name": "..."}
}
```

The `stop` on `</tool_call>` is what makes the rollout end *at* the decision.

### Building terminal rows

`convert_sft_to_image_pivot.py` emits tool-call rows only. Terminal (`__answer__`) rows are
built separately; follow these rules so their rewards are unambiguous.

1. **Where to cut.** Take a trajectory whose next assistant turn is the final answer (no tool
   call). The prefix is everything up to, and including, the last tool response before that
   turn. Drop the demonstration's answer turn itself.
2. **Gold answer.** Use the dataset's ground-truth answer, not the demonstration's text:
   `"expected_action": {"name": "__answer__", "arguments": {"answer": "<gold>"}}`. It must be a
   non-empty string, an int, or a finite float (anything else scores `expected_action_invalid`).
   Keep only trajectories whose demonstrated answer matches the gold, so "answer here" is
   actually the right decision.
3. **Ask for `\boxed{}`, and grade only `\boxed{}`.** End the first user message with:

   ```
   Provide your final answer in \boxed{Answer} format.
   ```

   Replace any other format instruction in the question (for example "Answer with the number
   only") with this sentence, so a row never asks for two formats. Set
   `"extraction_mode": "boxed"` on the row. Only the last `\boxed{}` after the last `</think>`
   is graded; a bare `3` or an `Answer: 3` line scores 0 (`answer_missing`). This matches the
   image-tools GRPO rows, whose prompts use the same sentence with `extraction_mode: boxed`.
4. **Case.** Leave `case_sensitive` unset (`false`) unless the answer is genuinely
   case-sensitive. The default normalizes case, quotes, trailing punctuation, numbers and LaTeX.
5. **Metadata.** Record provenance, e.g.
   `{"source_id": "...", "turn_index": 7, "tool_name": "__answer__", "terminal_row": true}`.

A terminal row:

```json
{
  "agent_ref": {"type": "responses_api_agents", "name": "image_tools_pivot_agent"},
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "...image-tool system prompt..."},
      {"role": "user", "content": [
        {"type": "input_text", "text": "How many yellow blocks are used in the figure? Provide your final answer in \\boxed{Answer} format."},
        {"type": "input_image", "image_url": "...", "detail": "auto"}
      ]},
      ...assistant tool call / tool response turns...
    ],
    "metadata": {"extra_body": "{\"stop\": [\"</tool_call>\"], \"include_stop_str_in_output\": true}"}
  },
  "expected_action": {"name": "__answer__", "arguments": {"answer": "2"}},
  "extraction_mode": "boxed",
  "metadata": {"source_id": "...", "turn_index": 3, "tool_name": "__answer__", "terminal_row": true}
}
```

Before training, validate the rows against the task-data schema, with the config's dataset
paths pointing at your files. A misspelled `extraction_mode` is reported as a row error (a
warning in `train_preparation`; add `+task_data_validation=error` to make it fail):

```bash
gym dataset collate "+config_paths=[resources_servers/image_tools/configs/image_tools_pivot.yaml]" \
    +output_dirpath=<output_dir> +mode=train_preparation
```

## Gotchas

- **`license` is required** on `train`/`validation` dataset entries. Without it,
  pydantic falls through to `BenchmarkDatasetConfig` and the server is silently
  demoted to an "almost-server" that never starts — the job dies minutes in with a
  confusing cascade about `prepare_script`. Run
  `python tools/preflight_pivot_config.py <config.yaml>` to catch this offline.
- Source SFT data may elide older tool-produced images, replacing them with
  `[N earlier produced image(s) omitted to bound context]`. Those pivots ask the
  model to decide about an image it can no longer see, which caps achievable reward.

## Testing

```bash
ng_test +entrypoint=resources_servers/image_tools
```
