# MM-BrowseComp

Multimodal BrowseComp ([paper](https://arxiv.org/abs/2508.13186),
[data](https://github.com/MMBrowseComp/MM-BrowseComp)): 400 multi-hop web-research questions whose
solution needs image or video content; 243 carry one or two question images. Answers are short and graded
by the BrowseComp answer judge (`correct: yes|no` against the reference answer).

Rows follow the BrowseComp format (`question`, `ground_truth`, system prompt, BrowseComp tools). A question
with images gets a user turn of `input_text` plus one `input_image` data URL per image, so the policy
must be vision-capable. The official "provide your problem-solving roadmap" prefix is replaced by
BrowseComp's `Explanation / Exact Answer / Confidence` suffix. Each row also keeps `id`, `category`,
`subtask`, `level`, `num_images` and the checklist, for per-slice scoring and the official checklist judge.

## Harnesses

| config | harness | notes |
| --- | --- | --- |
| `benchmarks/mm_browsecomp/config.yaml` | `browsecomp_agent` (flagship) | the image rides in `body.input` through every call and context reset; the tools stay text-only |
| `benchmarks/mm_browsecomp/configs/mm_browsecomp_opencode.yaml` | `opencode_sandboxed_agent` (OpenCode in a sandbox) | `attach_images: true` uploads the images and passes `-f`; the model entry is declared image-capable |

## Prepare

```bash
# offline: a dir with MMBrowseComp_400.jsonl and images/
MM_BROWSECOMP_DIR=/path/to/MM-BrowseComp ng_prepare_benchmark "+config_paths=[benchmarks/mm_browsecomp/config.yaml]"
```

Without `MM_BROWSECOMP_DIR` the jsonl and the 251 images are downloaded from GitHub into `data/cache`.
The prepared file embeds the images (about 260 MB); it is git-ignored. Do not publish the decrypted rows.

## Run

Same commands as `benchmarks/browsecomp/README.md`, with `benchmarks/mm_browsecomp/config.yaml` in place of
the BrowseComp config. For OpenCode, compose `benchmarks/mm_browsecomp/configs/mm_browsecomp_opencode.yaml` as in
`benchmarks/browsecomp/configs/browsecomp_opencode.yaml`.
