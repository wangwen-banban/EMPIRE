# Learning-rate schedule configuration

`empire/training/fsdp.py` maintains separate optimizer groups for the planner backbone and DiT actor. The explicit fields below are preferred.

| Field | Purpose |
| --- | --- |
| `backbone_lr_scheduler_type` | `constant`, `warmup_cosine`, `freeze_constant`, or `freeze_warmup_cosine` |
| `dit_lr_scheduler_type` | `constant` or `warmup_cosine` |
| `backbone_freeze_steps` | Absolute optimizer steps, a ratio in `[0, 1]`, or `-1` for the full run |
| `backbone_warmup_steps` | Absolute optimizer steps or a ratio in `[0, 1]` |
| `dit_warmup_steps` | Absolute optimizer steps or a ratio in `[0, 1]` |
| `backbone_min_lr_rate` | Final cosine multiplier for the planner groups |
| `dit_min_lr_rate` | Final cosine multiplier for the actor groups |

`lr_scheduler_type`, `warmup_ratio`, and `llm_freeze_step` remain compatibility fallbacks. Explicit component fields take precedence.

The final recipes use the established `backbone-freeze-warmup` compatibility schedule. Stage II additionally enforces freezing through `training_stage=stage2`; a scheduler value alone is not the freeze contract.

Example for a user-created actor-only schedule:

```json
{
  "trainer": {
    "backbone_lr_scheduler_type": "freeze_constant",
    "dit_lr_scheduler_type": "constant",
    "backbone_freeze_steps": -1,
    "action_model_learning_rate": 0.0001
  }
}
```
