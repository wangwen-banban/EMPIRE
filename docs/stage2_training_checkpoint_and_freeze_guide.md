# Stage II checkpoint and freeze guide

## Initialization

1. Train `empire/configs/train_stage1_planner.json`.
2. Set `STAGE1_CHECKPOINT` to the resulting weights file or checkpoint directory accepted by `scripts/train.py`.
3. Launch `empire/configs/train_stage2_actor.json`.

The public Stage II recipe starts fresh (`resume=false`). `STAGE1_CHECKPOINT` initializes model weights; it is not an optimizer-resume path. To resume a user-created run, explicitly set `resume=true` and `model_load_path` in a local config that is not committed.

## Freeze contract

The final Stage II recipe declares:

```json
{
  "train_condition_encoders": false,
  "train_setup": {
    "freeze_option": "freeze_llm_train_connect",
    "training_stage": "stage2"
  }
}
```

At Stage II setup, the PaliGemma planner, vision encoder, depth encoder, mapper, and field-of-view encoder are frozen. The DiT-B actor and its condition projector remain trainable. Verify the printed trainable-parameter summary before starting a long run.

## Checkpoint loading order

`scripts/train.py` uses this order:

1. If `resume=true` and a run checkpoint exists, load that checkpoint plus optimizer and scheduler state.
2. Otherwise, if `pretrain_path` is set, load Stage I model weights.
3. Otherwise, initialize from the configured Hugging Face backbones.

The checked-in recipes follow path 2 for Stage II and path 3 for Stage I.

## Reproducibility values

- Stage I: 1 epoch, 10,171 steps, global batch 64, planner LR `1e-5`.
- Stage II: 4 epochs, 40,684 steps, global batch 64, actor LR `1e-4`.
- Stage II sampling: 8 candidates, 4 Euler steps.
