# Flow-matching DiT usage

The Stage II actor predicts 60 future hand-motion frames with a causal DiT. Planner hidden states are supplied through span cross-attention; the current two-hand state is an initial condition. The final method concatenates depth features with RGB features before planner conditioning.

The relevant config block is:

```json
{
  "action_model": {
    "model_type": "DiT-B",
    "causal_attention": true,
    "condition_mode": "vlm_cross_attention",
    "training_objective": "flow_matching",
    "flow_matching_t_sampling": "beta",
    "flow_matching_beta_alpha": 1.5,
    "flow_matching_beta_beta": 1.0,
    "flow_matching_sampling_steps": 4
  },
  "repeated_diffusion_steps": 8
}
```

During training, the policy samples a time value, interpolates between noise and the target trajectory, and learns the velocity field. During inference, four Euler updates produce one trajectory. Repeating the sampler eight times supports best-of-8 evaluation.

Model-size comparisons are configured with `model_type`:

- `DiT-M`: paper small variant (46M parameters)
- `DiT-B`: final base variant (182M parameters)
- `DiT-L`: large variant (637M parameters)

Use the named files under `empire/configs/ablations/` rather than changing dimensions by hand.
