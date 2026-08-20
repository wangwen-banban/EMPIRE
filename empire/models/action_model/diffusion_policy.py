from empire.models.action_model.dit import DiT
from empire.models.action_model import create_diffusion
from . import gaussian_diffusion as gd
from empire.datasets.dataset_utils import ActionFeature
import torch
from torch import nn

def DiT_T(**kwargs):
    return DiT(depth=3, hidden_size=256, num_heads=4, **kwargs)
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_M(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

DiT_models = {'DiT-S': DiT_S, 'DiT-M': DiT_M, 'DiT-B': DiT_B, 'DiT-T': DiT_T, 'DiT-L': DiT_L}

class DiffusionPolicy(nn.Module):
    def __init__(
        self, 
        token_size, 
        model_type='DiT-B', 
        in_channels=192, 
        future_action_window_size=16, 
        past_action_window_size=0, 
        use_state=None, 
        action_type='angle',
        diffusion_steps=100,
        state_dim=None,
        loss_type='human',
        causal_attention=True,
        training_objective='diffusion',
        flow_matching_sampling_steps=None,
        flow_matching_t_sampling='uniform',
        flow_matching_t_min=0.0,
        flow_matching_t_max=1.0,
        flow_matching_beta_alpha=2.0,
        flow_matching_beta_beta=2.0,
        condition_mode='cognition_token',
        cross_attention_every_n_layers=1,
        loss_component_weights=None,
    ):
        super().__init__()
        # SimpleMLP takes in x_t, timestep, and condition, and outputs predicted noise.
        self.in_channels = in_channels
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", 
                                        noise_schedule = 'squaredcos_cap_v2', 
                                        diffusion_steps=self.diffusion_steps, 
                                        sigma_small=True, 
                                        learn_sigma = False
                                        ) 
        #self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'linear', diffusion_steps=100, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.use_state = use_state
        self.action_type = action_type
        objective_aliases = {
            "ddpm": "diffusion",
            "epsilon": "diffusion",
            "eps": "diffusion",
            "flow": "flow_matching",
            "rectified_flow": "flow_matching",
        }
        self.training_objective = objective_aliases.get(training_objective, training_objective)
        if self.training_objective not in {"diffusion", "flow_matching"}:
            raise ValueError(f"Unknown training_objective: {training_objective}")
        self.flow_matching_sampling_steps = flow_matching_sampling_steps
        self.flow_matching_t_sampling = flow_matching_t_sampling
        self.flow_matching_t_min = float(flow_matching_t_min)
        self.flow_matching_t_max = float(flow_matching_t_max)
        self.flow_matching_beta_alpha = float(flow_matching_beta_alpha)
        self.flow_matching_beta_beta = float(flow_matching_beta_beta)
        if self.flow_matching_t_sampling not in {"uniform", "beta"}:
            raise ValueError(f"Unknown flow_matching_t_sampling: {flow_matching_t_sampling}")
        if not (0.0 <= self.flow_matching_t_min < self.flow_matching_t_max <= 1.0):
            raise ValueError(
                "flow_matching_t_min/max must satisfy "
                f"0 <= min < max <= 1, got {self.flow_matching_t_min}, {self.flow_matching_t_max}"
            )
        if self.flow_matching_beta_alpha <= 0 or self.flow_matching_beta_beta <= 0:
            raise ValueError(
                "flow_matching_beta_alpha and flow_matching_beta_beta must be positive, "
                f"got {self.flow_matching_beta_alpha}, {self.flow_matching_beta_beta}"
            )
        
        # Get loss components and hand group mapping from ActionFeature
        if loss_type != 'human':
            raise ValueError(f"Only human hand loss is supported, got: {loss_type}")
        self.loss_components = ActionFeature.get_loss_components(action_type)
        if loss_component_weights is not None:
            unknown_components = set(loss_component_weights) - set(self.loss_components)
            if unknown_components:
                raise ValueError(
                    "Unknown loss component weights: "
                    f"{sorted(unknown_components)}; available components are "
                    f"{sorted(self.loss_components)}"
                )
            overridden_components = {}
            for name, (start, end, default_weight) in self.loss_components.items():
                weight = float(loss_component_weights.get(name, default_weight))
                if weight < 0:
                    raise ValueError(
                        f"Loss component weight must be non-negative, got {name}={weight}"
                    )
                overridden_components[name] = (start, end, weight)
            self.loss_components = overridden_components
        self.net = DiT_models[model_type](
            token_size = token_size, 
            action_dim = in_channels, 
            class_dropout_prob = 0.1, 
            learn_sigma = learn_sigma, 
            future_action_window_size = future_action_window_size, 
            past_action_window_size = past_action_window_size,
            use_state = use_state,
            state_dim=state_dim,
            causal_attention=causal_attention,
            condition_mode=condition_mode,
            cross_attention_every_n_layers=cross_attention_every_n_layers,
        )

    def sample_flow_matching_time(self, batch_size, device):
        if self.flow_matching_t_sampling == "uniform":
            base_t = torch.rand(batch_size, device=device)
        elif self.flow_matching_t_sampling == "beta":
            alpha = torch.tensor(self.flow_matching_beta_alpha, device=device)
            beta = torch.tensor(self.flow_matching_beta_beta, device=device)
            base_t = torch.distributions.Beta(alpha, beta).sample((batch_size,))
        else:
            raise ValueError(f"Unknown flow_matching_t_sampling: {self.flow_matching_t_sampling}")

        t_range = self.flow_matching_t_max - self.flow_matching_t_min
        return self.flow_matching_t_min + base_t * t_range

    def _masked_component_losses(self, pred, target, x_mask):
        assert pred.shape == target.shape

        square_delta = (pred - target) ** 2 * x_mask

        def mask_loss(from_dim, to_dim):
            s = square_delta[:, :, from_dim:to_dim].sum()
            n = x_mask[:, :, from_dim:to_dim].sum()
            return s / n if n > 0 else 0

        component_losses = {}
        component_counts = {}

        for name, (start, end, weight) in self.loss_components.items():
            component_losses[name] = mask_loss(start, end) * weight
            # A zero weight disables supervision for this component without
            # changing x_mask, noising, or the action-model input semantics.
            component_counts[name] = (
                x_mask[:, :, start].sum()
                if weight > 0
                else x_mask.new_zeros(())
            )

        total_count = sum(component_counts.values())
        if total_count == 0:
            loss = square_delta[0, 0, 0]
        else:
            loss = sum(
                component_losses[k] * component_counts[k]
                for k in component_counts.keys()
            ) / total_count

        return {
            "loss": loss,
            **component_losses,
        }

    # Given condition z and ground truth token x, x_mask, compute loss
    def loss(self, x, z, x_mask, state=None, state_mask=None, z_mask=None):
        if self.training_objective == "flow_matching":
            return self.flow_matching_loss(x, z, x_mask, state, state_mask, z_mask)

        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)
        
        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)
        x_t = x_t * x_mask
        x_t = torch.cat([x_t, x_mask], dim=2) # [B, T, D]

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z, state, state_mask, z_mask=z_mask)

        assert noise_pred.shape == noise.shape == x.shape

        return self._masked_component_losses(noise_pred, noise, x_mask)

    def flow_matching_loss(self, x, z, x_mask, state=None, state_mask=None, z_mask=None):
        noise = torch.randn_like(x)
        solver_t = self.sample_flow_matching_time(x.size(0), x.device)
        t_view = solver_t.to(dtype=x.dtype).view(-1, 1, 1)

        x_t = (1.0 - t_view) * x + t_view * noise
        x_t = x_t * x_mask
        x_t = torch.cat([x_t, x_mask], dim=2)

        model_timestep = solver_t * float(self.diffusion.num_timesteps)
        velocity_target = noise - x
        velocity_pred = self.net(x_t, model_timestep, z, state, state_mask, z_mask=z_mask)

        return self._masked_component_losses(velocity_pred, velocity_target, x_mask)
    
    # Given condition and noise, sample x using reverse diffusion process
    def sample(self, 
            action_features,
            cfg_scale,
            current_state,
            current_state_mask,
            use_ddim,
            num_ddim_steps,
            action_masks,
            action_feature_masks=None,
        ):
        B = action_features.shape[0]
        noise = torch.randn(action_features.shape[0], self.future_action_window_size+1, 
                self.in_channels,  device=action_features.device)   #[B, T, D]

        x_mask = action_masks.to(action_features.device)
        if action_feature_masks is None:
            action_feature_masks = torch.ones(
                action_features.shape[:2],
                dtype=torch.bool,
                device=action_features.device,
            )
        else:
            action_feature_masks = action_feature_masks.to(action_features.device, dtype=torch.bool)

        if self.training_objective == "flow_matching":
            return self.flow_matching_sample(
                action_features,
                cfg_scale,
                current_state,
                current_state_mask,
                num_ddim_steps,
                x_mask,
                action_feature_masks,
            )

        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, action_features.shape[1], -1) #[B, S, D]
            z = torch.cat([action_features, uncondition], 0)
            z_mask = torch.cat([action_feature_masks, action_feature_masks], 0)
            cfg_scale = cfg_scale

            if self.use_state == 'DiT':
                model_kwargs = dict(
                    z=z, x_mask=x_mask, 
                    cfg_scale=cfg_scale, state=current_state, 
                    state_mask=current_state_mask,
                    z_mask=z_mask,
                )
            else:
                model_kwargs = dict(z=z, x_mask=x_mask, cfg_scale=cfg_scale, z_mask=z_mask)
            sample_fn = self.net.forward_with_cfg
        else:
            z = action_features
            z_mask = action_feature_masks
            def sample_fn(x, t, z, x_mask, state=None, state_mask=None, z_mask=None):
                x = x * x_mask
                x = torch.cat([x, x_mask], dim=2)
                return self.net(x, t, z, state, state_mask, z_mask=z_mask)

            if self.use_state == 'DiT':
                model_kwargs = dict(z=z, x_mask=x_mask, state=current_state, state_mask=current_state_mask, z_mask=z_mask)
            else:
                model_kwargs = dict(z=z, x_mask=x_mask, z_mask=z_mask)

        if use_ddim and num_ddim_steps is not None:
            if self.ddim_diffusion is None:
                self.create_ddim(ddim_step=num_ddim_steps)
            samples = self.ddim_diffusion.ddim_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_features.device,
                eta=0.0
            )
        else:
            samples = self.diffusion.p_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_features.device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return samples

    def flow_matching_sample(
        self,
        action_features,
        cfg_scale,
        current_state,
        current_state_mask,
        num_steps,
        x_mask,
        z_mask=None,
    ):
        B = action_features.shape[0]
        if z_mask is None:
            z_mask = torch.ones(
                action_features.shape[:2],
                dtype=torch.bool,
                device=action_features.device,
            )
        else:
            z_mask = z_mask.to(action_features.device, dtype=torch.bool)
        num_steps = (
            num_steps
            if num_steps is not None
            else self.flow_matching_sampling_steps
            if self.flow_matching_sampling_steps is not None
            else self.diffusion_steps
        )
        if num_steps <= 0:
            raise ValueError(f"flow matching sampler requires positive num_steps, got {num_steps}")
        x = torch.randn(
            B,
            self.future_action_window_size + 1,
            self.in_channels,
            device=action_features.device,
        )
        x = x * x_mask

        using_cfg = cfg_scale > 1.0
        if using_cfg:
            uncondition = self.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0).expand(B, action_features.shape[1], -1)
            z = torch.cat([action_features, uncondition], 0)
            model_z_mask = torch.cat([z_mask, z_mask], 0)
        else:
            z = action_features
            model_z_mask = z_mask

        dt = 1.0 / float(num_steps)
        for step in range(num_steps, 0, -1):
            solver_t = step / float(num_steps)
            model_timestep_value = solver_t * float(self.diffusion.num_timesteps)

            if using_cfg:
                model_x = torch.cat([x, x], 0)
                model_timestep = torch.full(
                    (2 * B,),
                    model_timestep_value,
                    device=x.device,
                )
                if self.use_state == 'DiT':
                    velocity = self.net.forward_with_cfg(
                        model_x,
                        model_timestep,
                        z,
                        x_mask,
                        cfg_scale,
                        current_state,
                        current_state_mask,
                        z_mask=model_z_mask,
                    )
                else:
                    velocity = self.net.forward_with_cfg(
                        model_x,
                        model_timestep,
                        z,
                        x_mask,
                        cfg_scale,
                        z_mask=model_z_mask,
                    )
                velocity, _ = velocity.chunk(2, dim=0)
            else:
                model_timestep = torch.full(
                    (B,),
                    model_timestep_value,
                    device=x.device,
                )
                model_x = torch.cat([x * x_mask, x_mask], dim=2)
                velocity = self.net(model_x, model_timestep, z, current_state, current_state_mask, z_mask=model_z_mask)

            x = (x - velocity * dt) * x_mask

        return x

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim"+str(ddim_step), 
            noise_schedule = 'squaredcos_cap_v2', 
            diffusion_steps=self.diffusion_steps, 
            sigma_small=True, 
            learn_sigma = False
        )
        return self.ddim_diffusion
