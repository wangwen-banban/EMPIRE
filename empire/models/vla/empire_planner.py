import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Callable, Union
import copy
import numpy as np
import json

from PIL import Image
from functools import partial

from empire.utils.tensor_utils import move_masked_to_left, get_mask_of_last_masked_index, move_masked_to_left_ids
from empire.models.vlm_builder import build_vlm
from empire.models.da_builder import build_depth_encoder, CAMapper, MLPMapper
from empire.datasets.dataset_utils import build_cot_instruction_prompt
from empire.utils.mesh_utils import load_mesh_and_sample_points, compute_hand_object_distance

class EMPIRE_Planner(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs=None,
        act_model_configs=None,
        fwd_pred_next_n=1,
        repeated_diffusion_steps:int = 8,
        use_state='DiT',
        use_fov=True,
        use_bf16=False,
        **kwargs,
    ):
        super().__init__()

        self.configs = configs
        self.train_setup_configs = train_setup_configs
        self.act_model_configs = act_model_configs
        self.use_state = use_state
        self.use_fov = use_fov
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.past_action_window_size = 0
        # chunk_size for action prediction
        self.chunk_size = self.configs.get("fwd_pred_next_n", 16)
        self.future_action_window_size = self.chunk_size-1
        self.state_mask_prob = self.configs.get("state_mask_prob", 0.1)
        self.action_type = self.configs['train_dataset'].get("action_type", "angle")
        self.use_depth = self.configs.get("use_depth", False)
        self.use_state = use_state
        self.use_fov = use_fov
        self.use_bf16 = use_bf16
        if self.action_type == 'angle':
            self.hand_dim = 51
        elif self.action_type == 'keypoints':
            self.hand_dim = 69

        # Initialize the tokenizer and VLM backbone
        self.tokenizer, self.backbone = self._init_backbone()
        self.depth_image_encoder_type = self.configs.get("depth_image_encoder_type", "CA")

        # Add special token for CoT training if needed
        self.use_cot = self.configs['train_dataset'].get('use_cot', False)
        # plan-span conditioning: when True, Stage-2 conditions the DiT only on the
        # hidden states of the GT-CoT suffix tokens (labels != -100) and excludes the LM
        # loss from the trained total. Default False keeps legacy behavior byte-identical.
        self.cot_condition_span = self.configs.get("cot_condition_span", False)
        self.sap_token_id = None

        # if self.use_cot or self.train_setup_configs.get("training_stage", None) == 'stage2' or training_stage == self.train_setup_configs.get("training_stage", None) == 'dit_only':
        #     # Add special token <sap> (split action prompt) to mark the split between instruction and response
        #     sap_token = "<sap>"
        #     special_tokens = {'additional_special_tokens': [sap_token]}

        #     # Add special tokens to tokenizer
        #     num_added = self.tokenizer.add_special_tokens(special_tokens)
        #     print(f"Added {num_added} special token(s): {sap_token}")


        #     # Save the sap_token_id for later use
        #     self.sap_token_id = self.tokenizer.convert_tokens_to_ids(sap_token)
        #     print(f"SAP token ID: {self.sap_token_id}")

        #     # Resize language model embeddings to accommodate the new special token
        #     # Note: only resize the language model, NOT the top-level PaliGemma model,
        #     # because PaliGemma has separate vocab configs for vision and text.
        #     if num_added > 0:
        #         self.model.language_model.resize_token_embeddings(len(self.tokenizer))
        #         print(f"Resized language model token embeddings to {len(self.tokenizer)}")
        # else:
        #     self.sap_token_id = None

        if self.use_depth:
            self.depth_encoder = self._init_depth_encoder()
            self.depth_image_encoder = self._init_depth_image_encoder(self.depth_image_encoder_type)
        else:
            self.depth_image_encoder = None
            self.depth_encoder = None
        if self.train_setup_configs is not None and self.train_setup_configs.get("reinit", False):
            initialize_param(self.backbone)

        # Determine if we should create the DiT action model
        # Stage 1 (VLM-only training): skip DiT creation
        # Stage 2 (DiT training): create DiT
        self.create_action_model = True
        if self.train_setup_configs is not None:
            training_stage = self.train_setup_configs.get("training_stage", None)
            if training_stage == "stage1" or training_stage == "vlm_only":
                self.create_action_model = False
                print("Stage 1 (VLM-only): Skipping DiT model creation")

        if self.create_action_model:
            self.act_model = self._init_act_model()
        else:
            self.act_model = None

        # Initialize MANO models for hand-object loss computation
        if self.configs.get('use_refine_loss', False):
            self.mano_model_right = self._init_mano_model(is_right=True)
            self.mano_model_left = self._init_mano_model(is_right=False)

        self.vlm_state_encoder = None
        if self.use_state == 'VLM':
            self.state_and_mask_dim = 2 * self.configs["state_encoder"]["state_dim"]
            self.vlm_state_encoder = self._init_state_encoder()

        if self.use_fov:
            self.fov_encoder = self._init_fov_encoder()

        # The cognition token is a latent planning token. In stage1 it is
        # supervised indirectly by CoT generation; in stage2 DiT reads it.
        self.cognition_token = None
        self.cognition_token_id = None
        training_stage = self.train_setup_configs.get("training_stage", None) if self.train_setup_configs is not None else None
        self.use_cognition_token = self.configs.get("use_cognition_token", None)
        if self.use_cognition_token is None:
            self.use_cognition_token = self.create_action_model or self.use_cot

        if self.use_cognition_token:
            # The `cognition_token_id` is set to an unused token ID.
            self.cognition_token_id = self.configs.get("cognition_token_id", 10)
            untied_cognition_token = self.configs.get("untied_cognition_token", True)

            # Use a separately learned `cognition_token_embedding` that does not share parameters with the word embedding matrix
            # if `untied_cognition_token` is True. We initialize this separate `cognition_token_embedding`
            # with the word embedding parameter corresponding to the specified `cognition_token_id`.

            if untied_cognition_token:
                print(f"Using separate cognition token")
                print(f"Cognition token id: {self.cognition_token_id}")
                init_id = self.configs.get("cognition_token_init_id", None)
                if init_id is None:
                    init_id = self.cognition_token_id
                print(f"Init cognition token with id={init_id}")
                ebd = self.model.get_input_embeddings().weight.data[init_id]
                self.cognition_token = nn.Parameter(ebd.clone())
            else:
                self.cognition_token = None
        else:
            print(f"Training stage {training_stage}: No cognition token needed")

    def _init_depth_image_encoder(self, depth_image_encoder_type="CA"):
        if depth_image_encoder_type == "CA":
            print('Using the CrossAttention Mapper!')
            depth_image_encoder = CAMapper(
                q_dim=self.model.config.vision_config.projection_dim,
                k_dim=self.depth_encoder.config.head.dim_in,
                head_num=self.configs.get("depth_image_encoder_head_num", 8),
                layer_num=self.configs.get("mapper_layer", 4),
                bias=self.configs.get("mapper_bias", False)
            )
        elif depth_image_encoder_type == "CONCAT":
            print('Using the MLP Mapper!')
            depth_image_encoder = MLPMapper(
                depth_dim=self.depth_encoder.config.head.dim_in,
                image_dim=self.model.config.vision_config.projection_dim,
                bias=self.configs.get("mapper_bias", False)
            )
        else:
            raise ValueError(f"depth_image_encoder_type {depth_image_encoder_type} not implemented")

        depth_image_encoder.trainable_params()
        return depth_image_encoder


    def _init_depth_encoder(self):
        depth_encoder = build_depth_encoder(self.configs)
        return depth_encoder

    def _init_backbone(self):
        processor, model = build_vlm(self.configs["vlm"])
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        return self.tokenizer, model

    def _init_fov_encoder(self):
        from empire.utils.nn_utils import MLPProjector
        fov_dim = 2 # fov_x, fov_y
        mlp = MLPProjector(fov_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_state_encoder(self):
        from empire.utils.nn_utils import MLPProjector
        mlp = MLPProjector(self.state_and_mask_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_act_model(self):
        from empire.models.action_model.diffusion_policy import DiffusionPolicy
        action_head = DiffusionPolicy(
            model_type = self.act_model_configs.get("model_type", 'DiT-B'),
            token_size = self.act_model_configs.get("token_size", -1),
            in_channels = self.act_model_configs.get("action_dim", 192),
            future_action_window_size = self.future_action_window_size,
            past_action_window_size = self.past_action_window_size,
            use_state = self.use_state,
            action_type = self.configs['train_dataset'].get("action_type", "angle"),
            diffusion_steps = self.act_model_configs.get("diffusion_steps", 100),
            state_dim = self.configs["state_encoder"]["state_dim"] if self.use_state=='DiT' else None,
            loss_type = self.configs.get("loss_type", "human"),
            causal_attention = self.act_model_configs.get("causal_attention", True),
            training_objective = self.act_model_configs.get("training_objective", "diffusion"),
            flow_matching_sampling_steps = self.act_model_configs.get("flow_matching_sampling_steps", None),
            flow_matching_t_sampling = self.act_model_configs.get("flow_matching_t_sampling", "uniform"),
            flow_matching_t_min = self.act_model_configs.get("flow_matching_t_min", 0.0),
            flow_matching_t_max = self.act_model_configs.get("flow_matching_t_max", 1.0),
            flow_matching_beta_alpha = self.act_model_configs.get("flow_matching_beta_alpha", 2.0),
            flow_matching_beta_beta = self.act_model_configs.get("flow_matching_beta_beta", 2.0),
            condition_mode = self.act_model_configs.get("condition_mode", "cognition_token"),
            cross_attention_every_n_layers = self.act_model_configs.get("cross_attention_every_n_layers", 1),
            loss_component_weights = self.act_model_configs.get("loss_component_weights", None),
        )

        for param in action_head.parameters():
            assert param.dtype == torch.float32, f"Loaded diffusion action model parameter not in full precision: {param}"

        return action_head

    def _init_mano_model(self, is_right=True):
        """Initialize MANO model for hand-object loss computation.

        Args:
            is_right: True for right hand, False for left hand

        Returns:
            MANO model instance or None if not configured
        """
        try:
            from empire.models.mano import create_mano_layer

            mano_model_path = self.configs.get("mano_model_path")
            if not mano_model_path:
                raise ValueError(
                    "use_refine_loss requires an explicit mano_model_path; "
                    "obtain MANO assets separately under the publisher's terms"
                )
            mano_layer = create_mano_layer(mano_model_path, is_right=is_right)
            print(f"Initialized MANO model for {'right' if is_right else 'left'} hand")
            return mano_layer
        except Exception as e:
            print(f"Warning: Failed to initialize MANO model: {e}")
            print("Hand-object loss will be disabled")
            return None

    def trainable_params_setup(self):
        """
        Setup trainable parameters based on freeze_option and training_stage.
        Training stages:
        - Stage 1 (vlm_only): VLM-only training without DiT model
        - Stage 2 (dit_only): Freeze VLM backbone, only train DiT
        - Default: Train both VLM backbone and DiT
        """
        model = self.model
        model.config.use_cache = False

        # Get training stage from config
        training_stage = self.train_setup_configs.get("training_stage", None)

        if training_stage == "stage1" or training_stage == "vlm_only":
            # Stage 1: VLM-only training (DiT is not created)
            print("Training Stage 1: VLM-only training (no DiT model)")

            if self.train_setup_configs.get("freeze_option", "full_finetune") == "full_finetune":
                model.requires_grad_(True)
                self.vision_tower.requires_grad_(True)
                self.word_embedding.requires_grad_(True)

            if self.train_setup_configs.get("freeze_option", "only_head_and_token") == "only_head_and_token":
                model.requires_grad_(False)
                self.vision_tower.requires_grad_(False)
                self.word_embedding.requires_grad_(False)

            if self.train_setup_configs.get("freeze_option", "freeze_vision_encoder") == "freeze_vision_encoder":
                model.requires_grad_(True)
                self.word_embedding.requires_grad_(True)
                self.vision_tower.requires_grad_(False)
                
            if self.train_setup_configs.get("freeze_option", None) == "freeze_llm_train_connect":
                model.requires_grad_(False)
                self.vision_tower.requires_grad_(True)
                self.word_embedding.requires_grad_(False)
                print("Freezing LLM and training connection in Stage 1")

            # Stage 1 has no DiT model
            if self.act_model is None:
                print("No DiT model in Stage 1 - VLM-only training")

        elif training_stage == "stage2" or training_stage == "dit_only":
            # Stage 2: Freeze VLM backbone, only train DiT
            print("Training Stage 2: Freezing VLM backbone, training DiT only")

            # Freeze all VLM backbone components
            model.requires_grad_(False)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(False)

            # Only train DiT in Stage 2
            if self.act_model is not None:
                self.act_model.requires_grad_(True)
                print("DiT parameters trainable in Stage 2")
            else:
                raise ValueError("Stage 2 requires DiT model to be created!")

        else:
            # Default: Train both VLM and DiT (original behavior)
            print("Training Default: Training both VLM backbone and DiT")

            if self.train_setup_configs.get("freeze_option", "full_finetune") == "full_finetune":
                model.requires_grad_(True)
                self.vision_tower.requires_grad_(True)
                self.word_embedding.requires_grad_(True)

            if self.train_setup_configs.get("freeze_option", "only_head_and_token") == "only_head_and_token":
                model.requires_grad_(False)
                self.vision_tower.requires_grad_(False)
                self.word_embedding.requires_grad_(False)

            if self.train_setup_configs.get("freeze_option", "freeze_vision_encoder") == "freeze_vision_encoder":
                model.requires_grad_(True)
                self.word_embedding.requires_grad_(True)
                self.vision_tower.requires_grad_(False)

            if self.train_setup_configs.get("freeze_option", None) == "freeze_llm_train_connect":
                model.requires_grad_(False)
                self.vision_tower.requires_grad_(False)
                self.word_embedding.requires_grad_(False)

            if self.act_model is not None:
                self.act_model.requires_grad_(True)

        train_condition_encoders = self.configs.get("train_condition_encoders", None)
        if train_condition_encoders is None:
            train_condition_encoders = training_stage not in ("stage2", "dit_only")
        print(f"Condition encoders trainable: {train_condition_encoders}")

        if self.use_state == 'VLM' and self.vlm_state_encoder is not None:
            self.vlm_state_encoder.requires_grad_(train_condition_encoders)
        
        if self.use_fov:
            self.fov_encoder.requires_grad_(train_condition_encoders)

        if self.cognition_token is not None:
            train_cognition_token = self.configs.get("train_cognition_token", None)
            if train_cognition_token is None:
                train_cognition_token = training_stage not in ("stage2", "dit_only")
            self.cognition_token.requires_grad_(train_cognition_token)
            print(f"Cognition token trainable: {train_cognition_token}")

        if self.depth_encoder is not None:
            self.depth_encoder.requires_grad_(False)

        if self.depth_image_encoder is not None:
            self.depth_image_encoder.requires_grad_(train_condition_encoders)

    @property
    def image_processor(self):
        return self.model.processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def model(self):
        return self.backbone

    def compute_hand_object_loss(
        self,
        pred_actions: torch.Tensor,
        gt_actions: torch.Tensor,
        current_state: torch.Tensor,
        obj_mesh_paths: List[str],
        action_masks: torch.Tensor,
        num_sample_points: int = 1000,
    ) -> Optional[torch.Tensor]:
        """
        Compute MSE loss between predicted and ground truth hand-object distances.

        Args:
            pred_actions: Predicted actions of shape (B*T, action_dim)
            gt_actions: Ground truth actions of shape (B*T, action_dim)
            current_state: Current state of shape (B*T, state_dim)
            obj_mesh_paths: List of object mesh paths for each batch
            action_masks: Action mask of shape (B*T, action_dim), 1=compute loss, 0=ignore
            num_sample_points: Number of points to sample from object mesh

        Returns:
            MSE loss scalar, or None if no valid meshes
        """
        from empire.utils.mesh_utils import compute_hand_object_mse_loss_from_actions

        device = pred_actions.device

        # Get model configuration
        use_rel = self.configs.get('train_dataset', {}).get('use_rel', False)
        rel_mode = self.configs.get('rel_mode', 'step')
        use_obj_rel = self.configs.get('train_dataset', {}).get('use_obj_rel', False)

        # For now, we only support angle mode with proper MANO conversion
        if self.action_type != 'angle':
            print(f"Warning: hand_object_loss only supported for angle mode, got {self.action_type}")
            return None

        # Check if MANO models are available
        mano_model_right = getattr(self, 'mano_model_right', None)
        mano_model_left = getattr(self, 'mano_model_left', None)

        if mano_model_right is None and mano_model_left is None:
            print("Warning: MANO model not initialized, skipping hand_object_loss")
            return None

        loss = compute_hand_object_mse_loss_from_actions(
            pred_actions=pred_actions,
            gt_actions=gt_actions,
            current_state=current_state,
            obj_mesh_paths=obj_mesh_paths,
            mano_model_right=mano_model_right,
            mano_model_left=mano_model_left,
            action_masks=action_masks,
            action_type=self.action_type,
            use_rel=use_rel,
            rel_mode=rel_mode,
            use_obj_rel=use_obj_rel,
            num_sample_points=num_sample_points,
            device=device
        )

        return loss

    def _forward_act_model(
        self,
        vlm_features: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        mode: str = "train",
        repeated_diffusion_steps: int = 1,
        cfg_scale: float = 5.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        obj_mesh_paths: Optional[List[str]] = None,
        condition_span_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        
        actions = None
        action_loss = None

        B = vlm_features.shape[0]
        action_features, action_feature_masks = self.extract_action_condition(
            vlm_features, attention_mask, condition_span_mask=condition_span_mask
        )
        model_dtype = next(self.act_model.net.parameters()).dtype
        action_features = action_features.to(model_dtype)

        action_features_repeated = action_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_feature_masks_repeated = action_feature_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
        action_masks_repeated = action_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)

        action_features_repeated = action_features_repeated.view(B*repeated_diffusion_steps, action_features.shape[1], action_features.shape[-1])
        action_feature_masks_repeated = action_feature_masks_repeated.view(B*repeated_diffusion_steps, action_feature_masks.shape[1])
        action_masks_repeated = action_masks_repeated.view(B*repeated_diffusion_steps, action_masks.shape[1], action_masks.shape[2])

        if self.use_state == 'DiT':
            current_state_repeated = current_state.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_repeated = current_state_repeated.view(B*repeated_diffusion_steps, 1, current_state.shape[1])
            current_state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_mask_repeated = current_state_mask_repeated.view(B*repeated_diffusion_steps, 1, current_state_mask.shape[1])
        else:
            current_state_repeated = None
            current_state_mask_repeated = None

        if mode == "train":
            actions_repeated = action_labels.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            actions_repeated = actions_repeated.view(B*repeated_diffusion_steps, action_labels.shape[1], action_labels.shape[2])
            if self.use_state == 'DiT':
                action_loss = self.act_model.loss(
                    actions_repeated,
                    action_features_repeated,
                    action_masks_repeated,
                    current_state_repeated,
                    current_state_mask_repeated,
                    z_mask=action_feature_masks_repeated,
                )
            else:
                action_loss = self.act_model.loss(
                    actions_repeated,
                    action_features_repeated,
                    action_masks_repeated,
                    z_mask=action_feature_masks_repeated,
                )

            # Compute hand-object distance loss if enabled and object meshes are provided
            hand_object_loss = None
            use_refine_loss = self.configs.get('use_refine_loss', False)

            if use_refine_loss and obj_mesh_paths is not None and len(obj_mesh_paths) > 0:
                # For refine_loss, we need to sample predicted actions from the model
                # then compare hand-object distances between predicted and ground truth

                # Expand obj_mesh_paths to match repeated_diffusion_steps
                obj_mesh_paths_expanded = []
                for mesh_path in obj_mesh_paths:
                    obj_mesh_paths_expanded.extend([mesh_path] * repeated_diffusion_steps)

                # Sample predicted actions using the diffusion model
                # Use the same approach as eval mode: loop through batch and sample each
                with torch.no_grad():
                    # Temporarily set to eval mode to ensure consistent behavior with eval
                    was_training = self.act_model.training
                    self.act_model.eval()

                    pred_actions_sampled_list = []
                    for i in range(B):
                        # For each sample, create repeated features (like in eval mode)
                        single_action_features_repeated = action_features[i:i+1]
                        single_action_feature_masks_repeated = action_feature_masks[i:i+1]

                        # action_masks[i]: [T, D] -> repeat to [repeated_steps, T, D]
                        single_action_masks_repeated = action_masks[i:i+1].unsqueeze(0).repeat(1, 1, 1, 1)
                        single_action_masks_repeated = single_action_masks_repeated.view(1, action_masks.shape[1], action_masks.shape[2])

                        if self.use_state == 'DiT':
                            # current_state[i]: [1, D] -> repeat to [repeated_steps, 1, D]
                            single_current_state_repeated = current_state[i:i+1].unsqueeze(0).repeat(1, 1, 1)
                            single_current_state_repeated = single_current_state_repeated.view(1, 1, current_state.shape[1])

                            # current_state_mask[i]: [1, D] -> repeat to [repeated_steps, 1, D]
                            single_current_state_mask_repeated = current_state_mask[i:i+1].unsqueeze(0).repeat(1, 1, 1)
                            single_current_state_mask_repeated = single_current_state_mask_repeated.view(1, 1, current_state_mask.shape[1])

                            # Call sample exactly like eval mode
                            # all input shape or value
                            print(f"cfg_scale : {cfg_scale}, use_ddim: {use_ddim}, num_ddim_steps, {num_ddim_steps}, single_action_features_repeated shape {single_action_features_repeated.shape}, single_current_state_repeated shape {single_current_state_repeated.shape}, single_current_state_mask_repeated shape {single_current_state_mask_repeated.shape}, single_action_masks_repeated shape {single_action_masks_repeated.shape}")
                            # print(self.act_model.net.training)
                            # self.act_model.net.eval()  # set to eval mode for sampling
                            # print(self.act_model.net.training)
                            pred_actions_single = self.act_model.sample(
                                single_action_features_repeated,
                                cfg_scale,
                                single_current_state_repeated,
                                single_current_state_mask_repeated,
                                use_ddim,
                                num_ddim_steps,
                                single_action_masks_repeated,
                                single_action_feature_masks_repeated,
                            )
                            # self.act_model.net.train()  # set to eval mode for sampling
                            
                        else:
                            # Call sample without state
                            pred_actions_single = self.act_model.sample(
                                single_action_features_repeated,
                                cfg_scale,
                                None,
                                None,
                                use_ddim,
                                num_ddim_steps,
                                single_action_masks_repeated,
                                single_action_feature_masks_repeated,
                            )

                        pred_actions_sampled_list.append(pred_actions_single)

                    # Restore original training mode
                    self.act_model.train(was_training)

                    # Stack all sampled actions: [B, repeated_steps, T+1, D]
                    pred_actions_sampled = torch.stack(pred_actions_sampled_list, dim=0)
                    # Reshape to [B*repeated_steps, T+1, D]
                    pred_actions_sampled_expanded = pred_actions_sampled.view(B*repeated_diffusion_steps, pred_actions_sampled.shape[2], pred_actions_sampled.shape[3])
                self.act_model.train(True)
                # Use first timestep for comparison
                pred_actions_for_dist = pred_actions_sampled_expanded[:, 0, :]  # [B*repeated_steps, D]
                gt_actions_for_dist = actions_repeated[:, 0, :]  # [B*repeated_steps, D]

                # Reshape current_state to match
                current_state_for_dist = current_state_repeated[:, 0, :] if self.use_state == 'DiT' else None

                # Reshape action_masks to match: [B*repeated_diffusion_steps, action_dim]
                action_masks_for_dist = action_masks_repeated[:, 0, :]  # [B*repeated_diffusion_steps, action_dim]

                hand_object_loss = self.compute_hand_object_loss(
                    pred_actions_for_dist,
                    gt_actions_for_dist,
                    current_state_for_dist,
                    obj_mesh_paths_expanded,
                    action_masks_for_dist,
                    num_sample_points=1000,
                )

                # Add hand-object loss to the loss dict if computed
                if hand_object_loss is not None:
                    action_loss['refine_loss'] = hand_object_loss
                    action_loss['loss'] = action_loss['loss'] + hand_object_loss

            return actions, action_loss
        else:
            # evaluate mode
            # sample multiple action chunks
            # print(f"cfg_scale : {cfg_scale}, use_ddim: {use_ddim}, num_ddim_steps, {num_ddim_steps}, action_features_repeated shape {action_features_repeated.shape}, current_state_repeated shape {current_state_repeated.shape}, current_state_mask_repeated shape {current_state_mask_repeated.shape}, action_masks_repeated shape {action_masks_repeated.shape}")
            actions = self.act_model.sample(
                action_features_repeated,
                cfg_scale,
                current_state_repeated,      #[B*sample_times, 1, D]
                current_state_mask_repeated, #[B*sample_times, 1, D]
                use_ddim,
                num_ddim_steps,
                action_masks_repeated, # ori x_mask
                action_feature_masks_repeated,
            )

            return actions, action_loss

    def extract_cognition_token(self, output_hs, attention_mask): # essentially get the embedding of the last non-padding token
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, output_hs.size(-1))
        action_features = output_hs.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        return action_features

    def extract_action_condition(self, output_hs, attention_mask, condition_span_mask=None):
        condition_mode = self.act_model_configs.get("condition_mode", "cognition_token")
        condition_aliases = {
            "single_token": "cognition_token",
            "last_token": "cognition_token",
            "all_vlm_tokens": "vlm_cross_attention",
            "cross_attention": "vlm_cross_attention",
        }
        condition_mode = condition_aliases.get(condition_mode, condition_mode)

        if attention_mask is None:
            attention_mask = torch.ones(
                output_hs.shape[:2],
                dtype=torch.bool,
                device=output_hs.device,
            )
        else:
            attention_mask = attention_mask.to(output_hs.device, dtype=torch.bool)

        if condition_span_mask is not None:
            condition_span_mask = condition_span_mask.to(output_hs.device, dtype=torch.bool)

        if condition_mode == "cognition_token":
            action_features = self.extract_cognition_token(output_hs, attention_mask)
            action_feature_masks = torch.ones(
                action_features.shape[:2],
                dtype=torch.bool,
                device=output_hs.device,
            )
            return action_features, action_feature_masks

        if condition_mode == "vlm_cross_attention":
            max_condition_tokens = self.act_model_configs.get("max_condition_tokens", None)
            if max_condition_tokens is not None:
                max_condition_tokens = int(max_condition_tokens)
                if max_condition_tokens <= 0:
                    max_condition_tokens = output_hs.shape[1]
                max_condition_tokens = min(max_condition_tokens, output_hs.shape[1])
                output_hs = output_hs[:, :max_condition_tokens]
                attention_mask = attention_mask[:, :max_condition_tokens]
                if condition_span_mask is not None:
                    condition_span_mask = condition_span_mask[:, :max_condition_tokens]

            if condition_span_mask is not None:
                # plan-span conditioning: restrict the DiT cross-attention to the
                # CoT-suffix tokens only. condition_span_mask aligns 1:1 with output_hs
                # (same positions), so AND-ing it with the validity mask keeps exactly the
                # valid CoT-suffix positions.
                if condition_span_mask.shape != attention_mask.shape:
                    raise ValueError(
                        f"condition_span_mask shape {tuple(condition_span_mask.shape)} does not "
                        f"match attention_mask shape {tuple(attention_mask.shape)}; cannot align "
                        "the CoT-span mask with the VLM condition tokens."
                    )
                attention_mask = attention_mask & condition_span_mask
                if not attention_mask.any(dim=1).all():
                    raise ValueError(
                        "cot_condition_span=True but at least one sample has zero selected "
                        "CoT-span condition tokens after masking (labels != -100 selected "
                        "nothing). Ensure the batch was built with use_cot=true and a non-empty "
                        "GT CoT suffix."
                    )

            if not attention_mask.any(dim=1).all():
                raise ValueError("Each sample must contain at least one valid VLM condition token.")
            return output_hs, attention_mask

        raise ValueError(f"Unknown action condition_mode: {condition_mode}")

    def interactive_depth_features(self, image_features, depth_features):
        # assert image_features.shape == depth_features.shape, f"image_features shape {image_features.shape} does not match depth_features shape {depth_features.shape}"
        # interactive depth features to match image features size
        if self.depth_image_encoder_type == "CONCAT":
            depth_features = self.depth_image_encoder(depth_features)
            return depth_features
        else:
            image_features = self.depth_image_encoder(image_features, depth_features, depth_features)
        return image_features

    def prepare_vlm_input_embeddings(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        B = input_ids.shape[0]
        word_embeds = self.model.get_input_embeddings()(input_ids)
        input_ids_mask = attention_mask

        # Store original labels for later processing. Some non-CoT batches use
        # a zero tensor as a placeholder; treat it as no language supervision.
        labels_original = labels
        if labels_original is not None and torch.all(labels_original == 0):
            labels_original = None

        # Build extra condition embeddings. When CoT labels are present these
        # are inserted before the first supervised answer token, so the answer
        # can attend to them under the causal mask.
        additional_embeds_list = []
        additional_masks_list = []

        if self.use_state == 'VLM' and self.vlm_state_encoder is not None and current_state is not None and current_state_mask is not None:
            current_state = current_state * current_state_mask.to(current_state.dtype)
            state_embeds = self.vlm_state_encoder(
                torch.cat([current_state, current_state_mask.to(current_state.dtype)], dim=1)
            )
            state_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)

            additional_embeds_list.append(state_embeds.unsqueeze(1))
            additional_masks_list.append(state_ids_mask)

        if self.use_fov:
            fov_embeds = self.fov_encoder(fov)
            fov_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)

            additional_embeds_list.append(fov_embeds.unsqueeze(1))
            additional_masks_list.append(fov_ids_mask)

        if self.use_cognition_token and self.cognition_token_id is not None:
            cog_ids = torch.ones_like(input_ids[:, 0:1]) * self.cognition_token_id # [B, 1]
            cog_embeds = self.model.get_input_embeddings()(cog_ids)
            cog_ids_mask = torch.ones_like(cog_ids, dtype=torch.bool)

            if hasattr(self, 'cognition_token') and self.cognition_token is not None:
                assert self.cognition_token.shape[0] == cog_embeds.shape[-1], f"cognition token shape {self.cognition_token.shape} does not match cog embeds shape {cog_embeds.shape}"
                cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)

            additional_embeds_list.append(cog_embeds)
            additional_masks_list.append(cog_ids_mask)

        # Note: Here we only use `self.cognition_token_id` as a placeholder for the token corresponding to the FOV or states (if any) input.
        # In practice, the embedding passed to the LLM will be replaced with the actual FOV or states (if any) embedding.
        # For Stage 1 (VLM-only), use a default token ID; for Stage 2, use the actual cognition_token_id
        placeholder_token_id = self.cognition_token_id if (self.use_cognition_token and self.cognition_token_id is not None) else 10
        num_additional_tokens = sum(embeds.shape[1] for embeds in additional_embeds_list)

        if num_additional_tokens > 0:
            additional_embeds = torch.cat(additional_embeds_list, dim=1)
            additional_masks = torch.cat(additional_masks_list, dim=1)
            additional_tokens = torch.full(
                (B, num_additional_tokens),
                placeholder_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            if labels_original is not None:
                supervised_label_mask = labels_original != -100
                valid_lengths = input_ids_mask.to(torch.long).sum(dim=1)
                first_label_indices = supervised_label_mask.to(torch.long).argmax(dim=1)
                insert_positions = torch.where(
                    supervised_label_mask.any(dim=1),
                    first_label_indices,
                    valid_lengths,
                )
                additional_labels = torch.full(
                    (B, num_additional_tokens),
                    -100,
                    dtype=labels_original.dtype,
                    device=labels_original.device,
                )
            else:
                insert_positions = torch.full(
                    (B,),
                    input_ids.shape[1],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                additional_labels = None

            inputs_embeds = []
            inputs_masks_original = []
            input_ids_extended = []
            labels_extended_list = []
            for batch_idx in range(B):
                insert_pos = int(insert_positions[batch_idx].item())
                inputs_embeds.append(torch.cat([
                    word_embeds[batch_idx, :insert_pos],
                    additional_embeds[batch_idx],
                    word_embeds[batch_idx, insert_pos:],
                ], dim=0))
                inputs_masks_original.append(torch.cat([
                    input_ids_mask[batch_idx, :insert_pos],
                    additional_masks[batch_idx],
                    input_ids_mask[batch_idx, insert_pos:],
                ], dim=0))
                input_ids_extended.append(torch.cat([
                    input_ids[batch_idx, :insert_pos],
                    additional_tokens[batch_idx],
                    input_ids[batch_idx, insert_pos:],
                ], dim=0))
                if labels_original is not None:
                    labels_extended_list.append(torch.cat([
                        labels_original[batch_idx, :insert_pos],
                        additional_labels[batch_idx],
                        labels_original[batch_idx, insert_pos:],
                    ], dim=0))

            inputs_embeds = torch.stack(inputs_embeds, dim=0)
            inputs_masks_original = torch.stack(inputs_masks_original, dim=0)
            input_ids = torch.stack(input_ids_extended, dim=0)
            labels_extended = torch.stack(labels_extended_list, dim=0) if labels_original is not None else None
        else:
            inputs_embeds = word_embeds
            inputs_masks_original = input_ids_mask
            labels_extended = labels_original

        inputs_embeds, attention_mask = move_masked_to_left(inputs_embeds, inputs_masks_original)
        input_ids, sequence_mask = move_masked_to_left_ids(input_ids, inputs_masks_original)

        # Apply the same reordering to labels
        if labels_extended is not None:
            labels, _ = move_masked_to_left_ids(labels_extended, inputs_masks_original, pad_zero=False)
            # Pad to match input_ids length
            if labels.shape[1] < input_ids.shape[1]:
                pad_len = input_ids.shape[1] - labels.shape[1]
                labels = torch.cat([labels, torch.full((B, pad_len), -100, dtype=labels.dtype, device=labels.device)], dim=1)
            elif labels.shape[1] > input_ids.shape[1]:
                labels = labels[:, :input_ids.shape[1]]
        else:
            labels = None

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        # Merge text and images
        if pixel_values is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                # import pdb
                # pdb.set_trace()
                image_features = self.model.get_image_features(pixel_values) # pixel_values: [B, 3, H, W]
                # get the depth features
                if self.use_depth:
                    pixel_values = pixel_values.unsqueeze(1) # [B, 1, 3, H, W]
                    da_layers = self.configs.get("da_layer", [23])
                    if da_layers is None:
                        da_layers = [23]
                    elif isinstance(da_layers, int):
                        da_layers = [da_layers]
                    elif isinstance(da_layers, str):
                        da_layers = [int(da_layers)]
                    else:
                        da_layers = [int(layer) for layer in da_layers]
                    if len(da_layers) == 0:
                        raise ValueError("configs['da_layer'] must contain at least one layer")

                    target_da_layer = da_layers[-1]
                    depth_output = self.depth_encoder(pixel_values, None, None, export_feat_layers=da_layers)
                    # import pdb
                    # pdb.set_trace()
                    if self.depth_image_encoder_type == "CONCAT":
                        feat_key = f"feat_layer_{target_da_layer}"
                        depth_aux = depth_output.get("aux", {})
                        if feat_key not in depth_aux:
                            raise KeyError(
                                f"Depth feature key {feat_key} not found in depth_output['aux']. "
                                f"Available keys: {list(depth_aux.keys())}. Requested da_layer={da_layers}."
                            )
                        depth_features = depth_aux[feat_key] # [B, 1, 16, 16, 1024]
                        depth_features = depth_features.squeeze(1) # [B, 16, 16, 1024]
                        depth_patch_nums = depth_features.shape[1] * depth_features.shape[2]
                        depth_features = depth_features.reshape(B, depth_patch_nums, -1) # [B, 16*16, 1024]
                        depth_features = self.depth_image_encoder(depth_features) # [B, 16*16, 2304]
                        # Pad the attention mask and input_ids with depth_patch_nums
                        num_depth_tokens = depth_patch_nums
                        num_image_tokens = image_features.shape[1]
                        
                        # 1. Create depth tokens
                        depth_tokens = torch.full((B, num_depth_tokens), self.model.config.image_token_index, dtype=input_ids.dtype, device=input_ids.device)

                        # 2. Create depth embeds
                        depth_embeds = self.model.get_input_embeddings()(depth_tokens)

                        # 3. Create depth mask
                        depth_mask = torch.ones((B, num_depth_tokens), dtype=attention_mask.dtype, device=attention_mask.device)

                        # 4. Concatenate
                        input_ids = torch.cat([input_ids[:, :num_image_tokens], depth_tokens, input_ids[:, num_image_tokens:]], dim=1)
                        inputs_embeds = torch.cat([inputs_embeds[:, :num_image_tokens], depth_embeds, inputs_embeds[:, num_image_tokens:]], dim=1)
                        attention_mask = torch.cat([attention_mask[:, :num_image_tokens], depth_mask, attention_mask[:, num_image_tokens:]], dim=1)
                        sequence_mask = torch.cat([sequence_mask[:, :num_image_tokens], depth_mask, sequence_mask[:, num_image_tokens:]], dim=1)

                        # 5. Synchronize labels if provided
                        if labels is not None:
                            # Insert ignore tokens (-100) for depth tokens in labels
                            # Note: labels has already been extended with additional_tokens, so we use it directly
                            depth_labels = torch.full((B, num_depth_tokens), -100, dtype=labels.dtype, device=labels.device)
                            labels = torch.cat([labels[:, :num_image_tokens], depth_labels, labels[:, num_image_tokens:]], dim=1)


                        # 6. Update position_ids and cache_position
                        past_seen_tokens = 0
                        cache_position = torch.arange(
                            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                        )
                        position_ids = cache_position.unsqueeze(0) + 1

                        depth_features = depth_features / (self.model.config.text_config.hidden_size**0.5)
                        image_features = torch.cat([image_features, depth_features], dim=1)
                    else:
                        cls_feat_key = f"feat_layer_{target_da_layer}_cls"
                        depth_aux_cls = depth_output.get("aux_cls", {})
                        if cls_feat_key in depth_aux_cls:
                            depth_features = depth_aux_cls[cls_feat_key].squeeze(1) # [B, 1, D]
                            depth_features = depth_features.repeat(1, image_features.shape[1], 1)
                        else:
                            feat_key = f"feat_layer_{target_da_layer}"
                            depth_aux = depth_output.get("aux", {})
                            if feat_key not in depth_aux:
                                raise KeyError(
                                    f"Depth feature key {feat_key} not found in depth_output['aux'], and "
                                    f"Depth CLS feature key {cls_feat_key} not found in depth_output['aux_cls']. "
                                    f"Available aux keys: {list(depth_aux.keys())}. "
                                    f"Available aux_cls keys: {list(depth_aux_cls.keys())}. "
                                    f"Requested da_layer={da_layers}."
                                )
                            depth_features = depth_aux[feat_key] # [B, 1, H, W, D]
                            if depth_features.dim() == 5:
                                depth_features = depth_features.squeeze(1) # [B, H, W, D]
                            if depth_features.dim() == 4:
                                depth_features = depth_features.reshape(B, -1, depth_features.shape[-1]) # [B, H*W, D]
                            elif depth_features.dim() != 3:
                                raise ValueError(f"Unexpected depth feature shape for {feat_key}: {depth_features.shape}")
                        depth_features = depth_features / (self.model.config.text_config.hidden_size**0.5)
                        image_features = self.interactive_depth_features(image_features, depth_features)

            special_image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        is_training = labels is not None
        token_type_ids = None
        if is_training:
            token_type_ids = torch.zeros_like(attention_mask, dtype=torch.long)
            token_type_ids[labels != -100] = 1

        causal_mask = self.model._update_causal_mask(
            attention_mask, token_type_ids, None, cache_position, input_ids, inputs_embeds, is_training
        )
        return {
            "attention_mask": causal_mask,
            "attention_mask_2d": attention_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
            "labels": labels,
        }, sequence_mask

    def prepare_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_aligned_labels: bool = False,
        **kwargs,
    ):

        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
                pixel_values,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                labels=labels,
                **kwargs,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model.language_model(
                past_key_values=None,
                use_cache=use_cache,
                output_hidden_states=True,
                num_logits_to_keep=0, # can be modified
                **vlm_inputs
            )

        output_hs = outputs.hidden_states[-1]

        # Debug: Check shapes if using CoT
        # if self.use_cot and labels is not None:
        #     print(f"Debug: input_ids shape in vlm_inputs: {vlm_inputs.get('input_ids', 'N/A').shape if 'input_ids' in vlm_inputs else 'N/A'}")
        #     if hasattr(outputs, 'logits') and outputs.logits is not None:
        #         print(f"Debug: logits shape: {outputs.logits.shape}")
        #     print(f"Debug: labels shape: {labels.shape}")

        # When requested (cot_condition_span) also return the reordered+extended label
        # tensor that aligns position-by-position with output_hs. vlm_inputs["labels"] was
        # produced by the SAME move_masked_to_left permutation as inputs_embeds (and had
        # depth/additional tokens inserted at the same positions), so (labels != -100)
        # selects exactly the output_hs positions of the supervised CoT-suffix tokens.
        if return_aligned_labels:
            return output_hs, inputs_masks, outputs.loss, vlm_inputs.get("labels", None)
        return output_hs, inputs_masks, outputs.loss

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        mode="train",
        obj_mesh_path: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):

        if mode == "cot_generate":
            return {
                "cot_predictions": self.generate_cot_from_preprocessed_inputs(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    current_state_mask=current_state_mask,
                    current_state=current_state,
                    fov=fov,
                    max_new_tokens=kwargs.get("max_new_tokens", 128),
                    temperature=kwargs.get("temperature", 1.0),
                    do_sample=kwargs.get("do_sample", False),
                    top_k=kwargs.get("top_k", 50),
                    top_p=kwargs.get("top_p", 0.9),
                    skip_special_tokens=kwargs.get("skip_special_tokens", False),
                )
            }

        assert mode == "train", f"mode {mode} not supported in the forward function."
    
        # if labels is not None:
        #     print('Using cot ...')
        # else:
        #     print('Not using cot ...')
        loss = {}
        # plan-span (cot_condition_span): also fetch the reordered labels aligned with output_hs
        # so the DiT cross-attention can be restricted to the CoT-suffix tokens only.
        vlm_features_out = self.prepare_vlm_features(
            pixel_values,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache,
            labels,
            return_aligned_labels=self.cot_condition_span,
            **kwargs,
        )
        if self.cot_condition_span:
            output_hs, inputs_masks, llm_loss, aligned_labels = vlm_features_out
        else:
            output_hs, inputs_masks, llm_loss = vlm_features_out
            aligned_labels = None

        # Build the CoT-span mask over output_hs positions: (labels != -100) marks the
        # supervised GT-CoT suffix tokens, and aligned_labels lines up 1:1 with output_hs.
        condition_span_mask = None
        if self.cot_condition_span and aligned_labels is not None:
            condition_span_mask = (aligned_labels != -100)

        # Only compute action loss if DiT model exists
        if self.act_model is not None:
            _, action_loss = self._forward_act_model(
                vlm_features = output_hs,
                action_labels = action_labels,
                attention_mask = inputs_masks,
                action_masks = action_masks,
                current_state = current_state,
                current_state_mask = current_state_mask,
                mode = mode,
                repeated_diffusion_steps = self.repeated_diffusion_steps,
                obj_mesh_paths=obj_mesh_path,
                condition_span_mask=condition_span_mask,
            )
            self._update_loss(loss, action_loss)
        else:
            # Stage 1: No DiT model, only VLM loss
            pass
        if llm_loss is not None and labels is not None:
            # print(f"LLM loss: {llm_loss.item()}")
            # Always record llm_loss for logging/metrics visibility.
            self._update_loss(loss, {"llm_loss": llm_loss})
            # plan-span: the GT CoT is consumed purely as a DiT conditioning signal, so the LM
            # loss must NOT contribute to the trained/back-propagated total. All other
            # modes keep the legacy behavior of adding llm_loss to the total.
            _lm_w = float(self.configs.get('lm_loss_weight', 0.0) or 0.0)
            if _lm_w > 0.0:
                # EXPERIMENT: keep the LM loss ON in Stage-2 at weight lm_loss_weight (e.g. 0.5),
                # even for plan-span (cot_condition_span), so the VLM front (LLM + connectors) keeps
                # training on the GT-CoT. Default (flag absent -> 0.0) preserves legacy behavior.
                loss['loss'] = loss.get('loss', 0) + _lm_w * llm_loss
            elif not self.cot_condition_span:
                loss['loss'] = loss.get('loss', 0) + llm_loss
        return loss

    def generate_cot_from_preprocessed_inputs(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.9,
        skip_special_tokens: bool = False,
    ) -> Union[str, List[str]]:
        """Generate CoT from already-tokenized training prompts during the training loop."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        device = input_ids.device
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.to(device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        if current_state is not None:
            current_state = current_state.to(device)
        if current_state_mask is not None:
            current_state_mask = current_state_mask.to(device)
        if fov is not None:
            fov = fov.to(device)

        vlm_inputs, _ = self.prepare_vlm_input_embeddings(
            pixel_values,
            input_ids,
            attention_mask,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
        )

        inputs_embeds = vlm_inputs["inputs_embeds"]
        position_ids = vlm_inputs["position_ids"]
        cache_position = vlm_inputs["cache_position"]
        attention_mask_2d = vlm_inputs["attention_mask_2d"]

        from transformers import DynamicCache
        past_key_values = DynamicCache()

        generated_tokens = []
        stop_token = self.tokenizer.eos_token_id
        finished = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
        B = inputs_embeds.shape[0]

        for _ in range(max_new_tokens):
            causal_mask = self.model._update_causal_mask(
                attention_mask_2d,
                token_type_ids=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                is_training=False,
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    num_logits_to_keep=1,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_token_logits.shape[-1])
                    topk_values = torch.topk(next_token_logits, top_k, dim=-1).values
                    topk_threshold = topk_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(
                        next_token_logits < topk_threshold,
                        float("-inf"),
                    )

                probs = torch.softmax(next_token_logits, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                if top_p is not None and top_p < 1.0:
                    mask = probs_sum - probs_sort > top_p
                    probs_sort[mask] = 0.0

                probs_denom = probs_sort.sum(dim=-1, keepdim=True)
                probs_sort = torch.where(
                    probs_denom > 0,
                    probs_sort / probs_denom,
                    torch.full_like(probs_sort, 1.0 / probs_sort.shape[-1]),
                )
                next_token_idx = torch.multinomial(probs_sort, num_samples=1)
                next_token = torch.gather(probs_idx, -1, next_token_idx).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            if stop_token is not None:
                next_token = torch.where(finished, torch.full_like(next_token, stop_token), next_token)

            generated_tokens.append(next_token)

            if stop_token is not None:
                finished = finished | next_token.eq(stop_token)
            if stop_token is not None:
                local_finished = finished.all()
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    global_finished = torch.tensor(
                        1 if bool(local_finished.item()) else 0,
                        device=inputs_embeds.device,
                        dtype=torch.int32,
                    )
                    torch.distributed.all_reduce(global_finished, op=torch.distributed.ReduceOp.MIN)
                    if bool(global_finished.item()):
                        break
                elif local_finished:
                    break

            next_token_tensor = next_token.unsqueeze(-1)
            inputs_embeds = self.model.get_input_embeddings()(next_token_tensor)
            attention_mask_2d = torch.cat(
                [
                    attention_mask_2d,
                    torch.ones((B, 1), dtype=attention_mask_2d.dtype, device=attention_mask_2d.device),
                ],
                dim=1,
            )
            cache_position = cache_position[-1:] + 1
            position_ids = cache_position.unsqueeze(0) + 1

        if not generated_tokens:
            return "" if B == 1 else [""] * B

        generated_tokens = torch.stack(generated_tokens, dim=1)
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=skip_special_tokens)
        return decoded[0] if len(decoded) == 1 else decoded

    def image_preprocess(self, image: Image, size: Tuple[int, int] = (224, 224)) -> Image:
        width, height = image.size

        return image
        

    def predict_action(
        self, 
        image, 
        instruction: str, 
        current_state, 
        current_state_mask, 
        use_ddim=True, 
        num_ddim_steps=10, 
        cfg_scale=5.0, 
        action_mask_torch=None, 
        fov=None, 
        sample_times=1, 
        use_cache=False
    ) -> np.ndarray:
        """
        support B = 1 only for now
        instruction: str
        current_state: normalized current hand action, [B, D]
        current_state_mask: [B, D]
        action_mask_torch: [B, T, D]
        fov: [B, 2]
        return: predicted normalized hand action, [sample_times, T, D]
        """

        B = current_state.shape[0]
        assert B == 1, f"Batch size {B} not supported in predict_action for now."

        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        prefix = '<image>'
        model_inputs = self.processor(text=prefix + instruction, images=image, return_tensors="pt").to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        # Preprocess Image
        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        if action_mask_torch is None:
            x_mask = torch.zeros(B, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
            # x_mask[:, :, :102] = 1.0 # predict dual hand actions
            x_mask[:, :, 51:102] = 1.0 # predict right hand actions
            # x_mask[:, :, 0:51] = 1.0 # predict left hand actions
        else:
            x_mask = action_mask_torch.to(input_ids.device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
        current_state_mask = current_state_mask.to(input_ids.device)
        current_state = current_state.to(input_ids.device)
        fov = fov.to(input_ids.device) if fov is not None else None

        output_hs, inputs_masks, _ = self.prepare_vlm_features(
            pixel_value,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache=use_cache,
        )
        # handle multiple samples for one input
        samples, _ = self._forward_act_model(
            vlm_features = output_hs,
            attention_mask = inputs_masks,
            action_masks = x_mask,
            current_state = current_state,
            current_state_mask = current_state_mask,
            mode = "eval",
            repeated_diffusion_steps = sample_times,
            cfg_scale = cfg_scale,
            use_ddim = use_ddim,
            num_ddim_steps = num_ddim_steps,
        )
        action_np = samples.cpu().numpy() * x_mask.cpu().numpy()    # sample_times x T x D
        return action_np

    def predict_cot(
        self,
        image,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.9,
        current_state=None,
        current_state_mask=None,
        fov=None,
    ) -> str:
        """
        Generate chain-of-thought description for a given image and instruction.
        Used for Stage 1 (VLM-only) evaluation without DiT.
        Follows the official PaliGemma inference pattern using model.generate().

        Args:
            image: PIL Image or numpy array of shape (H, W, 3)
            instruction: task instruction string
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            do_sample: whether to use sampling (False = greedy decoding)
            top_k: top-k filtering parameter
            top_p: nucleus sampling parameter
            current_state: normalized current state, [B, D] (optional)
            current_state_mask: [B, D] (optional)
            fov: field of view, [B, 2] (optional)

        Returns:
            Generated CoT text string
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

        # Keep inference prompt aligned with training, but deterministic for reproducibility.
        prefix = '<image>' + build_cot_instruction_prompt(instruction, randomize=False) #+ '<sap>'
        model_inputs = self.processor(text=prefix, images=image, return_tensors="pt").to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        # Preprocess pixel values
        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)

        if current_state is not None:
            current_state = current_state.to(input_ids.device)
        if current_state_mask is not None:
            current_state_mask = current_state_mask.to(input_ids.device)
        if fov is not None:
            fov = fov.to(input_ids.device)

        # Prepare VLM input embeddings (handles depth, fov, state, image tokens)
        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
            pixel_value,
            input_ids,
            attention_mask,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
        )

        inputs_embeds = vlm_inputs['inputs_embeds']
        # Use the pre-built causal mask / positions from prepare_vlm_input_embeddings.
        # This is important for PaliGemma generate() when inputs_embeds are passed,
        # otherwise the underlying prepare_inputs_for_generation may see empty input_ids
        # and build an invalid causal mask (sequence_length=0).
        causal_attention_mask = vlm_inputs['attention_mask']
        position_ids = vlm_inputs['position_ids']
        cache_position = vlm_inputs['cache_position']
        input_len = inputs_embeds.shape[1]

        attention_mask = torch.ones((inputs_embeds.shape[0], input_len), dtype=torch.bool).to(inputs_embeds.device)
        # import pdb
        # pdb.set_trace()
        # Try generation using language_model
        generation = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False
        )

        # Decode only the generated tokens
        # When inputs_embeds is provided without input_ids, the generated sequence starts with dummy tokens of length input_len
        cot_text = self.tokenizer.decode(generation[0], skip_special_tokens=False)

        return cot_text

    def predict_cot_low_level(
        self,
        image,
        instruction: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.9,
        current_state=None,
        current_state_mask=None,
        fov=None,
    ) -> Union[str, List[str]]:
        """
        Low-level auto-regressive generation loop mimicking PaliGemma's custom Inference.py.
        This manually iterates token by token, updating the KV cache and inputs.
        """
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

        prefix = '<image>' + build_cot_instruction_prompt(instruction, randomize=False)
        model_inputs = self.processor(text=prefix, images=image, return_tensors="pt").to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)

        if current_state is not None:
            current_state = current_state.to(input_ids.device)
        if current_state_mask is not None:
            current_state_mask = current_state_mask.to(input_ids.device)
        if fov is not None:
            fov = fov.to(input_ids.device)

        # Prepare VLM input embeddings (handles depth, fov, state, image tokens)
        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
            pixel_value,
            input_ids,
            attention_mask,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
        )

        inputs_embeds = vlm_inputs['inputs_embeds']
        position_ids = vlm_inputs['position_ids']
        cache_position = vlm_inputs['cache_position']
        attention_mask_2d = vlm_inputs["attention_mask_2d"]

        from transformers import DynamicCache
        past_key_values = DynamicCache()

        generated_tokens = []
        stop_token = self.tokenizer.eos_token_id
        finished = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)

        B = inputs_embeds.shape[0]

        # Start auto-regressive generation
        for step in range(max_new_tokens):
            causal_mask = self.model._update_causal_mask(
                attention_mask_2d,
                token_type_ids=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                is_training=False,
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    num_logits_to_keep=1,
                )
                
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :] # [B, vocab_size]
            
            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_token_logits.shape[-1])
                    topk_values = torch.topk(next_token_logits, top_k, dim=-1).values
                    topk_threshold = topk_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < topk_threshold, float("-inf"))

                probs = torch.softmax(next_token_logits, dim=-1)

                # Apply top_p (nucleus sampling)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                if top_p is not None and top_p < 1.0:
                    mask = probs_sum - probs_sort > top_p
                    probs_sort[mask] = 0.0

                probs_denom = probs_sort.sum(dim=-1, keepdim=True)
                probs_sort = torch.where(
                    probs_denom > 0,
                    probs_sort / probs_denom,
                    torch.full_like(probs_sort, 1.0 / probs_sort.shape[-1]),
                )

                next_token_idx = torch.multinomial(probs_sort, num_samples=1)
                next_token = torch.gather(probs_idx, -1, next_token_idx).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            if stop_token is not None:
                next_token = torch.where(finished, torch.full_like(next_token, stop_token), next_token)

            generated_tokens.append(next_token)

            if stop_token is not None:
                finished = finished | next_token.eq(stop_token)
            if stop_token is not None and finished.all():
                break
                
            # Prepare inputs for the next token step 
            # (We only feed the newly predicted token embed from now on, not the whole sequence)
            next_token_tensor = next_token.unsqueeze(-1) # [B, 1]
            inputs_embeds = self.model.get_input_embeddings()(next_token_tensor)
            attention_mask_2d = torch.cat(
                [attention_mask_2d, torch.ones((B, 1), dtype=attention_mask_2d.dtype, device=attention_mask_2d.device)],
                dim=1,
            )
            cache_position = cache_position[-1:] + 1
            position_ids = cache_position.unsqueeze(0) + 1

        if not generated_tokens:
            return "" if B == 1 else [""] * B

        generated_tokens = torch.stack(generated_tokens, dim=1)
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return decoded[0] if len(decoded) == 1 else decoded

    def predict_cot_action(
        self,
        image,
        instruction: str,
        current_state,
        current_state_mask,
        action_mask_torch=None,
        fov=None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.9,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        cfg_scale: float = 5.0,
        sample_times: int = 1,
    ) -> Tuple[Union[str, List[str]], np.ndarray]:
        """
        Joint CoT and action inference using one shared VLM prefix pass.

        The prefix is the same CoT-style prompt used by predict_cot_low_level.
        The final prefix hidden state feeds DiT action sampling, and the same
        KV cache is reused to continue low-level CoT generation.
        """
        B = current_state.shape[0]
        assert B == 1, f"Batch size {B} not supported in predict_cot_action for now."

        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

        prefix = '<image>' + build_cot_instruction_prompt(instruction, randomize=False)
        model_inputs = self.processor(text=prefix, images=image, return_tensors="pt").to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        if action_mask_torch is None:
            x_mask = torch.zeros(B, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
            x_mask[:, :, 51:102] = 1.0
        else:
            x_mask = action_mask_torch.to(input_ids.device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
        current_state = current_state.to(input_ids.device)
        current_state_mask = current_state_mask.to(input_ids.device)
        fov = fov.to(input_ids.device) if fov is not None else None

        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
            pixel_value,
            input_ids,
            attention_mask,
            current_state_mask=current_state_mask,
            current_state=current_state,
            fov=fov,
        )

        inputs_embeds = vlm_inputs['inputs_embeds']
        position_ids = vlm_inputs['position_ids']
        cache_position = vlm_inputs['cache_position']
        attention_mask_2d = vlm_inputs["attention_mask_2d"]

        from transformers import DynamicCache
        past_key_values = DynamicCache()

        causal_mask = self.model._update_causal_mask(
            attention_mask_2d,
            token_type_ids=None,
            past_key_values=past_key_values,
            cache_position=cache_position,
            input_ids=None,
            inputs_embeds=inputs_embeds,
            is_training=False,
        )

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=True,
                num_logits_to_keep=1,
            )

        past_key_values = outputs.past_key_values
        output_hs = outputs.hidden_states[-1]

        samples, _ = self._forward_act_model(
            vlm_features=output_hs,
            attention_mask=inputs_masks,
            action_masks=x_mask,
            current_state=current_state,
            current_state_mask=current_state_mask,
            mode="eval",
            repeated_diffusion_steps=sample_times,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )
        action_np = samples.detach().cpu().numpy() * x_mask.detach().cpu().numpy()

        generated_tokens = []
        stop_token = self.tokenizer.eos_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=inputs_embeds.device)

        for step in range(max_new_tokens):
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k is not None and top_k > 0:
                    top_k_cur = min(top_k, next_token_logits.shape[-1])
                    topk_values = torch.topk(next_token_logits, top_k_cur, dim=-1).values
                    topk_threshold = topk_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < topk_threshold, float("-inf"))

                probs = torch.softmax(next_token_logits, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                if top_p is not None and top_p < 1.0:
                    mask = probs_sum - probs_sort > top_p
                    probs_sort[mask] = 0.0

                probs_denom = probs_sort.sum(dim=-1, keepdim=True)
                probs_sort = torch.where(
                    probs_denom > 0,
                    probs_sort / probs_denom,
                    torch.full_like(probs_sort, 1.0 / probs_sort.shape[-1]),
                )

                next_token_idx = torch.multinomial(probs_sort, num_samples=1)
                next_token = torch.gather(probs_idx, -1, next_token_idx).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            if stop_token is not None:
                next_token = torch.where(finished, torch.full_like(next_token, stop_token), next_token)

            generated_tokens.append(next_token)

            if stop_token is not None:
                finished = finished | next_token.eq(stop_token)
            if stop_token is not None and finished.all():
                break
            if step == max_new_tokens - 1:
                break

            next_token_tensor = next_token.unsqueeze(-1)
            inputs_embeds = self.model.get_input_embeddings()(next_token_tensor)
            attention_mask_2d = torch.cat(
                [attention_mask_2d, torch.ones((B, 1), dtype=attention_mask_2d.dtype, device=attention_mask_2d.device)],
                dim=1,
            )
            cache_position = cache_position[-1:] + 1
            position_ids = cache_position.unsqueeze(0) + 1

            causal_mask = self.model._update_causal_mask(
                attention_mask_2d,
                token_type_ids=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                is_training=False,
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    num_logits_to_keep=1,
                )
            past_key_values = outputs.past_key_values

        if not generated_tokens:
            cot_text = "" if B == 1 else [""] * B
        else:
            generated_tokens = torch.stack(generated_tokens, dim=1)
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            cot_text = decoded[0] if len(decoded) == 1 else decoded

        return cot_text, action_np

    def predict_cot_then_action(
        self,
        image,
        instruction: str,
        current_state,
        current_state_mask,
        use_ddim=True,
        num_ddim_steps=10,
        cfg_scale=5.0,
        action_mask_torch=None,
        fov=None,
        sample_times=1,
        use_cache=False,
        max_new_tokens: int = 96,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[Union[str, List[str]], np.ndarray]:
        """
        Online CoT-conditioned action prediction (2-pass, correctness-first).

        Pass 1: the VLM autoregressively generates a chain-of-thought (CoT) from the
                prefix '<image>' + build_cot_instruction_prompt(instruction, randomize=False),
                using the same proven low-level generation loop as predict_cot_action /
                predict_cot_low_level (DynamicCache, _update_causal_mask, argmax/sampling,
                EOS stop). The generated tokens are decoded into `cot_text`.
        Pass 2: the DiT is conditioned on the hidden states of [image + prefix + generated
                CoT] -- NOT on the prefix alone. This re-runs the EXACT predict_action body
                with instruction = build_cot_instruction_prompt(instruction) + cot_text, so
                that prepare_vlm_features encodes the full [image + prefix + cot] sequence
                exactly like training (which conditions the DiT on output_hs over
                [image + prefix + cot_suffix]).

        instruction: RAW coarse instruction string (NOT a CoT-injected one).
        current_state: normalized current hand action, [B, D]
        current_state_mask: [B, D]
        action_mask_torch: [B, T, D]
        fov: [B, 2]
        return: (cot_text, action_np) where action_np is the predicted normalized action,
                [sample_times, T, D].
        """

        B = current_state.shape[0]
        assert B == 1, f"Batch size {B} not supported in predict_cot_then_action for now."

        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            elif image.ndim == 4:
                image = [Image.fromarray(im) for im in image]
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

        # =====================================================================
        # Pass 1: generate the CoT with the proven low-level auto-regressive loop
        #         (identical mechanism to predict_cot_action / predict_cot_low_level).
        #         Prefix / current_state / current_state_mask / fov are used exactly
        #         as predict_cot_action does.
        # =====================================================================
        cot_prefix = '<image>' + build_cot_instruction_prompt(instruction, randomize=False)
        cot_inputs = self.processor(text=cot_prefix, images=image, return_tensors="pt").to('cuda')
        cot_pixel_value = cot_inputs['pixel_values']
        cot_input_ids = cot_inputs['input_ids']

        if isinstance(cot_pixel_value, torch.Tensor):
            cot_pixel_value = cot_pixel_value.to('cuda')
        elif isinstance(cot_pixel_value, dict):
            cot_pixel_value = {
                k: torch.stack([cot_pixel_value[idx][k] for idx in range(len(cot_input_ids))]).to('cuda') for k in cot_pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(cot_pixel_value)}")
        if cot_pixel_value.dim() == 5:
            cot_pixel_value = cot_pixel_value.view(-1, *cot_pixel_value.shape[2:])

        cot_attention_mask = torch.ones_like(cot_input_ids, dtype=torch.bool).to(cot_input_ids.device)
        cot_current_state = current_state.to(cot_input_ids.device)
        cot_current_state_mask = current_state_mask.to(cot_input_ids.device)
        cot_fov = fov.to(cot_input_ids.device) if fov is not None else None

        vlm_inputs, _ = self.prepare_vlm_input_embeddings(
            cot_pixel_value,
            cot_input_ids,
            cot_attention_mask,
            current_state_mask=cot_current_state_mask,
            current_state=cot_current_state,
            fov=cot_fov,
        )

        inputs_embeds = vlm_inputs['inputs_embeds']
        position_ids = vlm_inputs['position_ids']
        cache_position = vlm_inputs['cache_position']
        attention_mask_2d = vlm_inputs["attention_mask_2d"]

        from transformers import DynamicCache
        past_key_values = DynamicCache()

        generated_tokens = []
        stop_token = self.tokenizer.eos_token_id
        finished = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
        Bgen = inputs_embeds.shape[0]

        for step in range(max_new_tokens):
            causal_mask = self.model._update_causal_mask(
                attention_mask_2d,
                token_type_ids=None,
                past_key_values=past_key_values,
                cache_position=cache_position,
                input_ids=None,
                inputs_embeds=inputs_embeds,
                is_training=False,
            )

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                outputs = self.model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    num_logits_to_keep=1,
                )

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                if top_k is not None and top_k > 0:
                    top_k_cur = min(top_k, next_token_logits.shape[-1])
                    topk_values = torch.topk(next_token_logits, top_k_cur, dim=-1).values
                    topk_threshold = topk_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < topk_threshold, float("-inf"))

                probs = torch.softmax(next_token_logits, dim=-1)
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                if top_p is not None and top_p < 1.0:
                    mask = probs_sum - probs_sort > top_p
                    probs_sort[mask] = 0.0

                probs_denom = probs_sort.sum(dim=-1, keepdim=True)
                probs_sort = torch.where(
                    probs_denom > 0,
                    probs_sort / probs_denom,
                    torch.full_like(probs_sort, 1.0 / probs_sort.shape[-1]),
                )
                next_token_idx = torch.multinomial(probs_sort, num_samples=1)
                next_token = torch.gather(probs_idx, -1, next_token_idx).squeeze(-1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)

            if stop_token is not None:
                next_token = torch.where(finished, torch.full_like(next_token, stop_token), next_token)

            generated_tokens.append(next_token)

            if stop_token is not None:
                finished = finished | next_token.eq(stop_token)
            if stop_token is not None and finished.all():
                break
            if step == max_new_tokens - 1:
                break

            next_token_tensor = next_token.unsqueeze(-1)
            inputs_embeds = self.model.get_input_embeddings()(next_token_tensor)
            attention_mask_2d = torch.cat(
                [attention_mask_2d, torch.ones((Bgen, 1), dtype=attention_mask_2d.dtype, device=attention_mask_2d.device)],
                dim=1,
            )
            cache_position = cache_position[-1:] + 1
            position_ids = cache_position.unsqueeze(0) + 1

        if not generated_tokens:
            cot_text = "" if Bgen == 1 else [""] * Bgen
        else:
            generated_tokens = torch.stack(generated_tokens, dim=1)
            # skip_special_tokens=True drops the trailing <eos> but keeps the literal
            # <cot>/</cot> tags (regular text tokens), matching predict_cot_action.
            decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            cot_text = decoded[0] if len(decoded) == 1 else decoded

        # cot_text is a str for B == 1 (asserted above).
        cot_text_str = cot_text if isinstance(cot_text, str) else (cot_text[0] if len(cot_text) > 0 else "")

        # =====================================================================
        # Pass 2: condition the DiT on [image + prefix + generated CoT].
        #         EXACT predict_action body, except the generated CoT is supplied via
        #         the processor `suffix=` argument so the tokenization reproduces
        #         training (dataset.py:210-217) token-for-token: text is
        #         '<image>' + build_cot_instruction_prompt(instruction) and suffix is
        #         cot_text (kept WITH any <cot>/</cot> tags). The processor appends the
        #         prompt's trailing newline and the suffix's <eos>, yielding input_ids
        #         = [img][bos]<image>{prompt}{newline}{cot}<eos>, identical in structure
        #         to dataset.py's processor(text=[lang], suffix=[cot], images=imgs) call.
        #         The returned `labels`/`token_type_ids` are ignored (eval, no LM loss).
        # =====================================================================
        cot_prompt_text = '<image>' + build_cot_instruction_prompt(instruction, randomize=False)
        model_inputs = self.processor(
            text=[cot_prompt_text],
            suffix=[cot_text_str],
            images=(image if isinstance(image, list) else [image]),
            return_tensors="pt",
        ).to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        # Preprocess Image
        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        if action_mask_torch is None:
            x_mask = torch.zeros(B, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
            # x_mask[:, :, :102] = 1.0 # predict dual hand actions
            x_mask[:, :, 51:102] = 1.0 # predict right hand actions
            # x_mask[:, :, 0:51] = 1.0 # predict left hand actions
        else:
            x_mask = action_mask_torch.to(input_ids.device)

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
        current_state = current_state.to(input_ids.device)
        current_state_mask = current_state_mask.to(input_ids.device)
        fov = fov.to(input_ids.device) if fov is not None else None

        # plan-span conditioning deployable analog: when enabled, restrict the DiT to
        # the GENERATED-CoT span. The processor `suffix=[cot_text_str]` call above produced
        # labels that mark exactly the generated-CoT(+eos) tokens (labels != -100), while
        # image/prefix (and any cognition token inserted by prepare_vlm_input_embeddings)
        # stay -100. Passing labels also reproduces the training-time VLM layout (prefix-LM
        # mask; cognition token -- if any -- inserted before the CoT), keeping
        # generate-then-condition consistent with the cot_condition_span train/eval path.
        condition_span_mask = None
        if self.cot_condition_span:
            cot_labels = model_inputs['labels'] if 'labels' in model_inputs else None
            if cot_labels is None:
                raise ValueError(
                    "cot_condition_span=True but the processor returned no 'labels' for the "
                    "generated CoT suffix; cannot build the CoT-span condition mask."
                )
            cot_labels = cot_labels.to(input_ids.device)
            output_hs, inputs_masks, _, aligned_labels = self.prepare_vlm_features(
                pixel_value,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                use_cache=use_cache,
                labels=cot_labels,
                return_aligned_labels=True,
            )
            if aligned_labels is not None:
                condition_span_mask = (aligned_labels != -100)
        else:
            output_hs, inputs_masks, _ = self.prepare_vlm_features(
                pixel_value,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                use_cache=use_cache,
            )
        # handle multiple samples for one input
        samples, _ = self._forward_act_model(
            vlm_features = output_hs,
            attention_mask = inputs_masks,
            action_masks = x_mask,
            current_state = current_state,
            current_state_mask = current_state_mask,
            mode = "eval",
            repeated_diffusion_steps = sample_times,
            cfg_scale = cfg_scale,
            use_ddim = use_ddim,
            num_ddim_steps = num_ddim_steps,
            condition_span_mask = condition_span_mask,
        )
        action_np = samples.cpu().numpy() * x_mask.cpu().numpy()    # sample_times x T x D
        return cot_text, action_np

    @torch.no_grad()
    def predict_cot_then_action_batched(
        self, images, instructions, current_state, current_state_mask,
        action_mask_torch, fov=None, sample_times=1, max_new_tokens=96,
        cfg_scale=5.0, use_ddim=True, num_ddim_steps=10,
    ):
        """Batched (bs>1) prefix-LM single-pass CoT->DiT, one forward path per window.

        Unlike predict_cot_then_action (2-pass, B==1: generate CoT under a CAUSAL prefill,
        then RE-ENCODE under the prefix-LM mask for DiT conditioning), this does a single
        PREFIX-LM pass: bidirectional prefill over [image+depth+prompt+fov], lockstep
        generation that captures each CoT token's hidden state, then conditions the DiT on
        the captured CoT-span states (no re-encode). Validated equivalent to the 2-pass
        (teacher-forced span-hs abs diff ~1e-4; CoT text identical; MPJPE within best-of-8
        sampling noise). Variable prefix lengths -> LEFT-pad so all end at the same column;
        variable CoT lengths -> keep generating past eos (force-eos for finished samples),
        flag each sample's eos step, and cut each sample's CoT-span at its flag.

        images: list[np.ndarray] (len B); instructions: list[str] (len B);
        current_state/current_state_mask: [B,D]; action_mask_torch: [B,T,D]; fov: [B,2].
        Returns (cot_texts: list[str], action_np: np.ndarray [B, sample_times, T, D]).
        """
        from transformers import DynamicCache
        dev = current_state.device
        B = len(images)

        # ---- per-sample prefix embeddings (depth/fov/image; no prefill) ----
        embeds_list = []
        for i in range(B):
            img = images[i]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            prefix = '<image>' + build_cot_instruction_prompt(instructions[i], randomize=False)
            inp = self.processor(text=prefix, images=img, return_tensors='pt').to(dev)
            pv = inp['pixel_values'].to(dev)
            if pv.dim() == 5:
                pv = pv.view(-1, *pv.shape[2:])
            ids = inp['input_ids']
            attn = torch.ones_like(ids, dtype=torch.bool)
            vlm_inputs, _ = self.prepare_vlm_input_embeddings(
                pv, ids, attn, current_state_mask=current_state_mask[i:i + 1],
                current_state=current_state[i:i + 1],
                fov=(fov[i:i + 1] if fov is not None else None), labels=None)
            embeds_list.append(vlm_inputs['inputs_embeds'])

        Lps = [e.shape[1] for e in embeds_list]
        Lmax = max(Lps); H = embeds_list[0].shape[-1]; dtype = embeds_list[0].dtype
        minv = torch.finfo(dtype).min

        # ---- LEFT-pad stack: all prefixes end at column Lmax-1 ----
        embeds = torch.zeros((B, Lmax, H), dtype=dtype, device=dev)
        valid = torch.zeros((B, Lmax), dtype=torch.bool, device=dev)
        for i, e in enumerate(embeds_list):
            L = e.shape[1]; embeds[i, Lmax - L:] = e[0]; valid[i, Lmax - L:] = True
        pos = valid.long().cumsum(-1)                       # real tokens 1..L, pad 0
        cache_pos = torch.arange(Lmax, device=dev)

        # ---- bidirectional prefill (each query attends to its own valid keys) ----
        key_ok = valid[:, None, None, :].expand(B, 1, Lmax, Lmax)
        prefill_mask = torch.zeros((B, 1, Lmax, Lmax), dtype=dtype, device=dev).masked_fill(~key_ok, minv)
        pkv = DynamicCache()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
            out = self.model.language_model(
                inputs_embeds=embeds, attention_mask=prefill_mask, position_ids=pos,
                past_key_values=pkv, cache_position=cache_pos, use_cache=True,
                output_hidden_states=True, num_logits_to_keep=1)
        pkv = out.past_key_values
        prefill_hs = out.hidden_states[-1]                  # [B,Lmax,H]
        next_logits = out.logits[:, -1, :]                  # [B,V] (left-pad: last col = last real)

        # ---- lockstep generation; keep going past eos, flag each sample's eos step ----
        eos = self.tokenizer.eos_token_id
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        eos_step = torch.full((B,), -1, dtype=torch.long, device=dev)
        cot_hs, gen_tok = [], []
        cur_pos = pos[:, -1:] + 1                            # [B,1] per-sample next position
        total = Lmax
        for step in range(max_new_tokens):
            tok = torch.argmax(next_logits, dim=-1)          # [B]
            tok = torch.where(finished, torch.full_like(tok, eos), tok)
            gen_tok.append(tok)
            emb = self.model.get_input_embeddings()(tok.unsqueeze(-1))
            key_val = torch.cat([valid, torch.ones((B, total - Lmax + 1), dtype=torch.bool, device=dev)], dim=1)
            gen_mask = torch.zeros((B, 1, 1, total + 1), dtype=dtype, device=dev).masked_fill(
                ~key_val[:, None, None, :], minv)
            cur_cache = torch.tensor([total], device=dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.use_bf16):
                out = self.model.language_model(
                    inputs_embeds=emb, attention_mask=gen_mask, position_ids=cur_pos,
                    past_key_values=pkv, cache_position=cur_cache, use_cache=True,
                    output_hidden_states=True, num_logits_to_keep=1)
            pkv = out.past_key_values
            cot_hs.append(out.hidden_states[-1][:, -1, :])   # [B,H]
            next_logits = out.logits[:, -1, :]
            newly = (~finished) & tok.eq(eos)
            eos_step[newly] = step
            finished = finished | tok.eq(eos)
            total += 1; cur_pos = cur_pos + 1
            if bool(finished.all()):
                break
        G = len(cot_hs)
        eos_step[eos_step < 0] = G - 1                       # never-eos -> full length
        cot_hs = torch.stack(cot_hs, dim=1)                 # [B,G,H]
        gen_tok = torch.stack(gen_tok, dim=1)               # [B,G]

        # ---- assemble output_hs + per-sample CoT-span mask ----
        output_hs = torch.cat([prefill_hs, cot_hs], dim=1)  # [B, Lmax+G, H]
        Ltot = Lmax + G
        attn_valid = torch.cat([valid, torch.ones((B, G), dtype=torch.bool, device=dev)], dim=1)
        span = torch.zeros((B, Ltot), dtype=torch.bool, device=dev)
        for i in range(B):
            span[i, Lmax: Lmax + int(eos_step[i].item()) + 1] = True
        cot_texts = [self.tokenizer.decode(gen_tok[i, :int(eos_step[i].item()) + 1], skip_special_tokens=True)
                     for i in range(B)]

        # ---- DiT (batched); unpack REP-MAJOR [K*B] -> [K,B] -> per-window [B,K] ----
        x_mask = action_mask_torch.to(dev)
        samples, _ = self._forward_act_model(
            vlm_features=output_hs, attention_mask=attn_valid, action_masks=x_mask,
            current_state=current_state, current_state_mask=current_state_mask, mode='eval',
            repeated_diffusion_steps=sample_times, cfg_scale=cfg_scale, use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps, condition_span_mask=span)
        samples = samples.cpu().numpy()                     # [K*B, T, D] rep-major (rep*B+b)
        K = sample_times; Tn, Dn = samples.shape[-2], samples.shape[-1]
        samples_r = samples.reshape(K, B, Tn, Dn)           # [K, B, T, D]
        xm = x_mask.cpu().numpy()                           # [B, T, D]
        action_np = np.stack([samples_r[:, b] * xm[b] for b in range(B)], axis=0)  # [B, K, T, D]
        return cot_texts, action_np

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """

        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss
