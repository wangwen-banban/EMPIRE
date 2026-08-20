# Third-party notices

This distribution contains code under more than one license. The root `LICENSE` applies to the VITRA-derived portion only and is not a statement that every bundled file is MIT-licensed. License material redistributed with the release is under [`LICENSES/`](LICENSES/).

## VITRA-derived code

- Origin: VITRA research-code lineage.
- License: MIT, preserved verbatim in `LICENSE`.
- Copyright notice: Microsoft Corporation.

## Depth Anything 3 and DINOv2

- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- DINOv2: https://github.com/facebookresearch/dinov2
- Bundled location: `empire/models/depth_anything_3/` and the top-level import shim.
- License: Apache License 2.0. Source-file copyright and attribution headers are retained.
- License copy: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
- No model weights, service code, user interface, gallery, or optional exporter modules are bundled.

## DiT and OpenAI diffusion utilities

- DiT: https://github.com/facebookresearch/DiT
- OpenAI guided diffusion: https://github.com/openai/guided-diffusion
- OpenAI improved diffusion: https://github.com/openai/improved-diffusion
- Bundled location: `empire/models/action_model/`.
- `dit.py` is adapted from DiT and remains subject to Creative Commons Attribution-NonCommercial 4.0. This restriction can prohibit commercial use. Attribution and the complete legal text are in [LICENSES/DiT-CC-BY-NC-4.0.txt](LICENSES/DiT-CC-BY-NC-4.0.txt).
- OpenAI-derived diffusion utilities are MIT-licensed; the redistributed notice is in [LICENSES/OpenAI-MIT.txt](LICENSES/OpenAI-MIT.txt).
- Original source headers and upstream links are retained in the affected files.

## MANO and SMPL-X

- MANO project: https://mano.is.tue.mpg.de/
- SMPL-X implementation: https://github.com/vchoutas/smplx
- The release contains only a small runtime adapter to the separately installed `smplx` package. It contains no HaMeR source and no MANO weights, regressors, or model assets.
- MANO model files have separate access and use terms, including non-commercial restrictions. Users must obtain them from the official publisher and provide their location through `MANO_MODEL_PATH` or an explicit configuration value.
- The optional `manopth` backend is not bundled and requires explicit code and model-root environment variables.

## Other dependencies and model downloads

Python dependencies named in `pyproject.toml` are installed separately and retain their own licenses. In particular, the `data`/`public` extras pin EasternJournalist `utils3d` for camera augmentation and declare `trimesh` for object-mesh runtime utilities; neither project is copied into this repository. PaliGemma 2 and Depth Anything 3 weights are fetched by public Hugging Face model ID and retain their publisher terms. No third-party model weights are distributed in this repository.
