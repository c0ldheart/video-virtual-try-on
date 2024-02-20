# video-virtual-try-on

diffusion model baesd video-virtual-try-on

![result visualization](assets/allres1-10.gif)

NOTE: This is a personal experimental project. The code has not been properly organized (updated according to personal circumstances), there is no ready-to-use demo. Thanks to [ladi-vton](https://github.com/miccunifi/ladi-vton), [DisCo](https://github.com/Wangt-CN/DisCo) for the inspiration, and any related discussions are welcome.

## Instructions
- Improved 3D denoising UNet, improved face reconstruction performance and temporal consistency, using [VVT](https://competitions.codalab.org/competitions/23472) dataset, fine-tuning based on [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- Preprocessing dataset based on [Densepose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose), [Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing). The processed files will be uploaded soon
- Based on a pre-trained self-developed clothing deformation model, images of the warped clothing will also be released. The deformation model can be replaced by the existing graph virtual try-on deformation model
## Release Plan
- [x] Test results visualization
- [x] Code for model
- [x] Trained checkpoint
- [] Warping model
- [] Code for training
- [] Code for evaluation script (dataset, full pipeline)

## Model checkpoint
- [UNet](https://drive.google.com/file/d/1-q8B2pe9sWEst4859paJjJyXuPmyh03y/view?usp=drive_link)
- [Vision Encoder Projector](https://drive.google.com/file/d/1lC7XyK9DJw7gt6-66BK1cgv_dX5Pb0-z/view?usp=drive_link)

