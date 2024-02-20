# video-virtual-try-on

diffusion model baesd video-virtual-try-on

![result visualization](assets/allres1-10.gif)

## Instructions
- Improved 3D denoising UNet, improved face reconstruction performance and temporal consistency, using [VVT](https://competitions.codalab.org/competitions/23472) dataset, fine-tuning based on [stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting)
- Preprocessing dataset based on [Densepose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose), [Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing). The processed files will be uploaded soon
- Based on a pre-trained self-developed clothing deformation model, images of the warped clothing will also be released. The deformation model can be replaced by the existing graph virtual try-on deformation model
## Release Plan
- [x] Test results visualization
- [x] Code for model
- [] Trained checkpoint
- [] Warping model
- [] Code for training
- [] Code for evaluation script (dataset, full pipeline)

<!-- ## Model checkpoint
- [unet (alipan)](https://drive.google.com/file/d/1-)
- [vision encoder projector (alipan)](https://drive.google.com/file/d/1-) -->

