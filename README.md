# StylizedGS: Controllable Stylization for 3D Gaussian Splatting

<div align="center">
<a href="https://kristen-z.github.io/stylizedgs/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2404.05220" target="_blank" rel="noopener noreferrer"> <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF"></a>
<a href="https://drive.google.com/file/d/1PuDanf0JnpyCRtiurPIESG80pU_nEBIj/view"> <img src="https://img.shields.io/badge/Demo-blue" alt="Demo"></a>

<p>
    <a href="https://kristen-z.github.io/">Dingxi Zhang</a>   
    ·
    <a href="http://people.geometrylearning.com/yyj/">Yu-Jie Yuan</a>
    · 
    <a href="https://seancomeon.github.io/">Zhuoxun Chen</a>
    ·
    <a href="https://people.wgtn.ac.nz/fanglue.zhang">Fang-Lue Zhang</a>
    ·
    <a href="https://lynnho.github.io/">Zhenliang He</a>
    ·
    <a href="https://people.ucas.edu.cn/~sgshan">Shiguang Shan</a>
    ·
    <a href="http://www.geometrylearning.com/lin/">Lin Gao<sup>*</sup></a>
</p>

<p><b>Jittor Version</b></p>

<p>Institute of Computing Technology, Chinese Academy of Sciences</p>

<img src="./assets/teaser.jpg" alt="[Teaser Figure]" style="zoom:80%;" />
</div>
Given a 2D style image, the proposed StylizedGS method can stylize the pre-trained 3D Gaussian Splatting to match the desired style with detailed geometric features and satisfactory visual quality within a few minutes. We also enable users to control several perceptual factors, such as color, the style pattern size (scale), and the stylized regions (spatial), during the stylization to enhance the customization capabilities.


## Jittor Setup
The pytorch version of StylizedGS is also available [here](https://github.com/Kristen-Z/StylizedGS).
### Installation
First, clone this repository to your local machine, and install the dependencies (jittor and other basic python package). 

```bash
git clone https://github.com/Kristen-Z/StylizedGS-Jittor.git --recursive
conda create -n stylizedgs python=3.10
conda activate stylizedgs
python3.10 -m pip install jittor
pip install -r requirements.txt
```

Then, Compile the submodules based on C++ and Cuda. 

```bash
cd gaussian_renderer_new/diff_gaussian_rasterization
cmake .
make -j
cd ../..
cd scene/simple_knn
cmake .
make -j
```
### Data Preparation
We evaluate the dataset on [LLFF](https://bmild.github.io/llff/), [Tanks and Temples](https://www.tanksandtemples.org/) and [MipNeRF-360](https://jonbarron.info/mipnerf360/) datasets. For convenience, a small subset of preprocessed scene data and reference style images is provided [here](https://drive.google.com/file/d/1U7MTzKAFNY0XbJ4tnr8BwsFHKbedOOyw/view?usp=sharing).

To use custom data, please follow the instructions in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/README.md#processing-your-own-scenes) to process your own scenes.

The `datasets` folder is organized as follows:
```
datasets
|---llff
|   |---flower
|   |---horns
|   |---...
|---tandt
|---mipnerf360
|---styles
|   |---0.jpg
|   |---1.jpg
|   |---...
```

## Quick Start
### Train Origin 3DGS
```python
python train.py -s datasets/llff/flower -m output/style-flower-jittor
```

### For general stylization
```python
python train_style_depth.py -s datasets/llff/flower \
                -m output/style-flower-jittor \
                --point_cloud output/style-flower-jittor/point_cloud/iteration_30000/point_cloud.ply \
                --style datasets/styles/14.jpg \ # the Van Gogh's Starry Night style
                --histgram_match 
```

### For color control

```python
python train_style_depth.py -s datasets/llff/flower \
                -m output/style-flower-jittor \
                --point_cloud output/style-flower-jittor/point_cloud/iteration_30000/point_cloud.ply \
                --style datasets/styles/14.jpg \ # the Van Gogh's Starry Night style
                --histgram_match \
                --preserve_color
                # --second_style datasets/style/1.jpg # input the color style image
```

### For spatial control
#### 1. process images to get masks and save them as ".npy" format.
```
# pytorch version

pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

python gen_lang_masks.py --image_dir datasets/llff/flower/images/ --output_dir datasets/llff/flower/masks --text_prompt 'A bunch of flower'
```

#### 2. transfer STYLE2 to the masked region while other regions be STYLE1
```python
python train_style_spatial_load.py -s datasets/llff/flower \
                -m output/style-flower-jittor \
                --point_cloud output/style-flower-jittor/point_cloud/iteration_30000/point_cloud.ply \
                --style datasets/styles/14.jpg \ # the Van Gogh's Starry Night style
                --mask_dir datasets/llff/flower/masks \
                --second_style datasets/style/1.jpg
```

### For scale control
```python
python train_style_depth.py -s datasets/llff/flower \
                -m output/style-flower-jittor \
                --point_cloud output/style-flower-jittor/point_cloud/iteration_30000/point_cloud.ply \
                --style datasets/styles/14.jpg \ # the Van Gogh's Starry Night style
                --scale_level 2 # option: 0-2, the scale of style pattern 
```

### Render
```python
python render.py -s datasets/llff/flower -m output/style-flower-jittor
```
## Problem
If you can't download vgg from Jittor, you can downdload vgg ckpt from Pytorch and run "converter.py" to get Jittor version checkpoint

## Acknowledgements
Our work is based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [ARF](https://github.com/Kai-46/ARF-svox2). We thank the authors for their great work and open-sourcing the code.

Many thanks to Botao Zhang (@[zbtzbt44](https://github.com/zbtzbt44)) for providing this Jittor version of the implementation.

## Citation
```
@article{zhang2024stylizedgs,
  title={Stylizedgs: Controllable stylization for 3d gaussian splatting},
  author={Zhang, Dingxi and Yuan, Yu-Jie and Chen, Zhuoxun and Zhang, Fang-Lue and He, Zhenliang and Shan, Shiguang and Gao, Lin},
  journal={arXiv preprint arXiv:2404.05220},
  year={2024}
}
```