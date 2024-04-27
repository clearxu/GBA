# Generalization Boosted Adapter for Open-Vocabulary Segmentation

This repo contains the code for our paper [**Generalization Boosted Adapter for Open-Vocabulary Segmentation**]

<div align="center">
  <img src="imgs/teaser.jpg" width="100%" height="100%"/>
</div><br/>

*GBA* is an universal model for open-vocabulary image segmentation problems, consisting of a class-agnostic segmenter, in-vocabulary classifier, out-of-vocabulary classifier. With everything built upon a shared single frozen convolutional CLIP model,*GBA* not only achieves state-of-the-art performance on various open-vocabulary segmentation benchmarks, but also enjoys a much lower training (10 days with 4 A6000) and testing costs compared to prior arts.


## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for GBA](datasets/README.md).

See [Getting Started with  GBA](GETTING_STARTED.md).



## Acknowledgement

[Mask2Former](https://github.com/facebookresearch/Mask2Former)

[ODISE](https://github.com/NVlabs/ODISE)

[FCCLIP](https://github.com/bytedance/fc-clip)

[FreeSeg](https://github.com/bytedance/FreeSeg)
