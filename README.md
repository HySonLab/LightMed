<div align="center">
  <table>
    <tr>
      <td><img src="logo.jpg" width="150"></td>
      <td><h1>LightMed: <br>A PyTorch Implementation</h1></td>
    </tr>
  </table>
</div>

<p align="center">
<a href="https://arxiv.org/abs/" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2409.10594-b31b1b.svg?style=flat" /></a>
      <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.x %20%7C%202.x-673ab7.svg" alt="Tested PyTorch Versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
</p>

<p align="center">
<img src="LightMed.jpg" width="600"> <br>
Yes, I am LightMed!
</p>

🎉 This is a PyTorch/GPU implementation of the paper **LightMed (LM)**, which learning on frequecy domain.

**LightMed**

 📝[[Paper](https://www.biorxiv.org/content/10.1101/2024.09.28.615584v1.full.pdf)] </>[[code](https://github.com/HySonLab/LightMed)]

# LightMed: A Lightweight and Robust FFT-Based Model for Adversarially Resilient Medical Image Segmentation
An efficient FFT-based model for medical image segmentation. The algorithm is elaborated on our paper [LightMed: A Lightweight and Robust FFT-Based Model for Adversarially Resilient Medical Image Segmentation](https://www.biorxiv.org/content/10.1101/2024.09.28.615584v1.full.pdf)

## Requirement

``pip install -r requirement.txt``


## Example Cases
### Melanoma Segmentation from Skin Images (2018)
1. Download ISIC_2018 dataset we processing from https://www.kaggle.com/datasets/haminhhieu/isic-2018-data. You must download and your dataset folder under "data" should be like:

~~~
ISIC_2018
---- image_train.npy  
---- mask_train.npy 
----dataset
|   ----test_0
|   |   |images_test.npy
|   |   |masks_test.npy
|   ----test_1
|   |   |images_test.npy
|   |   |masks_test.npy
|   ----test_2
|   |   |images_test.npy
|   |   |masks_test.npy
|   ----test_3
|   |   |images_test.npy
|   |   |masks_test.npy       
|   ----test_4
|   |   |images_test.npy
|   |   |masks_test.npy 
 
~~~
    
2. For training, example run: ``python train.py --num_epochs 300 --batch_size 16 -- image_size 256 -- work_dir *folder save weight*``

3. For evaluation noise, example run: `` python test.py --model_paths *folder you save checkpoint* --test_dataset_paths *folder test_ dataset* --image_size 256``

4. For evaluation attack, example run: ``python test_fgsm_attack.py --model_path *folder you save checkpoint* --test_dataset_path *folder test dataset* --attack``

### Other datasets we used
- [ISIC 2017](https://www.kaggle.com/datasets/phmvittin/isic-2017-rerun)
- [Lung-Covid19](https://www.kaggle.com/datasets/haminhhieu/lung-data/data)
- [pH2](https://www.kaggle.com/datasets/haminhhieu/skin-lesion-dataset)
- [ACDC](https://www.kaggle.com/datasets/haminhhieu/acdc-dataset-lightmed)
### Run on  your own dataset
We suggest you following this notebook to set up your own dataset
[Link repair Dataset](https://www.kaggle.com/code/haminhhieu/skin-lesion-segmentation-using-unet/notebook).
Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.

## Thanks
Code copied a lot from [soleilssss/ FFCNet](https://github.com/soleilssss/FFCNet), [soleilssss/ AFACNet](https://github.com/soleilssss/AFACNet), [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet), and [adam-dziedzic/bandlimited-cnns](https://github.com/adam-dziedzic/bandlimited-cnns)

## Please cite our work!

```bibtex
@article {Pham2024.09.28.615584,
	author = {Pham, Viet Tien and Ha, Minh Hieu and Bui, Bao V. Q. and Hy, Truong Son},
	title = {LightMed: A Light-weight and Robust FFT-Based Model for Adversarially Resilient Medical Image Segmentation},
	elocation-id = {2024.09.28.615584},
	year = {2024},
	doi = {10.1101/2024.09.28.615584},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Accurate and reliable medical image segmentation is essential for computer-aided diagnosis and formulating appropriate treatment plans. However, real-world challenges such as suboptimal image quality and computational resource constraints hinder the effective deployment of deep learning-based segmentation models. To address these issues, we propose LightMed, a novel efficient neural architecture based on Fast Fourier Transform (FFT). Different from prior works, our model directly learns on the frequency domain, harnessing its resilience to noise and uneven brightness, which common artifacts found in medical images. By focusing on low-frequency image components, we significantly reduce computational complexity while preserving essential image features. Our deep learning architecture extracts discriminative features directly from the Fourier domain, leading to improved segmentation accuracy and robustness compared to traditional spatial domain methods. Additionally, we propose a new benchmark incorporating various levels of Gaussian noise to assess susceptibility to noise attacks. The experimental results demonstrate that LightMed not only effectively eliminates noise and consistently achieves accurate image segmentation but also shows robust resistance to imperceptible adversarial attacks compared to other baseline models. Our new benchmark datasets and source code are publicly available at https://github.com/HySonLab/LightMedCompeting Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/09/30/2024.09.28.615584},
	eprint = {https://www.biorxiv.org/content/early/2024/09/30/2024.09.28.615584.full.pdf},
	journal = {bioRxiv}
}
```

