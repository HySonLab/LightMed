<div align="center">
  <table>
    <tr>
      <td><img src="logo.jpg" width="150"></td>
      <td><h1>FFTMed: <br>A PyTorch Implementation</h1></td>
    </tr>
  </table>
</div>
<p align="center">
<a href="">
    <img src="https://img.shields.io/badge/bioRxiv-2024.09.28.615584-b31b1b.svg?style=flat" />
      <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.x %20%7C%202.x-673ab7.svg" alt="Tested PyTorch Versions"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
</p>

<p align="center">
<img src="FFTMed.jpg" width="600"> <br>
Yes, I am FFTMed!
</p>

🎉 This is a PyTorch/GPU implementation of the paper **FFTMed**, which learning on frequecy domain.

The paper is published at Scientific Reports - Nature: https://www.nature.com/articles/s41598-025-21799-5

**FFTMed**

 📝[[Paper]] </>[[code](https://github.com/HySonLab/LightMed)]

# FFTMed: Leveraging Fast-Fourier Transform for a Lightweight and Adversarial-Resilient Medical Image Segmentation Framework
An efficient FFT-based model for medical image segmentation. The algorithm is elaborated on our paper [FFTMed: Leveraging Fast-Fourier Transform for a Lightweight and Adversarial-Resilient Medical Image
Segmentation Framework]

## Requirement

``pip install -r requirement.txt``


## Example Cases
### Melanoma Segmentation from Skin Images (2018)
1. Download ISIC_2018 dataset we processing from (https://zenodo.org/records/15310397). You must download and your dataset folder under "data" should be like:

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
- [Data](https://zenodo.org/records/15310397)

### Run on  your own dataset
We suggest you following this notebook to set up your own dataset
[Link repair Dataset](https://www.kaggle.com/code/haminhhieu/skin-lesion-segmentation-using-unet/notebook).
Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.

## Thanks
Code copied a lot from [soleilssss/ FFCNet](https://github.com/soleilssss/FFCNet), [soleilssss/ AFACNet](https://github.com/soleilssss/AFACNet), [JCruan519/EGE-UNet](https://github.com/JCruan519/EGE-UNet), and [adam-dziedzic/bandlimited-cnns](https://github.com/adam-dziedzic/bandlimited-cnns)


