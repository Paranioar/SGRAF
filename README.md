# SGRAF
*PyTorch implementation for AAAI2021 paper of [**“Similarity Reasoning and Filtration for Image-Text Matching”**](https://drive.google.com/file/d/1tAE_qkAxiw1CajjHix9EXoI7xu2t66iQ/view?usp=sharing).* 

*It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) and [Awesome_Matching](https://github.com/Paranioar/Awesome_Matching_Pretraining_Transfering).* 

*We have released two versions of SGRAF: **Branch `main` for python2.7**; **Branch `python3.6` for python3.6**.*  

*If any problems, please contact me at r1228240468@gmail.com. (r1228240468@mail.dlut.edu.cn is deprecated)*

## Introduction

**The framework of SGRAF:**

<img src="./fig/model.png" width = "100%" height="50%">

**The updated results (Better than the original paper)**
<table>
   <tr> <td rowspan="2">Dataset</td> <td rowspan="2", align="center">Module</td> 
        <td colspan="3", align="center">Sentence retrieval</td> <td colspan="3", align="center">Image retrieval</td> </tr>
   <tr> <td>R@1</td><td>R@5</td><td>R@10</td> <td>R@1</td><td>R@5</td><td>R@10</td> </tr>
   <tr> <td rowspan="3">Flick30k</td>
        <td>SAF</td> <td>75.6</td><td>92.7</td><td>96.9</td> <td>56.5</td><td>82.0</td><td>88.4</td> </tr>
   <tr> <td>SGR</td> <td>76.6</td><td>93.7</td><td>96.6</td> <td>56.1</td><td>80.9</td><td>87.0</td> </tr>
   <tr> <td>SGRAF</td> <td>78.4</td><td>94.6</td><td>97.5</td> <td>58.2</td><td>83.0</td><td>89.1</td> </tr>
   <tr> <td rowspan="3">MSCOCO1k</td>
        <td>SAF</td> <td>78.0</td><td>95.9</td><td>98.5</td> <td>62.2</td><td>89.5</td><td>95.4</td> </tr>
   <tr> <td>SGR</td> <td>77.3</td><td>96.0</td><td>98.6</td> <td>62.1</td><td>89.6</td><td>95.3</td> </tr>
   <tr> <td>SGRAF</td> <td>79.2</td><td>96.5</td><td>98.6</td> <td>63.5</td><td>90.2</td><td>95.8</td> </tr>
   <tr> <td rowspan="3">MSCOCO5k</td>
        <td>SAF</td> <td>55.5</td><td>83.8</td><td>91.8</td> <td>40.1</td><td>69.7</td><td>80.4</td> </tr>
   <tr> <td>SGR</td> <td>57.3</td><td>83.2</td><td>90.6</td> <td>40.5</td><td>69.6</td><td>80.3</td> </tr>
   <tr> <td>SGRAF</td> <td>58.8</td><td>84.8</td><td>92.1</td> <td>41.6</td><td>70.9</td><td>81.5</td> </tr>
  

   
</table> 

## Requirements 
We recommended the following dependencies for ***Branch `main`***.

*  Python 2.7  
*  [PyTorch (>=0.4.1)](http://pytorch.org/)    
*  [NumPy (>=1.12.1)](http://www.numpy.org/)   
*  [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)  
*  Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```

## Download data and vocab
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features and vocabularies, which can be downloaded by using:

```bash
https://www.kaggle.com/datasets/kuanghueilee/scan-features
```
Another download link is available below：

```bash
https://drive.google.com/drive/u/0/folders/1os1Kr7HeTbh8FajBNegW8rjJf6GIhFqC
```

## Pre-trained models and evaluation
The pretrained models are only for **Branch `python3.6`(python3.6)**, not for **Branch `main`(python2.7)**.  
Modify the **model_path**, **data_path**, **vocab_path** in the `evaluation.py` file. Then run `evaluation.py`:

```bash
python evaluation.py
```

Note that `fold5=True` is only for evaluation on mscoco1K (5 folders average) while `fold5=False` for mscoco5K and flickr30K. Pretrained models and Log files can be downloaded from [Flickr30K_SGRAF](https://drive.google.com/file/d/1OBRIn1-Et49TDu8rk0wgP0wKXlYRk4Uj/view?usp=sharing) and [MSCOCO_SGRAF](https://drive.google.com/file/d/1SpuORBkTte_LqOboTgbYRN5zXhn4M7ag/view?usp=sharing).

## Training new models from scratch
Modify the **data_path**, **vocab_path**, **model_name**, **logger_name** in the `opts.py` file. Then run `train.py`:

For MSCOCO:

```bash
(For SGR) python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SGR
(For SAF) python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SAF
```

For Flickr30K:

```bash
(For SGR) python train.py --data_name f30k_precomp --num_epochs 40 --lr_update 30 --module_name SGR
(For SAF) python train.py --data_name f30k_precomp --num_epochs 30 --lr_update 20 --module_name SAF
```

## Reference

If SGRAF is useful for your research, please cite the following paper:

      @inproceedings{Diao2021SGRAF,
         title={Similarity reasoning and filtration for image-text matching},
         author={Diao, Haiwen and Zhang, Ying and Ma, Lin and Lu, Huchuan},
         booktitle={Proceedings of the AAAI conference on artificial intelligence},
         volume={35},
         number={2},
         pages={1218--1226},
         year={2021}
      }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).  


