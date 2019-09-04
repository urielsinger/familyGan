# Progressive Face Super Resolution
Deokyun Kim, Minseon Kim, Gihyun Kwon, and Dae-shik Kim, [Progressive Face Super-Resolution via Attention to Facial Landmark](https://arxiv.org/abs/1908.08239), The British Machine Vision Conference 2019 (BMVC 2019)


### Prerequisites
* Python 3.6
* Pytorch 1.0.0
* CUDA 9.0 or higher

### Data Preparation

* [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

create a folder:

```bash
 mkdir dataset

```
and then, download dataset. Anno & Img.


## Test

```bash
$ python eval.py --data-path './dataset' --checkpoint-path 'CHECKPOINT_PATH/****.ckpt'
```
<br/>
