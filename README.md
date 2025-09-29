# Learning to Ball: Composing Policies for Long-Horizon Basketball Moves

This is the official implementation for _*Learning to Ball: Composing Policies for Long-Horizon Basketball Moves*_. [[Webpage](https://pei-xu.github.io/basketball)] [[arXiv](http://arxiv.org/abs/2509.22442)] [[Youtube](https://www.youtube.com/2RBFIjjmR2I)] [[SIGGRAPH Asia'25](#)] [[TOG](#)]

This implementation is based on 

- _*Composite Motion Learning with Task Control*_ 
[[arXiv](https://arxiv.org/abs/2305.03286)]
[[Youtube](https://youtu.be/mcRAxwoTh3E)]
[[webpage](https://pei-xu.github.io/CompositeMotion)]

- _*A GAN-Like Approach for Physics-Based Imitation Learning and Interactive Character Control*_
[[arXiv](https://arxiv.org/abs/2105.10066)]
[[Youtube](https://www.youtube.com/watch?v=VHMyvDD3B_o)]
[[webpage](https://pei-xu.github.io/ICCGAN)]

![](doc/teaser.png)

## Citation
If you use the code or provided motions for your work, please consider citing our papers:

    @article{basketball,
        author = {Xu, Pei and Wu, Zhen and Wang, Ruocheng and Sarukkai, Vishnu and Fatahalian, Kayvon and Karamouzas, Ioannis and Zordan, Victor and Liu, C. Karen},
        title = {Learning to Ball: Composing Policies for Long-Horizon Basketball Moves},
        journal = {ACM Transactions on Graphics},
        publisher = {ACM New York, NY, USA},
        year = {2024},
        volume = {44},
        number = {6},
        doi = {10.1145/3763367}
    }

## Code Usage

### Dependencies
- Pytorch 2.1.2
- IsaacGym Pr4

We recommend to install all the requirements through Conda by

    $ conda create --name <env> --file requirements.txt -c pytorch -c conda-forge -c nvidia

IsaacGym Pr4 is available from the [official site](https://developer.nvidia.com/isaac-gym) and can be installed through `pip`.


### Policy Training

    $ python main.py <configure_file> --ckpt <checkpoint_dir>

We provide our configure files in `cfg` folder for reference. To reproduce the examples shown in the paper, e.g. `shoot`, please run the training by

    $ python main.py cfg/shoot.py --ckpt ckpt_shoot

The training results (model and log) will be generated in the `ckpt_shoot` folder.

The training can be done on a single GPU. Use `--device` option to specify the device used for training (default: 0).

### Evaluation

    $ python main.py <configure_file> --ckpt <checkpoint_dir> --test

We provide pretrained policy models in `pretrained` folder. To evaluate a pretrained policy, e.g. `shoot`, please run

    $ python main.py cfg/shoot.py --ckpt pretrained/shoot --test

Please visit our [webpage](https://pei-xu.github.io/basketball) for animated results.

