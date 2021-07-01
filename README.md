# FINN.no Recommender Systems Slate Dataset
> Repository containing the recommender systems slates dataset


We release the *FINN.no recommender systems slate dataset* to improve recommender systems research.
The dataset includes both search and recommendation interactions between users and the platform over a 30 day period.
The dataset has logged both exposures and clicks, *including interactions where the user did not click on any of the items in the slate*.
To our knowledge there exist no such large-scale dataset, and we hope this contribution can help researchers constructing improved models and improve offline evaluation metrics.

![A visualization of a presented slate to the user on the frontpage of FINN.no](finn-frontpage.png)

For each user u and interaction step t we recorded all items in the visible slate ![equ](https://latex.codecogs.com/gif.latex?a_t^u(s_t^u) ) (up to the scroll length ![equ](https://latex.codecogs.com/gif.latex?s_t^u)), and the user's click response ![equ](https://latex.codecogs.com/gif.latex?c_t^u).
The dataset consists of 37.4 million interactions, |U| ≈ 2.3) million  users and |I| ≈ 1.3 million items that belong to one of G = 290 item groups. For a detailed description of the data please see the [paper](https://arxiv.org/abs/2104.15046).

![A visualization of a presented slate to the user on the frontpage of FINN.no](interaction_illustration.png)

FINN.no is the leading marketplace in the Norwegian classifieds market and provides users with a platform to buy and sell general merchandise, cars, real estate, as well as house rentals and job offerings.
For questions, email simen.eide@finn.no or file an issue.

## Organization
The repository is organized as follows:
- The dataset is placed in (`data/`).
- The code open sourced from the article ["Dynamic Slate Recommendation with Gated Recurrent Units and Thompson Sampling"](https://arxiv.org/abs/2104.15046) is found in (`code/`). However, we are in the process of making the data more generally available which makes the code incompatible with the current (newer) version of the data. Please use [the v1.0 release of the repository](https://github.com/finn-no/recsys-slates-dataset/tree/v1.0) for a compatible version of the code and dataset.

## Download and prepare dataset
The data files can either be obtained by cloning this repository with git lfs, or (preferably) use the [datahelper.download_data_files()](https://github.com/finn-no/recsys-slates-dataset/blame/transform-to-numpy-arrays/datahelper.py#L3) function which downloads the same dataset from google drive.
For pytorch users, they can directly use the `dataset_torch.load_dataloaders()` to get ready-to-use dataloaders for training, validation and test datasets.

## Quickstart dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finn-no/recsys-slates-dataset/blob/master/quickstart-finn-recsys-slate-data.ipynb)
We provide a quickstart jupyter notebook that runs on Google Colab (quickstart-finn-recsys-slate-data.ipynb) which includes all necessary steps above.

NB: This quickstart notebook is currently incompatible with the main branch. 
We will update the notebook as soon as we have published a pip-package. In the meantime, please use [the v1.0 release of the repository](https://github.com/finn-no/recsys-slates-dataset/tree/v1.0)

## Citations
This repository accompany the paper ["Dynamic Slate Recommendation with Gated Recurrent Units and Thompson Sampling"](https://arxiv.org/abs/2104.15046) by Simen Eide, David S. Leslie and Arnoldo Frigessi.
The article is under review, and the pre-print can be obtained [here](https://arxiv.org/abs/2104.15046).

If you use either the code, data or paper, please consider citing the paper.

```
@article{eide2021dynamic,
      title={Dynamic Slate Recommendation with Gated Recurrent Units and Thompson Sampling}, 
      author={Simen Eide and David S. Leslie and Arnoldo Frigessi},
      year={2021},
      eprint={2104.15046},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

# Todo
This repository is currently *work in progress*, and we will provide descriptions and tutorials. Suggestions and contributions to make the material more available is welcome.
There are some features of the repository that we are working on:

- [x] Release the dataset as numpy objects instead of pytorch arrays. This will help non-pytorch users to more easily utilize the data
- [x] Maintain a pytorch dataset for easy usage
- [ ] Create a pip package for easier installation and usage. the package should download the dataset using a function.
- [ ] Make the quickstart guide compatible with the pip package and numpy format.
- [ ] Add easily useable functions that compute relevant metrics such as hitrate, log-likelihood etc.
- [ ] Distribute the data on other platforms such as kaggle.
- [ ] Add a short description of the data in the readme.md directly.

As the current state is in early stage, it makes sense to allow the above changes non-backward compatible. 
However, this should be completed within the next couple of months.


This file will become your README and also the index of your documentation.

## Install

`pip install your_project_name`

## How to use

Fill me in please! Don't forget code examples:

```python
1+1
```




    2


