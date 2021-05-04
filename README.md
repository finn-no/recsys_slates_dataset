# FINN.no Recommender Systems Slate Dataset
This repository accompany the paper *"Dynamic Slate Recommendation with Gated Recurrent Units and Thompson Sampling"* by Simen Eide, David S. Leslie and Arnoldo Frigessi.
The article is under review, and the pre-print can be obtained [here](https://arxiv.org/abs/2104.15046).

The repository is split into the dataset (`data/`) and the accompanying code for the paper (`code/`).

We release the *FINN.no recommender systems slate dataset* to improve recommender systems research.
The dataset includes both search and recommendation interactions between users and the platform over a 30 day period.
The dataset has logged both exposures and clicks, *including interactions where the user did not click on any of the items in the slate*.

For each user $u$ and interaction step $t$ we recorded all items in the visible slate $a_t^u(s_t^u)$ (up to the scroll length $s_t^u$), and the user's click response $c_t^u$.
The dataset consists of $37.4$ million interactions,  $|U| \approx 2.3$ million  users and $|I| \approx 1.3$ million items that belong to one of $|G| = 290$ item groups.

FINN.no is the leading marketplace in the Norwegian classifieds market and provides users with a platform to buy and sell general merchandise, cars, real estate, as well as house rentals and job offerings.

This repository is currently *work in progress*, and we will provide descriptions and tutorials. Suggestions and contributions to make the material more available is welcome.

For questions, email simen.eide@finn.no or file an issue.

## Download and prepare dataset

The data file is compressed, unzip by the following command: `gunzip -c data.pt.gz >data.pt`

1. Install git-lfs: This repository uses `git-lfs` to store the dataset. Therefore you need the git-lfs package in addition to github. See [https://git-lfs.github.com/] for installation instructions.
2. Clone the repository
3. The data file is compressed, unzip by the following command: `gunzip -c data.pt.gz >data.pt`

## Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finn-no/recsys-slates-dataset/blob/master/quickstart-finn-recsys-slate-data.ipynb)
We provide a quickstart jupyter notebook that runs on Google Colab (quickstart-finn-recsys-slate-data.ipynb).
