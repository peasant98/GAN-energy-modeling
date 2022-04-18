# GAN Energy Modeling

Energy Modeling via GANs (Generative Adversarial Networks). This work presents a comprehensive evaluation of GANs for building energy modeling, titled "**Evaluating Performance of Different Generative Adversarial Networks for Large-Scale Building Power Demand Prediction**"

Paper currently in submission.

**A comprehensive summary of each model is provided here: [Link](model_archs/README.md)**

You can find summarized results in the `results/` folder.


=========================================================================================

Contributors:
- Matthew Strong
- Yunyang Ye
- Yingli Lou
- Satish Upadhyaya

Before running anything, make sure to unzip the csv files that are necessary to run the code:

```sh
unzip training_data/data_collect.zip

unzip training_data/power_new.zip
```

Next, install the necessary python packages:

```sh
pip install -r requirements.txt

```

Now, you can run training for each type of GAN. The scripts listed below train GANs on the training dataset, create generated samples, then evaluate these samples on the simulation (ground truth data).

## GAN
```sh
python gan/gan.py
```

## CGAN
```sh
python cgan/main_cgan.py
```

## ACGAN
```sh
python acgan/main_acgan.py
```

## SGAN
```sh
python sgan/main_sgan.py
```

## InfoGAN
```sh
python infogan/infogan.py
```


