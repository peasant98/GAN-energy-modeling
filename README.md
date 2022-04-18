# GAN-energy-modeling

Energy Modeling via GANs (Generative Adversarial Networks).

Paper currently in submission to Applied Energy.

Contributors:
- Matthew Strong
- Yunyang Ye
- Yingli Lou
- Satish Upadhyaya

Before running anything, make sure to unzip the csv files that are necessary to run the code:

```sh
unzip data/data_collect.zip

unzip data/power_new.zip
```

Next, install the necessary python packages:

```sh
pip install -r requirements.txt

```

Now, you can run training for each type of GAN:

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
python infogan/main_infogan.py
```

## GAN
```sh
python gan/main.py
```
