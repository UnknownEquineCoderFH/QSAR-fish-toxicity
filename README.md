# Detecting Overfit: QSAR fish toxicity

## Description

This is a project for the course **Machine Learning, Artificial Intelligence and Big Data Analytics** at the [University of Applied Sciences in Krems](https://www.fh-krems.ac.at/en/).

The goal of this project is to detect overfitting in a regression model.
The original model that was provided by the lecturer contained chemical properties of wine and the quality of the wine.

Instead of using the wine dataset, I used the [QSAR fish toxicity dataset](https://archive-beta.ics.uci.edu/dataset/504/qsar+fish+toxicity) from the UCI Machine Learning Repository.

## The Model

This dataset was used to develop quantitative regression QSAR models to predict acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow) on a set of 908 chemicals. 

The dataset as-is contains no missing values.

![Fish](https://upload.wikimedia.org/wikipedia/commons/f/f7/Pimephales_promelas2.jpg)
##### The fish in question


LC50 data, which is the concentration that causes death in 50% of test fish over a test duration of 96 hours, was used as model response.

The model comprised 6 molecular descriptors: MLOGP (molecular properties), CIC0 (information indices), GATS1i (2D autocorrelations), NdssC (atom-type counts), NdsCH ((atom-type counts), SM1_Dz(Z) (2D matrix-based descriptors).

For this project, I used the following 4 features: MLOGP, CIC0, GATS1i, SM1_Dz(Z), as well as the target variable LC50.

## Run

The project was written using python 3.11.
It is possible that any other python version >= 3.9 will work as well.

To run the project, you need to install the dependencies first:

```bash
virtualenv venv -p python3.11

source venv/bin/activate

pip install -r requirements.txt
```

Then, you can run the project:

```bash
python src/main.py
```

## Results

As it can be seen in the [exported plot](temp-plot.html), as well as in the console output, as the model complexity increases, both the training and test error decrease.

This means that the model is able to fit the training data and the test data better and better.
(No overfitting happens).
