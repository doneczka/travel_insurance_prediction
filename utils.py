
# Basic data manipulation and visualization
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from typing import List, Set, Dict, Tuple, Optional
from numpy import float64
from sklearn.preprocessing import StandardScaler

# Statistical inference
import statsmodels.api as sm
import statsmodels.stats.weightstats as smweight
import statsmodels.stats.proportion as smprop

# Models
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
import xgboost as xg

# Metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import plot_confusion_matrix


def create_boxplot(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns boxplot for numerical features of a dataset.

    Boxplots give us a good understanding of how data are spread out in our dataset.

    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a boxplot for.
    Returns: subplots of boxplots for each of the features.
    
    '''
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for i, (ax, curve) in enumerate(zip(axs.flat, list_of_columns)):
        sns.boxplot(y=data[curve], color='darkorange', ax = ax,
                showmeans=True,  meanprops={"marker":"o",
                                            "markerfacecolor":"black", 
                                            "markeredgecolor":"black",
                                            "markersize":"6"},
                                 flierprops={'marker':'o', 
                                             'markeredgecolor':'darkgreen'})

        ax.set_title(list_of_columns[i])
        ax.set_ylabel('') 


def create_histplot(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns histomgrams for numerical features of a dataset.

    Histomgrams give us a good understanding of how data are distributed.

    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a histomgram for.
    Returns: subplots of histomgrams for each of the features.
    
    '''
    fig, axes = plt.subplots(
    1, 3, figsize=(16, 6), gridspec_kw={"hspace": 0.75, "wspace": 0.25})
    for i, ax in enumerate(axes.flatten()):
        sns.histplot(
        data=data, x=data[list_of_columns[i]], ax=ax, kde=True
        )
        ax.ticklabel_format(style="plain")
        ax.set_xlabel('')
        ax.set_title(
        f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
        ax.ticklabel_format(style='sci')

    sns.despine(left=True)
    plt.show()  

def create_countplot(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns countplot for categorical features of a dataset.

    Countplots give us a good understanding of how many instances are represented by specific discrete feature.

    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a countplot for.
    Returns: subplots of countplots for each of the features.
    
    '''
    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    for i, ax in enumerate(axes.flatten()):
        sns.countplot(data=data, x=data[list_of_columns[i]], ax=ax)
        ax.set_xlabel('')
        ax.set_title(
        f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
    
        sns.despine(left=True)

def create_catplot(data: pd.DataFrame, column: str) -> plt.figure:
    '''Function returns catplot for categorical features of a dataset.

    Catplots give us a good understanding of how many instances are represented by specific numerical feature
    with a devision to supgroups.

    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a catplot for.
    Returns: subplots of catplots for each of the features.
    '''
    g = sns.catplot(x=column, data=data, aspect=2.0, kind='count',
                    hue='Travel_Insurance')

    g.set_axis_labels(' ', 'Number of customers', labelpad=10)
    g.legend.set_title('With or Without Insurance')
    g.fig.suptitle(f'{column} and Insurance')
    g.ax.margins(.05)

def create_countplot_with_hue(data: pd.DataFrame, list_of_columns: List) -> plt.figure:
    '''Function returns countplot for categorical features of a dataset.

    Countplots give us a good understanding of how many instances are represented by specific discrete feature
    with devision to supgroups.

    Arg: data: pd.DataFrame - input data
         list_of_columns: feature we want to plot a countplot for.
    Returns: subplots of countplots for each of the features.

    '''

    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    for i, ax in enumerate(axes.flatten()):
        sns.countplot(data=data, x=data[list_of_columns[i]], hue = 'Travel_Insurance', ax=ax)
        ax.set_xlabel('')
        ax.set_title(
        f"{' '.join(list_of_columns[i].split('_'))}", fontsize=13, y=1.03)
    
        sns.despine(left=True)

def create_heatmap(df: pd.DataFrame, 
                   size_of_figure: Tuple[int, int]) -> plt.figure:
    """Function creates heatmap of correlation of feature from a given dataframe.
  
    Arg: df - pd.DataFrame - dataframe with analyzed features
       size_of_figure - Tuple[int] - desired figure size
       
    Return: plt.figure - heatmap.
    """
  
    corr_data = df
    corr = corr_data.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(size_of_figure))

    cmap = sns.color_palette("coolwarm", as_cmap=True)
    heatmap = sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=1,
    vmin=-1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    annot=True,
    )

    heatmap.set_title(
    f'Correlation heatmap of data attributes',
    fontdict={"fontsize": 16},
    pad=12,
    )
    plt.xlabel("")
    plt.ylabel("");

def ml_model(model, 
            X_train: np.array, 
            X_test: np.array, 
            y_train: np.array, 
            y_test: np.array) -> str:
    '''Function build a machine learning model, fit it, makes prediction and returns the acuracy score.

    Arg: model - type of machine learning model, eg. LogisticRegression(),
        X_train: np.array, 
        X_test: np.array, 
        y_train: np.array, 
        y_test: np.array - numpy arrays with train/test data
    Return: string sentence with the accury score
    '''
    model=model
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return f'The accuracy of the {model} is {accuracy_score(prediction, y_test)}'

def create_confusion_matrix(dict_of_models: Dict, 
            X_train: np.array, 
            X_test: np.array, 
            y_train: np.array, 
            y_test: np.array) -> plt.figure:
    '''Functions that create confusion matrix for machine learning model outcome
    
    Arg: dict_of_models: dictonary of models with names as keys and models as valus.
        X_train: np.array, 
        X_test: np.array, 
        y_train: np.array, 
        y_test: np.array - numpy arrays with train/test data
    Return: confusion matric - plt.figure
    '''
    f, ax=plt.subplots(1, 6, figsize=(24, 4))
    i=0
    for key, value in dict_of_models.items():
        y_pred = cross_val_predict(value, X_test, y_test, cv=6)
        sns.heatmap(confusion_matrix(y_test, y_pred), ax=ax[i], annot=True, fmt='2.0f')
        ax[i].set_title(f'Matrix for {key}')
        i+=1
