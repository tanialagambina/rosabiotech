"""
********************************************************************************
                    Data Analysis & Visualisation Package
********************************************************************************
This package contains a selection of functions that allow the user to analyse
the data and results visually, as well as other random supporting analysis
functions that aren't suited to train_model.py and processing.py.

Functions included within this package:
- plot_pairplot
- plot_boxplot
- plot_fingerprint
- plot_roc_curve
- calc_chi_squared
- bagged_tree_feature_importances
- calc_feature_importances_kbest
- recursive_feature_elimination
- lr_feature_selection
- plot_superplots
- calc_feature_ranking
- plot_model_comparisons
- plot_background_signal
- plot_background_signal_per_plate
- array_2_fingerprint
-


AUTHOR:
    Tania LaGambina
********************************************************************************
"""

import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import warnings
from itertools import chain
from fpdf import FPDF
from datetime import date
from production import processing, train_model


def plot_pairplot(
        data,
        target,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Function to plot a pairplot of the data. This includes a scatter plot for
    every feature against every other feature, as well as a distplot for each
    of the features, with data points separated by a point of interest, for
    example the target analyte.

    INPUTS:
    - data: The
    - target: The point of interest which you want to focus on with the pairplot.
        For example, one of the analytes can be chosen, like "TeaType"

    AUTHOR:
        Tania LaGambina
    """

    if True in (data.columns == target):
        fig = plt.figure(figsize=[30, 30])
        sns.set(style='ticks')
        if ('colors' in keyword_parameters.keys()):
            colors = keyword_parameters['colors']
            palette = sns.set_palette(sns.color_palette(colors))
        else:
            palette = 'magma'

        if ('hue_order' in keyword_parameters.keys()):
            hue_order = keyword_parameters['hue_order']
        else:
            hue_order = data[target].unique()

        print('Plotting figure...')
        plot = sns.pairplot(
                data.loc[data[target].isin(hue_order)],
                hue=target,
                palette=palette,
                hue_order=hue_order
                )
        fig = plot.fig
        plt.show()

        return fig
    else:
        AttributeError('The point of interest input selected is not in the '
                'dataframe.')



def plot_boxplot(
            data,
            target,
            *positional_parameters,
            **keyword_parameters
        ):
    """
    Function to plot a box plot for each of the columns in the dataframe. This
    allows us to visualise where any outliers may be and to understand the
    spread of the data a bit better.

    INPUTS:
    - data: The fluorescence data frame
    - target: The column in which we wish to compare different boxplots
    - save: True or False depending on whether the plot should be saved

    AUTHOR:
        Tania LaGambina
    """

    analytes_list = data[target].unique()
    if len(analytes_list) > 1:
        f, ax = plt.subplots(
            math.ceil(len(analytes_list)/2),
            2,
            figsize=(27, 9*math.ceil(len(analytes_list)/2))
            )
        f.tight_layout(pad=10.0)
    else:
        f, ax = plt.subplots(
            1,
            1,
            figsize=(10, 5)
            )
        f.tight_layout(pad=10.0)
    for analyte in np.linspace(0, np.subtract(len(analytes_list), 1), len(analytes_list)):
        parsed_data = copy.deepcopy(data.loc[data[target]==analytes_list[analyte.astype(int)]])
        parsed_data.drop(columns=target, axis=1, inplace=True)
        parsed_data = copy.deepcopy(parsed_data.select_dtypes(['number']))
        dyes_list = parsed_data.columns.str.split().str[-1]
        dyes = dyes_list.unique()
        dye_data = {}
        for dye in dyes:
            data_dye = copy.deepcopy(parsed_data[parsed_data.columns[parsed_data.columns.str.split().str[-1]==dye]])
            cols = data_dye.columns
            data_dye.columns = [''.join(col.split(' ')[:-1]) for col in cols]
            data_dye['Dye'] = dye
            dye_data[dye] = data_dye
        dat = pd.concat(dye_data, sort=True)
        dat.reset_index(drop=True, inplace=True)
        colors = ["#2E64FE", "#DF013A", "#800080", "#3FBF5F"]
        customPalette = sns.set_palette(sns.color_palette(colors))
        mdf = pd.melt(
            dat,
            id_vars='Dye',
            var_name='aHB',
            value_name='Fluorescence'
            )
        plt.subplot(
            math.ceil(len(analytes_list)/2),
            2,
            analyte.astype(int)+1
            )
        if ('order' in keyword_parameters.keys()):
            order = keyword_parameters['order']
            chart = sns.boxplot(
                data=mdf,
                x='aHB',
                y='Fluorescence',
                hue='Dye',
                palette=customPalette,
                order=order
                )
        else:
            chart = sns.boxplot(
                data=mdf,
                x='aHB',
                y='Fluorescence',
                hue='Dye',
                palette=customPalette
                )
        chart.set_ylim([0, 2])
        chart.set_title(analytes_list[analyte.astype(int)])
        chart.set_xticklabels(
            chart.get_xticklabels(),
            rotation=60,
            horizontalalignment='right',
            fontweight='light',
            fontsize='small'
        )
        # chart.legend().set_visible(False)
    if ('save' in keyword_parameters.keys()):
        if keyword_parameters['save'] is True:
            cwd = os.getcwd()
            filename = "/Results/"
            os.makedirs(os.path.dirname(cwd+filename), exist_ok=True)
            f.savefig(cwd+filename+'Boxplot for {}.png'.format(target), bbox_inches='tight')
        else:
            pass
    else:
        pass

    return f


def plot_roc_curve(
        x_test,
        y_test,
        y_true_label,
        clf,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    Function to plot the ROC curve for a binary classification problem.
    It allows us to visualise the skill of the model.
    It is the sensitivity (true positive rate) versus the false negative rate.
    For disease diagnoses, we would prefer a higher sensitivity, as the
    consequences of missing a diagnosis are much higher, than being cautious
    and giving a false positive. Thus, high sensitivity and negative predictive
    value are required in a screening setting.

    INPUTS:
    - x_test: The input test data defined in the train test split
    - y_test: The target test data defined in the train test split
    - clf: The trained model

    OUTPUTS:

    AUTHOR:
        Tania LaGambina, based on https://machinelearningmastery.com/
                                  roc-curves-and-precision-recall-curves-
                                  for-classification-in-python/
    """
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder, binarize

    # Generating a 'no skill' prediction
    ns_probs = [0 for _ in range(len(y_test))]

    probs = clf.predict_proba(x_test)

    # Keep probabilities for the positive outcome only
    probs = probs[:,1]

    # Calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    probs_auc = roc_auc_score(y_test, probs)
    # Summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model: ROC AUC=%.3f' % (probs_auc))
    if ('save' in keyword_parameters.keys()) and (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
        path = keyword_parameters['path']
        with open(path+"Model Results.txt", "a") as f:
            print('Model: ROC AUC=%.3f' % (probs_auc), file=f)
    # Converting the non binary values to binary using label encoder
    labelencoder = LabelEncoder()
    y_test = labelencoder.fit_transform(y_test)

    # Calculate ROC curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs, pos_label=1)
    mod_fpr, mod_tpr, _ = roc_curve(y_test, probs, pos_label=1)

    # Plot the ROC curve for the model
    plt.clf()
    fig = plt.figure(figsize=[5,5])

    plt.style.use('default')
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(mod_fpr, mod_tpr, marker='.', label='Model', linewidth=2.5, color='black')
    ax = plt.gca()
    ax.axis('square')
    plt.xlabel('False Positive Rate (1-Specificity)', fontsize=17)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=17)

    plt.legend()
    ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')
    plt.show()

    return fig, probs_auc


def calc_chi_squared(data, std_data, target):
    """
    The pearson's chi-squared statistical hypothesis is an example of a test
    for independence between categorical variables. A table summarization of
    two categorical variables is called a contingency table - because the
    content is used to help determine whether one variable is contingent upon or
    depends on the other variable. As our data is mostly dealing with continuous
    variables (even though it is a categorical classification problem), we must
    treat the data in some way first to make it 'categorical'. To do this, I
    chose to round the numeric fluorescence values to one decimal place. I then
    iterated through the different peptides and made a contingency table for
    each, versus the analyte of interest. The chi squared result was then
    calculated for each, and from this we can determine whether the value is
    dependent or independent on the prediction of the analyte. This section
    then goes on to create a seperate dataframe that removes those peptides
    that are calculated to be independent on the type of analyte. This could
    be used later on for the machine learning.

    This function takes in the data which the chi squared result is desired,
    and returns the result, as well as a new dataframe the removes the columns
    that are not contributing to the final signal.

    INPUTS:
    - data: The dataframe in which you want to get the chi squared result. It
            should be in the form ['No Pep', 'GRP22', ... , 'Analyte']
    - std_data: The standardised data frame, so the dependent standardised data
            can be produced
    - target: The target in which you want to measure the data with the chi
            squared statistic against. I.e. 'Analyte'

    OUTPUTS:
    - chi_squared_result: A dataframe reporting the results of the chi squared
            test. It gives a result for each of the peptides it tested as
            Dependent or Independent: Dependent meaning the peptide is
            contributing to the result, Independent meaning it is not
    - dependent_data: A processed version of the 'data' input. It is the same
            data, but with the peptides that are independent to the final
            result removed
    - dependent_standardised_data: A processed version of the 'std_data' input

    AUTHOR:
        Tania LaGambina
    """

    # chi-squared test with similar proportions
    from scipy.stats import chi2_contingency, chi2

    chi_squared_result = pd.DataFrame()
    peptides = []
    results = []
    columns = data.select_dtypes(['number']).columns.tolist()
    for peptide in columns:
        # Creating the contingency table
        data_crosstab = pd.crosstab(
            data.round(1)[target], data.round(1)[peptide], margins=False)
        stat, p, dof, expected = chi2_contingency(data_crosstab)
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        alpha = 1.0 - prob
        if p <= alpha:
            peptides.append(peptide)
            results.append('Dependent')
        else:
            peptides.append(peptide)
            results.append('Independent')

    chi_squared_result['Peptide'] = peptides
    chi_squared_result['Result'] = results

    dependent_data = copy.deepcopy(
        data.drop(
        chi_squared_result.loc[
        chi_squared_result.Result=='Independent'].Peptide.tolist(), axis=1))
    dependent_standardised_data = copy.deepcopy(
        std_data.drop(
        chi_squared_result.loc[
        chi_squared_result.Result=='Independent'].Peptide.tolist(), axis=1))

    return chi_squared_result, dependent_data, dependent_standardised_data

def bagged_tree_feature_importances(x_data, y_data):
    """
    This function takes the feature_importances_ functionality from the
    ExtraTreesClassifier to rank the features in terms of most significant.
    It is worth noting however, when running, each run may result in different
    results. This is to be expected if here are redundant or highly correlated
    features within the dataset. The bagged trees method will tend to suppress
    redundant features. If a feature A is used already, and feature B does not
    add any new information over A it won't get used - and it won't contribute
    to feature importances, even though it is in general a highly predictive
    value.
    The features that get picked as important among the redundant features is
    random.

    AUTHOR:
        Tania LaGambina
    """
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier(n_estimators=250)
    model.fit(x_data, y_data)
    feat_importances = pd.Series(
        model.feature_importances_,
        index=x_data.columns
        )
    # feat_importances.nlargest(15).plot(kind='barh')
    # plt.show()

    return feat_importances


def calc_feature_importances_kbest(
        x_train,
        y_train,
        features
        ):
    """
    Function to determine the 'best' features within the dataset. It scores
    all of the features using a function (f_classif or mutual_info_classif)
    to do so. This function then goes on to plot these best performing
    features for interpretiblity.

    INPUTS:
    - x_train: The training input data, which contains only the
            fluorescence values and not the corresponding class
            labels.
    - y_train: The target data for the training set that contains the
            class labels for a particular reading.
    - features: The input features that are used as predictors for the
            model training. I.e. the peptides.
    OUTPUTS:
    - score_df: A dataframe that cotains the f score calculated for each of
            the features/peptides. The f score is the ratio of explained to
            unexplained variance in the signal

    AUTHOR:
        Based on original by Kathryn Shelley, Woolfson Group
        Developed for Rosa Biotech by Tania LaGambina
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    model = SelectKBest(score_func=f_classif, k='all')
    model.fit(x_train, y_train)

    score_df = pd.DataFrame({'Feature': features, 'Score': model.scores_})
    score_df = score_df.sort_values(
        by=['Score'],
        axis=0,
        ascending=False
        ).reset_index(drop=True)

    return score_df


def recursive_feature_elimination(x_data, y_data):
    """
    This function uses the recursive feature elimination wrapper function with
    the logistic regression model to rank the most important features in the
    dataset, with 1 being the most important feature. A bar chart is returned
    with the highest ranking features.

    AUTHOR:
        Tania LaGambina
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    # define RFE
    rfe = RFE(estimator=LogisticRegression(solver='liblinear', multi_class='auto'), n_features_to_select=1)
    # fit RFE
    rfe.fit(x_data, y_data)
    # summarize all features
    # for i in range(x_data.shape[1]):
    # 	print('Column: %s, Selected %s, Rank: %.3f' % (x_data.columns[i], rfe.support_[i], rfe.ranking_[i]))
    # print(rfe.ranking_)
    feat_importances = pd.Series(
        rfe.ranking_,
        index=x_data.columns
        )
    # feat_importances.nsmallest(15).plot(kind='barh')
    # plt.gca().invert_yaxis()
    # plt.show()

    return feat_importances


def lr_feature_selection(x_data, y_data):
    """
    Making use of a Logistic Regression model with a l1 regularisation
    parameter to determine which features will have weights set to 0 in the
    course of the regularisation

    AUTHOR:
        Tania LaGambina
    """
    from sklearn.linear_model import LogisticRegression, Lasso
    from sklearn.feature_selection import SelectFromModel

    model = LogisticRegression(penalty='l1', solver='liblinear').fit(x_data, y_data)
    importances = model.coef_[0]
    feat_importances = pd.Series(
        data=np.abs(importances),
        index=x_data.columns
        )
    return feat_importances

def random_forest_regressor_feature_selection(x_data, y_data):
    """
    Feature importance calculated from importance of feature determined in
    a random forest regressor model

    AUTHOR:
        Tania LaGambina
    """
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(x_data, y_data)

    # features = model.get_booster().feature_names
    importances = model.feature_importances_
    model_feature_importances_df = pd.DataFrame(
        zip(x_data, importances),
        columns=['feature', 'importance']
    ).set_index('feature')

    return model_feature_importances_df

def xgb_regressor_feature_selection(x_data, y_data):
    """
    Feature importance calculated from importance of feature determined in
    a XGB regressor model

    AUTHOR:
        Tania LaGambina
    """
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.fit(x_data, y_data)

    features = model.get_booster().feature_names
    importances = model.feature_importances_
    model_feature_importances_df = pd.DataFrame(
        zip(features, importances),
        columns=['feature', 'importance']
    ).set_index('feature')

    return model_feature_importances_df

def knn_regressor_permutation_analysis_feature_selection(x_data, y_data):
    """
    Permutation analysis feature selection with k nearest neighbours regressor
    as model used

    AUTHOR:
        Tania LaGambina
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.inspection import permutation_importance
    model = KNeighborsRegressor()
    model.fit(x_data, y_data)
    results = permutation_importance(model, x_data, y_data, scoring='neg_mean_squared_error')
    importances = results.importances_mean
    model_feature_importances_df = pd.DataFrame(
        zip(x_data.columns, importances),
        columns=['feature', 'importance']
    ).set_index('feature')

    return model_feature_importances_df

def plot_superplots(
                parsed_data,
                category,
                hue,
                *positional_parameters,
                **keyword_parameters
            ):
    """
    This function plots superplots, first come across here:
    https://huygens.science.uva.nl/SuperPlotsOfData/
    These plots allow the user to view the data in a higher dimension.

    This function plots a superplot for each of the peptide-dye combinations,
    and splits each plot up by the 'category'. The individual plots are then
    coloured by the 'hue', which both need to be pre determined

    INPUTS:
        - parsed_data: The unscaled parsed data also containing the meta data
                that is required for the superplot
        - category: The column in which you would like to split each superplot
                up by along the x axis
        - hue: The column in parsed_data that will be used to colour the data
        - save: True or False required, depending on if you wish to save each
                of the output figures
        - path: (keyword_parameters) The path which you would like to save the
                individual output figures if save is True
        - palette: (keyword_parameters) The palette used to colour the plots

    AUTHOR:
        Tania LaGambina
    """
    warnings.filterwarnings("ignore")

    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    else:
        palette = 'magma'

    if ('size' in keyword_parameters.keys()):
        size = keyword_parameters['size']
    else:
        size = 5

    for peptide in parsed_data.select_dtypes(['number']).columns.tolist():
        if ('figsize' in keyword_parameters.keys()):
            figsize = keyword_parameters['figsize']
            f, ax = plt.subplots(figsize=figsize)
        else:
            f, ax = plt.subplots(figsize=(4, 5))
        ax.set_ylim([-0,2])
        ax = sns.swarmplot(
            x=category,
            y=peptide,
            hue=hue,
            data = parsed_data[[peptide, category, hue]],
            size=size,
            edgecolor='w',
            palette=palette
        )

        sns.boxplot(
            showmeans=True,
            meanline=True,
            meanprops={'visible': False},
            medianprops={'ls': '-', 'lw': 0.8},
            whiskerprops={'visible': False},
            zorder=10,
            showfliers=False,
            showbox=True,
            showcaps=True,
            capprops={'ls': '-', 'lw': 0.8},
            boxprops={'ls': '-', 'lw': 0.8},
            x=category,
            y=peptide,
            data = parsed_data[[peptide, category, hue]],
            ax=ax,
            color='white'
        )
        ax.legend(bbox_to_anchor=(1, 0.5), loc='center left')


def calc_feature_ranking(
        standardised_parsed_data,
        target,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    This function is a wrapper function that contains three different feature
    importance analysis techniques into one function. It them combines the
    results of these and returns a ranking, which can then be used to select
    which features should be used in an ML model.

    Note - for best and most unbiased results, only the train set should be used
    in this function to determine the most important feaures.

    INPUTS:
        - standardised_parsed_data: The scaled and standardised_parsed_data
        - target: The target column by which you want to see if the other
                features are important to it

    OUTPUTS:
        - ranking_df: A dataframe that contains the top 16 features and their
                respective ranking

    AUTHOR:
        Tania LaGambina
    """
    target_type = standardised_parsed_data[target].dtypes

    if target_type == 'object':

        # Bagged tree feature importances
        feat_importances_bt = bagged_tree_feature_importances(
            standardised_parsed_data.select_dtypes(['number']),
            standardised_parsed_data[[target]].iloc[:,0]
        )

        # ANOVA feature importances
        features = standardised_parsed_data.select_dtypes(['number']).columns.tolist()
        anova_scores = analysis.calc_feature_importances_kbest(
            standardised_parsed_data.select_dtypes(['number']),
            standardised_parsed_data[target],
            features
        )

        # Recursive Feature Elimination feature importances
        feat_importances_rfe = recursive_feature_elimination(
            standardised_parsed_data.select_dtypes(['number']),
            standardised_parsed_data[[target]].iloc[:,0]
        )

        # Logistic regression feature importances
        feat_importances_lr = lr_feature_selection(
            standardised_parsed_data.select_dtypes(['number']),
            standardised_parsed_data[[target]].iloc[:,0]
        )

        if ('n' in keyword_parameters.keys()):
            n = keyword_parameters['n']
            if standardised_parsed_data.select_dtypes(['number']).shape[1] < n:
                raise ValueError(f"To many features given for ranking")
            # Combining the different methods
            bt_df = feat_importances_bt.nlargest(n).to_frame()
            bt_df.insert(1, "RANK1", list(map(int, list(np.linspace(n, 1, n)))), True)
            bt_df = bt_df[['RANK1']]

            a_df = anova_scores.nlargest(n, 'Score').Feature.to_frame()
            a_df.insert(1, "RANK2", list(map(int, list(np.linspace(n, 1, n)))), True)
            a_df.set_index('Feature', inplace=True)

            rfe_df = feat_importances_rfe.nsmallest(n).to_frame()
            rfe_df.insert(1, "RANK3", list(map(int, list(np.linspace(n, 1, n)))), True)
            rfe_df = rfe_df[['RANK3']]

            lr_df = feat_importances_lr.nlargest(n).to_frame(name='coefs')
            lr_df.insert(1, "RANK4", list(map(int, list(np.linspace(n, 1, n)))), True)
            lr_df = lr_df.loc[~(lr_df.coefs==0)]
            lr_df = lr_df[['RANK4']]

        else:
            # Combining the different methods
            bt_df = feat_importances_bt.nlargest(16).to_frame()
            bt_df.insert(1, "RANK1", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            bt_df = bt_df[['RANK1']]

            a_df = anova_scores.nlargest(16, 'Score').Feature.to_frame()
            a_df.insert(1, "RANK2", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            a_df.set_index('Feature', inplace=True)

            rfe_df = feat_importances_rfe.nsmallest(16).to_frame()
            rfe_df.insert(1, "RANK3", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            rfe_df = rfe_df[['RANK3']]

            lr_df = feat_importances_lr.nlargest(16).to_frame(name='coefs')
            lr_df.insert(1, "RANK4", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            lr_df = lr_df.loc[~(lr_df.coefs==0)]
            lr_df = lr_df[['RANK4']]

        ranking_df = pd.concat([rfe_df, bt_df, a_df, lr_df], axis=1)
        ranking_df.fillna(0, inplace=True)
        ranking_df['SUM'] = ranking_df.loc[:,['RANK1', 'RANK2', 'RANK3', 'RANK4']].sum(axis=1)
        ranking_df.sort_values(by='SUM', axis=0, ascending=False, inplace=True)
        ranking_df = copy.deepcopy(ranking_df[['SUM']])

    elif target_type == 'float64':
        # Random forest regressor feature importances
        feat_importances_rf = random_forest_regressor_feature_selection(
            standardised_parsed_data.drop(columns=[target], axis=1).select_dtypes(['number']),
            standardised_parsed_data[[target]].iloc[:,0]
        )

        # XGB regressor feature importances
        feat_importances_xgb = xgb_regressor_feature_selection(
            standardised_parsed_data.drop(columns=[target], axis=1).select_dtypes(['number']),
            standardised_parsed_data[[target]].iloc[:,0]
        )

        # Permutation analysis with KNN feature importances
        feat_importances_knnpa = knn_regressor_permutation_analysis_feature_selection(
            standardised_parsed_data.drop(columns=[target], axis=1).select_dtypes(['number']),
            standardised_parsed_data[[target]].iloc[:,0]
        )

        if ('n' in keyword_parameters.keys()):
            n = keyword_parameters['n']
            if standardised_parsed_data.select_dtypes(['number']).shape[1] < n:
                raise ValueError(f"To many features given for ranking")
            # Combining the different methods
            rf_df = feat_importances_rf['importance'].nlargest(n).to_frame()
            rf_df.insert(1, "RANK1", list(map(int, list(np.linspace(n, 1, n)))), True)
            rf_df = rf_df[['RANK1']]

            xgb_df = feat_importances_xgb['importance'].nlargest(n).to_frame()
            xgb_df.insert(1, "RANK2", list(map(int, list(np.linspace(n, 1, n)))), True)
            xgb_df = xgb_df[['RANK2']]

            knnpa_df = feat_importances_knnpa['importance'].nlargest(n).to_frame()
            knnpa_df.insert(1, "RANK3", list(map(int, list(np.linspace(n, 1, n)))), True)
            knnpa_df = knnpa_df[['RANK3']]

        else:
            # Combining the different methods
            rf_df = feat_importances_rf['importance'].nlargest(16).to_frame()
            rf_df.insert(1, "RANK1", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            rf_df = rf_df[['RANK1']]

            xgb_df = feat_importances_xgb['importance'].nlargest(16).to_frame()
            xgb_df.insert(1, "RANK2", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            xgb_df = xgb_df[['RANK2']]

            knnpa_df = feat_importances_knnpa['importance'].nlargest(16).to_frame()
            knnpa_df.insert(1, "RANK3", [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], True)
            knnpa_df = knnpa_df[['RANK3']]


        ranking_df = pd.concat([rf_df, xgb_df, knnpa_df], axis=1)
        ranking_df.fillna(0, inplace=True)
        ranking_df['SUM'] = ranking_df.loc[:,['RANK1', 'RANK2', 'RANK3']].sum(axis=1)
        ranking_df.sort_values(by='SUM', axis=0, ascending=False, inplace=True)
        ranking_df = copy.deepcopy(ranking_df[['SUM']])
    else:
        raise ValueError('Please ensure the target data type is either float64 for continuuous or object for categorical')

    return ranking_df

def plot_model_comparisons(
        results,
        names,
        max_key,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    The function plots boxplots of the cross validation scores for each model
    in the comparison

    INPUTS:
        - results: A dictionary of the cross validation results for the optimum
            number of features for each model
        - names: The names of each model to label the x axis
        - save: True or False depending on whether you wish the resulting
            boxplot to be saved

    AUTHOR:
        Tania LaGambina
    """

    plt.boxplot(results, labels=names, showmeans=True)
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xticklabels(
        labels=names,
        rotation=60
    )
    ax.set_ylim([0, 1.01])

    fig.set_size_inches(10, 7)

    plt.show()

    if ('save' in keyword_parameters.keys()):
        save = keyword_parameters['save']
        if save is True:
            cwd = os.getcwd()
            path = cwd+'/Results/' #It is noted this is machine dependant - find a more versatile way
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path+'Model Comparison for {} Features.png'.format(max_key), bbox_inches='tight')
        else:
            pass

def plot_background_signal(
        queried_data,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    This function plots the water blanks as a boxplot for all of the data in
    the extracted dataset. Here, obvious outliers can be spotted and the general
    trend of the background data can be interpreted.

    INPUTS:
    - queried_data: The data extracted straight from the database
    - save: A true/false input that will determine whether the figure should be
            saved into the results folder

    AUTHOR:
        Tania LaGambina
    """

    data = copy.deepcopy(queried_data)
    data.rename(columns={'PLATE_ID':'EXPERIMENT_ID'}, inplace=True)
    data.replace({'Buffer':'No Pep',
                         'FRET C7':'DPH+C7',
                         'FRET NR':'DPH+NR',
                         'DPH + C7':'DPH+C7',
                         'DPH + NR':'DPH+NR'
                         }, inplace=True)
    data.drop_duplicates(['ROW_LOC', 'COLUMN_LOC', 'EXPERIMENT_ID'], keep='last', inplace=True)

    queried_data_w = copy.deepcopy(data.loc[data.ANALYTE_ID.isin(['Blank'])])
    queried_data_w.loc[:, ('PEPTIDE_DYE_ID')] = queried_data_w.loc[:, ('PEPTIDE_ID')]+' '+queried_data_w.loc[:, ('DYE_ID')]
    queried_data_w.sort_values(by=['PEPTIDE_DYE_ID'], inplace=True)

    fig = plt.figure(figsize=(17,5))
    chart = sns.boxplot(x=queried_data_w['PEPTIDE_DYE_ID'], y=queried_data_w['FLUORESCENCE'])
    chart.set_xticklabels(
        labels= chart.get_xticklabels(),
        rotation=60
    );
    plt.show()

    if ('save' in keyword_parameters.keys()):
        save = keyword_parameters['save']
        if save is True:
            cwd = os.getcwd()
            path = cwd+'/Results/'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path+'Background Signal Boxplot', bbox_inches='tight')


def plot_background_signal_per_plate(
        queried_data,
        *positional_parameters,
        **keyword_parameters
            ):
    """
    This function plots the water blanks as a boxplot for all of the data in
    the extracted dataset. Here, obvious outliers can be spotted and the general
    trend of the background data can be interpreted.

    INPUTS:
    - queried_data: The data extracted straight from the database
    - save: A true/false input that will determine whether the figure should be
            saved into the results folder

    AUTHOR:
        Tania LaGambina
    """
    data = copy.deepcopy(queried_data)
    data.rename(columns={'PLATE_ID':'EXPERIMENT_ID'}, inplace=True)
    data.replace({'Buffer':'No Pep',
                         'FRET C7':'DPH+C7',
                         'FRET NR':'DPH+NR',
                         'DPH + C7':'DPH+C7',
                         'DPH + NR':'DPH+NR'
                         }, inplace=True)
    data.drop_duplicates(['ROW_LOC', 'COLUMN_LOC', 'EXPERIMENT_ID'], keep='last', inplace=True)

    try:
        plates = data.EXPERIMENT_ID.unique()
    except:
        print('Check for an EXPERIMENT_ID column')

    for plate in plates:
        queried_data_plate = copy.deepcopy(data.loc[data.EXPERIMENT_ID.isin([plate])])
        queried_data_w = copy.deepcopy(queried_data_plate.loc[queried_data_plate.ANALYTE_ID.isin(['Blank'])])
        queried_data_w.loc[:, ('PEPTIDE_DYE_ID')] = queried_data_w.loc[:, ('PEPTIDE_ID')]+' '+queried_data_w.loc[:, ('DYE_ID')]
        queried_data_w.sort_values(by=['PEPTIDE_DYE_ID'], inplace=True)
        fig = plt.figure(figsize=(17,5))
        chart = sns.boxplot(x=queried_data_w['PEPTIDE_DYE_ID'], y=queried_data_w['FLUORESCENCE'])
        chart.set_xticklabels(
            labels= chart.get_xticklabels(),
            rotation=60
        );
        plt.show()
        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
            if save is True:
                cwd = os.getcwd()
                path = cwd+'/Results/Background Signal Boxplot per Plate/'
                os.makedirs(os.path.dirname(path), exist_ok=True)
                fig.savefig(path+'Background Signal Boxplot for {}'.format(plate), bbox_inches='tight')

def array_2_1_fingerprint(
        parsed_data,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    Fingerprint for Array 2.1 new dye and peptide layout - the layout used in the
    publications

    AUTHOR:
        Tania LaGambina
    """
    mean_parsed_data = copy.deepcopy(parsed_data.select_dtypes(['number']))
    mean_parsed_data = copy.deepcopy(mean_parsed_data.median())
    mean_parsed_data = pd.DataFrame(mean_parsed_data).T

#     if mean_parsed_data.shape[1] != 31:
#         raise ValueError('Input data shape is incorrect (Need (x, 31))')

    mean_parsed_data['No Pep None'] = 0
    mean_parsed_data = mean_parsed_data[[
        'No Pep DPH', 'I24D DPH', 'GRP52 NR',
        'No Pep NR', 'I24E DPH', 'RBBc029 NR',
        'No Pep DPH+NR', 'I24K DPH', 'RBBc030 NR',
        'No Pep DPH+C7', 'RBBb001 DPH', 'RBBd036 NR',

        'GRP51 DPH', 'RBBb034 DPH', 'RBBb031 DPH+NR',
        'GRP52 DPH', 'RBBc028 DPH', 'RBBd031 DPH+NR',
        'CC-Hex DPH', 'RBBd005 DPH', 'GRP51 DPH+C7',
        'CC-Hept DPH', 'RBBd036 DPH', 'RBBd030 DPH+C7'
    ]]

    mean_parsed_data_1 = mean_parsed_data[['No Pep DPH', 'I24D DPH', 'GRP52 NR']].values
    mean_parsed_data_2 = mean_parsed_data[['No Pep NR', 'I24E DPH', 'RBBc029 NR']].values
    mean_parsed_data_3 = mean_parsed_data[['No Pep DPH+NR', 'I24K DPH', 'RBBc030 NR']].values
    mean_parsed_data_4 = mean_parsed_data[['No Pep DPH+C7', 'RBBb001 DPH', 'RBBd036 NR']].values
    mean_parsed_data_5 = mean_parsed_data[['GRP51 DPH', 'RBBb034 DPH', 'RBBb031 DPH+NR']].values
    mean_parsed_data_6 = mean_parsed_data[['GRP52 DPH', 'RBBc028 DPH', 'RBBd031 DPH+NR']].values
    mean_parsed_data_7 = mean_parsed_data[['CC-Hex DPH', 'RBBd005 DPH', 'GRP51 DPH+C7']].values
    mean_parsed_data_8 = mean_parsed_data[['CC-Hept DPH', 'RBBd036 DPH', 'RBBd030 DPH+C7']].values
    grid = pd.concat([
        pd.DataFrame(mean_parsed_data_1),
        pd.DataFrame(mean_parsed_data_2),
        pd.DataFrame(mean_parsed_data_3),
        pd.DataFrame(mean_parsed_data_4),
        pd.DataFrame(mean_parsed_data_5),
        pd.DataFrame(mean_parsed_data_6),
        pd.DataFrame(mean_parsed_data_7),
        pd.DataFrame(mean_parsed_data_8)
    ])

    grid.reset_index(drop=True, inplace=True)

    dph_grid = pd.DataFrame([
        [True, True, False],
        [False, True, False],
        [False, True, False],
        [False, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False],
        [True, True, False]
    ])
    nr_grid = pd.DataFrame([
        [False, False, True],
        [True, False, True],
        [False, False, True],
        [False, False, True],
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, False]
    ])
    dphnr_grid = pd.DataFrame([
        [False, False, False],
        [False, False,  False],
        [True, False, False],
        [False, False, False],
        [False, False, True],
        [False, False, True],
        [False, False, False],
        [False, False, False]
    ])
    dphc7_grid = pd.DataFrame([
        [False, False, False],
        [False, False, False],
        [False, False, False],
        [True, False, False],
        [False, False, False],
        [False, False, False],
        [False, False, True],
        [False, False, True]
    ])

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

#     data_none = np.ma.masked_array(grid, ~none_grid)
    data_dph = np.ma.masked_array(grid, ~dph_grid)
    data_nr = np.ma.masked_array(grid, ~nr_grid)
    data_dphnr = np.ma.masked_array(grid, ~dphnr_grid)
    data_dphc7 = np.ma.masked_array(grid, ~dphc7_grid)
    fig = plt.figure()
#     ax = fig.add_subplots(figsize=[5,8])
    fig, ax = plt.subplots(figsize=[5,8])

    if ('vmax' in keyword_parameters.keys()):
        vmax = keyword_parameters['vmax']
    else:
        vmax = 1

    if ('greyscale' in keyword_parameters.keys()):
        if keyword_parameters['greyscale'] is True:
#             img1 = ax.imshow(data_none, cmap="Greys", vmin=0, vmax=1)
            img2 = ax.imshow(data_dph, cmap="Greys" ,vmin=0, vmax=vmax)
            img3 = ax.imshow(data_nr, cmap="Greys", vmin=0, vmax=vmax)
            img4 = ax.imshow(data_dphnr, cmap="Greys", vmin=0, vmax=vmax)
            img5 = ax.imshow(data_dphc7, cmap="Greys", vmin=0, vmax=vmax)

        else:
#             img1 = ax.imshow(data_none, cmap="Greys", vmin=0, vmax=1)
            img2 = ax.imshow(data_dph, cmap="Blues" ,vmin=0, vmax=vmax)
            img3 = ax.imshow(data_nr, cmap="Reds", vmin=0, vmax=vmax)
            img4 = ax.imshow(data_dphnr, cmap="Purples", vmin=0, vmax=vmax)
            img5 = ax.imshow(data_dphc7, cmap="Greens", vmin=0, vmax=vmax)


    else:
#         img1 = ax.imshow(data_none, cmap="Greys", vmin=0, vmax=1)
        img2 = ax.imshow(data_dph, cmap="Blues" ,vmin=0, vmax=vmax)
        img3 = ax.imshow(data_nr, cmap="Reds", vmin=0, vmax=vmax)
        img4 = ax.imshow(data_dphnr, cmap="Purples", vmin=0, vmax=vmax)
        img5 = ax.imshow(data_dphc7, cmap="Greens", vmin=0, vmax=vmax)


    if ('colorbar' in keyword_parameters.keys()):
        if keyword_parameters['colorbar'] is True:
            bar1 = plt.colorbar(img2)
            bar1.set_label('Scaled Dye Fluorescence')

    # bar2 = plt.colorbar(img3)
    # bar3 = plt.colorbar(img4)
    # bar4 = plt.colorbar(img5)
    if ('title' in keyword_parameters.keys()):
        title = keyword_parameters['title']
        ax.set_title(title)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


    if ('labels' in keyword_parameters.keys()):
        if keyword_parameters['labels'] is True:
            xvalues = np.array([0, 1, 2])
            yvalues = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            xx, yy = np.meshgrid(xvalues, yvalues)
            xcords = np.concatenate(xx)
            ycords = np.concatenate(yy)

            labels_list = list(mean_parsed_data.keys())
            labels = [i.split(' ')[:-1] for i in labels_list]

            n=0
            for j in enumerate(labels):

                x = xcords[n]
                y = ycords[n]

                n = n+1
                plt.text(x, y, ' '.join(j[1]),
                        ha='center',va='center',
                        size=10,color='k')
    plt.show()

    return fig

def plot_raw_data_boxplot(
            queried_data,
            *positional_parameters,
            **keyword_parameters
        ):
    """
    Function that produces a seaborn boxplot for the queried data of Array 2.0.
    This is the raw unprocessed data.

    INPUTS:
        - queried_data: The raw data extracted straight from the database. The
                output of processing.pull_data_from_db
        - hue: The variable which will colour the boxplot, i.e. ANALYTE_ID

    AUTHOR:
        Tania LaGambina
    """
    data = copy.deepcopy(queried_data)

    data.rename(columns={'PLATE_ID':'EXPERIMENT_ID'}, inplace=True)
    data.replace({'Buffer':'No Pep',
                         'FRET C7':'DPH+C7',
                         'FRET NR':'DPH+NR',
                         'DPH + C7':'DPH+C7',
                         'DPH + NR':'DPH+NR'
                         }, inplace=True)
    data.drop_duplicates(['ROW_LOC', 'COLUMN_LOC', 'EXPERIMENT_ID'], keep='last', inplace=True)

    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(30, 10))

    data['aHB'] = data['PEPTIDE_ID']+' '+data['DYE_ID']
    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    elif ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        palette = sns.set_palette(sns.color_palette(colors))
    else:
        palette = 'magma'
    if ('peptide_list' in keyword_parameters.keys()):
        peptide_list = keyword_parameters['peptide_list']
    else:
        peptide_list = ['CC-Hept DPH', 'CC-Hex DPH', 'GRP35 DPH', 'GRP46 NR', 'GRP51 DPH',
           'GRP51 DPH+C7', 'GRP51 DPH+NR', 'GRP51 NR', 'GRP52 DPH',
           'GRP52 DPH+C7', 'GRP52 NR', 'I24D DPH', 'I24E DPH', 'I24K DPH',
           'No Pep DPH', 'No Pep DPH+C7', 'No Pep DPH+NR', 'No Pep NR',
           'No Pep None', 'RBBb001 DPH', 'RBBb031 DPH+NR', 'RBBb034 DPH',
           'RBBc028 DPH', 'RBBc029 NR', 'RBBc030 NR', 'RBBd005 DPH',
           'RBBd030 DPH+C7', 'RBBd031 DPH', 'RBBd031 DPH+NR', 'RBBd036 DPH',
           'RBBd036 NR', 'RBBd039 DPH+C7']

    if ('hue' in keyword_parameters.keys()):
        hue = keyword_parameters['hue']
        if ('hue_order' in keyword_parameters.keys()):
            hue_order = keyword_parameters['hue_order']
        else:
            hue_order = list(data[hue].unique())
        g = sns.boxplot(
            x='aHB', y='FLUORESCENCE', hue=hue,
            data=data.sort_values(by=[hue]),
            palette=palette,
            hue_order=hue_order,
            order=peptide_list)
    else:
        g = sns.boxplot(
            x='aHB', y='FLUORESCENCE',
            data=data,
            palette=palette,
            order=peptide_list)

    if ('ylim' in keyword_parameters.keys()):
        ylim = keyword_parameters['ylim']
        g.set_ylim(ylim)
    else:
        # g.set_ylim([0,2])
        pass
    g.set_xticklabels(labels=g.get_xticklabels(), rotation = 45, ha="right")
    plt.show()

    return fig

def plot_raw_data_superplots(
            queried_data,
            *positional_parameters,
            **keyword_parameters
            ):
    """
    Function to plot superplots of all barrel-dye combinations on one plot for
    the raw queried data. This allows the user to look at individual readings
    across a plate, instead of a median version. There is an optional input
    of 'hue' which users can then plot the data split up in various ways (i.e.
    ANALYTE_ID)

    INPUTS:
        - queried_data: The raw data extracted straight from the database. The
                output of processing.pull_data_from_db
        - hue: The variable which will colour the boxplot, i.e. ANALYTE_ID

    AUTHOR:
        Tania LaGambina
    """
    data = copy.deepcopy(queried_data)
    warnings.filterwarnings("ignore")

    data.rename(columns={'PLATE_ID':'EXPERIMENT_ID'}, inplace=True)
    data.replace({'Buffer':'No Pep',
                         'FRET C7':'DPH+C7',
                         'FRET NR':'DPH+NR',
                         'DPH + C7':'DPH+C7',
                         'DPH + NR':'DPH+NR'
                         }, inplace=True)
    data.drop_duplicates(['ROW_LOC', 'COLUMN_LOC', 'EXPERIMENT_ID'], keep='last', inplace=True)
    if ('peptide_list' in keyword_parameters.keys()):
        peptide_list = keyword_parameters['peptide_list']
    else:
        peptide_list = ['CC-Hept DPH', 'CC-Hex DPH', 'GRP35 DPH', 'GRP46 NR', 'GRP51 DPH',
           'GRP51 DPH+C7', 'GRP51 DPH+NR', 'GRP51 NR', 'GRP52 DPH',
           'GRP52 DPH+C7', 'GRP52 NR', 'I24D DPH', 'I24E DPH', 'I24K DPH',
           'No Pep DPH', 'No Pep DPH+C7', 'No Pep DPH+NR', 'No Pep NR',
           'No Pep None', 'RBBb001 DPH', 'RBBb031 DPH+NR', 'RBBb034 DPH',
           'RBBc028 DPH', 'RBBc029 NR', 'RBBc030 NR', 'RBBd005 DPH',
           'RBBd030 DPH+C7', 'RBBd031 DPH', 'RBBd031 DPH+NR', 'RBBd036 DPH',
           'RBBd036 NR', 'RBBd039 DPH+C7']
    f, ax = plt.subplots(figsize=(30, 10))
    data['aHB'] = data['PEPTIDE_ID']+' '+data['DYE_ID']


    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    elif ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        palette = sns.set_palette(sns.color_palette(colors))
    else:
        palette = 'magma'
    if ('hue' in keyword_parameters.keys()):
        hue = keyword_parameters['hue']
        if ('hue_order' in keyword_parameters.keys()):
            hue_order = keyword_parameters['hue_order']
        else:
            hue_order = list(data[hue].unique())
        ax = sns.swarmplot(
            x='aHB',
            y='FLUORESCENCE',
            hue=hue,
        #         kind='swarm',
            data = data,
            edgecolor='w',
            size=6,
            hue_order=hue_order,
            order=peptide_list,
            palette=palette,
            linewidth=1,
            ax=ax
        )
    else:
        ax = sns.swarmplot(
            x='aHB',
            y='FLUORESCENCE',
        #         kind='swarm',
            data = data,
            order = peptide_list,
            edgecolor='w',
            size=6,
            palette=palette,
            linewidth=1,
            ax=ax
        )
    if ('title' in keyword_parameters.keys()):
        title = keyword_parameters['title']
        ax.set_title(title)

    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={'visible': False},
        medianprops={'ls': '-', 'lw': 0.8},
        whiskerprops={'visible': False},
        zorder=10,
        showfliers=False,
        showbox=True,
        showcaps=True,
        capprops={'ls': '-', 'lw': 0.8},
        boxprops={'ls': '-', 'lw': 0.8},
        x='aHB',
        y='FLUORESCENCE',
        data = data,
        ax=ax,
        color='white',
        order=peptide_list
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha="right")

    ax.legend(handles[:6], labels[:6])

    return f

def meta_data_distribution_superplot(
        parsed_data,
        x,
        y,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    Function that will plot meta data distributions for a dataset. Suitable for
    use with e.g. AGE, SEX, BMI, DIAGNOSIS.

    INPUTS:
        - parsed_data: The dataset that contains the meta data
        - x: The value that will split the x axis. Has to be categorical
        - y: The value which is measured per x. Has to be continuous
        - hue: A categorical variable that can be used to colour the plot based
            on the value
        - boxplot: True or False, an arguement that will determine if a boxplot
            outline will be drawn on top of the swarmplot

    AUTHOR:
        Tania LaGambina
    """
    data = copy.deepcopy(parsed_data)

    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        f, ax = plt.subplots(figsize=figsize)
    else:
        f, ax = plt.subplots(figsize=(8, 5))

    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    else:
        palette = 'magma'

    if ('hue' in keyword_parameters.keys()):
        hue = keyword_parameters['hue']
        ax = sns.swarmplot(
                x=x,
                y=y,
                hue=hue,
                data = data,
                edgecolor='w',
                size=7,
                palette=palette,
                linewidth=1,
                ax=ax
            )
    else:
        ax = sns.swarmplot(
                x=x,
                y=y,
                data = data,
                edgecolor='w',
                size=7,
                palette=palette,
                linewidth=1,
                ax=ax
            )

    if ('boxplot' in keyword_parameters.keys()):
        boxplot = keyword_parameters['boxplot']
        if boxplot is True:
            sns.boxplot(
                showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'ls': '-', 'lw': 0.8},
                whiskerprops={'visible': False},
                zorder=10,
                showfliers=False,
                showbox=True,
                showcaps=True,
                capprops={'ls': '-', 'lw': 0.8},
                boxprops={'ls': '-', 'lw': 0.8},
                x=x,
                y=y,
                data=data,
                ax=ax,
                color='white'
            )
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha="right")
    ax.set_xticks([])

    ax.legend(handles[:6], labels[:6])

    plt.show()

    return ax

def meta_data_distribution_boxplot(
        parsed_data,
        x,
        y,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    Function that will plot meta data distributions for a dataset. Suitable for
    use with e.g. AGE, SEX, BMI, DIAGNOSIS.

    INPUTS:
        - parsed_data: The dataset that contains the meta data
        - x: The value that will split the x axis. Has to be categorical
        - y: The value which is measured per x. Has to be continuous
        - hue: A categorical variable that can be used to colour the plot based
            on the value
        - boxplot: True or False, an arguement that will determine if a boxplot
            outline will be drawn on top of the swarmplot. If not specified
            default = False
        - figsize: In the format (x, y), to change the shape of the plot. If not
            determined will go with a default value
        - palette: The color palette of the plot (defualt is magma)

    AUTHOR:
        Tania LaGambina
    """
    data = copy.deepcopy(parsed_data)

    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        f, ax = plt.subplots(figsize=figsize)
    else:
        f, ax = plt.subplots(figsize=(8, 5))

    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    elif ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        palette = sns.set_palette(sns.color_palette(colors))
    else:
        palette = 'magma'


    if ('hue' in keyword_parameters.keys()):
        hue = keyword_parameters['hue']
        if ('hue_order' in keyword_parameters.keys()):
            hue_order = keyword_parameters['hue_order']
        else:
            hue_order = list(data[hue].unique())
        ax = sns.boxplot(
                x=x,
                y=y,
                # hue=hue,
                data=data,
                hue_order=hue_order,
                palette=palette,
                order=hue_order,
                ax=ax
                # showmeans=True
            )
    else:

        ax = sns.boxplot(
                x=x,
                y=y,
                data = data,
                palette=palette,
                ax=ax
                # showmeans=True
            )


    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha="right")
    ax.legend([], [], frameon=False)


    plt.show()
    return f

def plot_boxplot_full_array(
        data,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Boxplot function that plots all peptide dye combinations for Array 2.0
    on one axis. This differs from the plot_boxplot function as it plots the
    all the data on one plot, and doesn't seperate between plots.

    INPUTS:
        data: The dataset wanting to be plotted. Usually parsed_data. Needs to be
            dataframe format. Only permits Array 2.0 data.
        hue: A keyword_parameter, determines the column in the dataframe that
            will be used to colour the boxplots on the figure
        palette: A predefined python palette input as a string. E.g. 'tab20'
        colors: A list of colors in hex format, e.g ['#7f7f7f', #000000]
        line: Boolean True or False. If True, plots a dashed line across y=1
        ylim: In format [0,2]. Used to determine the limits of the y axis
        title: String input. An optional title to the plot

    AUTHOR:
        Tania LaGambina
    """

    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(30, 10))

    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    elif ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        palette = sns.set_palette(sns.color_palette(colors))
    else:
        palette = 'magma'

    if ('peptide_list' in keyword_parameters.keys()):
        peptide_list = keyword_parameters['peptide_list']
    else:
        peptide_list = ['CC-Hept DPH', 'CC-Hex DPH', 'GRP35 DPH', 'GRP46 NR', 'GRP51 DPH',
           'GRP51 DPH+C7', 'GRP51 DPH+NR', 'GRP51 NR', 'GRP52 DPH',
           'GRP52 DPH+C7', 'GRP52 NR', 'I24D DPH', 'I24E DPH', 'I24K DPH',
           'No Pep DPH', 'No Pep DPH+C7', 'No Pep DPH+NR', 'No Pep NR',
           'RBBb001 DPH', 'RBBb031 DPH+NR', 'RBBb034 DPH',
           'RBBc028 DPH', 'RBBc029 NR', 'RBBc030 NR', 'RBBd005 DPH',
           'RBBd030 DPH+C7', 'RBBd031 DPH', 'RBBd031 DPH+NR', 'RBBd036 DPH',
           'RBBd036 NR', 'RBBd039 DPH+C7']

    if ('hue' in keyword_parameters.keys()):
        hue = keyword_parameters['hue']
        if ('hue_order' in keyword_parameters.keys()):
            hue_order = keyword_parameters['hue_order']
        else:
            hue_order = list(data[hue].unique())
        peptide_list.append(hue)
        mdf = pd.melt(
            data[peptide_list],
            id_vars=hue,
            var_name='x',
            value_name='FLUORESCENCE'
           )
        peptide_list.remove(hue)
        g = sns.boxplot(
            x='x',
            y='FLUORESCENCE',
            hue=hue,
            hue_order=hue_order,
            data=mdf.sort_values(
                by=[hue]), palette=palette, order=peptide_list)
    else:
        mdf = pd.melt(
            data[peptide_list],
            var_name='x',
            value_name='FLUORESCENCE'
           )
        g = sns.boxplot(
            x='x',
            y='FLUORESCENCE',
            data=mdf, palette=palette, order=peptide_list)

    if ('ylim' in keyword_parameters.keys()):
        ylim = keyword_parameters['ylim']
        g.set_ylim(ylim)
    else:
        g.set_ylim([0,2])

    if ('title' in keyword_parameters.keys()):
        title = keyword_parameters['title']
        g.set_title(title)

    if ('line' in keyword_parameters.keys()):
        line = keyword_parameters['line']
        if line is True:
            g.axhline(1, linestyle='--', color='#7f7f7f')

    if ('legend' in keyword_parameters.keys() and keyword_parameters['legend'] is False):
        plt.legend([],[], frameon=False)

    g.set_xticklabels(labels=g.get_xticklabels(), rotation = 45, ha="right")
    plt.show()

    return fig

def plot_superplot_full_array(
        data,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Boxplot function that plots all peptide dye combinations for Array 2.0
    on one axis. This differs from the plot_boxplot function as it plots the
    all the data on one plot, and doesn't seperate between plots.

    INPUTS:
        data: The dataset wanting to be plotted. Usually parsed_data. Needs to be
            dataframe format. Only permits Array 2.0 data.
        hue: A keyword_parameter, determines the column in the dataframe that
            will be used to colour the boxplots on the figure
        palette: A predefined python palette input as a string. E.g. 'tab20'
        colors: A list of colors in hex format, e.g ['#7f7f7f', #000000]
        line: Boolean True or False. If True, plots a dashed line across y=1
        ylim: In format [0,2]. Used to determine the limits of the y axis
        title: String input. An optional title to the plot

    AUTHOR:
        Tania LaGambina
    """
    warnings.filterwarnings("ignore")
    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        f, ax = plt.subplots(figsize=figsize)
    else:
        f, ax = plt.subplots(figsize=(30, 10))

    if ('ylim' in keyword_parameters.keys()):
        ylim = keyword_parameters['ylim']
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([0,2])

    if ('size' in keyword_parameters.keys()):
        size = keyword_parameters['size']
    else:
        size = 6

    if ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    elif ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        palette = sns.set_palette(sns.color_palette(colors))
    else:
        palette = 'magma'

    if ('peptide_list' in keyword_parameters.keys()):
        peptide_list = keyword_parameters['peptide_list']
    else:
        peptide_list = ['CC-Hept DPH', 'CC-Hex DPH', 'GRP35 DPH', 'GRP46 NR', 'GRP51 DPH',
           'GRP51 DPH+C7', 'GRP51 DPH+NR', 'GRP51 NR', 'GRP52 DPH',
           'GRP52 DPH+C7', 'GRP52 NR', 'I24D DPH', 'I24E DPH', 'I24K DPH',
           'No Pep DPH', 'No Pep DPH+C7', 'No Pep DPH+NR', 'No Pep NR',
           'RBBb001 DPH', 'RBBb031 DPH+NR', 'RBBb034 DPH',
           'RBBc028 DPH', 'RBBc029 NR', 'RBBc030 NR', 'RBBd005 DPH',
           'RBBd030 DPH+C7', 'RBBd031 DPH', 'RBBd031 DPH+NR', 'RBBd036 DPH',
           'RBBd036 NR', 'RBBd039 DPH+C7']

    if ('hue' in keyword_parameters.keys()):
        hue = keyword_parameters['hue']
        if ('hue_order' in keyword_parameters.keys()):
            hue_order = keyword_parameters['hue_order']
        else:
            hue_order = list(data[hue].unique())

        peptide_list.append(hue)
        mdf = pd.melt(
            data[peptide_list],
            id_vars=hue,
            var_name='aHB',
            value_name='FLUORESCENCE'
           )
        ax = sns.swarmplot(
            x='aHB',
            y='FLUORESCENCE',
            hue=hue,
            hue_order=hue_order,
            data=mdf,
            palette=palette,
            size=size,
            edgecolor='w',
            linewidth=1,
            ax=ax
               )
    else:
        mdf = pd.melt(
            data[peptide_list],
            var_name='aHB',
            value_name='FLUORESCENCE'
           )
        ax = sns.swarmplot(
            x='aHB',
            y='FLUORESCENCE',
        #         kind='swarm',
            edgecolor='w',
            size=size,
            linewidth=1,
            data=mdf,
            palette=palette,
            ax=ax
               )


    if ('title' in keyword_parameters.keys()):
        title = keyword_parameters['title']
        ax.set_title(title)

    if ('line' in keyword_parameters.keys()):
        line = keyword_parameters['line']
        if line is True:
            ax.axhline(1, linestyle='--', color='#7f7f7f')

    if ('legend' in keyword_parameters.keys()):
        legend = keyword_parameters['legend']
    else:
        legend = True
    # fig.set_size_inches(15,7)
    try:
        peptide_list.remove(hue)
    except:
        pass
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={'visible': False},
        medianprops={'ls': '-', 'lw': 0.8},
        whiskerprops={'visible': False},
        zorder=10,
        showfliers=False,
        showbox=True,
        showcaps=True,
        capprops={'ls': '-', 'lw': 0.8},
        boxprops={'ls': '-', 'lw': 0.8},
        x='aHB',
        y='FLUORESCENCE',
        data = mdf,
        ax=ax,
        color='white',
        order=peptide_list
    )
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation = 45, ha="right")

    if ('fontsize' in keyword_parameters.keys()):
        fontsize = keyword_parameters['fontsize']
    else:
        fontsize = 15

    ax.set_xlabel(xlabel='aHB', fontsize=30)
    if ('ylabel' in keyword_parameters.keys()):
        ylabel = keyword_parameters['ylabel']
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_ylabel(ylabel='Fluorescence', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    if legend is False:
        plt.legend([],[], frameon=False)

    plt.show()

    return f

def make_report(
        parsed_data,
        path,
        conn,
        project,
        title,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Function to create a pdf report at the end of a data analysis workbook.
    The report outlines key data visualisations and results for a particular
    analysis.

    INPUTS:
        parsed_data: The parsed_data dataframe used in the analysis notebook
        path: The path to the folder where all of the plots have been saved
            for the notebook
        conn: The database connection object
        project: A string that defines the overall project the analysis is part
            of, this will be placed at the header
        title: A string that is the title of the particular analysis

    AUTHOR:
        Tania LaGambina
    """
    # Extra information extraction
    today = date.today()
    todaydate = today.strftime("%B %d, %Y")
    parsed_data = processing.experiment_data_from_db(conn, parsed_data, 'peptide_layout_id', 'EXPERIMENT_ID')
    array = parsed_data.PEPTIDE_LAYOUT_ID.unique()[0]
    replicates = parsed_data.groupby(['ANALYTE_ID']).count().max()[0]
    parsed_data = processing.plate_data_from_db(conn, parsed_data, 'plate_made', 'EXPERIMENT_ID')
    number_of_batches = len(parsed_data.PLATE_MADE.unique())
    batches = parsed_data.PLATE_MADE.unique()


    class PDF(FPDF):
        # def header(self):
            # self.image('/Users/tanialagambina/Documents/Rosa Biotech logo M.png', x=170, y=8, w=30)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', '', 8)
            self.cell(0, 10, str(self.page_no()), 0, 0, 'C')
    pdf = PDF()
    # pdf.add_font('Arial Nova Light', '', "/Users/tanialagambina/anaconda3/lib/python3.10/site-packages/fpdf/Arial Nova Light.ttf", uni=True)

    pdf.add_page()
    #----------------------------------------------------------------------------------------------------------------------
    # Page 1

    pdf.set_font('Arial', '', 16)
    pdf.cell(40, 10, project, ln=1)
    # pdf.ln(1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(40, 5, '{} Report'.format(title), ln=1)
    # pdf.ln(1)
    pdf.set_font('Arial','', 10)
    if ('report_description' in keyword_parameters.keys()):
        report_description = keyword_parameters['report_description']
        pdf.cell(40, 5, report_description, ln=1)
        pdf.ln(2)
    else:
        pass
    pdf.cell(40, 5, '{}'.format(todaydate), ln=1)
    pdf.ln(2)
    pdf.set_font('Arial','', 10)

    pdf.cell(40, 5, 'The analyte IDs used in this analysis were:', ln=1)
    # pdf.ln(2)

    pdf.set_font('Courier','', 4)
    pdf.ln(1)

    f = open(path+"Analyte Info.txt", "r")
    for x in f:
        pdf.multi_cell(0, 4, txt = x)
    pdf.ln(1)
    pdf.set_font('Arial','', 10)
    pdf.cell(30, 5, 'Data visualisations on the full dataset', ln=1)
    # pdf.ln(3)
    pdf.set_font('Courier','', 8)

    # pdf.set_xy(x=10, y= 130)
    f = open(path+"Sample Number.txt", "r")
    n = 3
    ypos = pdf.get_y()
    for x in f:
        pdf.set_xy(x=15, y= ypos+n)
        n = n+3
        pdf.cell(0, 4, txt = x)
    pdf.ln(1)
    ypos = pdf.get_y()
    try:
        pdf.image(path+'AGE Boxplot.png', y=ypos+10, x=10, w=25)
        pdf.image(path+'BMI Boxplot.png', y=ypos+10, x=36,  w=25)
    except:
        pass
    # pdf.ln(5)
    # pdf.image(path+'Median Full Array Boxplot.png', y = 101, x = 64, w=130)
    pdf.image(path+'Median Full Array Boxplot.png', x = 64, w=130)
    pdf.ln(2)
    # pdf.image(path+'Median Full Array Superplot.png', w=205, y=190, x=3)
    pdf.image(path+'Median Full Array Superplot.png', w=190, x=10)
    pdf.set_font('Arial','', 10)
    # pdf.set_xy(x=10, y= 215)
    pdf.set_x(x=10)
    pdf.cell(17,10, '')
    pdf.cell(40,10, 'Full Dataset PCA')
    pdf.cell(18,10, '')
    pdf.cell(40,10, 'Full Dataset Feature Ranking')
    pdf.cell(23,10, '')
    pdf.cell(40,10, 'Top 3 Features Superplot')
    # pdf.image(path+'Full Dataset PCA Scatter.png', y= 225, x = 10, w=75) #If i define just width or height, will it automatically calc
    ypos = pdf.get_y()
    pdf.set_y(y=ypos+10)
    pdf.image(path+'Full Dataset PCA Scatter.png', x = 10, w=75) #If i define just width or height, will it automatically calc

    # pdf.ln(5)
    pdf.set_font('Courier','', 8)
    # ypos = pdf.get_y()
    # pdf.set_xy(x=100, y= 200)
    f = open(path+"Full Dataset Ranking.txt", "r")
    n = 7.5
    for x in f:
    #     pdf.set_xy(x=85, y= 220+n)
        pdf.set_xy(x=85, y= ypos+n)
        n = n+2.5
        pdf.cell(0, 4, txt = x)
    # pdf.ln(5)
    # pdf.image(path+'Top 3 Features Superplot.png', w=80, x=125, y=233)
    pdf.set_y(y=ypos+15)
    pdf.image(path+'Top 3 Features Superplot.png', w=76, x=125)

    pdf.ln(5)
    pdf.add_page()
    #------------------------------------------------------------------------------------------------------------------
    # Page 2

    pdf.ln(10)
    pdf.set_font('Arial','', 12)
    pdf.cell(40, 5, 'Model Training')
    pdf.ln(5)
    pdf.set_font('Arial','', 10)
    pdf.cell(40, 5, 'Train dataset top 3 features and train dataset PCA for optimimum number of input features', ln=1)
    # pdf.ln(10)
    pdf.set_font('Courier','', 8)
    pdf.image(path+'Top 3 Features Pairplot.png', w=75, y=37, x=25)
    try:
        pdf.image(path+'Train Dataset PCA Scatter.png', w=75, y=42, x=100)
    except:
        pass
    pdf.ln(2)
    pdf.image(path+'Model Score vs Number of Features.png', x=30, y=105, h=55)
    # pdf.ln(5)
    pdf.set_xy(x=10, y=160)
    pdf.set_font('Arial','', 10)

    pdf.cell(40,10, 'Best Model Results')

    pdf.image(path+'Confusion Matrix.png', x=30, y=170, h=45)
    try:
        pdf.image(path+'ROC Curve.png', x=100, y=170, h=45)
    except:
        pass

    # pdf.cell(20,10, path+"Model Results.txt")
    pdf.set_font('Courier','', 7)

    f = open(path+"Model Results.txt", "r")
    pdf.set_xy(x=10, y=220)
    for x in f:
        pdf.cell(0, 4, txt = x, ln = 1)

    pdf.add_page()
    #------------------------------------------------------------------------------------------------------------------
    # Page 3
    pdf.set_font('Arial','', 10)
    pdf.ln(5)
    pdf.cell(40, 10, 'Experiment Information')
    pdf.ln(10)
    pdf.set_font('Courier','', 8)

    if array == 'N':
        array = 'Array 2.1'
    elif array == 'H':
        array = 'Array 2.0'
    else:
        pass

    pdf.cell(40, 10, 'Array used: {}'.format(array))
    pdf.ln(5)
    pdf.cell(40, 10, 'Number of replicates: {}'.format(replicates))
    pdf.ln(5)
    pdf.cell(40, 10, 'Number of batches: {}'.format(number_of_batches))
    pdf.ln(5)
    pdf.cell(40, 10, 'Batches: {}'.format(batches))

    pdf.output(path+'{} Report.pdf'.format(title), 'F')
