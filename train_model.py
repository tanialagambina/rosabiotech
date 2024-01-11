"""
********************************************************************************
                            Machine Learning Package
********************************************************************************
This package contains the class RunML which contains all of the functions that
are required within our analysis to train a variety of different models on our
dataset, as well as providing functions that perform dimensionality reduction
(principal component analysis for example) and feature importance analysis.

Functions & classes included within this package:
    - split_train_test_data
    - calc_feature_correlations
********************************************************************************
"""

import numpy as np
import os
import copy
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score, LeaveOneOut, GroupKFold, KFold, RepeatedKFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math
from math import sqrt
from imblearn.pipeline import Pipeline

def split_train_test_data(
        data,
        test_analytes_list,
        test_seen,
        target_col,
        specific_analytes_col,
        shuffle
        ):
    """
    Function that splits the data into the training and test sets

    INPUTS:
    - data: The data which you wish to split. Usually standardised_parsed_data
    - test_analytes_list: If randomise is set to False, this argument lists the
            names of the data points to be set aside for model testing
    - test_seen: Boolean input. False if the test set removes an
            independent reading set, and True if a single reading from a
            set of repeats is used for the test set. I.e. it is a seen
            variable
    - target_col: The target data which we want to train our models towards.
            The input should be a string indicating the column in the
            dataframe where the target data is contained. I.e. 'Analyte'
    - specific_analytes_col: The column that contains the most fine grained
            detail about the different analytes, if needed for seperation

    OUTPUTS:

    AUTHOR:
        Based on original by Kathryn Shelley, Woolfson Group
        Developed for Rosa Biotech by Tania LaGambina
    """
    num_data_points = data.shape[0]

    test_set = []
    train_set = []
    test_data = []
    split_data = {}
    if shuffle is True:
        data = data.iloc[np.random.permutation(data.index)].reset_index(drop=True)
    else:
        pass
    if test_seen is False: # Mostly in this group, ensures the same analyte ids are not being included in train and test sets
        test_data = copy.deepcopy(data.loc[
            data[specific_analytes_col].isin(test_analytes_list)].reset_index(drop=True))
        train_data = copy.deepcopy(data.loc[
            data[specific_analytes_col].isin(test_analytes_list)==False].reset_index(drop=True))
        split_data['cv'] = GroupKFold(n_splits=len(train_data[target_col].unique()))
        if len(test_data) == 0:
            raise ValueError('No analytes have been selected for the test set: Change value of test_analytes_list')
    else:
        num_points = round((len(data)*0.3)/len(data[specific_analytes_col].unique()))
        num_points = 0
        test_analytes_list = data[specific_analytes_col].unique().tolist()
        for string in test_analytes_list:
            test_dat = copy.deepcopy(data.loc[data[specific_analytes_col]==string])
            test_dat.reset_index(drop=True)
            test_data.append(test_dat.iloc[0:num_points, :])
        test_data = pd.concat(test_data, axis=0)

        all_data = copy.deepcopy(data.merge(test_data.drop_duplicates(), how='left', indicator=True))
        train_data = copy.deepcopy(all_data.loc[all_data['_merge']=='left_only'])
        train_data.drop(columns=['_merge'], inplace=True)
        split_data['cv'] = KFold(n_splits=4)

    tnd = copy.deepcopy(train_data.drop(target_col, axis=1))
    split_data['x_train'] = copy.deepcopy(
        tnd.select_dtypes(['number'])).to_numpy()
    tsd = copy.deepcopy(test_data.drop(target_col, axis=1))
    split_data['x_test'] = copy.deepcopy(
        tsd.select_dtypes(['number'])).to_numpy()

    split_data['y_train'] = np.array(
        train_data[target_col].tolist())
    split_data['y_test'] = np.array(
        test_data[target_col].tolist())
    split_data['analytes_test'] = np.array(
        test_data[specific_analytes_col].to_list()
    )

    return split_data


def calc_feature_correlations(data, *positional_parameters, **keyword_parameters):
    """
    Function to calculate the feature correlations between the different
    input features, and in this case, the fluorescence responses from each
    of the peptides.
    This is important as we do not wish for our input features for ML to be
    highly correlated. This is due to the fact many models assume that
    features are independant, and if they are not it would be a problem/

    INPUTS:
    - data: The data set that contains the fluourescence values and
                their corresponding class labels for the defined
                analytes

    OUTPUTS:
    - fig: The resulting correlation matrix which can then be saved outside
        the function

    AUTHOR:
        Tania LaGambina
    """
    feature_corr_df = data.select_dtypes(['number']) # the dataframe may include non numeric values like ids
    correlation_matrix = feature_corr_df.corr(method='pearson')

    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
    else:
        figsize = [40,30]

    if ('annot' in keyword_parameters.keys()):
        annot = keyword_parameters['annot']
    else:
        annot = True
    if ('cmap' in keyword_parameters.keys()):
        cmap = keyword_parameters['cmap']
    else:
        cmap = 'afmhot'

    if ('vmin' in keyword_parameters.keys()):
        vmin = keyword_parameters['vmin']
    else:
        vmin=0
    if ('vmax' in keyword_parameters.keys()):
        vmax = keyword_parameters['vmax']
    else:
        vmax=1

    fig = plt.figure(figsize=figsize)
    if ('absolute_values' in keyword_parameters.keys()):
        if keyword_parameters['absolute_values'] is False:
            sns.heatmap(
                data=correlation_matrix,
                vmin=vmin,
                vmax=vmax,
                square=True,
                annot=annot,
                cmap=cmap
                )
    else:
        sns.heatmap(
            data=correlation_matrix.abs(), # we don't really care about positive or negative correlations, only the strength of the correlation
            vmin=vmin,
            vmax=vmax,
            square=True,
            annot=annot,
            cmap=cmap
            )
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.show()
    sns.reset_defaults()

    return fig

def calc_feature_correlations_relplot(data):
    """
    Function to calculate the feature correlations between the different
    input features, and in this case, the fluorescence responses from each
    of the peptides.
    This is important as we do not wish for our input features for ML to be
    highly correlated. This is due to the fact many models assume that
    features are independant, and if they are not it would be a problem.
    This function plots the correlations in the form of a relplot instead of the
    usual heatmap. This uses sizes of points on the plot to convey the strength
    of the correlation, as well as the colour of these points

    INPUTS:
    - data: The data set that contains the fluourescence values and
                their corresponding class labels for the defined
                analytes

    OUTPUTS:
    - fig: The figure which can then be saved outside of the function in the
        desired location

    AUTHOR:
        Tania LaGambina
    """
    feature_corr_df = data.select_dtypes(['number'])
    correlation_matrix = feature_corr_df.corr(method='pearson').abs().stack().reset_index(name="correlation") # for a relplot, the correlation matrix needs to be in long form
    sns.set_theme(style="whitegrid")

    g = sns.relplot(
        data=correlation_matrix,
        x="level_0",
        y="level_1",
        hue="correlation",
        size="correlation",
        palette="vlag",
        hue_norm=(0, 1),
        edgecolor=".7",
        height=10,
        legend=False,
        sizes=(50, 250),
        size_norm=(-0, 0.8)
        )
    g.set(xlabel="", ylabel="", aspect="equal")
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.show()
    fig = g.ax.get_figure()
    sns.reset_defaults()

    return fig

def pca_scatter(
        pca_x_data,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    This function plots the components of PCA against each other, and
    colours the points depending on the value of the y_data. This allows
    the user to then clearly see whether there is any seperation between the
    different classes from transforming the data

    INPUTS:
    - pca_x_data: The transformed numeric x data. This can be either pca_x_train,
        pca_x_test, or a combination of the both
    - y_data: The corresponding target data associated with the pca_x_data
    - colors: A list of colors for the different groups on the PCA. E.g.
        colors = ["#B0ABFB", "#5D53EA", "#1B0FC2"]
    - hue_order: The list of unique y_data values, in the order you want them
        to be displayed (can be used to match color to label)

    AUTHOR:
        Tania LaGambina

    """
    df = pd.DataFrame(
        data=pca_x_data[0:,0:],
        index=[i for i in range(pca_x_data.shape[0])],
        columns=['Component '+str(i+1) for i in range(pca_x_data.shape[1])]
        )

    if ('y_data' in keyword_parameters.keys()):
        y_data = keyword_parameters['y_data']
        y_df = pd.DataFrame(
            data=y_data[0:],
            columns=['Target']
        )
        y_df.reset_index(drop=True, inplace=True)
        pca_scatter_df = pd.concat([df, y_df], axis=1)
    else:
        pca_scatter_df = df

    if ('donor' in keyword_parameters.keys()):
        donor = keyword_parameters['donor']
        donor_df = pd.DataFrame(
            data=donor[0:],
            columns=['Donor']
            )
        donor_df.reset_index(drop=True, inplace=True)
        pca_scatter_df = pd.concat([pca_scatter_df, donor_df], axis=1)
    else:
        pca_scatter_df = pca_scatter_df

    # pca_scatter_df = pd.concat([df, y_df], axis=1)
    plt.clf()
    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        fig, ax = plt.subplots(
            figsize=figsize
            )
    else:
        fig, ax = plt.subplots(
            figsize=(6, 6)
            )
    sns.set(style="white")
    sns.set_context("paper", rc={
        "font.size":17,
        "axes.titlesize":17,
        "axes.labelsize":17,
        "xtick.labelsize":15,
        "ytick.labelsize":15
    })
    # labels = str(y_df['Target'].unique())

    if ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        if len(colors) > 1:
            customPalette = sns.set_palette(sns.color_palette(colors))
        else:
            customPalette = colors
    elif ('palette' in keyword_parameters.keys()):
        customPalette = keyword_parameters['palette']
    else:
        customPalette = 'magma'

    if ('hue_order' in keyword_parameters.keys()):
        hue_order = keyword_parameters['hue_order']
    else:
        if ('y_data' in keyword_parameters.keys()):
            hue_order = np.unique(y_df.Target)
        else:
            pass

    if ('size' in keyword_parameters.keys()):
        size = keyword_parameters['size']
    else:
        size = 450

    if ('donor' in keyword_parameters.keys()):
        if ('style_order' in keyword_parameters.keys()):
            style_order = keyword_parameters['style_order']
        else:
            style_order = np.unique(donor_df.Donor)
        if ('y_data' in keyword_parameters.keys()):
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                hue_order = hue_order,
                hue="Target",
                style="Donor",
                style_order=style_order,
                s=size,
                ax=ax,
                data=pca_scatter_df,
                palette=customPalette
                            )
        else:
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                style="Donor",
                style_order=style_order,
                s=size,
                ax=ax,
                data=pca_scatter_df
                # palette=customPalette
                            )
    else:
        if ('y_data' in keyword_parameters.keys()):
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                hue_order = hue_order,
                hue="Target",
                s=size,
                ax=ax,
                data=pca_scatter_df,
                palette=customPalette
                )
        else:
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                # hue_order = hue_order,
                # hue="Target",
                s=size,
                ax=ax,
                data=pca_scatter_df
                # palette=customPalette
                )
#     ax.axis('square')
#     ax.set_ylim(min(pca_scatter_df["Component 2"])-(0.1*max(pca_scatter_df["Component 2"])), max(pca_scatter_df["Component 2"])+(0.1*max(pca_scatter_df["Component 2"])))
#     ax.set_xlim(min(pca_scatter_df["Component 1"])-(0.1*max(pca_scatter_df["Component 1"])), max(pca_scatter_df["Component 1"])+(0.1*max(pca_scatter_df["Component 1"])))

    if ('title' in keyword_parameters.keys()):
        title = keyword_parameters['title']
        ax.set_title(title)
    else:
        pass
#     ax.set_aspect('equal', 'datalim')
    ax.margins(0.1)

#     ax.axis('square')
    if ('y_data' in keyword_parameters.keys()):
        h,l = ax.get_legend_handles_labels()
        leg = ax.legend(
            h[0:len(hue_order)+1],
            l[0:len(hue_order)+1],
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=17,
            ncol=1,
            markerscale=3
            )

    plt.show()
    # sns.reset_defaults()

    return fig

def lda_scatter(
        lda_x_data,
        y_data,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    This function plots the components of PCA against each other, and
    colours the points depending on the value of the y_data. This allows
    the user to then clearly see whether there is any seperation between the
    different classes from transforming the data

    INPUTS:
    - pca_x_data: The transformed numeric x data. This can be either pca_x_train,
        pca_x_test, or a combination of the both
    - y_data: The corresponding target data associated with the pca_x_data
    - colors: A list of colors for the different groups on the PCA. E.g.
        colors = ["#B0ABFB", "#5D53EA", "#1B0FC2"]
    - hue_order: The list of unique y_data values, in the order you want them
        to be displayed (can be used to match color to label)

    AUTHOR:
        Tania LaGambina
    """
    df = pd.DataFrame(
        data=lda_x_data[0:,0:],
        index=[i for i in range(lda_x_data.shape[0])],
        columns=['Linear Discriminant '+str(i+1) for i in range(lda_x_data.shape[1])]
        )
    y_df = pd.DataFrame(
        data=y_data[0:],
        columns=['Target']
    )
    y_df.reset_index(drop=True, inplace=True)
    if ('donor' in keyword_parameters.keys()):
        donor = keyword_parameters['donor']
        donor_df = pd.DataFrame(
            data=donor[0:],
            columns=['Donor']
            )
        donor_df.reset_index(drop=True, inplace=True)
        lda_scatter_df = pd.concat([df, y_df, donor_df], axis=1)
    else:
        lda_scatter_df = pd.concat([df, y_df], axis=1)

    # pca_scatter_df = pd.concat([df, y_df], axis=1)
    plt.clf()
    fig, ax = plt.subplots(
        figsize=(6, 6)
        )
    sns.set(style="white")
    sns.set_context("paper", rc={
        "font.size":17,
        "axes.titlesize":17,
        "axes.labelsize":17,
        "xtick.labelsize":15,
        "ytick.labelsize":15
    })
    labels = str(y_df['Target'].unique())

    if ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        if len(colors) > 1:
            customPalette = sns.set_palette(sns.color_palette(colors))
        else:
            customPalette = colors
    elif ('palette' in keyword_parameters.keys()):
        customPalette = keyword_parameters['palette']
    else:
        customPalette = 'magma'

    if ('hue_order' in keyword_parameters.keys()):
        hue_order = keyword_parameters['hue_order']
    else:
        hue_order = np.unique(y_df.Target)

    if ('donor' in keyword_parameters.keys()):
        if ('style_order' in keyword_parameters.keys()):
            style_order = keyword_parameters['style_order']
        else:
            style_order = np.unique(donor_df.Donor)
        sns.scatterplot(
            x="Linear Discriminant 1",
            y="Linear Discriminant 2",
            hue_order = hue_order,
            hue="Target",
            style="Donor",
            style_order=style_order,
            s=450,
            ax=ax,
            data=lda_scatter_df,
            palette=customPalette
                        )
    else:
        sns.scatterplot(
            x="Linear Discriminant 1",
            y="Linear Discriminant 2",
            hue_order = hue_order,
            hue="Target",
            s=450,
            ax=ax,
            data=lda_scatter_df,
            palette=customPalette
            )
    # ax.axis('square')
    ax.set_ylim(min(lda_scatter_df["Linear Discriminant 2"])-1, max(lda_scatter_df["Linear Discriminant 2"])+1)
    ax.set_xlim(min(lda_scatter_df["Linear Discriminant 1"])-1, max(lda_scatter_df["Linear Discriminant 1"])+1)
    # ax.axis('square')
    # plt.legend(title='Target', labels=hue_order)
    # ax.set_aspect('datalim')
    # ax.axis('square')

    # ax.margins(1)
    h,l = ax.get_legend_handles_labels()
    leg = ax.legend(
        h[0:len(hue_order)+1],
        l[0:len(hue_order)+1],
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=17,
        ncol=1,
        markerscale=3
        )
    # if ('save' in keyword_parameters.keys()):
    #     if keyword_parameters['save'] is True:
    #         cwd = os.getcwd()
    #         path = cwd+'/Results/'
    #         os.makedirs(os.path.dirname(path), exist_ok=True)
    #         fig.savefig(path+'{} {} features LDA plot.png'.format(labels, str(len(self.x_train[0]))), bbox_inches='tight')
    #     else:
    #         pass
    # else:
    #     pass

    plt.show()
    # sns.reset_defaults()
    return fig

def run_pca_and_transform(x, *positional_parameters, **keyword_parameters):
    """
    The PCA function to output the transformed x input variables into PCA
    transformed data

    AUTHOR:
        Tania LaGambina
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    if ('seed' in keyword_parameters.keys()):
        random_state = keyword_parameters['seed']
    else:
        random_state = 42

    pipe = Pipeline([
            ('scaling', StandardScaler()),
            ('dimension_reduction', PCA(n_components=0.95, random_state=random_state))])
    pipe.fit(x)
    pca_x = pipe.transform(x)

    return pca_x

def run_pca_and_transform_traintest(x_train, x_test, *positional_parameters, **keyword_parameters):
    """
    The PCA function to output the transformed x input variables into PCA
    transformed data

    AUTHOR:
        Tania LaGambina
    """

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    if ('seed' in keyword_parameters.keys()):
        random_state = keyword_parameters['seed']
    else:
        random_state = 42
    pipe = Pipeline([
            ('scaling', StandardScaler()),
            ('dimension_reduction', PCA(n_components=0.95, random_state=random_state))])
    pipe.fit(x_train)
    pca_x_train = pipe.transform(x_train)
    pca_x_test = pipe.transform(x_test)

    return pca_x_train, pca_x_test

def run_lda_and_transform(x, y):
    """
    AUTHOR:
        Tania LaGambina
    """

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    pipe = Pipeline([
            ('scaling', StandardScaler()),
            ('dimension_reduction', LDA())])

    lda_x = pipe.fit_transform(x, y)

    return lda_x

def run_lda_and_transform_traintest(x_train, x_test, y_train):
    """
    AUTHOR:
        Tania LaGambina
    """

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import StandardScaler
    from imblearn.pipeline import Pipeline
    pipe = Pipeline([
            ('scaling', StandardScaler()),
            ('dimension_reduction', LDA())])

    # model.fit(x, y)
    lda_x_train = pipe.fit_transform(x_train, y_train)
    lda_x_test = pipe.transform(x_test)

    return lda_x_train, lda_x_test

def pca_scatter_labelled(
    pca_x_data,
    # y_data,
    id_data,
    *positional_parameters,
    **keyword_parameters
    ):
    """
    A version of the pca_scatter function which labels the individual data
    points depending on the labels you require, i.e. Analyte_ID

    INPUTS:
    - pca_x_data: The transformed pca_x_data, which can be taken from the
            run_pca_and_transform function. In an array form
    - y_data: The target data associated with the pca_x_data (this is the
            same target data that corresponds to x_data, the pca transformation
            doesn't change this). In an array form
    - id_data: The id_data that you wish the datapoints to be labelled by. In
            an array form. (To get this from a dataframe, can do e.g.
            df[['ANALYTE_ID']].values, if the id_data you wish to use is
            ANALYTE_ID. But it can be anything!)

    OUTPUTS:
    - fig: The scatterplot as a fig available to save outside of the function

    AUTHOR:
        Tania LaGambina
    """

    df = pd.DataFrame(
        data=pca_x_data[0:,0:],
        index=[i for i in range(pca_x_data.shape[0])],
        columns=['Component '+str(i+1) for i in range(pca_x_data.shape[1])]
        )
    id_df = pd.DataFrame(
        data=id_data[0:],
        columns=['ID']
        )
    id_df.reset_index(drop=True, inplace=True)

    if ('y_data' in keyword_parameters.keys()):
        y_data = keyword_parameters['y_data']
        y_df = pd.DataFrame(
            data=y_data[0:],
            columns=['Target']
        )
        y_df.reset_index(drop=True, inplace=True)
        pca_scatter_df = pd.concat([df, y_df, id_df], axis=1)
    else:
        pca_scatter_df = pd.concat([df, id_df], axis=1)

    if ('donor' in keyword_parameters.keys()):
        donor = keyword_parameters['donor']
        donor_df = pd.DataFrame(
            data=donor[0:],
            columns=['Donor']
            )
        donor_df.reset_index(drop=True, inplace=True)
        pca_scatter_df = pd.concat([pca_scatter_df, donor_df, id_df], axis=1)
    else:
        pca_scatter_df = pd.concat([df, id_df], axis=1)

    # pca_scatter_df = pd.concat([df, y_df], axis=1)
    plt.clf()
    if ('figsize' in keyword_parameters.keys()):
        figsize = keyword_parameters['figsize']
        fig, ax = plt.subplots(
            figsize=figsize
            )
    else:
        fig, ax = plt.subplots(
            figsize=(6, 6)
            )
    sns.set(style="white")
    sns.set_context("paper", rc={
        "font.size":17,
        "axes.titlesize":17,
        "axes.labelsize":17,
        "xtick.labelsize":15,
        "ytick.labelsize":15
    })
    # labels = str(y_df['Target'].unique())

    if ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        if len(colors) > 1:
            customPalette = sns.set_palette(sns.color_palette(colors))
        else:
            customPalette = colors
    elif ('palette' in keyword_parameters.keys()):
        customPalette = keyword_parameters['palette']
    else:
        customPalette = 'magma'

    if ('hue_order' in keyword_parameters.keys()):
        hue_order = keyword_parameters['hue_order']
    else:
        if ('y_data' in keyword_parameters.keys()):
            hue_order = np.unique(y_df.Target)
        else:
            pass

    if ('size' in keyword_parameters.keys()):
        size = keyword_parameters['size']
    else:
        size = 450

    if ('donor' in keyword_parameters.keys()):
        if ('style_order' in keyword_parameters.keys()):
            style_order = keyword_parameters['style_order']
        else:
            style_order = np.unique(donor_df.Donor)
        if ('y_data' in keyword_parameters.keys()):
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                hue_order = hue_order,
                hue="Target",
                style="Donor",
                style_order=style_order,
                s=size,
                ax=ax,
                data=pca_scatter_df,
                palette=customPalette
                            )
        else:
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                style="Donor",
                style_order=style_order,
                s=size,
                ax=ax,
                data=pca_scatter_df
                # palette=customPalette
                            )
    else:
        if ('y_data' in keyword_parameters.keys()):
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                hue_order = hue_order,
                hue="Target",
                s=size,
                ax=ax,
                data=pca_scatter_df,
                palette=customPalette
                )
        else:
            sns.scatterplot(
                x="Component 1",
                y="Component 2",
                # hue_order = hue_order,
                # hue="Target",
                s=size,
                ax=ax,
                data=pca_scatter_df
                # palette=customPalette
                )
    for i in range(pca_scatter_df.shape[0]):
        plt.text(
            x=pca_scatter_df["Component 1"][i],
            y=pca_scatter_df["Component 2"][i],
            s=pca_scatter_df.ID[i],
            fontdict=dict(size=11)
        )
    if ('title' in keyword_parameters.keys()):
        title = keyword_parameters['title']
        ax.set_title(title)
    else:
        pass
#     ax.set_aspect('equal', 'datalim')
    ax.margins(0.1)

#     ax.axis('square')
    if ('y_data' in keyword_parameters.keys()):
        h,l = ax.get_legend_handles_labels()
        leg = ax.legend(
            h[0:len(hue_order)+1],
            l[0:len(hue_order)+1],
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=17,
            ncol=1,
            markerscale=3
            )

    plt.show()
    # sns.reset_defaults()
    return fig


class RunML():

    def __init__(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        technique,
        cv,
        *positional_parameters,
        **keyword_parameters
        ):

        """
        Initialisaing the data that will be used to run ML
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.train_groups = y_train
        self.test_groups = y_test
        self.technique = technique
        self.cv = cv

        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass

        if ('random_state' in keyword_parameters.keys()):
            self.random_state = keyword_parameters['random_state']
        elif ('random_seed' in keyword_parameters.keys()):
            self.random_state = keyword_parameters['random_seed']
        elif ('seed' in keyword_parameters.keys()):
            self.random_state = keyword_parameters['seed']
        else:
            self.random_state = 42

    def run_algorithm(
            self,
            clf,
            run,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        Wrapper function containing parameter search and model training and
        testing. It allows the user to run different parts of the package
        algorithms by specifying the value of 'run'.

        INPUTS:
        - clf: Classifier of interest.
        - run: The type of run that you want to use the function for. Either
                to use it as a parameter search function ('randomsearch' and
                'gridsearch') or a training function 'train'.

        OUTPUTS:
        - search: The machine learning model and its parameters

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        if run == 'randomsearch':
            self.define_model_params(clf)
            search = self.run_randomized_search(clf)
            return search
        elif run == 'gridsearch':
            self.define_model_params(clf)
            search = self.run_grid_search(clf)
            return search
        elif run == 'train':
            if ('save' in keyword_parameters.keys()):
                save = keyword_parameters['save']
                if (save is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    search = self.train_model(clf, save=save, path=path)
                    accuracy, recall, precision, clf = self.test_model(search, save=save, path=path)
                else:
                    search = self.train_model(clf)
                    accuracy, recall, precision, clf = self.test_model(search)
            else:
                search = self.train_model(clf)
                accuracy, recall, precision, clf = self.test_model(search)
            return search, clf

        return search, clf

    def define_model_params(
            self,
            clf
            ):
        """
        Sets parameters to try with RandomizedSearchCV and GridSearchCV. Each
        different machine learning model has a set of hyperparameters
        associated with it that can be optimised, and this function gives the
        models a range to focus this optimisation

        INPUTS:
        - clf: The machine learning model object defined by the user

        OUTPUTS:
        - params: The defined parameters for the machine learning model to
                try within its optimisation

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        if type(clf['model']).__name__ == 'LogisticRegression':
            params = {
                'model__random_state':[self.random_state],
                'model__penalty': ['none', 'l2', 'l1', 'elasticnet'],
                'model__solver': ['sag', 'saga', 'newton-cg', 'lbfgs', 'liblinear'],
                'model__multi_class': ['ovr', 'multinomial'],
                'model__C': [100, 10, 1.0, 0.1, 0.01]
                }
            self.params = params
        elif type(clf['model']).__name__ == 'KNeighborsClassifier':
            neighbours = [3, 5, 7, 9]
            params = {
                'model__n_neighbors': list(range(3,21)),
                # 'model__leaf_size': list(range(1,50)),
                'model__weights': ['uniform', 'distance'],
                'model__metric' : ['minkowski','euclidean','manhattan'],
                'model__p': [1, 2]
                }
            self.params = params
        elif type(clf['model']).__name__ == 'SVC':
            params = {
                'model__random_state':[self.random_state],
                'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'model__degree': [0, 1, 2, 3, 4],
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': [0.001, 0.0001]

                }
            self.params = params
        elif type(clf['model']).__name__ == 'RandomForestClassifier':
            n_estimators = [int(x) for x in np.linspace(3, 200, 20)]
            min_samples_split = [2, 5, 10]
            min_samples_leaf = [1, 2, 4]
            params = {
                'model__bootstrap':[True, False],
                'model__n_estimators': n_estimators,
                'model__random_state':[self.random_state],
                'model__max_features': ['auto', 'sqrt'],
                'model__min_samples_split': min_samples_split,
                'model__min_samples_leaf': min_samples_leaf
                }
            self.params = params
        elif type(clf['model']).__name__ == 'MLPClassifier':
            # hidden_layer_sizes = np.arange(10, 15)
            hidden_layer_sizes = [(100,), (50,100,), (50,75,100,)]
            activation = ['tanh', 'relu', 'logistic', 'identity']
            solver = ['sgd', 'adam', 'lbfgs']
            alpha = [0.0001, 0.05]
            early_stopping = [True]
            max_iter = [300]
            learning_rate = ['constant', 'adaptive']
            params = {
                'model__random_state':[self.random_state],
                'model__hidden_layer_sizes': hidden_layer_sizes,
                'model__activation': activation,
                'model__max_iter': max_iter,
                'model__solver': solver,
                'model__alpha': alpha,
                'model__early_stopping': early_stopping,
                'model__learning_rate': learning_rate
                }
            self.params = params
        elif type(clf['model']).__name__ == 'Ridge':
            alpha = np.logspace(-2, 2, num=5)
            params = {'model__alpha': alpha}
        elif type(clf['model']).__name__ == 'SVR':
            params = {
                    'model__C': np.linspace(0,2,10),
                    'model__gamma': ['scale', 'auto'],
                    'model__random_state':[10],
                    'model__epsilon': np.linspace(0, 0.3, 17)
            }
            self.params = params
        elif type(clf['model']).__name__ == 'SGDRegressor':
            params = {
                    'model__alpha': np.linspace(0,1,17),
                    'model__penalty': ['l2', 'l1', 'elasticnet'],
                    'model__tol': np.linspace(1e-3, 1e-1, 17),
                    'model__loss': [
                        'squared_loss',
                        'huber',
                        'epsilon_insensitive',
                        'squared_epsilon_insensitive'
                        ]
            }
            self.params = params
        elif type(clf['model']).__name__ == 'Lasso':
            params = {'model__alpha': np.linspace(0,0.1,17)}
            self.params = params
        else:
            TypeError('Either an invalid method was selected, or the '
                'method you have selected does not require params')


    def run_randomized_search(
            self,
            clf
            ):
        """
        Function to explore a random list of parameters possible for a
        model. This is determined from a dictionary of params that determine
        where the searching takes place.

        INPUTS:
        - clf: The classifier model that has been elected for use.

        OUTPUTS:
        - random_search: The hyperparameters that produced the best result

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """
        from sklearn.model_selection import RandomizedSearchCV

        n_iter = 1
        for val in self.params.values():
            if isinstance(val, (list, np.ndarray)):
                n_iter *= len(val)
        n_iter = int(n_iter*0.2)
        if n_iter < 25:
            n_iter = 25

        random_search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=self.params,
            n_iter=n_iter,
            scoring='accuracy',
            # n_jobs=0,
            cv=self.cv,
            # iid=False, # depreciated in sklearn 0.24
            error_score=np.nan
            )
        random_search.fit(
            X=self.x_train,
            y=self.y_train,
            groups=self.train_groups
            )


        return random_search


    def run_grid_search(
            self,
            clf
            ):
        """
        Function to explore an exhaustive list of parameters possible for a
        model. This is determined from a dictionary of params that determine
        where the searching takes place.

        INPUTS:
        - clf: The classifier model that has been elected for use.

        OUTPUTS:
        - grid_search: The object containing the best parameters from grid
                search hyperparameter optimisation

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina

        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        grid_search = GridSearchCV(
            estimator=clf,
            param_grid=self.params,
            scoring='accuracy',
            # n_jobs=0,
            cv=self.cv,
            # iid=False, # depreciated in sklearn 0.24
            error_score=np.nan
            )
        grid_search.fit(
            X=self.x_train,
            y=self.y_train,
            groups=self.train_groups
            )


        return grid_search


    def train_model(
            self,
            clf,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        The core function that trains a machine learning model. This function
        is key within wrapper functions that train models.

        INPUTS:
        - clf: The classifier of choice

        OUTPUTS:
        - clf: The trained classifier

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        pipe = clf

        if len(self.y_train) > 7:
            scores = cross_val_score(
                estimator=pipe,
                X=self.x_train,
                y=self.y_train,
                groups=self.train_groups,
                scoring='accuracy',
                # cv=KFold(n_splits=4),
                # cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=2), # If the dummy models are not straight line, change to this
                # cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=self.random_state),
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                # cv=LeaveOneOut(),
                # n_jobs=-1,
                error_score=np.nan
                )
        else:
            scores = cross_val_score(
                estimator=pipe,
                X=self.x_train,
                y=self.y_train,
                groups=self.train_groups,
                scoring='accuracy',
                # cv=KFold(n_splits=4),
                # cv=RepeatedKFold(n_repeats=3, random_state=2), # if dummy models are not in straight line, change to this.
                # cv=RepeatedKFold(n_repeats=3, random_state=self.random_state),
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                # cv=LeaveOneOut(),
                # n_jobs=-1,
                error_score=np.nan
                )
        # print(self.train_groups)
        # print("Cross validation split used: {}".format(self.cv.n_splits))
        if ('save' in keyword_parameters.keys()) and (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            with open(path+"Model Results.txt", "a") as f:
                print('Model cross-validation score accuracy: %0.2f (%0.2f)' % (
                                scores.mean(), (scores.std())), file=f)
        print('\033[1m' + 'Model cross-validation score accuracy: %0.2f (%0.2f)' % (
                        scores.mean(), (scores.std())) + '\033[0m')
        # print(scores)
        self.scores = scores
        self.std = scores.std()
        self.cvmean = scores.mean()
        # self.standarderror = (scores.std())/sqrt(self.cv.n_splits)
        self.standarddeviation = (scores.std())

        pipe.fit(X=self.x_train, y=self.y_train)

        return pipe

    def test_model(
            self,
            clf,
            *positional_parameters,
            **keyword_parameters
        ):
        """
        Function for testing the specified produced model performance using the
        given test data set. Within this function, various performance metrics
        are calculated as well as a confusion matrix for the classification.
        The results are also saved in the results directory if we wish to go
        back and see what was historically produced.

        INPUTS:
        - clf: The trained machine learning model

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """
        from sklearn.metrics import (
            accuracy_score,
            recall_score,
            precision_score,
            confusion_matrix
            )
        from sklearn.utils.multiclass import unique_labels

        predictions = clf.predict(self.x_test)
        print(predictions)
        try:
            accuracy = accuracy_score(y_true=self.y_test, y_pred=predictions)
            if len(np.unique(self.y_test)) == 2:
                recall = recall_score(
                    y_true=self.y_test,
                    y_pred=predictions,
                    average='weighted'
                    )

                precision = precision_score(
                    y_true=self.y_test,
                    y_pred=predictions,
                    average='weighted'
                    )
                if ('save' in keyword_parameters.keys()):
                    if (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
                        path = keyword_parameters['path']
                        with open(path+"Model Results.txt", "a") as f: # the 'a' here means 'append' - i.e. the existing contents of
                    # the file won't be overwritten. If you want the file to be overwritten instead at the beginning, use 'w'
                            print('Recall: %0.2f' % (recall), file=f)
                            print('Precision: %0.2f' % (precision), file=f)
                    else:
                        print('Recall: %0.2f' % (recall))
                        print('Precision: %0.2f' % (precision))
                else:
                    print('Recall: %0.2f' % (recall))
                    print('Precision: %0.2f' % (precision))

            else:
                recall = np.nan
                precision = np.nan

            plt.clf()
            fig = plt.figure(figsize=[5,5])

            try:
                labels = self.labels
            except Exception as err:
                labels = unique_labels(self.y_test, predictions)
            sns.heatmap(
                data=confusion_matrix(
                        y_true=self.y_test,
                        y_pred=predictions,
                        labels=labels),
                cmap='RdPu',
                square=True,
                annot=True,
                annot_kws={
                    "size":50
                    # "rotation":350
                    },
                xticklabels=True,
                # rotation=30,
                cbar=False,
                yticklabels=True
                )
            if len(np.unique(self.y_test)) == 2:
                tn, fp, fn, tp = confusion_matrix(self.y_test, predictions).ravel()
                if ('save' in keyword_parameters.keys()):
                    if (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
                        path = keyword_parameters['path']
                        # with open(path+"Model Results.txt", "a") as f:
                        print('TN {}'.format(tn))
                        print('FP {}'.format(fp))
                        print('FN {}'.format(fn))
                        print('TP {}'.format(tp))
                    else:
                        print('TN {}'.format(tn))
                        print('FP {}'.format(fp))
                        print('FN {}'.format(fn))
                        print('TP {}'.format(tp))
                else:
                    print('TN {}'.format(tn))
                    print('FP {}'.format(fp))
                    print('FN {}'.format(fn))
                    print('TP {}'.format(tp))
                test_accuracy = (tn + tp)/(tn + fp + fn + tp)

                ci95 = 1.96*math.sqrt((test_accuracy*(1-test_accuracy))/(tn + fp + fn + tp))
                if ('save' in keyword_parameters.keys()):
                    if (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
                        path = keyword_parameters['path']
                        with open(path+"Model Results.txt", "a") as f:
                            print('The test set accuracy is %0.2f with a 95%% confidence interval of +/- %0.2f' % (test_accuracy, ci95), file=f)
                    else:
                        print('The test set accuracy is %0.2f with a 95%% confidence interval of +/- %0.2f' % (test_accuracy, ci95))
                else:
                    print('The test set accuracy is %0.2f with a 95%% confidence interval of +/- %0.2f' % (test_accuracy, ci95))

                # print(test_accuracy)
                # print(ci95)
            else:
                pass

            ax = plt.gca()
            ax.set(
                xticklabels=labels,
                yticklabels=labels,
                xlabel='Predicted label',
                ylabel='True label'
                )
            ax.tick_params(axis='both', which='major', labelsize=18.5)
            ax.tick_params(axis='both', which='minor', labelsize=18.5)
            plt.xticks(rotation=30)
            plt.yticks(rotation='horizontal')
            plt.show()

            if ('save' in keyword_parameters.keys()):
                if (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    fig.savefig(path+'Confusion Matrix.png', dpi=300, bbox_inches='tight')
                else:
                    pass
            else:
                pass

            return accuracy, recall, precision, clf
        except:
            accuracy = np.nan
            recall = np.nan
            precision = np.nan
            if ('save' in keyword_parameters.keys()):
                if (keyword_parameters['save'] is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    with open(path+"Predictions.txt", "w") as f:
                        print('The predictions are {}'.format(predictions), file=f)
                else:
                    print('The predictions are {}'.format(predictions))
            else:
                print('The predictions are {}'.format(predictions))

            return accuracy, recall, precision, clf


    def run_logistic_regression(
            self,
            hyperopt,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        Wrapper function for training with a Logistic Regression model.
        Logistic Regression is a sigmoid binary function, and multiple different
        classifiers can be created with a larger than binary target set to reach
        the desired predictive capabilities.
        Function calls on the run_algorithm function twice. Once for determining
        the optimum parameters, and the section time to train the optimized
        model.

        INPUTS:
        - hyperopt: Determine between the two different types of hyperparameter
            optimization. Either 'randomsearch' or 'gridsearch'.

        OUTPUTS:
        - search: The trained model object

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        from sklearn.linear_model import LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False

        hyperopt = hyperopt.lower().replace(' ', '')
        if hyperopt in ['randomsearch', 'gridsearch']:
            clf = LogisticRegression(
                max_iter=4000,
                class_weight='balanced',
                # random_state=random_seed
                )
            if self.technique == 'PCA':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', PCA(random_state=self.random_state)),
                        ('model', clf)]
                        )
            elif self.technique == 'None':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        # ('dimension_reduction', PCA(random_state=self.random_state)),
                        ('model', clf)]
                        )
            else:
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', LDA()),
                        ('model', clf)]
                        )
        else:
            TypeError('The wrong input for hyperopt has been entered. Select either '
            'randomsearch or gridsearch.')
        print("Tuning hyperparameters...")
        search = self.run_algorithm(
            pipe,
            run=hyperopt
            )
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        pipe.set_params(**search.best_params_)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )
        return clf


    def run_k_nearest_neighbours(
            self,
            hyperopt,
            *positional_parameters,
            **keyword_parameters
            ):
        """

        Wrapper function for training with a K Nearest Neighbours model.
        K nearest neighbours is a simple statisical model that works by comparing
        each data point to the points around it. Where the closest points are,
        this point will belong to that group. The number of neighbours are
        selected within the hyperparameter optimisation stage of this function.
        Function calls on the run_algorithm function twice. Once for determining
        the optimum parameters, and the section time to train the optimized
        model.

        INPUTS:
        - hyperopt: Determine between the two different types of hyperparameter
            optimization. Either 'randomsearch' or 'gridsearch'.

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina

        """

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False
        hyperopt = hyperopt.lower().replace(' ', '')
        if hyperopt in ['randomsearch', 'gridsearch']:
            clf = KNeighborsClassifier(metric='minkowski')
            if self.technique == 'PCA':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', PCA()),
                        ('model', clf)]
                        )
            elif self.technique == 'None':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        # ('dimension_reduction', PCA()),
                        ('model', clf)]
                        )
            else:
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', LDA()),
                        ('model', clf)]
                        )
        else:
            TypeError('The wrong input for hyperopt has been entered. Select either '
            'randomsearch or gridsearch.')
        print("Tuning hyperparameters...")
        search = self.run_algorithm(
            pipe,
            run=hyperopt
            )
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA()),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA()),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        pipe.set_params(**search.best_params_)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf


    def run_naive_bayes(
            self,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        Wrapper function for training with a Gaussian Naive Bayes model.
        Gaussian Naive Bayes is one of the simplest statistical models and is a
        slight derivative of the optimum bayes classifier (which real systems
        can't really be modelled by). It should work for simple datasets.
        Function calls on the run_algorithm function once - as it does not
        need to optimise parameters as it is too simple to have any.

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False
        clf = GaussianNB()
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA()),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA()),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )

        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf

    def run_lda(
            self,
            *positional_parameters,
            **keyword_parameters
            ):
        """

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Tania LaGambina
            26th September 2022
        """

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False

        clf = LinearDiscriminantAnalysis()
        pipe = Pipeline([
                ('standardising', StandardScaler()),
                ('model', clf)]
                )
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf

    def run_svc(
            self,
            hyperopt,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        Wrapper function for training with a Support Vector Classifier model.
        Support vector classifiers works on the idea that we can construct an
        optimal hyperplane between two perfectly separated classes - SVCs
        address the scenario where classes may not be separable by a linear
        boundary. This is a model which is a little more complex than the very
        simple statistical models.
        Function calls on the run_algorithm function twice. Once for determining
        the optimum parameters, and the section time to train the optimized
        model.

        INPUTS:
        - hyperopt: Determine between the two different types of hyperparameter
            optimization. Either 'randomsearch' or 'gridsearch'.

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Based original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        from sklearn.svm import SVC
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False
        random_seed = 5
        hyperopt = hyperopt.lower().replace(' ', '')
        if hyperopt in ['randomsearch', 'gridsearch']:
            clf = SVC(probability=True)
            if self.technique == 'PCA':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', PCA(random_state=self.random_state)),
                        ('model', clf)]
                        )
            elif self.technique == 'None':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        # ('dimension_reduction', PCA(random_state=self.random_state)),
                        ('model', clf)]
                        )
            else:
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', LDA()),
                        ('model', clf)]
                        )
        else:
            TypeError('The wrong input for hyperopt has been entered. Select either '
            'randomsearch or gridsearch.')
        print("Tuning hyperparameters...")
        search = self.run_algorithm(
            pipe,
            run=hyperopt
            )
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        pipe.set_params(**search.best_params_)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf


    def run_random_forest(
            self,
            hyperopt,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        Wrapper function for training with a Random Forest model.
        Random forests are tree based models that are based on the fact that
        feature space can be split dependant on some conditions. The more splits,
        the more complex the mdoel is that is produced.
        Function calls on the run_algorithm function twice. Once for determining
        the optimum parameters, and the section time to train the optimized
        model.

        INPUTS:
        - hyperopt: Determine between the two different types of hyperparameter
            optimization. Either 'randomsearch' or 'gridsearch'.

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Based on original by Kathryn Shelley, Woolfson Group
            Developed for Rosa Biotech by Tania LaGambina
        """

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False
        random_seed = 5
        hyperopt = hyperopt.lower().replace(' ', '')
        if hyperopt in ['randomsearch', 'gridsearch']:
            clf = RandomForestClassifier()
            if self.technique == 'PCA':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', PCA()),
                        ('model', clf)]
                        )
            elif self.technique == 'None':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        # ('dimension_reduction', PCA()),
                        ('model', clf)]
                        )
            else:
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', LDA()),
                        ('model', clf)]
                        )
        else:
            TypeError('The wrong input for hyperopt has been entered. Select either '
            'randomsearch or gridsearch.')
        print("Tuning hyperparameters...")
        search = self.run_algorithm(
            pipe,
            run=hyperopt
            )
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA()),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA()),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        pipe.set_params(**search.best_params_)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf


    def run_neural_network(
            self,
            hyperopt,
            *positional_parameters,
            **keyword_parameters
            ):
        """
        Function to train a neural network model using the sklearn MLPClassifier
        function.

        INPUTS:
        - hyperopt: Determine between the two different types of hyperparameter
            optimization. Either 'randomsearch' or 'gridsearch'.

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Tania LaGambina
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False
        hyperopt = hyperopt.lower().replace(' ', '')
        if hyperopt in ['randomsearch', 'gridsearch']:
            clf = MLPClassifier()
            if self.technique == 'PCA':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', PCA(random_state=self.random_state)),
                        ('model', clf)]
                        )
            elif self.technique == 'None':
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        # ('dimension_reduction', PCA(random_state=self.random_state)),
                        ('model', clf)]
                        )
            else:
                pipe = Pipeline([
                        ('standardising', StandardScaler()),
                        ('dimension_reduction', LDA()),
                        ('model', clf)]
                        )
        else:
            TypeError('The wrong input for hyperopt has been entered. Select either '
            'randomsearch or gridsearch.')
        print("Tuning hyperparameters...")
        search = self.run_algorithm(
            pipe,
            run=hyperopt
            )
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        pipe.set_params(**search.best_params_)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        print('The tuned model is:')
        print(pipe)
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf


    def run_dummy_classifier2(
            self,
            *positional_parameters,
            **keyword_parameters
            ):
        """

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Tania LaGambina
        """

        from sklearn.dummy import DummyClassifier
        from imblearn.pipeline import Pipeline
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False

        clf = DummyClassifier(strategy='uniform', random_state=self.random_state)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf


    def run_stacking(
                self,
                hyperopt,
                *positional_parameters,
                **keyword_parameters
                ):
        """
        A Stacked Generalisation meta model that incorporates predicitons from
        multiple models, and uses a meta model to decide which of these
        predictions to trust.

        Author:
            Tania LaGambina
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import StackingClassifier, RandomForestClassifier
        # from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import RepeatedKFold

        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False

        hyperopt = hyperopt.lower().replace(' ', '')

        if hyperopt in ['randomsearch', 'gridsearch']:
            lr_clf = LogisticRegression(
                # n_jobs=-1,
                max_iter=4000,
                class_weight='balanced'
                )
            knn_clf = KNeighborsClassifier(metric='minkowski')
            # nn_clf = MLPClassifier()
            rf_clf = RandomForestClassifier()
            svm_clf = SVC()
        else:
            TypeError('The wrong input for hyperopt has been entered. Select either '
            'randomsearch or gridsearch.')

        print("Tuning lr hyperparameters...")
        lr_search = self.run_algorithm(
            lr_clf,
            run=hyperopt
            )
        lr = LogisticRegression(
            **lr_search.best_params_,
            max_iter=4000,
            # n_jobs=-1,
            class_weight='balanced'
            )

        print("Tuning knn hyperparameters...")
        knn_search = self.run_algorithm(
            knn_clf,
            run=hyperopt
            )
        knn = KNeighborsClassifier(**knn_search.best_params_)

        # print("Tuning nn hyperparameters...")
        # nn_search = self.run_algorithm(
        #     nn_clf,
        #     run=hyperopt
        #     )
        # nn = MLPClassifier(**nn_search.best_params_)

        print("Tuning rf hyperparameters...")
        rf_search = self.run_algorithm(
            rf_clf,
            run=hyperopt
            )
        rf = RandomForestClassifier(**rf_search.best_params_)

        print("Tuning svm hyperparameters...")
        svm_search = self.run_algorithm(
            svm_clf,
            run=hyperopt
            )
        svm = SVC(**svm_search.best_params_)

        bayes = GaussianNB()

        level0 = list()
        level0.append(('lr', lr))
        level0.append(('knn', knn))
        level0.append(('bayes', bayes))
        # level0.append(('nn', nn))
        level0.append(('rf', rf))
        level0.append(('svm', svm))

        level1 = LogisticRegression()

        clf = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

        if ('labels' in keyword_parameters.keys()):
                self.labels = keyword_parameters['labels']
        else:
            pass
        print("Training model...")
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                pipe,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                pipe,
                run='train'
                )

        return clf

    def run_dummy_classifier(
            self,
            *positional_parameters,
            **keyword_parameters
            ):
        """

        OUTPUTS:
        - clf: The trained model object

        AUTHOR:
            Tania LaGambina
        """

        from sklearn.dummy import DummyClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from imblearn.pipeline import Pipeline
        if ('save' in keyword_parameters.keys()):
            save = keyword_parameters['save']
        else:
            save = False


        clf = DummyClassifier(random_state=self.random_state)
        print("Training model...")
        if ('labels' in keyword_parameters.keys()):
            self.labels = keyword_parameters['labels']
        else:
            pass
        if self.technique == 'PCA':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        elif self.technique == 'None':
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    # ('dimension_reduction', PCA(random_state=self.random_state)),
                    ('model', clf)]
                    )
        else:
            pipe = Pipeline([
                    ('standardising', StandardScaler()),
                    ('dimension_reduction', LDA()),
                    ('model', clf)]
                    )
        if ('path' in keyword_parameters.keys()):
            path = keyword_parameters['path']
            search, clf = self.run_algorithm(
                clf,
                run='train',
                save=save,
                path=path
                )
        else:
            search, clf = self.run_algorithm(
                clf,
                run='train'
                )

        return clf

def plot_model_scores_per_features(score):
    """
    Function to plot the model scores with respect to the number of features
    included in the model. This can then be used to determine the optimimum
    number of features to be included in the model.

    INPUTS:
        - score: The dictionary of model scroes and number of features
                associated with the scores

    AUTHOR:
        Tania LaGambina
    """
    lists = sorted(score.items())
    x, y = zip(*lists)
    plt.figure(figsize=[20,10])
    plt.plot(x,y)

    ax = plt.gca()
    ax.set(
        xticks=x,
        yticks=y,
        xlabel='Number of features',
        ylabel='Model CV accuracy')
    ax.set_ylim([0, 1])

    plt.show()

    return ax

def ml_wrapper(
        parsed_data,
        target_col,
        specific_analytes_col,
        ranking_df,
        test_analytes_list,
        low_feat_no,
        high_feat_no,
        test_seen,
        model_name,
        results,
        names,
        aucs,
        *positional_parameters,
        **keyword_parameters
    ):
    """
    Function that contains multiple ML functions to produce the pipeline.
    Brought together into a function for simplicity reasons.

    INPUTS:
        - parsed_data: The unstandardised scaled data
        - standardised_parsed_data: The standardised scaled data
        - target_col: The column in the dataframe that is the target feature to
            train the model towards
        - specific_analytes_col: Column which describes the different analytes
            in more fine depth
        - ranking_df: A dataframe of the top features ranked in order
        - test_analytes_list: The specific analytes that will be left out for
            the test set
        - low_feat_no: The lowest number of features to be included in the model
            when running through a loop to determine best number of features
        - best_feat_no: The optimum number of features (which can be determined
            at a later date) for which the cross validation score will be
            recorded
        - high_feat_no: The highest number of features to be included in the
            model when running through a loop to determine the best number of
            features
        = test_seen: True or False
        - model: The model used in the function. Will take string values like
            'Stacking', 'K Nearest Neighbours', 'Gaussian Naive Bayes'
        - colors (keyword_parameters): The specific colours to colour the groups
            on the pca_scatter. If not specified, will use a default palette
        - hue_order (keyword_parameters): The order of the analytes to be labelled
            useful if you wish a specific analyte to be a specific colour. If
            not specified, orders the analytes in the order they appear in the
            dataframe

    OUTPUTS:
        - score: A dictionary of the scores from the model per number of
            features. To be used to in plot_model_scores_per_features function
        - optimimum_score: The cross validation scores for the model at the
            optimum number of features. To be used in conjunction with the
            function that plots the cross validation scores for each model
            to compare the variance and pick the best model

    AUTHOR:
        Tania LaGambina
    """
    import numpy as np
    import pandas as pd
    from production import train_model, analysis
    import joblib

    score = {}
    if ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
    else:
        colors = [
            '#ff0000',
            '#ffa500',
            '#ffff00',
            '#008000',
            '#0000ff',
            '#4b0082',
            '#ee82ee'
        ]
    if ('hue_order' in keyword_parameters.keys()):
        hue_order = keyword_parameters['hue_order']
        # self.labels = hue_order
    else:
        hue_order = parsed_data[target_col].unique()

    if ('seed' in keyword_parameters.keys()):
        seed = keyword_parameters['seed']
    else:
        seed = 42
    # if ('path' in keyword_parameters.keys()):
    #     path = keyword_parameters['path']
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    # else:
    #     path = os.getcwd()+'/Results'

    for n in range(parsed_data.shape[0]):
        if n>low_feat_no and n<high_feat_no:
            print('{} input features'.format(n))
            parsed_data_n = parsed_data[
                ranking_df.nlargest(n, 'SUM').index.tolist()+[str(specific_analytes_col)]+[str(target_col)]
            ]
            print(parsed_data_n.columns)
            if ('save' in keyword_parameters.keys()):
                save = keyword_parameters['save']
            else:
                save = False
            models = {}
            if n<6:
                analysis.plot_pairplot(parsed_data_n.loc[parsed_data_n[target_col].isin(hue_order)], target_col, save=save, colors=colors, hue_order=hue_order)
            else:
                pass

            if (save is True) and ('path' in keyword_parameters.keys()):
                path = keyword_parameters['path']
                fig = analysis.plot_pairplot(parsed_data[
                    ranking_df.nlargest(3, 'SUM').index.tolist()+[str(specific_analytes_col)]+[str(target_col)]
                    ], target_col, save=False, colors=colors, hue_order=hue_order)
                fig.savefig(path+'Top 3 Features Pairplot.png', dpi=300, bbox_inches='tight', format='png')
                fig.savefig(path+'Top 3 Features Pairplot.svg', dpi=300, bbox_inches='tight', format='svg')

            split_data = train_model.split_train_test_data(
                data=parsed_data_n,
                test_analytes_list=test_analytes_list,
                test_seen=test_seen,
                target_col=target_col,
                specific_analytes_col=specific_analytes_col,
                shuffle=False
            )
            x_train = split_data['x_train']
            y_train = split_data['y_train']
            x_test = split_data['x_test']
            y_test = split_data['y_test']
            cv = split_data['cv']
            analytes = split_data['analytes_test']

            if (save is True) and ('path' in keyword_parameters.keys()):
                path = keyword_parameters['path']
                test_set = parsed_data_n.loc[parsed_data_n[specific_analytes_col].isin(test_analytes_list)].reset_index(drop=True)
                test_set.to_excel(path+'test_set.xlsx')

            if ('dimension_reduction' in keyword_parameters.keys()):
                technique = keyword_parameters['dimension_reduction']
                if technique == 'LDA':
                    lda_x_train, lda_x_test = train_model.run_lda_and_transform_traintest(x_train, x_test, y_train)
                    try:
                        if len(np.unique(y_train)) > 2:
                            scatter = train_model.lda_scatter(lda_x_data=lda_x_train, y_data=y_train, colors=colors, hue_order=hue_order)
                            if (save is True) and ('path' in keyword_parameters.keys()):
                                path = keyword_parameters['path']
                                scatter.savefig(path+'Train Dataset LDA Scatter.png', dpi=300, bbox_inches='tight', format='png')
                                scatter.savefig(path+'Train Dataset LDA Scatter.svg', dpi=300, bbox_inches='tight', format='svg')
                        else:
                            pass
                    except:
                        pass
                elif technique == 'None':
                    pass
                else:
                    technique = 'PCA'
                    pca_x_train, pca_x_test = train_model.run_pca_and_transform_traintest(x_train=x_train, x_test=x_test, seed=seed)
                    # pca = train_model.PCATools(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seed=seed)
                    # pca.run_pca()
                    # pca_x_train, pca_x_test = pca.run_pca_and_transform()
                    try:
                        scatter = train_model.pca_scatter(pca_x_data=pca_x_train, y_data=y_train, colors=colors, hue_order=hue_order)
                        if (save is True) and ('path' in keyword_parameters.keys()):
                            path = keyword_parameters['path']
                            scatter.savefig(path+'Train Dataset PCA Scatter.png', dpi=300, bbox_inches='tight', format='png')
                            scatter.savefig(path+'Train Dataset PCA Scatter.svg', dpi=300, bbox_inches='tight', format='svg')
                    except:
                        pass
            else:
                technique = 'PCA'
                pca_x_train, pca_x_test = train_model.run_pca_and_transform_traintest(x_train=x_train, x_test=x_test, seed=seed)
                # pca = train_model.PCATools(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, seed=seed)
                # pca.run_pca()
                # pca_x_train, pca_x_test = pca.run_pca_and_transform()
                try:
                    scatter = train_model.pca_scatter(pca_x_data=pca_x_train, y_data=y_train, colors=colors, hue_order=hue_order)
                    if (save is True) and ('path' in keyword_parameters.keys()):
                        path = keyword_parameters['path']
                        scatter.savefig(path+'Train Dataset PCA Scatter.png', dpi=300, bbox_inches='tight', format='png')
                        scatter.savefig(path+'Train Dataset PCA Scatter.svg', dpi=300, bbox_inches='tight', format='svg')
                except:
                    pass

            ml = train_model.RunML(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                technique=technique,
                labels=hue_order,
                cv=2,
                seed=seed
                )

            if str.lower(model_name) in ['knn', 'k neighbours', 'k neighbours','k nearest neighbours','k nearest neighbors']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_k_nearest_neighbours(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_k_nearest_neighbours(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['nn', 'neural network', 'neural net']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_neural_network(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_neural_network(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['svm', 'svc', 'support vector machine', 'support vector classifier']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_svc(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_svc(
                        hyperopt='gridsearch'
                        )

            elif str.lower(model_name) in ['rf', 'random forest', 'decision tree']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_random_forest(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_random_forest(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['stacking', 'stacked generalisation', 'ensemble', 'stacked generalization']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_stacking(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_stacking(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['lr', 'logistic regression']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_logistic_regression(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_logistic_regression(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['gnb', 'bayes', 'gaussian naive bayes', 'naive bayes']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_naive_bayes(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_naive_bayes(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['dummy', 'dummy_model', 'dummy model', 'dummy classifier', 'prior dummy', 'prior classifier']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_dummy_classifier(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_dummy_classifier(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['dummy2', 'dummy 2', 'random', 'random classifier', 'uniform classifier', 'uniform dummy']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_dummy_classifier2(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_dummy_classifier2(
                        hyperopt='gridsearch'
                        )
            elif str.lower(model_name) in ['lda', 'linear discriminant analysis', 'discriminant analysis']:
                if ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    model = ml.run_lda(
                        hyperopt='gridsearch',
                        save=save,
                        path=path
                        )
                else:
                    model = ml.run_lda(
                        hyperopt='gridsearch'
                        )
            else:
                print('Input a valid ML model')

            if (save is True) and ('path' in keyword_parameters.keys()):
                joblib.dump(model, path+'best_model.sav')

            if technique == 'LDA':
                learning_curve = train_model.plot_learning_curves(model, lda_x_train, y_train)
                if (save is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    learning_curve.savefig(path+'Learning Curves.png', dpi=300, bbox_inches='tight', format='png')
                    learning_curve.savefig(path+'Learning Curves.svg', dpi=300, bbox_inches='tight', format='svg')

            elif technique == 'None':
                learning_curve = train_model.plot_learning_curves(model, x_train, y_train)
                if (save is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    learning_curve.savefig(path+'Learning Curves.png', dpi=300, bbox_inches='tight', format='png')
                    learning_curve.savefig(path+'Learning Curves.svg', dpi=300, bbox_inches='tight', format='svg')
            else:
                learning_curve = train_model.plot_learning_curves(model, pca_x_train, y_train)
                if (save is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    learning_curve.savefig(path+'Learning Curves.png', dpi=300, bbox_inches='tight', format='png')
                    learning_curve.savefig(path+'Learning Curves.svg', dpi=300, bbox_inches='tight', format='svg')

            score[n] = ml.cvmean

            optimum_score = np.nan
            model_score = ml.cvmean
            std_score = ml.std
            if (save is True) and ('path' in keyword_parameters.keys()):
                path = keyword_parameters['path']
                with open(path+"Model Results.txt", "a") as f:
                    # print('The test analytes are: {}'.format(analytes), file=f)
                    print('The tuned final best model is {}'.format(model), file=f)
                with open(path+"Test Analyte IDs.txt", "w") as f:
                    print('The test analytes are: {}'.format(analytes), file=f)
            else:
                pass
            results.append(model_score)
            names.append(model_name+'_'+str(n))
            if ('stds' in keyword_parameters.keys()):
                stds = keyword_parameters['stds']
                stds.append(std_score)
            else:
                pass

            # else:
                # pass
            if len(parsed_data_n[target_col].unique())<3:
                if (save is True) and ('path' in keyword_parameters.keys()):
                    path = keyword_parameters['path']
                    fig, auc = analysis.plot_roc_curve(x_test, y_test, 1, model, save=save, path=path)
                    fig.savefig(path+'ROC Curve.png', dpi=300, bbox_inches='tight', format='png')
                    fig.savefig(path+'ROC Curve.svg', dpi=300, bbox_inches='tight', format='svg')
                else:
                    fig, auc = analysis.plot_roc_curve(x_test, y_test, 1, model)
                aucs.append(auc)
            else:
                pass
        else:
            pass
    if ('save' in keyword_parameters.keys()) and (keyword_parameters['save'] is True):
        if ('stds' in keyword_parameters.keys()):
            # return score, optimum_score, results, names, stds
            return results, names, stds, aucs

        else:
            # return score, optimum_score, results, names
            return results, names, aucs
    else:
        if ('stds' in keyword_parameters.keys()):
            # return score, optimum_score, results, names, stds
            return results, names, stds, aucs

        else:
            # return score, optimum_score, results, names
            return results, names, aucs

def determine_best_model(results, names):
    """
    Function for determining the best model from an ML Pipeline

    Author:
        Tania LaGambina
    """
    results_df = pd.DataFrame(data=[results,names])
    rdf = results_df.T
    rdf.columns=['accuracy', 'model']
    df2=rdf.query('accuracy == accuracy.max()')

    print(df2)
    return df2

def plot_model_features_comparison(
        results,
        names,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Function to plot the model cv score for multiple models over multiple
    numbers of features. Helps visualise the best and most consistent performing
    models, as well as the optimum numer of features.

    INPUTS:
        - results: A dictionary containing the model cv scores per number of
                feature, or can be subsituted with aucs, a dictionary containing
                the auc score per number of feature
        - names: The names of the model and corresponding number of features
        Both of these come from the ml_wrapper function
        - **ranking_df: If you want to include the peptide labels instead of the number of
                features, then you can use the ranking_df which is produced on the
                train set to iteratively add in the peptides into the pipeline. If you
                want to use this, add in ranking_df=ranking_df as an input. Note, as this
                is a keyword parameter, the function will ignore just 'ranking_df' added on
                its own

    AUTHOR:
        Tania LaGambina
    """

    results_df = pd.DataFrame(data=results, index=names)
    results_df['Model'] = results_df.index
    results_df.reset_index(inplace=True, drop=True)
    num_of_feats = list(x.split('_')[-1] for x in results_df.Model)
    results_df['Number of Features'] = num_of_feats
    results_df['Model'] = list(x.split('_')[0] for x in results_df.Model)
    melted_results_df = pd.melt(results_df, id_vars=['Model', 'Number of Features'], value_name='Score')
    melted_results_df.drop(columns=['variable'], axis=1, inplace=True)

    fig = plt.figure(figsize=[15,7])
    sns.set(style="white")
    sns.set_context("paper", rc={
        "font.size":17,
        "axes.titlesize":17,
        "axes.labelsize":17,
        "xtick.labelsize":15,
        "ytick.labelsize":15
    })

    if ('hue_order' in keyword_parameters.keys()):
        hue_order = keyword_parameters['hue_order']
    else:
        hue_order = melted_results_df['Model'].unique()

    if ('colors' in keyword_parameters.keys()):
        colors = keyword_parameters['colors']
        palette = sns.set_palette(sns.color_palette(colors))
    elif ('palette' in keyword_parameters.keys()):
        palette = keyword_parameters['palette']
    else:
        palette = 'tab10'

    ax = sns.swarmplot(
        x='Number of Features',
        y='Score',
        hue='Model',
        data = melted_results_df,
        hue_order = hue_order,
        palette = palette,
        size=8,
        edgecolor='w'
    )
    ax.set_ylim([0,1.1])
    if ('ranking_df' in keyword_parameters.keys()):
        ranking_df = keyword_parameters['ranking_df']
        labels = ranking_df.index
        ticks = np.arange(stop = 1 * len(labels), step=1)  # as many ticks as there are labels
        ax.set_xticks(ticks, labels, rotation=80)
    else:
        pass
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=15)
    plt.show()

    return fig


def plot_learning_curves(
        clf,
        data,
        target,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Function to plot learning curves for different models

    AUTHOR:
        Tania LaGambina
    """
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    if ('train_sizes' in keyword_parameters.keys()):
        train_sizes, train_scores, valid_scores = learning_curve(
            clf,
            data,
            target,
            # train_sizes=np.linspace(5, 55, num=50, dtype=int),
            train_sizes=keyword_parameters['train_sizes'],
            cv=5,
            scoring='accuracy'
        )
    else:
        try:
            train_sizes, train_scores, valid_scores = learning_curve(
                clf,
                data,
                target,
                cv=5,
                scoring='accuracy'
            )
        except:
            train_sizes = np.nan
            train_scores = np.nan
            valid_scores = np.nan
    try:
        train_scores_mean = train_scores.mean(axis=1)
        valid_scores_mean = valid_scores.mean(axis=1)
        # print('Mean training scores\n\n', pd.Series(train_scores_mean, index = train_sizes))
        # print('\n', '-' * 20) # separator
        # print('\nMean validation scores\n\n',pd.Series(valid_scores_mean, index = train_sizes))
        # plt.clf()
        fig = plt.figure(figsize=[7,6])
        plt.style.use('default')
        plt.plot(train_sizes, train_scores_mean, label = 'Training score', linewidth=1.5)
        plt.plot(train_sizes, valid_scores_mean, label = 'Validation score', linewidth=1.5)
        plt.ylabel('Accuracy', fontsize = 14)
        plt.xlabel('Training set size', fontsize = 14)
        plt.title('Learning curves', fontsize = 18, y = 1.03)
        plt.legend()
        plt.ylim(0.0, 1.1)
        plt.show()

        return fig
    except:
        pass


def plot_model_features_comparison_std(
        results,
        stds,
        names,
        *positional_parameters,
        **keyword_parameters
        ):
    """
    Model score vs number of features comparison with standard deviation included

    INPUTS:
        - results: The accuracy results dictionary for each model
        - stds: The standard deviation of each of the accuracy results
        - names: The names of each of the models (note - the name also
                includes the number of features put into the model)
        - color_dict: A dictionary containing colors associated with each model
                name. Default already included

    AUTHOR:
        Tania LaGambina
    """
    import matplotlib.pyplot as plt

    results_df = pd.DataFrame(data={'accuracy':results, 'std':stds}, index=names)
    results_df['model'] = results_df.index
    results_df.reset_index(drop=True, inplace=True)
    num_of_feats = list(x.split('_')[-1] for x in results_df.model)
    results_df['features'] = num_of_feats
    results_df['model'] = list(x.split('_')[0] for x in results_df.model)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_ylim([0,1])
    if ('color_dict' in keyword_parameters.keys()):
        color_dict = keyword_parameters['color_dict']
    else:
        color_dict = {
            'Stacking':'blue',
            'Logistic Regression':'orange',
            'Gaussian Naive Bayes':'green',
            'K Nearest Neighbors':'red',
            'SVM':'purple',
            'Prior Dummy':'brown',
            'Uniform Dummy':'pink'
        }
    if ('alpha' in keyword_parameters.keys()):
        alpha = keyword_parameters['alpha']
    else:
        alpha = 0.2

    if ('s' in keyword_parameters.keys()):
        s = keyword_parameters['s']
    else:
        s = 100

    for model in results_df.model.unique():
        color = color_dict[model]
        model_df = results_df.loc[results_df.model==model]
        ax.fill_between(
            model_df['features'],
            model_df['accuracy'] - 0.5*model_df['std'],
            model_df['accuracy'] + 0.5*model_df['std'],
            alpha=alpha,
            color=color
            )
        ax.scatter(
            model_df['features'],
            model_df['accuracy'],
            edgecolors='white',
            s=s,
            color=color,
            label=model
            )
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('CV Accuracy', fontsize = 14)
    plt.xlabel('Number of Features', fontsize = 14)

    plt.show()
