# version 1.0

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pycountry_convert as pc

# Sklearn
from sklearn import feature_selection, model_selection, ensemble, metrics
from sklearn.linear_model import LogisticRegression

def recognize_type(df, col, max_cat=20):
    '''
    Recognize whether a column is numerical or categorical.

    :parameter
        :param df: dataframe - input data
        :param col: str - name of the column to analyze
        :param max_cat: num - max number of unique values to recognize a column as categorical
    :return
        "cat" if the column is categorical or "num" otherwise
    '''
    if (df[col].dtype == "O") | (df[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"


def missing_data_and_types(df):
    '''
    Visualize columns type and missing data of a dataframe

    :parameter
        :param df: dataframe - input data
    :return
        visualization of column types and missing values
    '''
    # Classifies each column as numerical or categorical
    dic_cols = {col:recognize_type(df, col, max_cat=20) for col in df.columns}

    # Plots a visualization of the columns type and missing data
    heatmap = df.isnull()
    for k,v in dic_cols.items():
     if v == "num":
       heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
     else:
       heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)

    sns.heatmap(heatmap, cbar=False).set_title('Dataset Overview')

    # Figure size
    sns.set(rc = {'figure.figsize':(12,7.5)})

    plt.show()
    print("\033[1;37;40m Categorical ", "\033[1;30;41m Numeric ", "\033[1;30;47m NaN ")


def missing_values_table(df):
    '''
    Proportion of missing values of a dataframe
    '''
    # Table
    missing = df.isnull().sum()
    missing_percent = 100*missing/len(df)
    missing_table = pd.concat([missing, missing_percent], axis=1)
    missing_table.columns = ['missing values', '% of missing values']
    missing_table = missing_table.loc[missing_table['missing values'] != 0].sort_values('missing values', ascending=False)
    print('The dataset has total {} columns. \nThere are {} columns that have missing values.\n\n'.format(df.shape[1], missing_table.shape[0]))
    return missing_table


def missing_values_percentages(df):
    '''
    Proportion of missing values of a dataframe
    '''
    # Percentage of missing values
    ncounts = pd.DataFrame([df.isna().mean()]).T
    ncounts[0] *= 100

    # Rename columns
    ncounts = ncounts.rename(columns={0: 'train_missing', 1: 'test_missing'})

    # Create horizontal bar graph
    ncounts = ncounts[ncounts['train_missing'] != 0].sort_values(by=['train_missing'])
    ncounts.plot(kind='barh', title='% of Values Missing')
    plt.show()


def graph_cat_vs_label(category_column, label_column, data):
    '''
    Plot the count and percentage of each value of a categorical column vs the label
    :parameter
        :param data: dataframe - input data
        :param category_column: column of categorical type
        :param label_column: column of the label - target value
    :return
        A pair of bar plots
    '''
    #x, y = 'Male', "Clicked on Ad"
    x, y = category_column, label_column
    df = data
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle(x+"   vs   "+y, fontsize=20)

    # Counts
    ax1.title.set_text('Count')
    order = df.groupby(x)[y].count().index.tolist()
    ax1 = sns.countplot(x=x, hue=y, data=df, ax=ax1, palette='magma_r')

    #ax1.grid(True)

    # Percentages
    ax2.title.set_text('Percentage')
    a = df.groupby(x)[y].count()
    a= pd.DataFrame(a)
    a.rename(columns={y:"tot"}, inplace=True)
    a.reset_index()

    b = df.groupby([x,y])[y].count()
    b = pd.DataFrame(b)
    b.rename(columns={y:0}, inplace=True)
    b.reset_index()

    b = a.join(b, how='inner')

    b['%'] = b[0] / b["tot"] *100

    b.reset_index(inplace=True)

    sns.barplot(x=x, y='%', hue=y, data=b, ax=ax2, palette='magma_r')#.get_legend().remove()

    #ax2.grid(True)
    plt.show()


def country_list_to_continent(country_list):
    '''
    Converts a list of countries into a list of their respective continents
    
    :parameter
        param: country_list: list of countries
    :return
        list of respective continents
    '''
    continent_list = []
    for index, country in enumerate(country_list):
        special_cn = {
            'Palestinian Territory': 'AS',
            'British Indian Ocean Territory (Chagos Archipelago)': 'AS',
            'Korea': 'AS',
            'Bouvet Island (Bouvetoya)': 'AS',
            'Saint Helena': 'AF',
            'Svalbard & Jan Mayen Islands': 'EU',
            'Cote d\'Ivoire': 'AF',
            'Antarctica (the territory South of 60 deg S)': 'AN',
            'Pitcairn Islands': 'OC',
            'Libyan Arab Jamahiriya': 'AS',
            'Saint Barthelemy': 'SA',
            'Reunion': 'AF',
            'Netherlands Antilles': 'SA',
            'Slovakia (Slovak Republic)': 'EU',
            'Timor-Leste': 'AF',
            'Western Sahara': 'AF',
            'United States Minor Outlying Islands': 'NA',
            'Holy See (Vatican City State)': 'EU',
            'French Southern Territories': 'AN'
            }

        if type(country) == str:
            if country in special_cn:
                continent_list.append(special_cn[country])
            else:
                country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
                continent_name = pc.country_alpha2_to_continent_code(country_code)
                continent_list.append(continent_name)
        else:
           continent_list.append(None)
    
    return continent_list


def graph_time_vs_label(period, timestamp_column, label_column, data):
    '''
    Take a timestamp column of a dataframe, divide it into bins and plot vs the label
    
    :parameter
        param: period: Month, Week, Day, Hour
        param: timestamp_column
        param: label_column
        param: data: dataframe
    :return
        graph of bins vs target value
    '''
    df = data
    
    timestamp = timestamp_column
    label = label_column
    
    # Creates Date bins
    df[timestamp] = pd.to_datetime(df[timestamp])
    if period == 'Month':
        df['Month'] = df[timestamp].dt.month
    elif period == 'Day':
        df['Day'] = df[timestamp].dt.day
    elif period == 'Hour':
        df['Hour'] = df[timestamp].dt.hour
    elif period == 'Weekday':
        df['Weekday'] = df[timestamp].dt.dayofweek
    
    # Splits and counts bins by Label
    P = df[period].value_counts().sort_index()
    P0 = df[period][df[label] == 0].value_counts().sort_index()
    P1 = df[period][df[label] == 1].value_counts().sort_index()
    
    #
    fig, ax = plt.subplots()
    fig.suptitle(period+"   vs   "+label, fontsize=20)
    
    # 
    P0_count = pd.DataFrame(P0).reset_index()
    P0_count = P0_count.drop(['index'], axis=1)

    P1_count = pd.DataFrame(P1).reset_index()
    P1_count = P1_count.drop(['index'], axis=1)
    
    # Plots
    P0_count[period].plot(kind='bar', color='white', alpha=0.1)
    P0_count[period].plot(kind='line', marker='.', color='blue', ms=5, alpha=0.07)

    P1_count[period].plot(kind='bar', color='white', alpha=0.1)
    P1_count[period].plot(kind='line', marker='.', color='purple', ms=7)
    
    # x-axis labels
    if period == 'Month':
        lst = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
    elif period == 'Day':
        lst = list(range(32))
    elif period == 'Hour':
        lst = list(range(25))
    elif period == 'Weekday':
        lst = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # x-axis labels
    my_xticks = []
    for i in P.index:
        my_xticks.append(lst[i])
    
    plt.xticks(range(len(P.index)), my_xticks ) 

    plt.show()


def plot_confusion_matrix(cf_matrix):
    '''
    Plot Confusion Matrix
    '''
    group_names = ['TN','FP','FN','TP']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


def lasso_anova_features_importance(X,y):
    '''
    Plots Features importance using Lasso and Anova
    '''
    feature_names = X.columns
    X = X.values
    y = y.values

    # Anova
    selector = feature_selection.SelectKBest(score_func=  
                   feature_selection.f_classif, k=10).fit(X,y)
    anova_selected_features = feature_names[selector.get_support()]

    # Lasso regularization
    selector = feature_selection.SelectFromModel(estimator= 
                  LogisticRegression(C=1, penalty="l1", 
                  solver='liblinear'), max_features=10).fit(X,y)
    lasso_selected_features = feature_names[selector.get_support()]

    # Plot
    df_features = pd.DataFrame({"features":feature_names})
    df_features["anova"] = df_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
    df_features["num1"] = df_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
    df_features["lasso"] = df_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
    df_features["num2"] = df_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
    df_features["method"] = df_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
    df_features["selection"] = df_features["num1"] + df_features["num2"]
    sns.barplot(y="features", x="selection", hue="method", data=df_features.sort_values("selection", ascending=False), dodge=False)


def random_forest_features_importance(X,y):
    feature_names = X.columns.tolist()
    X = X.values
    y = y.values


    # Importance
    model = ensemble.RandomForestClassifier(n_estimators=100,
                          criterion="entropy", random_state=0)
    model.fit(X,y)
    importances = model.feature_importances_

    # Put in a pandas dtf
    df_importances = pd.DataFrame({"IMPORTANCE":importances, 
                "VARIABLE":feature_names}).sort_values("IMPORTANCE", 
                ascending=False)
    df_importances['cumsum'] = df_importances['IMPORTANCE'].cumsum(axis=0)
    df_importances = df_importances.set_index("VARIABLE")

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    df_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
                    kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    df_importances[["cumsum"]].plot(kind="line", linewidth=4, 
                                     legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(df_importances)), 
              xticklabels=df_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()


def ROC_curves_KfoldCV(X, y, model, num_splits):
    cv = model_selection.StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=0)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0,1,100)
    fig = plt.figure()
    i = 1
    for train, test in cv.split(X, y):
       prediction = model.fit(X[train],
                    y[train]).predict_proba(X[test])
       fpr, tpr, t = metrics.roc_curve(y[test], prediction[:, 1])
       tprs.append(np.interp(mean_fpr, fpr, tpr))
       roc_auc = metrics.auc(fpr, tpr)
       aucs.append(roc_auc)
       plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
       i = i+1

    plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC K-Fold Validation')
    plt.legend(loc="lower right")
    plt.show()