import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import os

def classifier_test(feature_df, target, classifier, params, iter_gen, iter_param, use_node_count, col_name, runs, rand_state, start_seed):
    full_results = []
    
    for ii in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(feature_df, target, test_size=0.2, random_state=start_seed + ii)

        for jj in iter_gen:
            params[iter_param] = jj
            
            if rand_state:
                params['random_state'] = start_seed + ii
            model = classifier(**params)
            model.fit(X_train, y_train)
            
            test_pred = model.predict(X_test)
            
            full_results.append({
                'Test Accuracy': accuracy_score(y_test, test_pred),
                'Test F1': f1_score(y_test, test_pred, average='weighted'),
                # 'Cost-Complexity Pruning Alpha': alpha,
                col_name: jj if not use_node_count else model.tree_.node_count,
                'Trial Num': ii
            })
            
    return pd.DataFrame(full_results)  


def stratify_dataframe(df, col, metric):
    new_df = df.groupby(col)[metric].quantile([0, 0.25, 0.5, 0.75, 1]).reset_index().rename(columns={'level_1': 'Level'})
    new_df['Level'] = np.select(
        [new_df['Level'].eq(0), new_df['Level'].eq(0.25), new_df['Level'].eq(0.5), new_df['Level'].eq(0.75), new_df['Level'].eq(1)],
        ['Min', '1st Quartile', 'Median', '3rd Quartile', 'Max'],
        default=''
    )
     
    return new_df   


def plot_classifier_results(classifier_name, results_df, col_name, use_log, plots_dir, include_f1=False):
    
    half_pal = sns.color_palette('rocket', 3)
    palette = half_pal[::-1] + half_pal[1:]
    
    if include_f1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.suptitle(f'Performance of {classifier_name} Across Test Sets')
    if use_log:
        plt.xscale('log')
    
    # Top Plot
    acc_strat_df = stratify_dataframe(results_df, col_name, 'Test Accuracy')
    topline = sns.lineplot(x=col_name, y='Test Accuracy', data=acc_strat_df, hue='Level', palette=palette, alpha=0.6, ax=ax1, legend=True)
    topline_arr = topline.get_lines()
    ax1.legend(topline_arr, title='Quantiles', labels=['Min', '_', '1st Q', '_', 'Median', '_', '3rd Q', '_', 'Max'], loc='lower right')
    ax1.fill_between(topline_arr[0].get_xdata(), topline_arr[0].get_ydata(), topline_arr[1].get_ydata(), color=palette[0], alpha=.1, label='_')
    ax1.fill_between(topline_arr[0].get_xdata(), topline_arr[1].get_ydata(), topline_arr[2].get_ydata(), color=palette[1], alpha=.1, label='_')
    ax1.fill_between(topline_arr[0].get_xdata(), topline_arr[2].get_ydata(), topline_arr[3].get_ydata(), color=palette[1], alpha=.1, label='_')
    ax1.fill_between(topline_arr[0].get_xdata(), topline_arr[3].get_ydata(), topline_arr[4].get_ydata(), color=palette[0], alpha=.1, label='_')
    
    # Bottom Plot
    if include_f1:
        f1_strat_df = stratify_dataframe(results_df, col_name, 'Test F1')
        botline = sns.lineplot(x=col_name, y='Test F1', data=f1_strat_df, hue='Level', palette=palette, alpha=0.6, ax=ax2, legend=True)
        botline_arr = botline.get_lines()
        ax2.legend(topline_arr, title='Quantiles', labels=['Min', '_', '1st Q', '_', 'Median', '_', '3rd Q', '_', 'Max'], loc='lower right')
        ax2.fill_between(botline_arr[0].get_xdata(), botline_arr[0].get_ydata(), botline_arr[1].get_ydata(), color=palette[0], alpha=.1, label='_')
        ax2.fill_between(botline_arr[0].get_xdata(), botline_arr[1].get_ydata(), botline_arr[2].get_ydata(), color=palette[1], alpha=.1, label='_')
        ax2.fill_between(botline_arr[0].get_xdata(), botline_arr[2].get_ydata(), botline_arr[3].get_ydata(), color=palette[1], alpha=.1, label='_')
        ax2.fill_between(botline_arr[0].get_xdata(), botline_arr[3].get_ydata(), botline_arr[4].get_ydata(), color=palette[0], alpha=.1, label='_')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{classifier_name.lower().replace(" ", "_")}_performance.png'))
    plt.show()