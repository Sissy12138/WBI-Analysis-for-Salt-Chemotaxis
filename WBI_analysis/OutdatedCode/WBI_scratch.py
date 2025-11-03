# 可视化所有行为变量的热图
def plot_correlation_heatmap_from_df(analysis_results_df,
                                     alpha_original=0.05,
                                     title="Neuron-Behavior Correlation Heatmap",
                                     fs=16):
    """
    根据 cross_correlation_shuffle_analysis 函数的DataFrame输出，绘制相关性热图。

    参数:
    analysis_results_df (pd.DataFrame): cross_correlation_shuffle_analysis 函数返回的DataFrame。
    alpha_original (float): 原始的显著性水平，用于与校正后的P值比较。默认为0.05。
    title (str): 图表的标题。

    """
    # 保证顺序一致
    original_neuron_order = analysis_results_df['Neuron'].unique()
    original_behavior_order = analysis_results_df['Behavior'].unique()

    # 2. 将 'Neuron' 和 'Behavior' 列转换为有序的分类类型
    analysis_results_df['Neuron'] = pd.Categorical(
        analysis_results_df['Neuron'], categories=original_neuron_order, ordered=True
    )
    analysis_results_df['Behavior'] = pd.Categorical(
        analysis_results_df['Behavior'], categories=original_behavior_order, ordered=True
    )
    # 将长格式的DataFrame转换为宽格式的矩阵
    true_correlations = analysis_results_df.pivot_table(
        index='Behavior', columns='Neuron', values='true_r'
    )
    p_values_corrected = analysis_results_df.pivot_table(
        index='Behavior', columns='Neuron', values='p_cor'
    )
    width = true_correlations.shape[1] * 1.2
    height = true_correlations.shape[0] * 0.8
    fig = plt.figure(figsize=(width, height))  # Create figure and axes explicitly
    # Create a divider for the existing axes. This makes it easy to append new axes next to it.
    ax = fig.add_axes([0.1, 0.1, 0.95, 0.9])  # 主图在左边
    # 添加 colorbar axes
    cbar_ax = fig.add_axes([0.8, 0.1, 0.03, 0.9])  # colorbar 精确位置（右侧）
    sns.heatmap(
        ax=ax,
        true_correlations,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        cbar_ax=cbar_ax,
        linewidths=.5,
        linecolor='grey',
        square=True
    )

    significance_levels = {
        0.001: "***",  # p < 0.001
        0.01: "**",  # p < 0.01
        0.05: "*",  # p < 0.05
    }
    sorted_levels = sorted(significance_levels.keys())
    # 在显著的格子上添加星号 (*) 标记
    for i in range(true_correlations.shape[0]):
        # 对于每个行为（每一行）
        for j in range(true_correlations.shape[1]):
            # 对于每个神经元(每一列)
            #             print(true_correlations.columns[j])
            # 检查P值是否有效且小于原始alpha
            p_val = p_values_corrected.iloc[i, j]
            true_r = true_correlations.iloc[i, j]

            # Check if p-value and correlation are valid
            if not pd.isna(p_val) and not pd.isna(true_r):
                asterisks = ""
                # Iterate through significance levels to find the appropriate number of asterisks
                for level in sorted_levels:
                    if p_val <= level:
                        asterisks = significance_levels[level]
                        # We found the highest level of significance, no need to check smaller p-values
                        break

                # If asterisks were assigned (meaning it's significant at alpha_original or lower)
                text_obj = ax.texts[i * true_correlations.shape[1] + j]
                current_text = text_obj.get_text()
                if asterisks and p_val <= alpha_original:  # Ensure it's below the general alpha as well

                    text_obj.set_text(f"{asterisks}")
                    text_obj.set_color('black')
                    text_obj.set_fontsize(10)
                else:
                    text_obj.set_text('')
    cbar_ax.tick_params(labelsize=fs * 0.6)  # Example: adjust colorbar tick font size
    cbar_ax.set_ylabel('Correlation Value', rotation=270, labelpad=15, fontsize=fs * 0.75)
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel("Neuron Index", fontsize=fs * 0.75)
    ax.set_ylabel("Neurons", fontsize=fs * 0.75)
    ax.tick_params(axis='x', rotation=45, labelsize=fs * 0.5)
    ax.tick_params(axis='y', rotation=0, labelsize=fs * 0.5)
    plt.tight_layout()
    plt.show()
