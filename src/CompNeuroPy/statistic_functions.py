import pingouin as pg
from CompNeuroPy import extra_functions as ef
from CompNeuroPy import system_functions as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def anova_between_groups(
    df: pd.DataFrame,
    dv: str,
    group_list: list[str],
    print_file: str = "./anova_between_groups.txt",
    fig_path: str = "./boxplot.png",
):
    """
    Perform a N-way ANOVA with post-hoc tests.
    Besides creating the print_file, it also saves the ANOVA table and the post-hoc tests
    in the same folder as the print_file. (names:
    anova_between_groups_equal_var_*group_name*.pkl, anova_between_groups_anova.pkl, and
    anova_between_groups_post_hoc.pkl)

    Args:
        df (pd.DataFrame):
            dataframe with the data
        dv (str):
            dependent variable
        group_list (list[str]):
            list of independent variables
        print_file (str):
            path to save the ANOVA table
        fig_path (str):
            path to save the box plot figure
    """
    ### print the dataframe
    if "/" in print_file:
        sf.create_dir("/".join(print_file.split("/")[:-1]))
    with open(print_file, "w") as file:
        ef.print_df(df, file=file)
        print("\n", file=file)

    ### This is a between subject design, so the first step is to test for equality of
    ### variances
    equal_var_list = []
    for group in group_list:
        equal_var = pg.homoscedasticity(data=df, dv=dv, group=group)
        ### print the dataframe equal_var
        with open(print_file, "a") as file:
            print(f"equal_var {group}:", file=file)
            ef.print_df(equal_var, file=file)
            print("\n", file=file)
        ### save the dataframe equal_var
        sf.save_variables(
            [equal_var],
            [f"anova_between_groups_equal_var_{group}"],
            path="/".join(print_file.split("/")[:-1]) if "/" in print_file else "./",
        )
        ### store if equal_var is True in equal_var_list
        equal_var_list.append(equal_var["equal_var"].iloc[0])

    if all(equal_var_list):
        ### If the groups have equal variances, we can use a regular N-way ANOVA
        anova = pg.anova(data=df, dv=dv, between=group_list)
    elif len(group_list) == 1:
        ### If one group has unequal variances, we can use a Welch's ANOVA
        anova = pg.welch_anova(data=df, dv=dv, between=group_list)
    else:
        ### If more than one group has unequal variances, fall back to regular N-way ANOVA
        anova = pg.anova(data=df, dv=dv, between=group_list)
    ### print the dataframe anova
    with open(print_file, "a") as file:
        print("ANOVA table:", file=file)
        ef.print_df(anova, file=file)
        print("\n", file=file)
    ### save the dataframe anova
    sf.save_variables(
        [anova],
        ["anova_between_groups_anova"],
        path="/".join(print_file.split("/")[:-1]) if "/" in print_file else "./",
    )

    ### post-hoc tests
    post_hoc = pg.pairwise_tests(
        data=df, dv=dv, between=group_list, padjust="fdr_bh", return_desc=True
    )
    ### print the dataframe post_hoc
    with open(print_file, "a") as file:
        print("Post-hoc tests:", file=file)
        ef.print_df(post_hoc, file=file)
        print("\n", file=file)
    ### save the dataframe post_hoc
    sf.save_variables(
        [post_hoc],
        ["anova_between_groups_post_hoc"],
        path="/".join(print_file.split("/")[:-1]) if "/" in print_file else "./",
    )

    # create boxplot for each group
    df.boxplot(column=dv, by=group_list, figsize=(12, 8))
    plt.tight_layout()
    if "/" in fig_path:
        sf.create_dir("/".join(fig_path.split("/")[:-1]))
    plt.savefig(fig_path, dpi=300)
    plt.close("all")
