import pingouin as pg
from CompNeuroPy import extra_functions as ef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def anova_between_groups(df: pd.DataFrame, dv: str, group_list: list[str]):
    """
    Perform a N-way ANOVA with post-hoc tests.

    Args:
        df (pd.DataFrame):
            dataframe with the data
        dv (str):
            dependent variable
        group_list (list[str]):
            list of independent variables
    """

    ef.print_df(df)
    print("\n")

    # 1. This is a between subject design, so the first step is to test for equality of variances
    for group in group_list:
        equal_var = pg.homoscedasticity(data=df, dv=dv, group=group)
        print(f"equal_var {group}:")
        ef.print_df(equal_var)
        print("\n")

    # 2. If the groups have equal variances, we can use a regular N-way ANOVA
    anova = pg.anova(data=df, dv=dv, between=group_list)
    print("ANOVA table:")
    ef.print_df(anova)
    print("\n")

    # 3. If there is a main effect, we can proceed to post-hoc tests
    post_hoc = pg.pairwise_tests(
        data=df, dv=dv, between=group_list, padjust="fdr_bh", return_desc=True
    )
    print("Post-hoc tests:")
    ef.print_df(post_hoc)
    print("\n")

    # create boxplot for each group
    df.boxplot(column=dv, by=group_list, figsize=(12, 8))
    plt.tight_layout()
    plt.savefig("boxplot.png", dpi=300)
