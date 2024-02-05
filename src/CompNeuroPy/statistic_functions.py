import pingouin as pg
from CompNeuroPy import extra_functions as ef


def tmp():

    # Load an example dataset comparing pain threshold as a function of hair color
    df = pg.read_dataset("anova2")
    ef.print_df(df)
    print("\n")

    # 1. This is a between subject design, so the first step is to test for equality of variances
    equal_var_blend = pg.homoscedasticity(data=df, dv="Yield", group="Blend")
    equal_var_crop = pg.homoscedasticity(data=df, dv="Yield", group="Crop")
    print("equal_var_blend:")
    ef.print_df(equal_var_blend)
    print("\n")
    print("equal_var_crop:")
    ef.print_df(equal_var_crop)
    print("\n")

    # 2. If the groups have equal variances, we can use a regular two-way ANOVA
    anova = pg.anova(data=df, dv="Yield", between=["Blend", "Crop"])
    print("ANOVA table:")
    ef.print_df(anova)
    print("\n")

    # 3. If there is a main effect, we can proceed to post-hoc tests
    post_hoc = pg.pairwise_tests(
        data=df, dv="Yield", between=["Blend", "Crop"], padjust="fdr_bh"
    )
    print("Post-hoc tests:")
    ef.print_df(post_hoc)
    print("\n")
