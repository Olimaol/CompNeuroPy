from ANNarchy import Synapse


class FactorSynapse(Synapse):
    """
    Synapse which scales the transmitted value by a specified factor. Factor is
    equivalent to the connection weight if weight==1.

    Parameters:
        max_trans (float, optional):
            Maximum value that can be transmitted. Default: None.
        mod_factor (float, optional):
            Factor by which the weight value is multiplied. Default: 0.
    """

    def __init__(self, max_trans: None | float = None, mod_factor: float = 0):
        super().__init__(
            parameters=f"""
            {f"max_trans  = {max_trans}" if max_trans is not None else ""}
            mod_factor = {mod_factor}
        """,
            equations="",
            pre_spike=f"""
            g_target += w * mod_factor {": max = max_trans" if max_trans is not None else ""}
        """,
            name="factor_synapse",
            description="""
            Synapse which scales the transmitted value by a specified factor. Factor is
            equivalent to the connection weight if weight==1.
        """,
        )


### create objects for backward compatibility
factor_synapse = FactorSynapse(max_trans=0)
factor_synapse_without_max = FactorSynapse()
