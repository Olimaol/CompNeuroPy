from ANNarchy import Synapse

factor_synapse = Synapse(
    parameters = """
        max_trans  = 0
        mod_factor = 0
    """,    
    equations = "",    
    pre_spike = """
        g_target += w * mod_factor : max = max_trans
    """,
    name = "factor_synapse",
    description = "Synapse which scales the transmitted value by a specified factor. Factor is equivalent to the connection weight if weight==1."
)
