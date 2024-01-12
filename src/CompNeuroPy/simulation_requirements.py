from ANNarchy import get_population, Population


class ReqPopHasAttr:
    """
    Checks if population(s) contains the attribute(s) (parameters or variables)
    """

    def __init__(self, pop, attr):
        """
        Args:
            pop (str or list of strings):
                population name(s)
            attr (str or list of strings):
                attribute name(s)
        """
        self.pop_name_list = pop
        self.attr_name_list = attr
        ### convert single strings into list
        if not (isinstance(pop, list)):
            self.pop_name_list = [pop]
        if not (isinstance(attr, list)):
            self.attr_name_list = [attr]

    def run(self):
        """
        Checks if population(s) contains the attribute(s) (parameters or variables)

        Raises:
            ValueError: if population(s) does not contain the attribute(s)
        """
        for attr_name in self.attr_name_list:
            for pop_name in self.pop_name_list:
                pop: Population = get_population(pop_name)
                if not (attr_name in pop.attributes):
                    raise ValueError(
                        "Population "
                        + pop_name
                        + " does not contain attribute "
                        + attr_name
                        + "!\n"
                    )


### old name for backwards compatibility, TODO: remove
req_pop_attr = ReqPopHasAttr
