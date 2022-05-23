from ANNarchy import get_population

class req_pop_attr:
    def __init__(self, pop, attr):
        """
            pop: string or list of strings, population name(s)
            attr: string or list of strings, attribute name(s)
        """
        self.pop=pop
        self.attr=attr
        ### convert single strings into list
        if not(isinstance(pop,list)): self.pop=[pop]  
        if not(isinstance(attr,list)): self.attr=[attr]  
    def run(self):
        """
            checks if population(s) contains the attribute(s) (parameters or variables)
        """
        for attr in self.attr:
            for pop in self.pop:
                assert attr in vars(get_population(pop))['attributes'], 'Error: Population '+pop+' does not contain attribute '+attr+'!\n'
