nameScope = ''
scopesUsed = set()

class name_scope(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global nameScope

        newScope = (nameScope + self.name + '/')

        if newScope not in scopesUsed:
            scopesUsed.add(self.name)
        else:
            print('Scope ' + newScope + ' already used!')
            exit(1)

        nameScope = newScope

    def __exit__(self, type, value, traceback):
        global nameScope

        loc = nameScope.find(self.name)
        nameScope = nameScope[:loc]

def modelGraphViz(model, fileName):

    from graphviz import Digraph

    graph = Digraph(format='png')

    modelGraphViz_(model, graph)

    graph.render(fileName)

    return graph

def modelGraphViz_(model, graph):

    if not getattr(model, 'fitComponents', None) or len(model.fitComponents) == 0:
        return

    graph.node(model.name, model.name)

    for c in model.fitComponents.values():
        graph.node(c.name, c.name)
        graph.edge(model.name, c.name)

        if not getattr(c, "getInitialParameterValuesAndStepSizes", None): # Is a distribution
            for p in c.getParameters().values():

                graph.node(p.name, p.name)
                graph.edge(c.name, p.name)

                # For constrained parameters
                for d in p.kw.values():
                    if getattr(d, 'name'):
                        graph.node(d.name, d.name)
                        graph.edge(p.name, d.name)

        modelGraphViz_(c, graph)
