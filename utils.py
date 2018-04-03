nameScope = ''

class name_scope(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global nameScope
        nameScope += self.name + '/'

    def __exit__(self, type, value, traceback):
        global nameScope
        loc = nameScope.find(self.name)
        nameScope = nameScope[:loc]
