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
