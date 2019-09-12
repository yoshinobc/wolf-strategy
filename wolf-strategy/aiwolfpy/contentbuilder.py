# 2.1
def ESTIMATE(target, role):
    return 'ESTIMATE Agent[' + "{0:02d}".format(target) + '] ' + role


def COMINGOUT(target, role):
    return 'COMINGOUT Agent[' + "{0:02d}".format(target) + '] ' + role

# 2.2


def DIVINE(target):
    return 'DIVINE Agent[' + "{0:02d}".format(target) + ']'


def GUARD(target):
    return 'GUARD Agent[' + "{0:02d}".format(target) + ']'


def VOTE(target):
    return 'VOTE Agent[' + "{0:02d}".format(target) + ']'


def ATTACK(target):
    return 'ATTACK Agent[' + "{0:02d}".format(target) + ']'

# 2.3


def DIVINED(target, species):
    return 'DIVINED Agent[' + "{0:02d}".format(target) + '] ' + species


def IDENTIFIED(target, species):
    return 'IDENTIFIED Agent[' + "{0:02d}".format(target) + '] ' + species


def GUARDED(target):
    return 'GUARDED Agent[' + "{0:02d}".format(target) + ']'


def VOTED(target):
    return 'VOTED Agent [' + "{0:02d}".format(target) + ']'


def ATTACKED(target):
    return 'ATTACKED Agent [' + "{0:02d}".format(target) + ']'


"""
# 2.4
def agree(talktype, day, id):
    return 'AGREE '+ talktype + ' day' + str(day) + ' ID:' + str(id)

def disagree(talktype, day, id):
    return 'DISAGREE '+ talktype + ' day' + str(day) + ' ID:' + str(id)
"""


def AGREE(talknumber):
    return 'AGREE' + str(talknumber)


def DISAGREE(talknumber):
    return 'DISAGREE' + str(talknumber)
# 2.5


def skip():
    return 'Skip'


def over():
    return 'Over'

# 3


def request(text):
    return 'REQUEST(' + text + ''
