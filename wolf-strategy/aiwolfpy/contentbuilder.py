# 2.1
def ESTIMATE(target, role):
    return 'ESTIMATE Agent[' + "{0:02d}".format(int(target)+1) + '] ' + role


def COMINGOUT(target, role):
    return 'COMINGOUT Agent[' + "{0:02d}".format(int(target)+1) + '] ' + role

# 2.2


def DIVINE(target):
    return 'DIVINE Agent[' + "{0:02d}".format(int(target)+1) + ']'


def GUARD(target):
    return 'GUARD Agent[' + "{0:02d}".format(int(target)+1) + ']'


def VOTE2(agent, target):
    if target == "ANY":
        return "{0:02d}".format(int(agent)+1) + ' VOTE ' + "ANY"
    else:
        return "{0:02d}".format(int(agent)+1) + ' VOTE Agent[' + "{0:02d}".format(int(target)+1) + ']'


def VOTE(target):
    if target == "ANY":
        return 'VOTE ' + "ANY"
    else:
        return 'VOTE Agent[' + "{0:02d}".format(int(target)+1) + ']'


def ATTACK(target):
    return 'ATTACK Agent[' + "{0:02d}".format(int(target)+1) + ']'

# 2.3


def DIVINED(target, species):
    return 'DIVINED Agent[' + "{0:02d}".format(int(target)+1) + '] ' + species


def IDENTIFIED(target, species):
    return 'IDENTIFIED Agent[' + "{0:02d}".format(int(target)+1) + '] ' + species


def GUARDED(target):
    return 'GUARDED Agent[' + "{0:02d}".format(int(target)+1) + ']'


def VOTED(target):
    return 'VOTED Agent [' + "{0:02d}".format(int(target)+1) + ']'


def ATTACKED(target):
    return 'ATTACKED Agent [' + "{0:02d}".format(int(target)+1) + ']'


"""
# 2.4
def agree(talktype, day, id):
    return 'AGREE '+ talktype + ' day' + str(day) + ' ID:' + str(id)

def disagree(talktype, day, id):
    return 'DISAGREE '+ talktype + ' day' + str(day) + ' ID:' + str(id)
"""


def AGREE(talktype, day, id):
    return 'AGREE ' + talktype + ' day' + str(day) + ' ID:' + str(id)


def DISAGREE(talktype, day, id):
    return 'DISAGREE ' + talktype + ' day' + str(day) + ' ID:' + str(id)
# 2.5


def skip():
    return 'Skip'


def over():
    return 'Over'

# 3


def REQUEST(target, text):
    if target == "ANY":
        return 'REQUEST ' + str(target) + " (" + text + ")"
    else:
        return 'REQUEST Agent[' + "{0:02d}".format(int(target)+1) + ']' + " ( " + text + " )"


def INQUIRE(target, text):
    if target == "ANY":
        return 'INQUIRE ' + str(target) + " (" + text + ")"
    else:
        return 'INQUIRE Agent[' + "{0:02d}".format(int(target)+1) + ']' + " ( " + text + " )"


def BECAUSE(sentence1, sentence2):
    return "BECAUSE (" + sentence1 + ") (" + sentence2 + ")"


def AND(sentence1, sentence2):
    return "AND (" + sentence1 + ") (" + sentence2 + ")"
