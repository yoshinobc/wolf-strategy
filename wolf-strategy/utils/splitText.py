import re


def splitText(text):
    temp = text.split()
    topic = temp[0]
    if topic == "ESTIMATE":
        target = int(temp[1][6:8]) - 1
        role = temp[2]
        return [topic, target, role]
    elif topic == "COMINGOUT":
        target = int(temp[1][6:8]) - 1
        role = temp[2]
        return [topic, target, role]
    elif topic == "DIVINATION":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "GUARDED":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "VOTE":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "ATTACK":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "DIVINED":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        species = temp[2]
        return [topic, target, species]
    elif topic == "IDENTIFIED":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        species = temp[2]
        return [topic, target, species]
    elif topic == "GUARDED":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "VOTED":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        species = temp[2]
        return [topic, target]
    elif topic == "ATTACKED":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "AGERR":
        return [topic]
    elif topic == "DISAGREE":
        return [topic]
    elif topic == "REQUEST":
        if temp[1][6:8] in "ANY":
            target = "ANY"
        else:
            target = int(temp[1][6:8]) - 1
        flag = False
        sentence = ""
        for t in text:
            if flag and t != ")":
                sentence += t
            if t == "(":
                flag = True
        return [topic, target, sentence]
    elif topic == "INQUIRE":
        sentence1 = temp[1]
        sentence2 = temp[2]
        return [topic, sentence1, sentence2]
    elif topic == "BECAUSE":
        sentence1 = temp[1]
        sentence2 = temp[2]
        return [topic, sentence1, sentence2]
    elif topic == "DAY":
        day = int(temp[1])
        sentence = temp[3]
        return [topic, day, sentence]
    elif topic == "NOT":
        return []
        sentences = parse_text(temp[1])
        return [topic, sentences]
    elif topic == "AND":
        return []
        sentences = parse_text(temp[1])
        return [topic, sentences]
    elif topic == "OR":
        return []
        sentences = parse_text(temp[1])
        return [topic, sentences]
    elif topic == "XOR":
        sentence1 = temp[1]
        sentence2 = temp[2]
        return [topic, sentence1, sentence2]
    else:
        return []


def parse_text_sub(text):
    stack = []
    for i, c in enumerate(text):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            start = stack.pop()
            yield (len(stack), text[start + 1:i])


def parse_text(text):
    text = "(" + text + ")"
    a = list(parse_text_sub(text))
    clist = []
    for i, content in a:
        clist.append(content)
    return list(set(clist))
