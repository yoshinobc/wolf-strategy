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
        target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "GUARDED":
        target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "VOTE":
        target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "ATTACK":
        target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "DIVINED":
        target = int(temp[1][6:8]) - 1
        species = temp[2]
        return [topic, target, species]
    elif topic == "IDENTIFIED":
        target = int(temp[1][6:8]) - 1
        species = temp[2]
        return [topic, target, species]
    elif topic == "GUARDED":
        target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "VOTED":
        target = int(temp[1][6:8]) - 1
        species = temp[2]
        return [topic, target]
    elif topic == "ATTACKED":
        target = int(temp[1][6:8]) - 1
        return [topic, target]
    elif topic == "AGERR":
        return [topic]
    elif topic == "DISAGREE":
        return [topic]
    elif topic == "REQUEST":
        target = int(temp[1][6:8]) - 1
        sentence = temp[2]
        return [topic, target, sentence]
    elif topic == "INQUIRE":
        target = int(temp[1][6:8]) - 1
        sentence = temp[2]
    elif topic == "BECAUSE":
        target = int(temp[1][6:8]) - 1
        sentence = temp[2]
        return [topic, target, sentence]
    elif topic == "DAY":
        day = int(temp[1])
        sentence = temp[3]
        return [topic, target, sentence]
    elif topic == "NOT":
        sentence = temp[1]
        return [topic, sentence]
    elif topic == "AND":
        sentence = temp[1]
        return [topic, sentence]
    elif topic == "OR":
        sentence1 = temp[1]
        sentence2 = temp[2]
        return [topic, sentence1, sentence2]
    elif topic == "XOR":
        sentence1 = temp[1]
        sentence2 = temp[2]
        return [topic, sentence1, sentence2]
    else:
        return []
