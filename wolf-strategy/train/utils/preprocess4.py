import os
import numpy as np
from . import splitText


class preprocess3(object):
    def __init__(self):
        self.agentNum = 5
        self.content_map = {
            "ESTIMATE": 0, "COMINGOUT": 1, "VOTE": 2, "DIVINED": 3, "IDENTIFIED": 4, "AGREE": 5, "DISAGREE": 6, "REQUEST": 7, "INQUIRE": 8, "BECAUSE": 9, "DEAD": 10, "execute": 11, "divined": 12, "DAY": 13
        }
        self.f_map = np.zeros(
            (self.agentNum, self.agentNum, len(self.content_map.keys())))
        self.y_map = np.zeros(20)
        self.is_divine = False
        self.is_finish = False

    def update_result(self):
        self.y_map1 = self.y_map[:4]
        self.y_map2 = self.y_map[4:8]
        self.y_map3 = self.y_map[8:12]
        self.y_map4 = self.y_map[12:16]
        self.y_map5 = self.y_map[16:20]
        self.is_finish = True

    def update_status(self, contents):
        if "VILLAGER" in contents[3]:
            agent = int(contents[2]) - 1
            self.y_map[agent * 4 + 0] = 1
            # self.sample.append(agent)

        elif "SEER" in contents[3]:
            agent = int(contents[2]) - 1
            self.y_map[agent * 4 + 1] = 1
            # self.calm.append(agent)

        elif "POSSESSED" in contents[3]:
            agent = int(contents[2]) - 1
            self.y_map[agent * 4 + 2] = 1
            # self.liar.append(agent)

        elif "WEREWOLF" in contents[3]:
            agent = int(contents[2]) - 1
            self.y_map[agent * 4 + 3] = 1
            # self.repel.append(agent)

    def update_divine(self, content):
        agent = int(content[2]) - 1
        op = int(content[3]) - 1
        self.f_map[agent][op][self.content_map[content[4]]] += 1

    def update_dead(self, content):
        agent = int(content[2]) - 1
        self.f_map[agent][agent][self.content_map[content[1]]] += 1

    def update_talk_content(self, agent, content):
        if len(content) == 0:
            return

        if content[0] == "AGREE" or content[0] == "DISAGREE":
            self.f_map[agent][agent][self.content_map[content[0]]] += 1
            return
        if type(content[1]) == int:
            op = int(content[1])
            self.f_map[agent][op][self.content_map[content[0]]] += 1
        elif content[1] == "ANY":
            for op in range(self.agentNum):
                self.f_map[agent][op][self.content_map[content[0]]] += 1
        else:
            self.update_talk_content(
                agent, splitText.splitText(content[1].replace("(", "").replace(")", "")))
            self.update_talk_content(
                agent, splitText.splitText(content[2].replace("(", "").replace(")", "")))

    def update_talk(self, contents):
        agent = int(contents[4])-1
        text = contents[5]
        texts = splitText.parse_text(text)
        for text in texts:
            content = splitText.splitText(text)
            if len(content) != 0:
                self.update_talk_content(agent, content)

    def update(self, file_name):
        f = open(file_name, mode="r")
        line = f.readline()
        #print("l", file_name, line)
        if line == "":
            return
        if line[3] == "SEER":
            self.is_divine = True
        while line:
            line = f.readline().rstrip(os.linesep)
            contents = line.split(",")
            if len(contents) == 1:
                continue
            elif contents[1] == "talk":
                self.update_talk(contents)
            elif contents[1] == "dead" or contents[1] == "execute":
                self.update_dead(contents)
            elif contents[1] == "divine" and self.is_divine:
                self.update_divine(contents)
            elif contents[1] == "status":
                self.update_status(contents)
            elif contents[1] == "result":
                self.update_result()
