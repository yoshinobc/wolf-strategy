import os
import numpy as np
from . import splitText
"""
estimate:0
co:1
vote:2
divined:3
identified:4
agree:5
disagree:6
request:7
inquire:8
because:9
"""


class preprocess1(object):
    def __init__(self):
        self.agentNum = 5
        self.content_map = {
            "ESTIMATE": 0, "COMINGOUT": 1, "VOTE": 2, "DIVINED": 3, "IDENTIFIED": 4, "AGREE": 5, "DISAGREE": 6, "REQUEST": 7, "INQUIRE": 8, "BECAUSE": 9, "DEAD": 10, "execute": 11, "divined": 12, "DAY": 13
        }
        self.f_map = np.zeros(
            (self.agentNum, self.agentNum, len(self.content_map.keys())))
        self.y_map = np.zeros((5, 5))
        self.is_divine = False

    def update_result(self):
        self.y_map = np.array(self.y_map).flatten()
        self.y_map1 = self.y_map[:5]
        self.y_map2 = self.y_map[5:10]
        self.y_map3 = self.y_map[10:15]
        self.y_map4 = self.y_map[15:20]
        self.y_map5 = self.y_map[20:25]

    def update_status(self, contents):
        if "Sample" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent][0] = 1

        elif "CALM" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent][1] = 1

        elif "Liar" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent][2] = 1

        elif "REPEL" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent][3] = 1

        elif "Follow" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent][4] = 1

    def update_divine(self, content):
        agent = int(content[2]) - 1
        op = int(content[3]) - 1
        self.f_map[agent][op][self.content_map[content[4]]] += 1

    def update_dead(self, content):
        agent = int(content[2]) - 1
        self.f_map[agent][agent][self.content_map[content[1]]] += 1

    def update_talk_content(self, agent, content):
        if len(content) == 0 or content[0] == "AGREE" or content[0] == "DISAGREE":
            return
        # print(content)
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
