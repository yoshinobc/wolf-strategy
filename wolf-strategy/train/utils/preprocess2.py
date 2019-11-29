"""
[agent, opponent, ESTIMATE , COMINGOUT, VOTE, DIVINED , IDENTIFIED , AGREE,
    DISAGREE , REQUEST , INQUIRE , BECAUSE , DEAD , execute , divined , DAY]
agent: 1 - 5,6ANY
opponent: 1 - 5,6ANY
estimate: 0, 1, 2, 3(werewolf, villager, seer, possessed)
comingout: 0, 1, 2, 3
vote: 0, 1
divined: 0, 1(werewolf,villager)
identified: 0, 1(werewolf,villager)
agree: 0, 1
disagree: 0, 1
request: 0, 1
inquire: 0, 1
because: 0, 1
dead: 0, 1
execute: 0, 1
divined(結果):0,1
day:0,1,2
"""
import numpy as np
import os
from . import splitText


class preprocess2(object):
    def __init__(self):
        self.agentNum = 5
        self.content_map = {
            "agent": 0, "opponent": 1,
            "ESTIMATE": 2, "COMINGOUT": 3, "VOTE": 4, "DIVINED": 5, "IDENTIFIED": 6, "AGREE": 7, "DISAGREE": 8, "REQUEST": 9, "INQUIRE": 10, "BECAUSE": 11, "attack": 12, "execute": 13, "divined": 14, "DAY": 15
        }
        self.f_maps = []
        self.y_map = np.zeros(25)
        self.is_divine = False
        self.is_finish = False

    def update_result(self):
        self.y_map1 = self.y_map[:5]
        self.y_map2 = self.y_map[5:10]
        self.y_map3 = self.y_map[10:15]
        self.y_map4 = self.y_map[15:20]
        self.y_map5 = self.y_map[20:25]
        self.is_finish = True

    def update_status(self, contents):
        if "Sample" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent*5+0] = 1

        elif "CALM" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent*5+1] = 1

        elif "Liar" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent*5+2] = 1

        elif "REPEL" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent*5+3] = 1

        elif "Follow" in contents[5]:
            agent = int(contents[2]) - 1
            self.y_map[agent * 5 + 4] = 1

    def update_talk_content(self, agent, content):
        if len(content) == 0:
            return
        if content[0] == "AGREE" or content[0] == "DISAGREE":
            self.f_map[self.content_map[content[0]]] = 1
            self.f_map[self.content_map["agent"]] = agent
            return
        if type(content[1]) == int:
            op = int(content[1])
            self.f_map[self.content_map["agent"]] = agent
            self.f_map[self.content_map["opponent"]] = op
            self.f_map[self.content_map[content[0]]] = 1
        elif content[1] == "ANY":
            self.f_map[self.content_map["agent"]] = agent
            self.f_map[self.content_map["opponent"]] = 5
            self.f_map[self.content_map[content[0]]] = 1
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

    def update_divine(self, content):
        agent = int(content[2]) - 1
        op = int(content[3]) - 1
        self.f_map[self.content_map["agent"]] = agent
        self.f_map[self.content_map["opponent"]] = op
        self.f_map[self.content_map[content[1]]] = 1

    def update_attack(self, content):
        agent = int(content[0]) - 1
        op = int(content[2]) - 1
        self.f_map[self.content_map["agent"]] = agent
        self.f_map[self.content_map["opponent"]] = op
        self.f_map[self.content_map[content[1]]] = 1

    def update_dead(self, content):
        agent = int(content[2]) - 1
        self.f_map[self.content_map["agent"]] = agent
        self.f_map[self.content_map[content[1]]] = 1

    def update(self, file_name):
        f = open(file_name, mode="r")
        line = f.readline()
        if line[3] == "SEER":
            self.is_divine = True
        while line:
            self.f_map = np.zeros(len(self.content_map))
            line = f.readline().rstrip(os.linesep)
            contents = line.split(",")
            print(contents)
            if len(contents) == 1:
                continue
            elif contents[1] == "talk":
                self.update_talk(contents)
            elif contents[1] == "execute":
                self.update_dead(contents)
            elif contents[1] == "attack" and contents[3] == "true":
                self.update_attack(contents)
            elif contents[1] == "divine" and self.is_divine:
                self.update_divine(contents)
            elif contents[1] == "status":
                self.update_status(contents)
            elif contents[1] == "result":
                self.update_result()
            if self.f_map.any():
                self.f_maps.append(self.f_map)
