import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
# ANYを対応させる


class Villager(object):
    def __init__(self, agent_name):
        self.myname = agent_name

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting, myrole):
        self.base_info = base_info
        self.game_setting = game_setting
        self.diff_data = diff_data
        self.playerNum = len(self.base_info["remainTalkMap"].keys())
        self.myrole = myrole
        self.suspicion = {}
        for i in range(self.playerNum):
            self.suspicion[str(i)] = 0
        self.AGREESentenceQue = deque([])
        self.DISAGREESentenceQue = deque([])
        self.RequestQue = deque([])
        self.day = 0
        self.agentIdx = self.base_info["agentIdx"]
        self.divineans = []

    def update_talk_suspicion_villager(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent - 1)] += 1
            self.suspicion[str(content[1] - 1)] += 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1] - 1)] -= 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == self.myrole:
            self.suspicion[str(agent - 1)] += 10

    def update_talk_suspicion_werewolf(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent - 1)] += 1
            self.suspicion[str(content[1] - 1)] -= 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1] - 1)] += 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "WEREWOLF":
            self.suspicion[str(agent - 1)] += 5
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "VILLAGER":
            self.suspicion[str(agent - 1)] -= 5

    def update_talk_agreedisagree(self, idx, content):
        if content[0] == "ESTIMATE":
            if content[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) + 1 and (content[2] == "WEREWOLF" or content[2] == "POSSESSED"):
                self.AGREESentenceQue.append(idx)
            else:
                self.DISAGREESentenceQue.append(idx)
        if content[0] == "VOTE":
            if content[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) + 1:
                self.AGREESentenceQue.append(idx)
            else:
                self.DISAGREESentenceQue.append(idx)

    def update_talk_request(self, content):
        pass
        # 他人のリクエストにどれぐらい答えるか

    def update_talk(self, row):
        text, agent = row[1]["text"], row[1]["agent"]
        content = splitText.splitText(text)
        if len(content) != 0:
            if self.myrole == "VILLAGER" or self.myrole == "SEER":
                self.update_talk_suspicion_villager(agent, content)
            else:
                self.update_talk_suspicion_werewolf(agent, content)
            self.update_talk_agreedisagree(row[1]["idx"], content)
            self.update_talk_request(content)

    def update_dead(self, row):
        self.suspicion[str(row[1]["agent"] - 1)] -= 1000

    def update_divine(self, row):
        text = row[1]["text"]
        content = splitText.splitText(text)
        self.divineans.append((content[1], content[2]))

    def update(self, base_info, diff_data, request):
        self.diff_data = diff_data
        for row in self.diff_data.iterrows():
            if row[1]["type"] == "talk":
                self.update_talk(row)
            if row[1]["type"] == "dead" or row[1]["type"] == "execute":
                self.update_dead(row)
            if row[1]["type"] == "divine":
                self.update_divine(row)

    def dayStart(self):
        self.talkCount = 0
        self.day += 1
        self.voteop = 1

    def vote(self):
        return self.voteop

    def finish(self):
        self.divineans = []
        return None

    def talk(self):
        if self.talkCount == 0 and self.day == 1:
            self.talkCount += 1
            return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif self.talkCount <= 3 and len(self.AGREESentenceQue) >= 1:
            self.talkCount += 1
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText)
        elif self.talkCount <= 6 and len(self.DISAGREESentenceQue) >= 1:
            self.talkCount += 1
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText)
        elif self.talkCount <= 7:
            self.talkCount += 1
            self.voteop = int(sorted(self.suspicion.items(),
                                     key=lambda x: x[1])[-1][0]) + 1
            return cb.VOTE(self.voteop)
        elif self.talkCount <= 8:
            self.talkCount += 1
            return cb.BECAUSE(cb.ESTIMATE(self.voteop, "WEREWOLF"), cb.VOTE(self.voteop))
        elif self.talkCount <= 9:
            self.talkCount += 1
            return cb.REQUEST("ANY", cb.VOTE(self.voteop))
        elif self.talkCount <= 10:
            self.talkCount += 1
            return cb.skip()
        else:
            return cb.skip()
            # INQUIREをつけるか
