import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb


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
        self.agentIdx = int(self.base_info["agentIdx"]) - 1
        self.suspicion = {}
        for i in range(self.playerNum):
            if i == self.agentIdx:
                self.suspicion[str(i)] = -1000
            else:
                self.suspicion[str(i)] = 0
        self.AGREESentenceQue = deque([])
        self.DISAGREESentenceQue = deque([])
        self.myESTIMATE = None
        self.RequestQue = deque([])
        self.day = -1
        self.divineans = None
        self.CoFlag = [False for _ in range(self.playerNum)]

    def update_talk_suspicion_villager(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 1
            self.suspicion[str(content[1])] += 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1])] -= 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == self.myrole:
            self.suspicion[str(agent)] += 5
        if content[0] == "COMINGOUT" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 5
        if content[0] == "COMINGOUT" and content[2] == "SEER":
            self.suspicion[str(agent)] += 2

    def update_talk_suspicion_werewolf(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 1
            self.suspicion[str(content[1])] -= 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1])] += 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 5
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "VILLAGER":
            self.suspicion[str(agent)] -= 5

    def update_talk_agreedisagree(self, agent, idx, content):
        idx = str(idx).zfill(3)
        if agent == self.agentIdx:
            if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
                # print("muESTIMATE")
                self.myESTIMATE = ("TALK", str(self.day).zfill(2), idx)
            return None
        if int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1]) == 0:
            return None
        if content[0] == "ESTIMATE":
            if content[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) and (content[2] == "WEREWOLF" or content[2] == "POSSESSED"):
                self.AGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
            else:
                self.DISAGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
        if content[0] == "VOTE" or content[0] == "VOTED":
            if content[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]):
                self.AGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
            else:
                self.DISAGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))

    def update_talk_request(self, row, content):
        idx = str(row[1]["idx"]).zfill(3)
        content_ = splitText.splitText(content[2])
        if len(content_) >= 1:
            if content_[0] == "COMINGOUT" and (content_[1] == "ANY" or content_[1] == self.agentIdx):
                self.co_rate = 2
                self.agree_co = ("TALK", str(self.day).zfill(2), idx)

    def update_talk(self, row):
        text, agent = row[1]["text"], int(row[1]["agent"]) - 1
        texts = splitText.parse_text(text)
        for text in texts:
            content = splitText.splitText(text)
            if len(content) != 0:
                if self.myrole == "VILLAGER" or self.myrole == "SEER":
                    self.update_talk_suspicion_villager(agent, content)
                else:
                    self.update_talk_suspicion_werewolf(agent, content)
                self.update_talk_agreedisagree(agent, row[1]["idx"], content)
                if content[0] == "REQUEST" and (content[1] == "ANY" or content[1] == self.agentIdx):
                    self.update_talk_request(row, content)

    def update_dead(self, row):
        self.suspicion[str(int(row[1]["agent"]) - 1)] -= 1000

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
        self.day += 1
        self.voteop = 1
        self.old_voteop = None
        self.isCo = False
        self.isVote = False
        self.isBecause = False
        self.isRequestVote = False
        self.isDivine = False
        self.co_rate = random.uniform(0, 1)

    def vote(self):
        return int(self.voteop)+1

    def finish(self):
        return None

    def talk(self):
        if self.co_rate != 2:
            self.co_rate = random.uniform(0, 1)
        if not self.isCo and self.day == 1 and self.co_rate >= 0.5:
            if self.co_rate == 2:
                return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "VILLAGER"))
            else:
                return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif not self.isVote:
            if int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1]) == 0:
                return cb.skip()
            self.voteop = int(sorted(self.suspicion.items(),
                                     key=lambda x: x[1])[-1][0])
            if self.old_voteop != self.voteop and self.old_voteop != None:
                return cb.DISAGREE(self.myESTIMATE[0], self.myESTIMATE[1], self.myESTIMATE[2])
            self.isVote = True
            self.old_voteop = self.voteop
            return cb.VOTE(self.voteop)

        elif not self.isBecause:
            if int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1]) == 0:
                return cb.skip()
            self.isBecause = True
            return cb.BECAUSE(cb.ESTIMATE(self.voteop, "WEREWOLF"), cb.VOTE(self.voteop))
        elif not self.isRequestVote:
            if int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1]) == 0:
                return cb.skip()
            self.isRequestVote = True
            return cb.REQUEST("ANY", cb.VOTE(self.voteop))
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 2:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        index = 0
        while True:
            if index == self.playerNum:
                return cb.skip()
            if not self.CoFlag[index]:
                self.CoFlag[index] = True
                return cb.REQUEST(index, cb.COMINGOUT(index, "ANY"))
            else:
                index += 1
        return cb.skip()
