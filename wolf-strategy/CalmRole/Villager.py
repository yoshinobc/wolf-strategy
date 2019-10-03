import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
# ANYを対応させる
# 意見が変わった時に報告をする
# DISAGREE AGREEに反応するエージェント
# FOLLOWするエージェント INQUIREするagent


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
        self.RequestQue = deque([])
        self.day = -1
        self.divineans = []
        self.CoFlag = [False for _ in range(self.playerNum)]

    def update_talk_suspicion_villager(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 1
            self.suspicion[str(content[1])] += 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1])] -= 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == self.myrole:
            self.suspicion[str(agent)] += 10
        if content[0] == "COMINGOUT" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 10
        if content[0] == "COMINGOUT" and content[2] == "SEER":
            self.suspicion[str(agent)] += 2

    def update_talk_suspicion_werewolf(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[int(agent)] += 1
            self.suspicion[int(content[1])] -= 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[int(content[1])] += 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "WEREWOLF":
            self.suspicion[int(agent)] += 5
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "VILLAGER":
            self.suspicion[int(agent)] -= 5

    def update_talk_agreedisagree(self, idx, content):
        idx = str(idx).zfill(3)
        if sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1] == 0:
            return None
        if content[0] == "ESTIMATE":
            if content[1] == sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0] and (content[2] == "WEREWOLF" or content[2] == "POSSESSED"):
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

    def update_talk_request(self, content):
        pass
        # 他人のリクエストにどれぐらい答えるか

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
                self.update_talk_agreedisagree(row[1]["idx"], content)
                self.update_talk_request(content)

    def update_dead(self, row):
        self.suspicion[str(int(row[1]["agent"]) - 1)] -= 1000

    def update_divine(self, row):
        text = row[1]["text"]
        content = splitText.splitText(text)
        if content[2] == "WEREWOLF" or content[2] == "POSSESSED":
            self.suspicion[content[1]] += 100
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
        self.isCo = False
        self.isVote = False
        self.isBecause = False
        self.isRequestVote = False
        self.isDivine = False

    def vote(self):
        return self.voteop

    def finish(self):
        self.divineans = []
        return None

    def talk(self):
        print(self.suspicion)
        if not self.isCo and self.day == 1:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif not self.isVote:
            self.voteop = int(sorted(self.suspicion.items(),
                                     key=lambda x: x[1])[-1][0])
            if sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1] == 0:
                return cb.skip()
            self.isVote = True
            return cb.VOTE(self.voteop)
        elif not self.isBecause:
            if sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1] == 0:
                return cb.skip()
            self.isBecause = True
            return cb.BECAUSE(cb.ESTIMATE(self.voteop, "WEREWOLF"), cb.VOTE(self.voteop))
        elif not self.isRequestVote:
            if sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1] == 0:
                return cb.skip()
            self.isRequestVote = True
            return cb.REQUEST("ANY", cb.VOTE(self.voteop))
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        for i, flag in enumerate(self.CoFlag):
            if not flag:
                return cb.REQUEST(i, cb.COMINGOUT(i, "VILLAGER"))
            else:
                return cb.skip()
        return cb.skip()
        # INQUIREをつけるか
