import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
# SEER, POSSESSEDはmodeを敵の占いCOにpriorityする
# ANYを対応させる


class Villager(object):
    def __init__(self, agent_name):
        self.myname = agent_name

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting, myrole):
        self.base_info = base_info
        self.game_setting = game_setting
        self.playerNum = len(self.base_info["remainTalkMap"].keys())
        self.myrole = myrole
        self.mode = -1
        self.repelTargetQue = deque([])
        self.AGREESentenceQue = deque([])
        self.DISAGREESentenceQue = deque([])
        self.WolfEstimateFlag = False
        self.day = -1
        self.voteIdxRandom = -1
        self.agentIdx = int(self.base_info["agentIdx"]) - 1

    def update_talk_repel(self, agent, content):
        if len(content) >= 2 and content[1] == "ANY":
            content[1] = str(self.agentIdx)
        if (content[0] == "VOTE" and str(content[1]) == str(self.agentIdx)) or (content[0] == "ESTIMATE" and str(content[1]) == self.agentIdx and (content[2] == "WEREWOLF" or content[2] == "POSSESSED")):
            if agent not in self.repelTargetQue:
                self.repelTargetQue.append(agent)

    def update_talk_agreedisagree(self, agent, idx, content):
        if agent == self.agentIdx:
            return None
        if content[0] == "ESTIMATE" and content[1] == self.mode:
            idx = str(idx).zfill(3)
            if content[2] == "WEREWOLF" or content[2] == "POSSESSED":
                self.AGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
            else:
                self.DISAGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
        if content[0] == "VOTE" and content[1] == self.mode:
            idx = str(idx).zfill(3)
            self.AGREESentenceQue.append(("TALK", str(self.day).zfill(2), idx))

    def update_talk_request(self, agent, idx, content):
        pass

    def update_talk(self, row):
        text, agent = row[1]["text"], int(row[1]["agent"])-1
        texts = splitText.parse_text(text)
        for text in texts:
            content = splitText.splitText(text)
            if len(content) != 0:
                self.update_talk_repel(agent, content)
                self.update_talk_agreedisagree(agent, row[1]["idx"], content)
                self.update_talk_request(agent, row[1]["idx"], content)
                if self.myrole == "WEREWOLF":
                    self.update_talk_divine(agent, content)

    def update_dead(self, row):
        if int(row[1]["agent"]) - 1 == self.mode:
            self.repelTargetQue.remove(self.mode)

    def update(self, base_info, diff_data, request):
        self.diff_data = diff_data
        for row in self.diff_data.iterrows():
            if row[1]["type"] == "talk":
                self.update_talk(row)
            if row[1]["type"] == "dead" or row[1]["type"] == "execute":
                self.update_dead(row)
        if len(self.repelTargetQue) == 0:
            self.mode = -1
        else:
            if self.mode != self.repelTargetQue[0]:
                self.isVote = False
            self.mode = self.repelTargetQue[0]

    def dayStart(self):
        self.day += 1
        self.isCo = False
        self.isVote = False
        self.isBecause = False
        self.isRequest = False

    def talk(self):
        if not self.isCo and self.day == 1 and random.uniform(0, 1) <= 0.5:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif not self.isVote:
            if self.mode == -1:
                return cb.skip()
            else:
                self.isVote = True
                return cb.VOTE(self.mode)
        elif not self.isBecause:
            if self.mode == -1:
                return cb.skip()
            else:
                self.isBecause = True
                return cb.BECAUSE(cb.ESTIMATE(self.mode, "WEREWOLF"), cb.VOTE(self.mode))
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        elif not self.isRequest:
            self.isRequest = True
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.REQUEST("ANY", cb.VOTE(self.mode))
        return cb.skip()

    def vote(self):
        if self.mode == -1:
            while True:
                self.voteIdxRandom = random.randint(0, self.playerNum - 1)
                if self.voteIdxRandom != self.agentIdx:
                    break
            return self.voteIdxRandom + 1
        return self.mode + 1

    def finish(self):
        return None
