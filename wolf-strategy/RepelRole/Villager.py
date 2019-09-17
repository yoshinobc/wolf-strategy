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
        self.RequestQue = deque([])
        self.WolfEstimateFlag = False
        self.day = 0
        self.agentIdx = self.base_info["agentIdx"]

    def update_talk_repel(self, agent, content):
        if (content[0] == "VOTE" and content[1] == self.agentIdx) or (content[0] == "ESTIMATE" and content[1] == self.agentIdx and (content[2] == "WEREWOLF" or content[2] == "POSSESSED")):
            self.repelTargetQue.append(agent)

    def update_talk_agreedisagree(self, idx, content):
        if content[0] == "ESTIMATE" and content[1] == self.mode:
            if content[2] == "WEREWOLF" or content[2] == "POSSESSED":
                self.AGREESentenceQue.append(idx)
            else:
                self.DISAGREESentenceQue.append(idx)

    def update_talk_request(self, agent, idx, content):
        if content[0] == "REQUEST":
            if agnet != self.mode and content[1] == self.agentIdx:
                requestContent = splitText.splitText(content[2])
                # 相手からのリクエストにどれぐらい答えるか

    def update_talk(self, row):
        text, agent = row[1]["text"], row[1]["agent"]
        content = splitText.splitText(text)
        if len(content) != 0:
            self.update_talk_repel(agent, content)
            self.update_talk_agreedisagree(row[1]["idx"], content)
            self.update_talk_request(agent, row[1]["idx"], content)
            if self.myrole == "WEREWOLF":
                self.update_talk_divine(agent, content)

    def update_dead(self, row):
        if row[1]["agent"] - 1 == self.mode:
            self.repelTargetQue.pop()

    def update(self, base_info, diff_data, request):
        self.diff_data = diff_data
        if len(self.repelTargetQue) == 0:
            self.mode = -1
        else:
            self.mode = self.repelTargetQue[0]
        for row in self.diff_data.iterrows():
            if row[1]["type"] == "talk":
                self.update_talk(row)
            if row[1]["type"] == "dead" or row[1]["type"] == "execute":
                self.update_dead(row)

    def dayStart(self):
        self.talkCount = 0
        self.day += 1

    def talk(self):
        if self.talkCount == 0 and self.day == 1:
            self.talkCount += 1
            return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif self.talkCount <= 1:
            self.talkCount += 1
            if self.mode == -1:
                self.voteIdxRandom = random.randint(0, self.playerNum) + 1
                return cb.VOTE(self.voteIdxRandom)
            else:
                return cb.VOTE(self.mode + 1)
        elif self.talkCount <= 2:
            self.talkCount += 1
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.BECAUSE(cb.ESTIMATE(self.mode + 1, "WEREWOLF"), cb.VOTE(self.mode+1))
        elif self.talkCount <= 6 and len(self.AGREESentenceQue) >= 1:
            self.talkCount += 1
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText)
        elif self.talkCount <= 9 and len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            self.talkCount += 1
            return cb.DISAGREE(DISAGREEText)
        elif self.talkCount <= 9:
            self.talkCount += 1
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.REQUEST("ANY", cb.VOTE(self.mode+1))
        elif self.talkCount <= 10:
            # INQUIREをつけるか
            return cb.skip()

        self.talkCount += 1
        return cb.skip()

    def vote(self):
        if self.mode == -1:
            return self.voteIdxRandom
        return self.mode

    def finish(self):
        return None
