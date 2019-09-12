import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
#SEER, POSSESSEDはmodeを敵の占いCOにpriorityする


class Villager(object):
    def __init__(self, agent_name):
        super().__init__(agent_name)

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

    def update_talk_repel(content):
        if (content[0] == "VOTE" and content[1] == self.agentIdx) or (content[0] == "ESTIMATE" and (content[2] == "WEREWOLF" or content[2] == "POSSESSED")):
            self.repelTargetQue.append(agent)

    def update_talk_agreedisagree(content):
        if content[0] == "ESTIMATE" and content[1] == self.mode:
            if content[2] == "WEREWOLF" or content[2] == "POSSESSED":
                self.AGREESentenceQue.append(row["idx"])
            else:
                self.DISAGREESentenceQue.append(row["idx"])

    def update_talk_request(content):
        if content[0] == "REQUEST":
            if agnet != self.mode and content[1] == self.agentIdx:
                requestContent = splitText(content[2])
                # 相手からのリクエストにどれぐらい答えるか

    def update_talk(self, row):
        text, agent = row["text"], row["agent"]
        content = splitText(text)
        update_talk_repel(content)
        update_talk_agreedisagree(content)
        update_talk_request(content)
        if self.myrole == "WEREWOLF":
            update_talk_divine(content)

    def update_dead(self, row):
        if row["agent"] - 1 == self.mode:
            self.repelTargetQue.pop()

    def update(self, base_info, diff_data, request):
        if len(self.repelTargetQue) == 0:
            self.mode = -1
        else:
            self.mode = self.repelTargetQue[0]
        for row in self.diff_data.iterrows():
            if row["type"] == "talk":
                update_talk(self, row)
            if row["type"] == "dead" or row["type"] == "execute":
                update_dead(self, row)

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
            return cb.BECAUSE(cb.ESTIMATE(self.mode + 1, "WEREWOLF"))
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
            return cb.REQUEST(o, "VOTE", self.mode + 1)
        elif self.talkCount <= 10:
            # INQUIREをつけるか

        self.talkCount += 1
        return cb.skip()

    def vote(self):
        if self.mode == -1:
            return self.voteIdxRandom
        return self.mode

    def finish(self):
        return None
