import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from CalmRole import Villager
# ANYを対応させる


class Seer(Villager.Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(base_info, diff_data, game_setting, myrole)

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
            if content[0] == "REQUEST" and content[1] == self.agentIdx:
                text2 = content[2]
                content2 = splitText(text2)
                if content2[0] == "DIVINE":
                    if content2[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) + 1:
                        self.requestdivine = content2[1]
                        self.AGREESentenceQue.append(idx)
                    else:
                        self.DISAGREESentenceQue.append(idx)

    def update(self, base_info, diff_data, request):
        super().update(base_info, diff_data, request)

    def update_talk_request(self, content):
        pass
        # 他人のリクエストにどれぐらい答えるか

    def dayStart(self):
        super().dayStart()
        self.requestdivine = -1

    def vote(self):
        return self.voteop

    def divine(self):
        if self.requestdivine == -1:
            return int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) + 1
        else:
            return self.requestdivine

    def finish(self):
        return None

    def talk(self):
        if self.talkCount == 0 and self.day == 1:
            self.talkCount += 1
            return cb.COMINGOUT(self.agentIdx, "SEER")
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
            return cb.DIVINED(self.divineans[0][0], self.divineans[0][1])
        else:
            return cb.skip()
            # INQUIREをつけるか
