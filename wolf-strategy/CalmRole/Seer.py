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
        idx = str(idx).zfill(3)
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
            if content[0] == "REQUEST" and content[1] == self.agentIdx:
                text2 = content[2]
                content2 = splitText(text2)
                if content2[0] == "DIVINE":
                    if content2[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]):
                        self.requestdivine = content2[1]
                        self.AGREESentenceQue.append(
                            ("TALK", str(self.day).zfill(2), idx))
                    else:
                        self.DISAGREESentenceQue.append(
                            ("TALK", str(self.day).zfill(2), idx))

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
            return int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0])
        else:
            return self.requestdivine

    def finish(self):
        return None

    def talk(self):
        if not self.isCo and self.day == 1:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "SEER")
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        elif not self.isVote:
            self.isVote = True
            self.voteop = int(sorted(self.suspicion.items(),
                                     key=lambda x: x[1])[-1][0])
            return cb.VOTE(self.voteop)
        elif not self.isBecause:
            self.isBecause = True
            return cb.BECAUSE(cb.ESTIMATE(self.voteop, "WEREWOLF"), cb.VOTE(self.voteop))
        elif not self.isRequestVote:
            self.isRequestVote = True
            return cb.REQUEST("ANY", cb.VOTE(self.voteop))
        elif not self.isDivine:
            self.isDivine = True
            return cb.DIVINED(self.divineans[0][0], self.divineans[0][1])
        for i, flag in enumerate(self.CoFlag):
            if not flag:
                return cb.REQUEST(i, cb.COMINGOUT(i, "VILLAGER"))
            else:
                return cb.skip()
        return cb.skip()
        # INQUIREをつけるか
