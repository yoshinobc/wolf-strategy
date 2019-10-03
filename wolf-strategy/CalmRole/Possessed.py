import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from CalmRole import Villager
# ANYを対応させる


class Possessed(Villager.Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(base_info, diff_data, game_setting, myrole)

    def update(self, base_info, diff_data, request):
        super().update(base_info, diff_data, request)

    def update_talk_request(self, content):
        pass
        # 他人のリクエストにどれぐらい答えるか

    def dayStart(self):
        super().dayStart()

    def vote(self):
        return self.voteop

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
            self.voteop = int(
                sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0])
            return cb.VOTE(self.voteop)
        elif not self.isBecause:
            self.isBecause = True
            return cb.BECAUSE(cb.ESTIMATE(self.voteop, "WEREWOLF"), cb.VOTE(self.voteop))
        elif not self.isRequestVote:
            self.isRequestVote = True
            return cb.REQUEST("ANY", cb.VOTE(self.voteop))
        elif not self.isDivine:
            self.isDivine = True
            return cb.DIVINED(int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]), "WEREWOLF")
            # INQUIREをつけるか
        for i, flag in enumerate(self.CoFlag):
            if not flag:
                return cb.REQUEST(i, cb.COMINGOUT(i, "VILLAGER"))
            else:
                return cb.skip()
        return cb.skip()
