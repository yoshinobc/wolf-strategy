import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from CalmRole import Villager
# ANYを対応させる


class Werewolf(Villager.Villager):
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

    def attack(self):
        return int(sorted(self.suspicion.items(), key=lambda x: x[1])[0][0]) + 1

    def finish(self):
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
            self.voteop = int(
                sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) + 1
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
