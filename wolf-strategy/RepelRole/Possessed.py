from aiwolfpy import contentbuilder as cb
from RepelRole import Villager
import random
from collections import deque
from utils import splitText


class Possessed(Villager.Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(base_info, diff_data, game_setting, myrole)

    def update(self, base_info, diff_data, request):
        super().update(base_info, diff_data, request)

    def dayStart(self):
        super().dayStart()
        self.isDivined = False

    def talk(self):
        if not self.isCo and self.day == 1:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "SEER")
        elif not self.isVote:
            self.isVote = True
            if self.mode == -1:
                if self.voteIdxRandom == -1:
                    self.voteIdxRandom = random.randint(1, self.playerNum-1)
                return cb.VOTE(self.voteIdxRandom)
            else:
                return cb.VOTE(self.mode)
        elif not self.isBecause:
            self.isBecause = True
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.BECAUSE(cb.ESTIMATE(self.mode, "WEREWOLF"), cb.VOTE(self.mode))
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(ISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        elif not self.isRequest:
            self.isRequest = True
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.REQUEST("ANY", cb.VOTE(self.mode))
        elif not self.isDivined:
            self.isDivined = True
            if self.mode != -1:
                return cb.DIVINED(self.mode, "WEREWOLF")
            else:
                return cb.skip()
        else:
            return cb.skip()
