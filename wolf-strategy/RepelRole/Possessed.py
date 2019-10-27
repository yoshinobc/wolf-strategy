from aiwolfpy import contentbuilder as cb
from RepelRole import Villager
import random
from collections import deque
from utils import splitText


class Possessed(Villager.Villager):

    def dayStart(self):
        super().dayStart()
        self.isDivined = False

    def talk(self):
        if not self.isCo and self.day == 1 and random.uniform(0, 1) <= 0.8:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "SEER")
        elif not self.isVote:
            if self.mode == -1:
                return cb.skip()
            else:
                self.isVote = True
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
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
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
