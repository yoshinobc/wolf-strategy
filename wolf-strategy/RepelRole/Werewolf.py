from aiwolfpy import contentbuilder as cb
from RepelRole import Villager
import random
from collections import deque
from utils import splitText


class Werewolf(Villager.Villager):

    def update_talk_divine(self, agent, content):
        # DIVINEDで自分を村人側ってするってことは仲間だから
        if content[0] == "DIVINED" and content[1] == self.agentIdx and (content[2] == "VILLAGER" or content[2] == "SEER") and agent == self.mode:
            self.repelTargetQue.remove(self.mode)
            self.mode = -1
            self.WolfEstimateFlag = True

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
            self.isBecause = True
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
        elif self.isRequest:
            self.isRequest = True
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.REQUEST("ANY", cb.VOTE(self.mode))
        else:
            return cb.skip()

    def attack(self):
        if self.mode == -1:
            return self.voteIdxRandom+1
        return self.mode
