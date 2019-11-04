import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from CalmRole import Villager
# ANYを対応させる


class Werewolf(Villager.Villager):

    def attack(self):
        return int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0])

    def talk(self):
        if self.co_rate != 1:
            self.co_rate = random.uniform(0, 1)
        if not self.isCo and self.day == 1 and self.co_rate >= 0.5:
            self.isCo = True
            if self.co_rate == 2:
                return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "VILLAGER"))
            else:
                return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE("TALK", AGREEText[0], AGREEText[1])
        elif len(self.DISAGREESentenceQue) >= 2:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE("TALK", DISAGREEText[0], DISAGREEText[1])
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
        index = 0
        while True:
            if index == self.playerNum:
                return cb.skip()
            if not self.CoFlag[index]:
                self.CoFlag[index] = True
                return cb.REQUEST(index, cb.COMINGOUT(index, "ANY"))
            else:
                index += 1
        return cb.skip()
