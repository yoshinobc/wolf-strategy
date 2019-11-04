import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from statistics import mode
from LiarRole import Villager


class Werewolf(Villager.Villager):

    def attack(self):
        return int(sorted(self.suspicion.items(), key=lambda x: x[1])[0][0]) + 1

    def talk(self):
        if self.co_rate != 2:
            self.co_rate = random.uniform(0, 1)
        if not self.isCo and self.day == 1 and self.co_rate >= 0.5:
            self.isCo = True
            if self.co_rate == 2:
                return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "VILLAGER"))
            else:
                return cb.COMINGOUT(self.agentIdx, "VILLAGER")

        elif not self.isVote and len(self.vote_list) == self.playerNum - 1:
            lists = []
            for i in range(len(self.vote_list)):
                lists.append(self.vote_list[i][1])
            self.isVote = True
            self.talk_voteop = mode(lists)
            for i in range(len(self.vote_list)):
                if self.vote_list[i][1] == self.talk_voteop:
                    self.because_list.append(self.vote_list[i][0])
            return cb.VOTE(self.talk_voteop)
        elif len(self.because_list) > 0 and self.isVote:
            agent = self.because_list[0]
            del self.because_list[0]
            return cb.BECAUSE(cb.VOTE2(agent, self.talk_voteop), cb.VOTE(self.talk_voteop))
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 2:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        index = 0
        while True:
            if index == self.playerNum:
                return cb.skip()
            if not self.istalk_vote[index]:
                self.istalk_vote[index] = True
                return cb.INQUIRE(index, cb.VOTE("ANY"))
            else:
                index += 1
        return cb.skip()
        # INQUIREをつけるか
