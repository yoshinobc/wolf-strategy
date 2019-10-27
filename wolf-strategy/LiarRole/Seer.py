import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from statistics import mode
from LiarRole import Villager


class Seer(Villager.Villager):

    def update_talk_agreedisagree(self, agent, idx, content):
        super().update_talk_agreedisagree(agent, idx, content)
        if content[0] == "REQUEST" and content[1] == self.agentIdx:
            text2 = content[2]
            content2 = splitText(text2)
            if content2[0] == "DIVINE":
                if content2[1] == self.talk_voteop:
                    self.requestdivine = content2[1]
                    self.AGREESentenceQue.append(
                        ("TALK", str(self.day).zfill(2), idx))
                else:
                    self.DISAGREESentenceQue.append(
                        ("TALK", str(self.day).zfill(2), idx))

    def dayStart(self):
        super().dayStart()
        self.requestdivine = -1

    def divine(self):
        if self.requestdivine == -1:
            return int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0])
        else:
            return self.requestdivine

    def talk(self):
        if not self.isCo and self.day == 1:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "SEER")
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
        elif not self.isDivine:
            self.isDivine = True
            agent, target = self.divineans[0][0], self.divineans[0][1]
            self.divineans = []
            return cb.DIVINED(agent, target)
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
