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
            content2 = splitText.splitText(text2)
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
            return int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0])+1
        else:
            return self.requestdivine+1

    def talk(self):
        self.talk_step += 1
        if self.co_rate != 2:
            self.co_rate = random.uniform(0, 1)
        if not self.isCo and self.day == 1 and self.co_rate >= 0:
            self.isCo = True
            if self.co_rate == 2:
                return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "SEER"))
            else:
                return cb.COMINGOUT(self.agentIdx, "SEER")
        elif not self.isVote and self.talk_step >= 5:
            lists = []
            for i in range(len(self.vote_list)):
                if self.vote_list[i] != -1:
                    lists.append(self.vote_list[i])
            self.isVote = True
            try:
                self.talk_voteop = mode(lists)
            except:
                for d in self.base_info["statusMap"].items():
                    if d[1] == "ALIVE":
                        self.talk_voteop = int(d[0]) - 1
                        break
            for i in range(len(self.vote_list)):
                if self.vote_list[i] == self.talk_voteop:
                    self.because_list.append(i)
            return cb.VOTE(self.talk_voteop)
        elif not self.isDivine:
            self.isDivine = True
            agent, target = int(
                sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]), "WEREWOLF"
            return cb.DIVINED(agent, target)
        elif len(self.because_list) > 0 and self.isVote:
            agent = self.because_list[0]
            del self.because_list[0]
            return cb.BECAUSE(cb.VOTE2(agent, self.talk_voteop), cb.VOTE(self.talk_voteop))
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            return cb.DISAGREE(DISAGREEText[0], DISAGREEText[1], DISAGREEText[2])
        index = 0
        if self.talk_step >= 3:
            while True:
                if index == self.playerNum:
                    return cb.skip()
                if not self.istalk_vote[index] and index != self.agentIdx and self.base_info["statusMap"][str(index+1)] == "ALIVE":
                    self.istalk_vote[index] = True
                    return cb.INQUIRE(index, cb.VOTE("ANY"))
                else:
                    index += 1
        return cb.skip()
