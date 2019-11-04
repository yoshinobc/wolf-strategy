import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from CalmRole import Villager
# calm divineans


class Seer(Villager.Villager):

    def update_divine(self, row):
        text = row[1]["text"]
        content = splitText.splitText(text)
        if content[2] == "WEREWOLF" or content[2] == "POSSESSED":
            self.suspicion[str(content[1])] += 100
        self.divineans = (content[1], content[2])

    def update_talk_agreedisagree(self, agent, idx, content):
        super().update_talk_agreedisagree(agent, idx, content)
        if content[0] == "REQUEST" and content[1] == self.agentIdx:
            text2 = content[2]
            content2 = splitText.splitText(text2)
            if content2[0] == "DIVINE":
                if content2[1] == int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]):
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
            return int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][0]) + 1
        else:
            return int(self.requestdivine) + 1

    def talk(self):
        if not self.isCo and self.day == 1:
            self.isCo = True
            if self.co_rate == 2:
                return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "SEER"))
            else:
                return cb.COMINGOUT(self.agentIdx, "SEER")
        elif len(self.AGREESentenceQue) >= 1:
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])
        elif len(self.DISAGREESentenceQue) >= 2:
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
            agent, role = self.divineans[0], self.divineans[1]
            self.divineans = None
            return cb.DIVINED(agent, role)
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
