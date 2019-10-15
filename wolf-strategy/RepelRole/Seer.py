from aiwolfpy import contentbuilder as cb
from RepelRole import Villager
import random
from collections import deque
from utils import splitText


class Seer(Villager.Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(base_info, diff_data, game_setting, myrole)

    def update_talk_request(self, agent, idx, content):
        idx = str(idx).zfill(3)
        if agent != self.mode and content[0] == self.agentIdx:
            requestContent = splitText.splitText(content[1])
            if requestContent[0] == "DIVINED" or requestContent[0] == "VOTE":
                self.DISAGREESentenceQue.append(
                    (requestContent[0], str(self.day).zfill(2), idx))

    def update_talk_repel(self, agent, content):
        super().update_talk_repel(agent, content)
        if (content[0] == "COMINGOUT" and content[2] == "SEER"):
            if int(content[1])-1 in self.repelTargetQue:
                self.repelTargetQue.remove(int(content[1])-1)

    def update(self, base_info, diff_data, request):
        super().update(base_info, diff_data, request)

    def talk(self):
        if not self.isCo and self.day == 1:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "SEER")
        elif not self.isVote:
            self.isVote = True
            if self.mode == -1:
                while True:
                    self.voteIdxRandom = random.randint(
                        0, self.playerNum - 1)
                    if self.voteIdxRandom != self.agentIdx:
                        break
                return cb.VOTE(self.voteIdxRandom)
            else:
                return cb.VOTE(self.mode)
        elif not self.isBecause:
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
        elif not self.isRequest:
            if self.mode == -1:
                return cb.skip()
            else:
                self.isRequest = True
                return cb.REQUEST("ANY", cb.VOTE(self.mode))
        elif not self.isDivined:
            if self.mode != -1:
                self.isDivined = True
                return cb.DIVINED(self.mode, "WEREWOLF")
            else:
                return cb.skip()
        else:
            return cb.skip()

    def vote(self):
        return super().vote()

    def divine(self):
        if self.mode == -1:
            return random.randint(1, self.playerNum)
        return self.mode

    def finish(self):
        return None

    def dayStart(self):
        super().dayStart()
        self.isDivined = False
