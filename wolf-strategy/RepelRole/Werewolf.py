from aiwolfpy import contentbuilder as cb
from RepelRole import Villager
import random
from collections import deque
from utils import splitText


class Werewolf(Villager.Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(base_info, diff_data, game_setting, myrole)

    def update_talk_divine(self, agent, content):
        # DIVINEDで自分を村人側ってするってことは仲間だから
        if content[0] == "DIVINED" and content[1] == self.agentIdx and (content[2] == "VILLAGER" or content[2] == "SEER") and agent == self.mode:
            self.mode = -1
            self.repelTargetQue.remove(self.mode)
            self.WolfEstimateFlag = True

    def update(self, base_info, diff_data, request):
        super().update(base_info, diff_data, request)

    def talk(self):
        print(self.repelTargetQue)
        if not self.isCo and self.day == 1:
            self.isCo = True
            return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif not self.isVote:
            self.isVote = True
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
        """
        elif self.talkCount <= 10:
            # INQUIREをつけるか
            return cb.skip()
        """

    def attack(self):
        if self.mode == -1:
            return self.voteIdxRandom
        return self.mode
