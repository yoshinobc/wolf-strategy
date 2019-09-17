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
            self.repelTargetQue.pop()
            self.WolfEstimateFlag = True

    def update(self, base_info, diff_data, request):
        super().update(base_info, diff_data, request)

    def talk(self):
        if self.talkCount == 0 and self.day == 1:
            self.talkCount += 1
            return cb.COMINGOUT(self.agentIdx, "VILLAGER")
        elif self.talkCount <= 1:
            self.talkCount += 1
            if self.mode == -1:
                self.voteIdxRandom = random.randint(0, self.playerNum) + 1
                return cb.VOTE(self.voteIdxRandom)
            else:
                return cb.VOTE(self.mode + 1)
        elif self.talkCount <= 2:
            self.talkCount += 1
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.BECAUSE(cb.ESTIMATE(self.mode + 1, "WEREWOLF"), cb.VOTE(self.mode+1))
        elif self.talkCount <= 6 and len(self.AGREESentenceQue) >= 1:
            self.talkCount += 1
            AGREEText = self.AGREESentenceQue.pop()
            return cb.AGREE(AGREEText)
        elif self.talkCount <= 9 and len(self.DISAGREESentenceQue) >= 1:
            DISAGREEText = self.DISAGREESentenceQue.pop()
            self.talkCount += 1
            return cb.DISAGREE(DISAGREEText)
        elif self.talkCount <= 9:
            self.talkCount += 1
            if self.mode == -1:
                return cb.skip()
            else:
                return cb.REQUEST("ANY", cb.VOTE(self.mode+1))
        elif self.talkCount <= 10:
            # INQUIREをつけるか
            return cb.skip()
        else:
            return cb.skip()

    def attack(self):
        if self.mode == -1:
            return self.voteIdxRandom
        return self.mode
