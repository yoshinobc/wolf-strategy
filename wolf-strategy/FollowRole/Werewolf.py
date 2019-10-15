import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from FollowRole import Villager


class Werewolf(Villager.Villager):
    def __init__(self, agent_name):
        self.myname = agent_name

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting, myrole):
        self.base_info = base_info
        self.game_setting = game_setting
        self.diff_data = diff_data
        self.playerNum = len(self.base_info["remainTalkMap"].keys())
        self.myrole = myrole
        self.agentIdx = int(self.base_info["agentIdx"]) - 1
        self.day = -1
        self.agree_co = None

    def dayStart(self):
        self.day += 1
        self.voteop = None
        self.isCo = True
        self.isVote = False
        self.request_vote = False
        self.divineop = None
        self.Agreeque = deque([])

    def finish(self):
        return None

    def attack(self):
        for d in self.base_info["statusMap"].items():
            if d[1] == "ALIVE":
                return int(d[0]) - 1

    def talk(self):
        if not self.isCo:
            self.isCo = True
            return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "VILLAGER"))
        elif not self.isVote and self.voteop != None:
            self.isVote = True
            return cb.AND(cb.AGREE(self.request_vote_agree[0], self.request_vote_agree[1], self.request_vote_agree[2]), cb.VOTE(self.voteop))
        elif len(self.Agreeque) >= 1:
            AGREEText = self.Agreeque.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])

        elif not self.request_vote:
            for d in self.base_info["statusMap"].items():
                if d[1] == "ALIVE":
                    return cb.INQUIRE(int(d[0])-1, cb.REQUEST(self.agentIdx, cb.VOTE("ANY")))
        else:
            return cb.skip()
