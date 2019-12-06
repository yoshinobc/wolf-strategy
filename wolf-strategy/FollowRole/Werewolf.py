import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from FollowRole import Villager
# calmのdivineansがoutofrange


class Werewolf(Villager.Villager):

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
        self.talk_turn = 0
        self.istalk_vote = [False for _ in range(self.playerNum)]

    def attack(self):
        for d in self.base_info["statusMap"].items():
            if d[1] == "ALIVE":
                return int(d[0])

    def talk(self):
        self.talk_turn += 2
        if not self.isCo:
            self.isCo = True
            return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "VILLAGER"))
        elif not self.isVote and self.voteop != None:
            self.isVote = True
            return cb.AND(cb.AGREE(self.request_vote_agree[0], self.request_vote_agree[1], self.request_vote_agree[2]), cb.VOTE(self.voteop))
        elif len(self.Agreeque) >= 1:
            AGREEText = self.Agreeque.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])

        elif not self.request_vote and self.talk_turn >= 3:
            for d in self.base_info["statusMap"].items():
                if d[1] == "ALIVE":
                    return cb.REQUEST(int(d[0]) - 1, cb.REQUEST(self.agentIdx, cb.VOTE("ANY")))
        index = 0
        while True:
            if self.talk_turn <= 3:
                return cb.skip()
            if index == self.playerNum:
                return cb.skip()
            if not self.istalk_vote[index] and self.base_info["statusMap"][str(index + 1)] == "ALIVE":
                self.istalk_vote[index] = True
                return cb.INQUIRE(index, cb.VOTE("ANY"))
            else:
                index += 1
        else:
            return cb.skip()
