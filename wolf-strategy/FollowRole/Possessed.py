import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from FollowRole import Villager


class Possessed(Villager.Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(base_info, diff_data, game_setting, myrole)

    def update_divine(self, row):
        text = row[1]["text"]
        content = splitText.splitText(text)
        self.divineans = (content[1], content[2])

    def update(self, base_info, diff_data, request):
        self.diff_data = diff_data
        self.base_info = base_info
        for row in self.diff_data.iterrows():
            if row[1]["type"] == "talk":
                self.update_talk(row)
            if row[1]["type"] == "dead" or row[1]["type"] == "execute":
                self.update_dead(row)
            if row[1]["type"] == "divine":
                self.update_divine(row)

    def dayStart(self):
        self.day += 1
        self.voteop = None
        self.isCo = True
        self.isVote = False
        self.request_vote = False
        self.Agreeque = deque([])
        self.divineop = None
        self.divineans = None
        self.isdivine = True

    def finish(self):
        return None

    def talk(self):
        if not self.isCo:
            self.isCo = True
            return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "SEER"))
        elif not self.isVote and self.voteop != None:
            self.isVote = True
            return cb.AND(cb.AGREE(self.request_vote_agree[0], self.request_vote_agree[1], self.request_vote_agree[2]), cb.VOTE(self.voteop))
        elif len(self.Agreeque) >= 1:
            AGREEText = self.Agreeque.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])

        elif not self.request_vote:
            for d in self.base_info["statusMap"].items():
                if d[1] == "ALIVE":
                    return cb.INQUIRE(int(d[0]) - 1, cb.REQUEST(self.agentIdx, cb.VOTE("ANY")))
        elif not self.isdivine:
            self.isdivine = True
            for d in self.base_info["statusMap"].items():
                if d[1] == "ALIVE":
                    return cb.DIVINE(int(d[0])-1, "WEREWOLF")
        else:
            return cb.skip()
