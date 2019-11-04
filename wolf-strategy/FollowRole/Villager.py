import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb


class Villager(object):
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

    def update_talk_request(self, row, content):
        idx = str(row[1]["idx"]).zfill(3)
        content_ = splitText.splitText(content[2])
        if len(content_) >= 1:
            if content_[0] == "VOTE" and self.voteop == None and content_[1] != self.agentIdx:
                if content_[1] == "ANY":
                    return
                self.voteop = content_[1]
                self.request_vote = True
                self.request_vote_agree = (
                    ("TALK", str(self.day).zfill(2), idx))
            elif content_[0] == "VOTE" and self.voteop == content_[1]:
                self.Agreeque.append(("TALK", str(self.day).zfill(2), idx))
            if content_[0] == "DIVINE":
                self.divineop = content_[1]
            if content_[0] == "COMINGOUT":
                self.isCo = False
                self.agree_co = ("TALK", str(self.day).zfill(2), idx)

    def update_talk(self, row):
        text, agent = row[1]["text"], int(row[1]["agent"]) - 1
        texts = splitText.parse_text(text)
        for text in texts:
            content = splitText.splitText(text)
            if len(content) != 0 and content[0] == "REQUEST" and (content[1] == "ANY" or content[1] == self.agentIdx):
                self.update_talk_request(row, content)

    def update_dead(self, row):
        if self.voteop == str(int(row[1]["agent"]) - 1):
            self.voteop = None

    def update(self, base_info, diff_data, request):
        self.diff_data = diff_data
        self.base_info = base_info
        for row in self.diff_data.iterrows():
            if row[1]["type"] == "talk":
                self.update_talk(row)
            if row[1]["type"] == "dead" or row[1]["type"] == "execute":
                self.update_dead(row)

    def dayStart(self):
        self.day += 1
        self.voteop = None
        self.isCo = False
        self.isVote = False
        self.request_vote = False
        self.divineop = None
        self.Agreeque = deque([])
        self.request_vote_agree = None
        self.talk_turn = 0

    def vote(self):
        if self.voteop == None:
            for d in self.base_info["statusMap"].items():
                if d[1] == "ALIVE" and int(d[0])-1 != int(self.agentIdx):
                    return int(d[0])

        return int(self.voteop)+1

    def finish(self):
        return None

    def talk(self):
        self.talk_turn += 1
        if self.isCo:
            self.isCo = True
            return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, self.myrole))
        elif not self.isVote and self.voteop != None:
            self.isVote = True
            return cb.AND(cb.AGREE(self.request_vote_agree[0], self.request_vote_agree[1], self.request_vote_agree[2]), cb.VOTE(self.voteop))
        elif len(self.Agreeque) >= 1:
            AGREEText = self.Agreeque.pop()
            return cb.AGREE(AGREEText[0], AGREEText[1], AGREEText[2])

        elif not self.request_vote and self.talk_turn >= 3:
            for d in self.base_info["statusMap"].items():
                if d[1] == "ALIVE":
                    return cb.REQUEST(int(d[0])-1, cb.REQUEST(self.agentIdx, cb.VOTE("ANY")))
        else:
            return cb.skip()
