import random
from collections import deque
from utils import splitText
from aiwolfpy import contentbuilder as cb
from statistics import mode


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
        self.suspicion = {}
        for i in range(self.playerNum):
            if i == self.agentIdx:
                self.suspicion[str(i)] = -1000
            else:
                self.suspicion[str(i)] = 0
        self.AGREESentenceQue = deque([])
        self.DISAGREESentenceQue = deque([])
        self.divineans = []

    def update_talk_suspicion_villager(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 1
            self.suspicion[str(content[1])] += 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1])] -= 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == self.myrole:
            self.suspicion[str(agent)] += 10
        if content[0] == "COMINGOUT" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 10
        if content[0] == "COMINGOUT" and content[2] == "SEER":
            self.suspicion[str(agent)] += 2

    def update_talk_suspicion_werewolf(self, agent, content):
        if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 1
            self.suspicion[str(content[1])] -= 2
        if content[0] == "ESTIMATE" and content[2] == "VILLAGER":
            self.suspicion[str(content[1])] += 1
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "WEREWOLF":
            self.suspicion[str(agent)] += 5
        if content[0] == "ESTIMATE" and content[1] == self.agentIdx and content[2] == "VILLAGER":
            self.suspicion[str(agent)] -= 5

    def update_talk_request(self, row, content):
        idx = str(row[1]["idx"]).zfill(3)
        content_ = splitText.splitText(content[2])
        if len(content_) >= 1:
            if content_[0] == "COMINGOUT" and (content_[1] == "ANY" or content_[1] == self.agentIdx):
                self.co_rate = 2
                self.agree_co = ("TALK", str(self.day).zfill(2), idx)

    def update_talk_agreedisagree(self, agent, idx, content):
        idx = str(idx).zfill(3)
        if agent == self.agentIdx:
            if content[0] == "ESTIMATE" and content[2] == "WEREWOLF":
                # print("muESTIMATE")
                self.myESTIMATE = ("TALK", str(self.day).zfill(2), idx)
            return None
        if int(sorted(self.suspicion.items(), key=lambda x: x[1])[-1][1]) == 0:
            return None
        if content[0] == "ESTIMATE" and self.talk_voteop != None:
            if content[1] == self.talk_voteop and (content[2] == "WEREWOLF" or content[2] == "POSSESSED"):
                self.AGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
            else:
                self.DISAGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
        if (content[0] == "VOTE" or content[0] == "VOTED") and self.talk_voteop != None:
            if content[1] == self.talk_voteop:
                self.AGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))
            else:
                self.DISAGREESentenceQue.append(
                    ("TALK", str(self.day).zfill(2), idx))

    def update_talk(self, row):
        text, agent = row[1]["text"], int(row[1]["agent"]) - 1
        texts = splitText.parse_text(text)
        for text in texts:
            content = splitText.splitText(text)
            if len(content) != 0:
                if content[0] == "VOTE":
                    self.istalk_vote[agent] = True
                    self.vote_list[agent] = content[1]
                if self.myrole == "VILLAGER" or self.myrole == "SEER":
                    self.update_talk_suspicion_villager(agent, content)
                else:
                    self.update_talk_suspicion_werewolf(agent, content)
                self.update_talk_agreedisagree(agent, row[1]["idx"], content)
                if content[0] == "REQUEST" and (content[1] == "ANY" or content[1] == self.agentIdx):
                    self.update_talk_request(row, content)

    def update_dead(self, row):
        self.suspicion[str(int(row[1]["agent"]) - 1)] -= 1000

    def update_divine(self, row):
        text = row[1]["text"]
        content = splitText.splitText(text)
        if content[2] == "WEREWOLF" or content[2] == "POSSESSED":
            self.suspicion[str(content[1])] += 100
        self.divineans.append((content[1], content[2]))

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
        self.voteop = -1
        self.old_voteop = None
        self.isCo = False
        self.isVote = False
        self.isBecause = False
        self.isRequestVote = False
        self.vote_list = [-1] * self.playerNum
        self.talk_voteop = None
        self.co_rate = random.uniform(0, 1)
        self.because_list = []
        self.isDivine = False
        self.talk_step = 0
        self.istalk_vote = [False for _ in range(self.playerNum)]

    def vote(self):
        return int(sorted(self.suspicion.items(),
                          key=lambda x: x[1])[-1][0]) + 1

    def finish(self):
        return None

    def talk(self):
        self.talk_step += 1
        if self.co_rate != 2:
            self.co_rate = random.uniform(0, 1)
        if not self.isCo and self.day == 1 and self.co_rate >= 0.5:
            self.isCo = True
            if self.co_rate == 2:
                return cb.AND(cb.AGREE(self.agree_co[0], self.agree_co[1], self.agree_co[2]), cb.COMINGOUT(self.agentIdx, "VILLAGER"))
            else:
                return cb.COMINGOUT(self.agentIdx, "VILLAGER")
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
        if self.talk_step >= 5:
            while True:
                if index == self.playerNum:
                    return cb.skip()
                if not self.istalk_vote[index] and index != self.agentIdx:
                    self.istalk_vote[index] = True
                    return cb.INQUIRE(index, cb.VOTE("ANY"))
                else:
                    index += 1
        return cb.skip()
        # INQUIREをつけるか
