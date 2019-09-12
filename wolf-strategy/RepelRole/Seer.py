class Seer(Villager):
    def __init__(self, agent_name):
        super().__init__(agent_name)

    def initialize(self, base_info, diff_data, game_setting, myrole):
        super().initialize(self, base_info, diff_data, game_setting, myrole)

    def update_talk_request(content):
        content[0] == "REQUEST":
        if agent != self.mode and content[1] == self.agentIdx:
            requestContent = splitText(content[2])
            if requestContent[0] == "DIVINED" or requestContent[0] == "VOTE":
                self.DISAGREESentenceQue.append(row["idx"])

    def update(self, base_info, diff_data, request):
        super().update(self, base_info, diff_data, request)

    def talk(self):
        elif self.myrole == "SEER":
            if self.talkCount == 0 and self.day == 1:
                self.talkCount += 1
                return cb.COMINGOUT(self.agentIdx, "SEER")
            elif self.talkCount <= 1:
                self.talkCount += 1
                if self.mode == -1:
                    self.voteIdxRandom = random.randint(0, self.playerNum) + 1
                    return cb.VOTE(self.voteIdxRandom)
                else:
                    return cb.VOTE(self.mode + 1)
            elif self.talkCount <= 2:
                self.talkCount += 1
                return cb.BECAUSE(cb.ESTIMATE(self.mode + 1, "WEREWOLF"))
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
                # randomの仕方に無駄がある
                return cb.REQUEST(random.randint(0, self.playerNum) + 1, "VOTE", self.mode + 1)
            elif self.talkCount <= 10:
                if self.mode != -1:
                    return cb.DIVINED(self.mode + 1, "WEREWOLF")

    def vote(self):
        return super().vote()

    def divine(self):
        if self.mode == -1:
            return random.randint(0, self.playerNum)
        return self.mode + 1

    def finish(self):
        return None

    def dayStart(self):
        super().dayStart()
