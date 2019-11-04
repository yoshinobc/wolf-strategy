#!/usr/bin/env python
from __future__ import print_function, division
import RepelRole.Villager
import RepelRole.Seer
import RepelRole.Werewolf
import RepelRole.Possessed


# this is main script
# simple version

import aiwolfpy
import aiwolfpy.contentbuilder as cb
import sys
myname = 'Liar'


class Liar(object):

    def __init__(self, agent_name):
        # myname
        self.myname = agent_name

    def getName(self):
        return self.myname

    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        # print(base_info)
        # print(diff_data)
        if self.base_info["myRole"] == "VILLAGER":
            self.role = RepelRole.Villager.Villager(self.myname)
        elif self.base_info["myRole"] == "WEREWOLF":
            self.role = RepelRole.Werewolf.Werewolf(self.myname)
        elif self.base_info["myRole"] == "SEER":
            self.role = RepelRole.Seer.Seer(self.myname)
        elif self.base_info["myRole"] == "POSSESSED":
            self.role = RepelRole.Possessed.Possessed(self.myname)
        self.role.initialize(base_info, diff_data,
                             game_setting, self.base_info["myRole"])

    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        self.role.update(base_info, diff_data, request)
        """
        try:
            self.role.update(base_info, diff_data, request)
        except Exception:
            print("error update")
            pass
        # print(base_info)
        # print(diff_data)
        """

    def dayStart(self):
        self.role.dayStart()
        """
        try:
            self.role.dayStart()
        except Exception:
            print("error dayStart")
            pass
        """

    def talk(self):
        return self.role.talk()

        """
        try:
            return self.role.talk()
        except Exception:
            print("error talk")
            return cb.over()
        """

    def whisper(self):
        return self.role.whisper()
        """
        try:
            return self.role.whisper()
        except Exception:
            print("error whisper")
            return cb.over()
        """

    def vote(self):
        return self.role.vote()
        """
        try:
            return self.role.vote()
        except Exception:
            print("error vote")
        return 1
        """

    def attack(self):
        return self.role.attack()
        """
        try:
            return self.role.attack()
        except Exception:
            print("error attack")
            return 1
        """

    def divine(self):
        return self.role.divine()
        """
        try:
            return self.role.divine()
        except Exception:
            print("error divine")
            return 1
        """

    def guard(self):
        return self.role.guard()
        """
        try:
            return self.role.guard()
        except Exception:
            print("error guard")
            return 1
        """

    def finish(self):
        return self.role.finish()


args = sys.argv
if len(args) == 7:    
    myname = myname+str(args[6])

agent = Liar(myname)

# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
