#!/usr/bin/env python
from __future__ import print_function, division

# this is main script
# simple version

import aiwolfpy
import aiwolfpy.contentbuilder as cb

myname = 'bc'


class Repel(object):

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
        self.role = Villager(self.myname)

    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        try:
            self.role.update(base_info, diff_data, request)
        except Exception:
            print("error update")
            pass
        # print(base_info)
        # print(diff_data)

    def dayStart(self):
        try:
            self.role.dayStart(base_info, diff_data, request)
        except Exception:
            print("error dayStart")
            pass

    def talk(self):
        try:
            return self.role.talk()
        except Exception:
            print("error talk")
            return cb.over()

    def whisper(self):
        try:
            return self.role.whisper()
        except Exception:
            print("error whisper")
            return cb.over()

    def vote(self):
        try:
            return self.role.vote()
        except Exception:
            print("error vote")
        return 1

    def attack(self):
        try:
            return self.role.attack()
        except Exception:
            print("error attack")
            return 1

    def divine(self):
        try:
            return self.role.divine()
        except Exception:
            print("error divine")
            return 1

    def guard(self):
        try:
            return self.role.guard()
        except Exception:
            print("error guard")
            return 1

    def finish(self):
        return self.role.finish()


agent = Repel(myname)


# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
