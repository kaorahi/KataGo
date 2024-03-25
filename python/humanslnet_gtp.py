#!/bin/python3

# copied & modified from play.py and humanslnet_gui.py

import sys
import subprocess
import json
import re
import atexit
import datetime
import heapq
import math
import random
from threading import Thread

from board import Board
from features import Features
from gamestate import GameState
from sgfmetadata import SGFMetadata

# Constants -----------------------------------------------------

board_size = 19
rules = GameState.RULES_JAPANESE

setmeta_command_aliases = ['hs-set-meta', 'meta']

# SGFMetadata -----------------------------------------------------

rank_options=[
            "KG","9d","8d","7d","6d","5d","4d","3d","2d","1d","1k","2k","3k","4k","5k","6k","7k","8k","9k","10k","11k","12k","13k","14k","15k","16k","17k","18k","19k","20k"
]
source_options=["KG","OGS","KGS","Fox","Tygem(Unused)","GoGoD","Go4Go"]
tc_options=["Blitz","Fast","Slow","Unknown"]
# date_options=[
#             1800,1825,1850,1875,1900,1915,1930,1940,1950,1960,1970,1980,1985,1990,1995,2000,2005,2008,2010,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023
# ]

meta_param = {
    "rank": "9d",
    # "rank": "1d/9d",  # B=1d, W=9d
    "source": "GoGoD",
    "tc": "Unknown",
    "date": 2020,
}

def make_sgfmeta(meta_param):
    ranks = meta_param["rank"].split('/')
    b_rank = ranks[0]
    w_rank = b_rank if len(ranks) < 2 else ranks[1]
    inverseBRank = case_insensitive_index(rank_options, b_rank)
    inverseWRank = case_insensitive_index(rank_options, w_rank)
    tc_index = case_insensitive_index(tc_options, meta_param["tc"])
    source_index = case_insensitive_index(source_options, meta_param["source"])
    return SGFMetadata(
        inverseBRank = inverseBRank,
        inverseWRank = inverseWRank,
        bIsHuman = inverseBRank != 0,
        wIsHuman = inverseWRank != 0,
        gameIsUnrated = False,
        gameRatednessIsUnknown = meta_param["source"] == "KGS",
        tcIsUnknown = meta_param["tc"] == "Unknown",
        tcIsByoYomi = meta_param["tc"] != "Unknown",
        mainTimeSeconds = [300,900,1800,0][tc_index],
        periodTimeSeconds = [10,15,30,0][tc_index],
        byoYomiPeriods = [5,5,5,0][tc_index],
        boardArea = 361,
        gameDate = datetime.date(int(meta_param["date"]),6,1),
        source = source_index,
    )

def case_insensitive_index(lis, elem):
    return [str(x).lower() for x in lis].index(elem.lower())

# GoClient -----------------------------------------------------

class GoClient():
    def __init__(self, server_command, game_state):
        self.server_command = server_command
        self.game_state = game_state
        self.board_size = self.game_state.board_size

        self.start_server()

    def start_server(self):
        print(f"Starting server with command: {self.server_command}")
        self.server_process = subprocess.Popen(
            self.server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        atexit.register(self.server_process.terminate)

        def print_stderr():
            while True:
                line = self.server_process.stderr.readline()
                if not line:
                    returncode = self.server_process.poll()
                    if returncode is not None:
                        return
                print(line,end="")

        t = Thread(target=print_stderr)
        t.daemon = True
        t.start()

        command = {"command": "start", "board_size": self.board_size, "rules": GameState.RULES_JAPANESE}
        self.send_command(command)
        response = self.receive_response()
        if response != {"outputs": ""}:
            self.handle_error(f"Unexpected response from server: {response}")

        for (pla,loc) in self.game_state.moves:
            command = {"command": "play", "pla": pla, "loc": loc}
            self.send_command(command)
            response = self.receive_response()
            if response != {"outputs": ""}:
                self.handle_error(f"Unexpected response from server: {response}")

    def send_command(self, command):
        print(f"Sending: {json.dumps(command)}", file=sys.stderr)
        self.server_process.stdin.write(json.dumps(command) + "\n")
        self.server_process.stdin.flush()

    def receive_response(self):
        print(f"Waiting for response")
        while True:
            returncode = self.server_process.poll()
            if returncode is not None:
                raise OSError(f"Server terminated unexpectedly with {returncode=}")
            response = self.server_process.stdout.readline().strip()
            if response != "":
                break
        print(f"Got response (first 100 chars): {str(response[:100])}", file=sys.stderr)
        # print(f"Got response: {str(response[:])}", file=sys.stderr)
        return json.loads(response)

    def handle_error(self, error_message):
        print(f"Error: {error_message}")
        self.server_process.terminate()

        sys.exit(1)

    def undo(self):
        gs = self.get_game_state()
        if not gs.can_undo():
            return False
        gs.undo()
        command = {"command": "undo"}
        self.send_command(command)
        response = self.receive_response()
        if response != {"outputs": ""}:
            self.handle_error(f"Unexpected response from server: {response}")
        return True

    def play(self, pla, loc):
        gs = self.game_state
        if not gs.board.would_be_legal(pla,loc):
            return False
        gs.play(pla,loc)
        command = {"command": "play", "pla": pla, "loc": loc}
        self.send_command(command)
        response = self.receive_response()
        if response != {"outputs": ""}:
            self.handle_error(f"Unexpected response from server: {response}")
        return True

    def clear(self):
        self.set_game_state(GameState(board_size, rules))

    def set_sgfmeta(self, sgfmeta):
        self.sgfmeta = sgfmeta

    def refresh_model(self):
        sgfmeta = self.sgfmeta
        command = {"command": "get_model_outputs", "sgfmeta": sgfmeta.to_dict()}
        self.send_command(command)
        response = self.receive_response()
        if "outputs" not in response:
            self.handle_error(f"Unexpected response from server: {response}")
        self.latest_model_response = response["outputs"]

    def get_game_state(self):
        return self.game_state

    def set_game_state(self, game_state):
        self.game_state = game_state
        self.clear_server_board()

    def clear_server_board(self):
        command = {"command": "start", "board_size": self.board_size, "rules": rules}
        self.send_command(command)
        response = self.receive_response()
        if response != {"outputs": ""}:
            self.handle_error(f"Unexpected response from server: {response}")

# Start GoClient -----------------------------------------------------

param_str = None
server_command = sys.argv[1:]

if len(server_command) >= 2 and server_command[0] == "-meta":
    param_str = server_command[1]
    server_command = server_command[2:]

if not server_command:
    print("Usage: python humanslnet_gui.py [-meta 'KEY=VAL,KEY=VAL,...'] <server_command>")
    sys.exit(1)

if param_str is not None:
    # Note:
    # Using JSON for param_str is troublesome due to Lizzie's ad-hoc
    # handling of quotes (splitCommand in Leelaz.java).
    # So naive 'key=val,key=val' is safer.
    pairs = [pair.split('=') for pair in param_str.split(',')]
    meta_param.update({key.strip(): value.strip() for key, value in pairs})

client = GoClient(server_command, GameState(board_size, rules))
client.set_sgfmeta(make_sgfmeta(meta_param))

# Basic parsing --------------------------------------------------------
colstr = 'ABCDEFGHJKLMNOPQRST'
def parse_coord(s,board):
    if s == 'pass':
        return Board.PASS_LOC
    return board.loc(colstr.index(s[0].upper()), board.size - int(s[1:]))

def str_coord(loc,board):
    if loc == Board.PASS_LOC:
        return 'pass'
    x = board.loc_x(loc)
    y = board.loc_y(loc)
    return '%c%d' % (colstr[x], board.size - y)

# Utils for GTP -----------------------------------------------------

# copied from katrain/core/utils.py in https://github.com/sanderland/katrain
def weighted_selection_without_replacement(items, pick_n):
    """For a list of tuples where the second element is a weight, returns random items with those weights, without replacement."""
    elt = [(math.log(random.random()) / (item[1] + 1e-18), item) for item in items]  # magic
    return [e[1] for e in heapq.nlargest(pick_n, elt)]  # NB fine if too small

def lz_analyze_output(items, value, gs):
    move = [str_coord(loc,gs.board) for loc, _ in items]
    prior10k = [int(prior * 10000) for _, prior in items]
    visits = prior10k
    # guarantee sum(visits) = 10000 to avoid cache in LizGoban
    visits[0] += 10000 - sum(prior10k)
    winrate10k = int(value[0] * 10000)
    infos = [f'info move {move[k]} visits {visits[k]} winrate {winrate10k} lcb {winrate10k} prior {prior10k[k]} order {k} pv {move[k]}' for k in range(len(items))]
    return ' '.join(infos)

# GTP Implementation -----------------------------------------------------

# Adapted from https://github.com/pasky/michi/blob/master/michi.py, which is distributed under MIT license
# https://opensource.org/licenses/MIT

known_commands = [
    # required commands
    'protocol_version',
    'name',
    'version',
    'known_command',
    'list_commands',
    'quit',
    'boardsize',
    'clear_board',
    'komi',
    'play',
    'genmove',
    # additional commands
    'undo',
    'lz-analyze',
    *setmeta_command_aliases,
]

while True:
    try:
        line = input().strip()
    except EOFError:
        break
    if line == '':
        continue
    command = [s.lower() for s in line.split()]
    if re.match('\d+', command[0]):
        cmdid = command[0]
        command = command[1:]
    else:
        cmdid = ''

    ret = ''
    if False:
        pass
    # required commands
    elif command[0] == "protocol_version":
        ret = '2'
    elif command[0] == "name":
        ret = 'HumanSLNetGTP'
    elif command[0] == "version":
        ret = '0.1'
    elif command[0] == "known_command":
        ret = 'true' if command[1] in known_commands else 'false'
    elif command[0] == "list_commands":
        ret = '\n'.join(known_commands)
    elif command[0] == "quit":
        print('=%s \n\n' % (cmdid,), end='')
        break
    elif command[0] == "boardsize":
        if int(command[1]) != board_size:
            print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], board_size), file=sys.stderr)
            ret = None
        board_size = int(command[1])
        client.set_game_state(GameState(board_size, rules))
    elif command[0] == "clear_board":
        client.clear()
    elif command[0] == "komi":
        print("Warning: Komi is not configurable", file=sys.stderr)
        # 'ret = None' causes annoying dialogs in Sabaki
        # ret = None
    elif command[0] == "play":
        pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
        gs = client.get_game_state()
        loc = parse_coord(command[2],gs.board)
        ok = client.play(pla,loc)
        if not ok:
            print("Warning: Illegal move %s" % (command[1],), file=sys.stderr)
            ret = None
    elif command[0] == "genmove":
        client.refresh_model()
        outputs = client.latest_model_response
        items = outputs["moves_and_probs0"]
        loc, prob = weighted_selection_without_replacement(items, 1)[0]
        gs = client.get_game_state()
        pla = gs.board.pla

        if len(command) > 1:
            pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
        client.play(pla,loc)
        ret = str_coord(loc,gs.board)
        print(f'Selected {ret} with probability {prob}', file=sys.stderr)
    # additional commands
    elif command[0] == "undo":
        ok = client.undo()
        if not ok:
            print('Warning: Cannot undo', file=sys.stderr)
            ret = None
    elif command[0] == "lz-analyze":
        client.refresh_model()
        outputs = client.latest_model_response
        items = outputs["moves_and_probs0"]
        items.sort(key=lambda mp: mp[1], reverse=True)
        value = outputs["value"]
        gs = client.get_game_state()
        ret = '\n' + lz_analyze_output(items, value, gs)
    elif command[0] in setmeta_command_aliases:
        pairs = [command[i:i+2] for i in range(1, len(command), 2)]
        for key, value in pairs:
            meta_param[key] = value
        sgfmeta = make_sgfmeta(meta_param)
        client.set_sgfmeta(sgfmeta)
        print(f'sgfmeta = {sgfmeta}', file=sys.stderr)
    # unknown
    else:
        print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
        ret = None

    if ret is not None:
        print('=%s %s\n\n' % (cmdid, ret,), end='')
    else:
        print('?%s ???\n\n' % (cmdid,), end='')
    sys.stdout.flush()

# if __name__ == "__main__":
#     main()
