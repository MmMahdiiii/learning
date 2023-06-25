# -*- coding: utf-8 -*-

# python imports
import random
import math
import pickle
import copy

# chillin imports
from chillin_client import RealtimeAI

# project imports
from ks.models import ECell, EDirection, Position
from ks.commands import ChangeDirection, ActivateWallBreaker

# neat imports
from neat import nn as neat_nn


# todo: create a class for handling the game including game_result


class AI(RealtimeAI):

    def __init__(self, world, nn):
        super(AI, self).__init__(world)
        self.nn = nn

    def calculate_score(self, new_world, is_us, current_agent):
        added_score = 0
        opp_added_score = 0
        agent_side = self.my_side if is_us else self.other_side
        wall_breaker_rem_time = current_agent.wall_breaker_rem_time
        wall_breaker_cooldown = current_agent.wall_breaker_cooldown

        destination = new_world.board[current_agent.position.y][current_agent.position.x]

        if destination == ECell.Empty:
            added_score += new_world.constants.wall_score_coefficient
        else:
            if (is_us and self.my_side == 'Yellow' and destination == ECell.YellowWall) or \
                    (not is_us and self.my_side == 'Yellow' and destination == ECell.BlueWall) or \
                    (is_us and self.my_side == 'Blue' and destination == ECell.BlueWall) or \
                    (not is_us and self.my_side == 'Blue' and destination == ECell.YellowWall):
                if wall_breaker_cooldown == 0 and wall_breaker_rem_time > 0:
                    added_score += 0
                    current_agent.wall_breaker_rem_time -= 1
                else:
                    if current_agent.health > 1:
                        added_score += 0
                        current_agent.health -= 1
                    else:
                        added_score += new_world.constants.my_wall_crash_score
            elif (is_us and self.my_side == 'Yellow' and destination == ECell.BlueWall) or \
                    (not is_us and self.my_side == 'Yellow' and destination == ECell.YellowWall) or \
                    (is_us and self.my_side == 'Blue' and destination == ECell.YellowWall) or \
                    (not is_us and self.my_side == 'Blue' and destination == ECell.BlueWall):
                added_score += new_world.constants.wall_score_coefficient
                opp_added_score -= new_world.constants.wall_score_coefficient
                if wall_breaker_cooldown == 0 and wall_breaker_rem_time > 0:
                    new_world.agents[agent_side].wall_breaker_rem_time -= 1
                else:
                    if current_agent.health > 1:
                        current_agent.health -= 1
                    else:
                        added_score += new_world.constants.enemy_wall_crash_score
            elif destination == ECell.AreaWall:
                added_score += new_world.constants.area_wall_crash_score

        return added_score, opp_added_score, current_agent

    def game_result(self, new_world, action, is_us):
        # creating agent my_side
        current_agent = copy.deepcopy(new_world.agents[self.my_side]) if is_us else \
            copy.deepcopy(new_world.agents[self.other_side])

        if is_us:
            if self.my_side == 'Yellow':
                new_world.board[current_agent.position.y][current_agent.position.x] = ECell.YellowWall
            elif self.my_side == 'Blue':
                new_world.board[current_agent.position.y][current_agent.position.x] = ECell.BlueWall
        else:
            if self.other_side == 'Yellow':
                new_world.board[current_agent.position.y][current_agent.position.x] = ECell.YellowWall
            elif self.other_side == 'Blue':
                new_world.board[current_agent.position.y][current_agent.position.x] = ECell.BlueWall

        # direction change
        if action == "up":
            current_agent.position.y -= 1
            current_agent.direction = EDirection.Up
        elif action == "down":
            current_agent.position.y += 1
            current_agent.direction = EDirection.Down
        elif action == "left":
            current_agent.position.x -= 1
            current_agent.direction = EDirection.Left
        elif action == "right":
            current_agent.position.x += 1
            current_agent.direction = EDirection.Right

        # calculate score
        current_score = new_world.scores[self.my_side] if is_us else new_world.scores[self.other_side]
        added_score_curr_agent, added_score_opp_curr_agent, current_agent = self.calculate_score(new_world, is_us,
                                                                                                 current_agent)
        final_score = current_score + added_score_curr_agent
        if is_us:
            new_world.scores[self.my_side] = final_score
            new_world.scores[self.other_side] += added_score_opp_curr_agent
        else:
            new_world.scores[self.other_side] = final_score
            new_world.scores[self.my_side] += added_score_opp_curr_agent

        # update wall breaker
        if current_agent.wall_breaker_rem_time == 0:
            current_agent.wall_breaker_cooldown = new_world.constants.wall_breaker_cooldown
            current_agent.wall_breaker_rem_time = new_world.constants.wall_breaker_duration

        # update world with agent
        if is_us:
            new_world.agents[self.my_side] = current_agent
        else:
            new_world.agents[self.other_side] = current_agent

        return new_world

    def get_actions(self, world, player):
        actions = []
        for direction in EDirection:
            if self.is_opposite_direction(direction, player.direction):
                continue
            if direction == EDirection.Up:
                # todo: check if we want to have Deliberate loss
                # todo: if index out of range error, might be here!
                ##
                y_destination = player.position.y - 1
                x_destination = player.position.x
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("up_off")
                    else:
                        # hitting our wall
                        if (self.my_side == "Yellow" and world.board[y_destination][
                            x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][
                                    x_destination] == ECell.BlueWall):
                            actions.append("up_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("up_on_enemy")
            elif direction == EDirection.Down:
                y_destination = player.position.y + 1
                x_destination = player.position.x
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("down_off")
                    else:
                        # hitting our wall
                        if (self.my_side == "Yellow" and world.board[y_destination][
                            x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][
                                    x_destination] == ECell.BlueWall):
                            actions.append("down_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("down_on_enemy")
            elif direction == EDirection.Left:
                y_destination = player.position.y
                x_destination = player.position.x - 1
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("left_off")
                    else:
                        # hitting our wall
                        if (self.my_side == "Yellow" and world.board[y_destination][
                            x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][
                                    x_destination] == ECell.BlueWall):
                            actions.append("left_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("left_on_enemy")
            elif direction == EDirection.Right:
                y_destination = player.position.y
                x_destination = player.position.x + 1
                if world.board[y_destination][x_destination] == ECell.AreaWall:
                    continue
                else:
                    if world.board[y_destination][x_destination] == ECell.Empty:
                        actions.append("right_off")
                    else:
                        # hitting our wall
                        ####
                        if (self.my_side == "Yellow" and world.board[y_destination][
                            x_destination] == ECell.YellowWall) or \
                                (self.my_side == "Blue" and world.board[y_destination][
                                    x_destination] == ECell.BlueWall):
                            actions.append("right_on_our")
                        # hitting enemy wall
                        else:
                            actions.append("right_on_enemy")
        return actions

    def is_opposite_direction(self, direction1, direction2):
        if direction1 == EDirection.Up and direction2 == EDirection.Down:
            return True
        if direction1 == EDirection.Down and direction2 == EDirection.Up:
            return True
        if direction1 == EDirection.Left and direction2 == EDirection.Right:
            return True
        if direction1 == EDirection.Right and direction2 == EDirection.Left:
            return True
        return False

    def initialize(self):
        pass

    def heuristic(self, world):
        # calculating distances of the agent in 8 directions to the nearest wall
        # if was enemy wall, then make enemy = 1

        def is_enemy_wall(X, Y):
            if self.other_side == 'Blue':
                return world.board[X][Y] == ECell.BlueWall
            else:
                return world.board[X][Y] == ECell.YellowWall

        print(world.agents[self.my_side].position.x, world.agents[self.my_side].position.y)

        x, y = world.agents[self.my_side].position.x, world.agents[self.my_side].position.y

        height = len(world.board[0])
        width = len(world.board)

        pos = []
        for i in range(8):
            pos.append([y, x])
        distances = [0] * 8
        enemies = [0] * 8

        # down
        pos[0][0] += 1
        while pos[0][0] < height and world.board[pos[0][0]][pos[0][1]] == ECell.Empty:
            pos[0][0] += 1

        # left down
        pos[1][0] += 1
        pos[1][1] -= 1
        while pos[1][0] < height and pos[1][1] >= 0 and world.board[pos[1][0]][pos[1][1]] == ECell.Empty:
            pos[1][0] += 1
            pos[1][1] -= 1

        # left
        pos[2][1] -= 1
        while pos[2][1] >= 0 and world.board[pos[2][0]][pos[2][1]] == ECell.Empty:
            pos[2][1] -= 1

        # left up
        pos[3][0] -= 1
        pos[3][1] -= 1
        while pos[3][0] >= 0 and pos[3][1] >= 0 and world.board[pos[3][0]][pos[3][1]] == ECell.Empty:
            pos[3][0] -= 1
            pos[3][1] -= 1

        # up
        pos[4][0] -= 1
        while pos[4][0] >= 0 and world.board[pos[4][0]][pos[4][1]] == ECell.Empty:
            pos[4][0] -= 1

        # right up
        pos[5][0] -= 1
        pos[5][1] += 1
        while pos[5][0] >= 0 and pos[5][1] < width and world.board[pos[5][0]][pos[5][1]] == ECell.Empty:
            pos[5][0] -= 1
            pos[5][1] += 1

        # down
        pos[6][1] += 1
        while pos[6][1] < width and world.board[pos[6][0]][pos[6][1]] == ECell.Empty:
            pos[6][1] += 1

        # right down
        pos[7][0] += 1
        pos[7][1] += 1
        while pos[7][0] < height and pos[7][1] < width and world.board[pos[7][0]][pos[7][1]] == ECell.Empty:
            pos[7][0] += 1
            pos[7][1] += 1

        for i in range(8):
            distances[i] = math.sqrt((pos[i][1] - x) ** 2 + (pos[i][0] - y) ** 2)
            if is_enemy_wall(pos[i][0], pos[i][1]):
                enemies[i] = 1

        information = distances + enemies + [world.agents[self.my_side].wall_breaker_cooldown,
                                             world.agents[self.my_side].wall_breaker_rem_time,
                                             world.agents[self.my_side].health,
                                             world.scores[self.my_side] - world.scores[self.other_side]]
        print(information)
        onehot_direction = [0] * 4
        onehot_direction[world.agents[self.my_side].direction.value] = 1
        information += onehot_direction
        return self.nn.activate(information)

    def decide(self):
        # self.i += 1

        # depth = 1
        # self.min_max_tree(depth, self.world)

        # test
        best_move = None
        actions = self.get_actions(self.world, player=self.world.agents[self.my_side])
        scores = [0] * len(actions)
        for i, action in enumerate(actions):
            next_direction = action.split("_")[0]
            # activate_state = action.split("_")[1:]
            new_world = copy.deepcopy(self.world)
            new_world = self.game_result(new_world, next_direction, is_us=True)
            scores[i] = self.heuristic(new_world)

        best_move = int(scores.index(max(scores)))
        move = actions[best_move].split("_")

        if move[1] == "on":
            self.send_command(ActivateWallBreaker())

        if move[0] == "right":
            self.send_command(ChangeDirection(EDirection.Right))
        elif move[0] == "left":
            self.send_command(ChangeDirection(EDirection.Left))
        elif move[0] == "up":
            self.send_command(ChangeDirection(EDirection.Up))
        elif move[0] == "down":
            self.send_command(ChangeDirection(EDirection.Down))



    # min max tree functions
    def min_max_tree(self, depth, world):

        best_move = None
        best_score = float('-inf')
        for action in self.get_actions(world, player=self.my_side):
            next_direction = action.split("_")[0]
            activate_state = action.split("_")[1:]
            new_world = copy.deepcopy(world)
            new_world = self.game_result(new_world, next_direction, is_us=True)
            new_score = self.min_val(depth, new_world)
            if new_score > best_score:
                best_score = new_score
                best_move = action
        # change agent direction to best move
        # todo: activate wall breaker if needed

        return best_score, best_move

    def min_val(self, depth, world):
        depth -= 1
        # min is max of other player
        if depth == 0:
            return self.heuristic(world)
        v = float('-inf')
        for action in self.get_actions(world, player=self.world.agents[self.other_side]):
            # deep copy from world
            new_world = copy.deepcopy(world)
            new_world = self.game_result(new_world, action, is_us=False)
            v2 = self.max_val(depth, new_world)
            if v2 > v:
                v = v2
        return v

    def max_val(self, depth, world):
        depth -= 1
        if depth == 0:
            return self.heuristic(world)
        v = float('-inf')
        for action in self.get_actions(world, player=self.world.agents[self.my_side]):
            # deep copy from world
            new_world = copy.deepcopy(world)
            new_world = self.game_result(new_world, action, is_us=True)
            v2 = self.min_val(depth, new_world)
            if v2 > v:
                v = v2
        return v
