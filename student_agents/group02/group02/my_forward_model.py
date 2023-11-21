'''Module to manage and advanced game state'''
from collections import defaultdict

import numpy as np

from pommerman import constants
from pommerman import characters
from pommerman import utility


class ForwardModel(object):
    @staticmethod
    def step(actions,
             curr_board,
             curr_agents,
             curr_bombs,
             curr_items,
             curr_flames,
             max_blast_strength=10):
        board_size = len(curr_board)

        # Tick the flames. Replace any dead ones with passages. If there is an
        # item there, then reveal that item.
        flames = []
        for flame in curr_flames:
            position = flame.position
            if flame.is_dead():
                item_value = curr_items.get(position)
                if item_value:
                    del curr_items[position]
                else:
                    item_value = constants.Item.Passage.value
                curr_board[position] = item_value
            else:
                flame.tick()
                flames.append(flame)
        curr_flames = flames

        # Redraw all current flames
        # Multiple flames may share a position and the map should contain
        # a flame until all flames are dead to avoid issues with bomb
        # movements and explosions.
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        # Step the living agents and moving bombs.
        # If two agents try to go to the same spot, they should bounce back to
        # their previous spots. This is complicated with one example being when
        # there are three agents all in a row. If the one in the middle tries
        # to go to the left and bounces with the one on the left, and then the
        # one on the right tried to go to the middle one's position, she should
        # also bounce. A way of doing this is to gather all the new positions
        # before taking any actions. Then, if there are disputes, correct those
        # disputes iteratively.
        # Additionally, if two agents try to switch spots by moving into each
        # Figure out desired next position for alive agents
        alive_agents = [agent for agent in curr_agents if agent.alive]
        desired_agent_positions = [agent.position for agent in alive_agents]

        for num_agent, agent in enumerate(alive_agents):
            position = agent.position
            # We change the curr_board here as a safeguard. We will later
            # update the agent's new position.
            curr_board[position] = constants.Item.Passage.value
            action = actions[agent.aid]

            if action == constants.Action.Stop.value:
                pass
            elif action == constants.Action.Bomb.value:
                position = agent.position
                if not utility.position_is_bomb(curr_bombs, position):
                    bomb = agent.maybe_lay_bomb()
                    if bomb:
                        curr_bombs.append(bomb)
            elif utility.is_valid_direction(curr_board, position, action):
                desired_agent_positions[num_agent] = agent.get_next_position(
                    action)

        # Gather desired next positions for moving bombs. Handle kicks later.
        desired_bomb_positions = [bomb.position for bomb in curr_bombs]

        for num_bomb, bomb in enumerate(curr_bombs):
            curr_board[bomb.position] = constants.Item.Passage.value
            if bomb.is_moving():
                desired_position = utility.get_next_position(
                    bomb.position, bomb.moving_direction)
                if utility.position_on_board(curr_board, desired_position) \
                   and not utility.position_is_powerup(curr_board, desired_position) \
                   and not utility.position_is_wall(curr_board, desired_position):
                    desired_bomb_positions[num_bomb] = desired_position

        # Position switches:
        # Agent <-> Agent => revert both to previous position.
        # Bomb <-> Bomb => revert both to previous position.
        # Agent <-> Bomb => revert Bomb to previous position.
        crossings = {}

        def crossing(current, desired):
            '''Checks to see if an agent is crossing paths'''
            current_x, current_y = current
            desired_x, desired_y = desired
            if current_x != desired_x:
                assert current_y == desired_y
                return ('X', min(current_x, desired_x), current_y)
            assert current_x == desired_x
            return ('Y', current_x, min(current_y, desired_y))

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                desired_position = desired_agent_positions[num_agent]
                border = crossing(agent.position, desired_position)
                if border in crossings:
                    # Crossed another agent - revert both to prior positions.
                    desired_agent_positions[num_agent] = agent.position
                    num_agent2, _ = crossings[border]
                    desired_agent_positions[num_agent2] = alive_agents[
                        num_agent2].position
                else:
                    crossings[border] = (num_agent, True)

        for num_bomb, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[num_bomb] != bomb.position:
                desired_position = desired_bomb_positions[num_bomb]
                border = crossing(bomb.position, desired_position)
                if border in crossings:
                    # Crossed - revert to prior position.
                    desired_bomb_positions[num_bomb] = bomb.position
                    num, is_agent = crossings[border]
                    if not is_agent:
                        # Crossed bomb - revert that to prior position as well.
                        desired_bomb_positions[num] = curr_bombs[num].position
                else:
                    crossings[border] = (num_bomb, False)

        # Deal with multiple agents or multiple bomb collisions on desired next
        # position by resetting desired position to current position for
        # everyone involved in the collision.
        agent_occupancy = defaultdict(int)
        bomb_occupancy = defaultdict(int)
        for desired_position in desired_agent_positions:
            agent_occupancy[desired_position] += 1
        for desired_position in desired_bomb_positions:
            bomb_occupancy[desired_position] += 1

        # Resolve >=2 agents or >=2 bombs trying to occupy the same space.
        change = True
        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                # Either another agent is going to this position or more than
                # one bomb is going to this position. In both scenarios, revert
                # to the original position.
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] > 1):
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for num_bomb, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[num_bomb]
                curr_position = bomb.position
                if desired_position != curr_position and \
                      (bomb_occupancy[desired_position] > 1 or agent_occupancy[desired_position] > 1):
                    desired_bomb_positions[num_bomb] = curr_position
                    bomb_occupancy[curr_position] += 1
                    change = True

        # Handle kicks.
        agent_indexed_by_kicked_bomb = {}
        kicked_bomb_indexed_by_agent = {}
        delayed_bomb_updates = []
        delayed_agent_updates = []

        # Loop through all bombs to see if they need a good kicking or cause
        # collisions with an agent.
        for num_bomb, bomb in enumerate(curr_bombs):
            desired_position = desired_bomb_positions[num_bomb]

            if agent_occupancy[desired_position] == 0:
                # There was never an agent around to kick or collide.
                continue

            agent_list = [
                (num_agent, agent) for (num_agent, agent) in enumerate(alive_agents) \
                if desired_position == desired_agent_positions[num_agent]]
            if not agent_list:
                # Agents moved from collision.
                continue

            # The agent_list should contain a single element at this point.
            assert (len(agent_list) == 1)
            num_agent, agent = agent_list[0]

            if desired_position == agent.position:
                # Agent did not move
                if desired_position != bomb.position:
                    # Bomb moved, but agent did not. The bomb should revert
                    # and stop.
                    delayed_bomb_updates.append((num_bomb, bomb.position))
                continue

            # NOTE: At this point, we have that the agent in question tried to
            # move into this position.
            if not agent.can_kick:
                # If we move the agent at this point, then we risk having two
                # agents on a square in future iterations of the loop. So we
                # push this change to the next stage instead.
                delayed_bomb_updates.append((num_bomb, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))
                continue

            # Agent moved and can kick - see if the target for the kick never had anyhing on it
            direction = constants.Action(actions[agent.aid])
            target_position = utility.get_next_position(desired_position,
                                                        direction)
            if utility.position_on_board(curr_board, target_position) and \
                       agent_occupancy[target_position] == 0 and \
                       bomb_occupancy[target_position] == 0 and \
                       not utility.position_is_powerup(curr_board, target_position) and \
                       not utility.position_is_wall(curr_board, target_position):
                # Ok to update bomb desired location as we won't iterate over it again here
                # but we can not update bomb_occupancy on target position and need to check it again
                # However we need to set the bomb count on the current position to zero so
                # that the agent can stay on this position.
                bomb_occupancy[desired_position] = 0
                delayed_bomb_updates.append((num_bomb, target_position))
                agent_indexed_by_kicked_bomb[num_bomb] = num_agent
                kicked_bomb_indexed_by_agent[num_agent] = num_bomb
                bomb.moving_direction = direction
                # Bombs may still collide and we then need to reverse bomb and agent ..
            else:
                delayed_bomb_updates.append((num_bomb, bomb.position))
                delayed_agent_updates.append((num_agent, agent.position))

        for (num_bomb, bomb_position) in delayed_bomb_updates:
            desired_bomb_positions[num_bomb] = bomb_position
            bomb_occupancy[bomb_position] += 1
            change = True

        for (num_agent, agent_position) in delayed_agent_updates:
            desired_agent_positions[num_agent] = agent_position
            agent_occupancy[agent_position] += 1
            change = True

        while change:
            change = False
            for num_agent, agent in enumerate(alive_agents):
                desired_position = desired_agent_positions[num_agent]
                curr_position = agent.position
                # Agents and bombs can only share a square if they are both in their
                # original position (Agent dropped bomb and has not moved)
                if desired_position != curr_position and \
                      (agent_occupancy[desired_position] > 1 or bomb_occupancy[desired_position] != 0):
                    # Late collisions resulting from failed kicks force this agent to stay at the
                    # original position. Check if this agent successfully kicked a bomb above and undo
                    # the kick.
                    if num_agent in kicked_bomb_indexed_by_agent:
                        num_bomb = kicked_bomb_indexed_by_agent[num_agent]
                        bomb = curr_bombs[num_bomb]
                        desired_bomb_positions[num_bomb] = bomb.position
                        bomb_occupancy[bomb.position] += 1
                        del agent_indexed_by_kicked_bomb[num_bomb]
                        del kicked_bomb_indexed_by_agent[num_agent]
                    desired_agent_positions[num_agent] = curr_position
                    agent_occupancy[curr_position] += 1
                    change = True

            for num_bomb, bomb in enumerate(curr_bombs):
                desired_position = desired_bomb_positions[num_bomb]
                curr_position = bomb.position

                # This bomb may be a boomerang, i.e. it was kicked back to the
                # original location it moved from. If it is blocked now, it
                # can't be kicked and the agent needs to move back to stay
                # consistent with other movements.
                if desired_position == curr_position and num_bomb not in agent_indexed_by_kicked_bomb:
                    continue

                bomb_occupancy_ = bomb_occupancy[desired_position]
                agent_occupancy_ = agent_occupancy[desired_position]
                # Agents and bombs can only share a square if they are both in their
                # original position (Agent dropped bomb and has not moved)
                if bomb_occupancy_ > 1 or agent_occupancy_ != 0:
                    desired_bomb_positions[num_bomb] = curr_position
                    bomb_occupancy[curr_position] += 1
                    num_agent = agent_indexed_by_kicked_bomb.get(num_bomb)
                    if num_agent is not None:
                        agent = alive_agents[num_agent]
                        desired_agent_positions[num_agent] = agent.position
                        agent_occupancy[agent.position] += 1
                        del kicked_bomb_indexed_by_agent[num_agent]
                        del agent_indexed_by_kicked_bomb[num_bomb]
                    change = True

        for num_bomb, bomb in enumerate(curr_bombs):
            if desired_bomb_positions[num_bomb] == bomb.position and \
               not num_bomb in agent_indexed_by_kicked_bomb:
                # Bomb was not kicked this turn and its desired position is its
                # current location. Stop it just in case it was moving before.
                bomb.stop()
            else:
                # Move bomb to the new position.
                # NOTE: We already set the moving direction up above.
                bomb.position = desired_bomb_positions[num_bomb]

        for num_agent, agent in enumerate(alive_agents):
            if desired_agent_positions[num_agent] != agent.position:
                agent.move(actions[agent.aid])
                if utility.position_is_powerup(curr_board, agent.position):
                    agent.pick_up(
                        constants.Item(curr_board[agent.position]),
                        max_blast_strength=max_blast_strength)

        # Explode bombs.
        exploded_map = np.zeros_like(curr_board)
        has_new_explosions = False

        for bomb in curr_bombs:
            bomb.tick()
            if bomb.exploded():
                has_new_explosions = True
            elif curr_board[bomb.position] == constants.Item.Flames.value:
                bomb.fire()
                has_new_explosions = True

        # Chain the explosions.
        while has_new_explosions:
            next_bombs = []
            has_new_explosions = False
            for bomb in curr_bombs:
                if not bomb.exploded():
                    next_bombs.append(bomb)
                    continue

                bomb.bomber.incr_ammo()
                for _, indices in bomb.explode().items():
                    for r, c in indices:
                        if not all(
                            [r >= 0, c >= 0, r < board_size, c < board_size]):
                            break
                        if curr_board[r][c] == constants.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        if curr_board[r][c] == constants.Item.Wood.value:
                            break

            curr_bombs = next_bombs
            for bomb in curr_bombs:
                if bomb.in_range(exploded_map):
                    bomb.fire()
                    has_new_explosions = True

        # Update the board's bombs.
        for bomb in curr_bombs:
            curr_board[bomb.position] = constants.Item.Bomb.value

        # Update the board's flames.
        flame_positions = np.where(exploded_map == 1)
        for row, col in zip(flame_positions[0], flame_positions[1]):
            curr_flames.append(characters.Flame((row, col)))
        for flame in curr_flames:
            curr_board[flame.position] = constants.Item.Flames.value

        # Kill agents on flames. Otherwise, update position on curr_board.
        for agent in alive_agents:
            if curr_board[agent.position] == constants.Item.Flames.value:
                agent.die()
            else:
                curr_board[agent.position] = utility.agent_value(agent.aid)

        return curr_board, curr_agents, curr_bombs, curr_items, curr_flames
