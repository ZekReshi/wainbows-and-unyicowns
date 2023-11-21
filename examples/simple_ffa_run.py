"""An example to show how to set up an pommerman game programmatically"""
import pommerman
from pommerman import agents
from group02 import group02_agent
import time


def main():
    """Simple function to bootstrap a game."""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents
    agent_list = [
        agents.SimpleAgent(),
        group02_agent.Group02Agent()
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(10):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        env.render()
        time.sleep(5)
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
