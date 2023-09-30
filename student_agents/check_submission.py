import argparse
import importlib
import random

import pommerman
from pommerman import agents


# this method is importing your installed agent, creating an agent object and calling the act-method once
# to verify everything is working as expected
def check_single_submission(group_nr):
    agent_module = "group{}.group{}_agent.Group{}Agent".format(group_nr, group_nr, group_nr)
    parts = agent_module.split('.')
    module = ".".join(parts[:-1])

    # import module
    try:
        module = importlib.import_module(module)
        print("- Successfully imported your agent module")
    except ModuleNotFoundError as e:
        print(str(e))
        print("Check if you installed your package. If so, please check "
              "your imports and the naming conventions. Importing your agent module failed!")
        return

    try:
        agent = getattr(module, parts[-1])
        print("- Successfully retrieved agent class")
    except AttributeError as e:
        print(str(e))
        print("Please check the name of your agent class. No correctly named agent class found in your package!")
        return

    try:
        # create agent object
        my_agent = agent()
        print("- Successfully instantiated agent")
    except Exception as e:
        print(str(e))
        print("Creating an agent object failed!")
        return

    # finally call the agent's act method once
    agent_list = [my_agent, agents.SimpleAgent()]
    env = pommerman.make('PommeFFACompetition-v0', agent_list)
    state = env.reset()
    try:
        action = my_agent.act(state[0], env.action_space)
        if type(action) is int:
            print("- Successfully called agent's act-method")
        else:
            print("act-method of your agent must return an integer value!")
            return
    except Exception as e:
        print(str(e))
        print("Calling act-method of agent failed!")
        return

    print("Your agent seems to stick to the conventions! "
          "For the moodle submission, zip your agent folder and upload to moodle")


def get_args():
    parser = argparse.ArgumentParser(description='Test agent package')
    parser.add_argument(
        '--gn', default='XX', help='your group number')
    args = parser.parse_args()

    group_nr = args.gn
    if len(group_nr) != 2:
        raise ValueError("group number has to consist of 2 digits")

    try:
        group_nr_int = int(group_nr)
        if not 0 <= group_nr_int < 100:
            raise ValueError()
    except ValueError:
        raise ValueError("The group number entered is not valid!")
    return group_nr


if __name__ == "__main__":
    random.seed(1)
    gn = get_args()
    check_single_submission(gn)
