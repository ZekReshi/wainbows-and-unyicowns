from shutil import copy2
import argparse
import os


def rename(group_nr, test_token):
    # src items
    template_name = "groupXX"
    src_folder = template_name
    src_package = os.path.join(template_name, template_name)
    src_package_init = os.path.join(src_package, "__init__.py")
    src_agent_module = os.path.join(src_package, "groupXX_agent.py")
    src_setup_file = os.path.join(src_folder, "setup.py")

    # dest items
    dest_folder = "group{}{}".format(group_nr, test_token)
    dest_package = os.path.join(dest_folder, dest_folder)
    dest_agent_module = os.path.join(dest_package, "group{}{}_agent.py".format(group_nr, test_token))
    dest_setup_file = os.path.join(dest_folder, "setup.py")

    # create dest folder and package
    if os.path.exists(dest_folder):
        print("Folder {} already exists".format(dest_folder))
        return False
    else:
        os.makedirs(dest_folder)
    os.makedirs(dest_package)

    # create resource folder
    os.makedirs(os.path.join(dest_package, "resources"))

    # copy files
    copy2(src_setup_file, dest_setup_file)
    copy2(src_package_init, dest_package)
    copy2(src_agent_module, dest_agent_module)

    # change content of files
    with open(dest_agent_module, "r") as module_file:
        content = module_file.read()

    content = content.replace("GroupXXAgent", "Group{}{}Agent".format(group_nr, test_token))

    with open(dest_agent_module, "w") as module_file:
        module_file.write(content)

    # change content of setup.py
    with open(dest_setup_file, "r") as module_file:
        content = module_file.read()

    content = content.replace("groupXX", "group{}{}".format(group_nr, test_token))

    with open(dest_setup_file, "w") as module_file:
        module_file.write(content)
    return True


def get_args():
    parser = argparse.ArgumentParser(description='Test agent package')
    parser.add_argument('--gn', default='XX', help='your group number')
    parser.add_argument('--t', nargs='?', default="",
                        help='create a test agent, provide a string that is appended to the name')
    args = parser.parse_args()

    group_nr = args.gn
    if len(group_nr) != 2:
        raise ValueError("The group number has to consist of 2 digits")

    try:
        group_nr_int = int(group_nr)
        if not 0 <= group_nr_int < 100:
            raise ValueError()
    except ValueError:
        raise ValueError("The group number entered is not valid!")
    return group_nr, args.t


if __name__ == "__main__":
    gn, t = get_args()
    if rename(gn, t):
        print("Successfully created new folder")
