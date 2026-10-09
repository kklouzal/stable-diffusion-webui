from modules import launch_utils

# Read by other modules as launch.<name>.
args = launch_utils.args
git_tag = launch_utils.git_tag
list_extensions = launch_utils.list_extensions


def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.start()


if __name__ == "__main__":
    main()
