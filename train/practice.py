import argparse
import collections


# create parser
import argparse
parser = argparse.ArgumentParser()


# positional argument
# **.py [-h] value
# - help: explaine about argument
# - type: type of argument
parser.add_argument("value", help="input value", type=int)


# optional argument
# **.py [-h] [--verbosity VERBOSITY] value
parser.add_argument("--verbosity", help="increase output verbosity")


# with action
# **.py [-h] [--verbosity VERBOSITY] [--verbose] value
# - action="store_true": true if the option is specified, false otherwise
# -v: short option
parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")


# get parsed args
args = parser.parse_args()
print(args)