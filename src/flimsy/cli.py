import argparse

from flimsy.pipeline import logging_utils

def get_parser():
    parser = argparse.ArgumentParser()

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()


if __name__ == "__main__":
    main()