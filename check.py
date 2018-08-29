#!/usr/bin/env python3
"""
Scripts in bin/ come from:
http://matt.might.net/articles/shell-scripts-for-passive-voice-weasel-words-duplicates/
"""
import sys
import os
import glob


def get_posts():
    articles = glob.glob('content/post/*.md')
    projects = glob.glob('content/project/*.md')

    return articles + projects


if __name__ == '__main__':
    if len(sys.argv[1:]) == 0:
        posts = get_posts()
    else:
        posts = sys.argv[1:]

    for p in posts:
        print('\n\n' + p)

        print('\t## Check weasels words:')
        os.system('./bin/check_weasels.sh ' + p)
        print('\t## Check passive voice:')
        os.system('./bin/check_passivevoice.sh ' + p)
        print('\t## Check duplicates')
        os.system('./bin/check_duplicates.pl ' + p)