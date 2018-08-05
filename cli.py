#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import click

from homoglyph_cnn.command import cnn_group

cli = cnn_group
cli = click.help_option()(cli)
cli = click.version_option()(cli)


def main():
    cli(auto_envvar_prefix='CNN')


if __name__ == '__main__':
    main()
