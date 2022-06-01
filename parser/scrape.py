import globals

from db_creation import truncate_tables, sql_create_tables
from collections import deque
from bfs import bfs_separated


def reset_globals():
    globals.Errors_list = {'Movies': [], 'Actors': []}
    globals.actor_visited = set()
    globals.movie_visited = set()
    globals.relations_to_upload = set()
    globals.actors_to_upload = set()
    globals.movies_to_upload = set()


def parse_imdb(actor_start_url, cache=False, truncate=False):
    reset_globals()
    if truncate:
        truncate_tables()

    sql_create_tables()
    movie_q = deque()
    actor_q = deque()
    actor_q.append(actor_start_url)
    return bfs_separated(actor_q=actor_q, movie_q=movie_q, cache=cache)
