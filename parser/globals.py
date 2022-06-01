import asyncio


class ParsingError(Exception):
    pass


Errors_list = {'Movies': [], 'Actors': []}
actor_visited = set()
movie_visited = set()

relations_to_upload = set()
actors_to_upload = set()
movies_to_upload = set()
cc = 4
sem = asyncio.Semaphore(4)
headers = {'Accept-language': 'en', 'X-FORWARDED-FOR': '134.199.245.152'}

