import numpy as np
import aiohttp
import asyncio
import tqdm
import globals
import nest_asyncio

from math import ceil
from bs4 import BeautifulSoup
from send_requests import get_soups
from parse_data import get_movies_by_actor_soup, get_actors_by_movie_soup
from db_creation import dump_table


def actor_cycle(actor_q: 'collections.deque', movie_q: 'collections.deque', samp: int):
    # global sesh
    while actor_q:
        current_actor_links = actor_q.popleft()
        print('batches in actor_q remaining: ', len(actor_q))
        print('len of an act batch: ', len(current_actor_links))
        cur_act_ids = np.array([i.split('/')[-2] for i in current_actor_links])

        new_act_mask = [x not in globals.actor_visited for x in cur_act_ids]
        current_actor_links = np.array(current_actor_links)[new_act_mask]

        for i in tqdm.tqdm(range(ceil(len(current_actor_links) / samp))):
            try:

                # batch of links to parse from all deque_pop (presumably just one)
                uploaded_links = current_actor_links[i * samp:(i + 1) * samp]
                try:
                    #nest_asyncio.apply()

                    current_movies_resps = asyncio.run(get_soups(uploaded_links, globals.sem))
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, KeyboardInterrupt) as sd:
                    print('Server disconnected')
                    if type(sd).__name__ == 'KeyboardInterrupt':
                        raise KeyboardInterrupt(f'_{i}')
                    else:
                        raise globals.ParsingError(f'sleep_{i}')

                movies_to_check = np.array([get_movies_by_actor_soup(BeautifulSoup(url, 'html.parser'), index=u)
                                            for u, url in enumerate(current_movies_resps)], dtype='object')

                actor_update = [x for y in movies_to_check[:, 1] for x in y if x[0] not in globals.actor_visited]
                rels_update = [x for y in movies_to_check[:, 2] for x in y if x[1] not in globals.actor_visited]

                globals.actor_visited.update([x[0] for x in actor_update])
                globals.actors_to_upload.update(actor_update)

                globals.relations_to_upload.update(rels_update)

                movs_np = np.array(list({mov_url for act in movies_to_check[:, 0] for mov_url in act}))
                filter_movies = [x.split('/')[-2] not in globals.movie_visited for x in movs_np]

                if len(globals.actors_to_upload) > 10000:
                    globals.actors_to_upload = dump_table(globals.actors_to_upload, 'actors')

                if len(globals.relations_to_upload) > 25000:
                    globals.relations_to_upload = dump_table(globals.relations_to_upload, 'relations')

            except (KeyboardInterrupt, globals.ParsingError) as error:

                if len(str(error)) > 0:
                    print(error)
                    idx = int(str(error).split('_')[-1])
                else:
                    idx = 0
                print('len of outstanding acts: ', len(current_actor_links[i * samp + idx:]))
                actor_q.insert(0, current_actor_links[i * samp + idx:])
                globals.actor_visited.update(cur_act_ids[i * samp: i * samp + idx])
                if type(error).__name__ == 'ParsingError':
                    raise globals.ParsingError('sleep_na')
                else:
                    raise KeyboardInterrupt

            movie_q.append(movs_np[filter_movies])

            # check actor as visited
            batch_act_ids = cur_act_ids[i * samp:(1 + i) * samp]
            globals.actor_visited.update(batch_act_ids)

    return actor_q, movie_q, 'b'


def movie_cycle(actor_q, movie_q, samp):
    # global sesh
    print('Initial movie_q len: ', len(movie_q))
    while movie_q:
        current_movie_links = movie_q.popleft()
        print('batches in movie_q remaining: ', len(movie_q))
        print('len of a mov batch: ', len(current_movie_links))
        cur_mov_ids = np.array([m.split('/')[-2] for m in current_movie_links])

        new_mov_mask = [x not in globals.movie_visited for x in cur_mov_ids]
        current_mov_links = current_movie_links[new_mov_mask]
        current_mov_links = [x + 'fullcredits' for x in current_mov_links]
        for i in tqdm.tqdm(range(ceil(len(current_mov_links) / samp))):
            try:
                uploaded_links = current_mov_links[i * samp:(i + 1) * samp]
                try:
                    current_act_resps = asyncio.run(get_soups(uploaded_links, globals.sem))
                except (aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, KeyboardInterrupt) as sd:
                    print('Server disconnected')
                    if type(sd).__name__ == 'KeyboardInterrupt':
                        raise KeyboardInterrupt(f'_{i}')
                    else:
                        raise globals.ParsingError(f'sleep_{i}')

                acts_to_check = np.array([get_actors_by_movie_soup(BeautifulSoup(url, 'html.parser'), url, index=u)
                                          for u, url in enumerate(current_act_resps)], dtype='object')

                movie_update = np.array(list({x for x in acts_to_check[:, 1]}))

                movie_update_mask = [x not in globals.movie_visited for x in movie_update]

                globals.movie_visited.update(movie_update[movie_update_mask])
                globals.movies_to_upload.update([(x, '\\N', '\\N') for x in movie_update[movie_update_mask]])

                acts_np = np.array(list({act_url for mov in acts_to_check[:, 0] for act_url in mov}))
                filter_acts = [x.split('/')[-2] not in globals.actor_visited for x in acts_np]

                if len(globals.movies_to_upload) > 10000:
                    globals.movies_to_upload = dump_table(globals.movies_to_upload, 'movies')

            except (KeyboardInterrupt, globals.ParsingError) as error:

                if len(str(error)) > 0:
                    idx = int(str(error).split('_')[-1])
                else:
                    idx = 0
                print('len of outstanding movs: ', len(current_movie_links[i * samp + idx:]))
                movie_q.insert(0, current_movie_links[i * samp + idx:])
                print('len of movie_q after insert: ', len(movie_q))
                globals.movie_visited.update(cur_mov_ids[i * samp:i * samp + idx])
                if type(error).__name__ == 'ParsingError':
                    raise globals.ParsingError('sleep_na')
                else:
                    raise KeyboardInterrupt

            actor_q.append(acts_np[filter_acts])

            # update visited movies
            batch_mov_ids = cur_mov_ids[i * samp:(1 + i) * samp]
            globals.movie_visited.update(batch_mov_ids)

    return actor_q, movie_q, 'a'


