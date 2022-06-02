import nest_asyncio
import asyncio
import time
import os
import globals
import datetime
import pickle

from db_creation import dump_table
from cycles import actor_cycle, movie_cycle


def bfs_separated(actor_q, movie_q, counter=0, samp=100, mode='a', cache=False, start_time=time.time()):
    if len(actor_q) == len(movie_q) == 0:
        print('Parsing_finished')
        return None
    try:

        asyncio.get_event_loop()
        nest_asyncio.apply()

        if mode == 'a':
            print('actor cycle...')
            actor_q, movie_q, mode = actor_cycle(actor_q, movie_q, samp)
            print('movie cycle...')
            actor_q, movie_q, mode = movie_cycle(actor_q, movie_q, samp)

        else:
            print('movie cycle...')
            actor_q, movie_q, mode = movie_cycle(actor_q, movie_q, samp)
            print('actor cycle...')
            actor_q, movie_q, mode = actor_cycle(actor_q, movie_q, samp)

        return bfs_separated(actor_q, movie_q, counter=counter + 1, samp=samp, mode=mode, cache=cache)

    # create separate func for caching and separate func for restarting
    except (KeyboardInterrupt, globals.ParsingError) as e:
        if cache:
            print(mode)
            if not os.path.exists('Cache'):
                os.makedirs('Cache')
            name_time = datetime.datetime.now().strftime("%d-%b_%H-%M-%S")
            fold_name = 'Cache\\' + name_time + mode
            os.makedirs(fold_name)

            # questionable performance
            for file in ((actor_q, 'actor_q'), (movie_q, 'movie_q'),
                         (globals.actor_visited, 'actor_visited'), (globals.movie_visited, 'movie_visited')):
                with open(fold_name + '\\' + file[1], "wb") as f:
                    pickle.dump(file[0], f)

            # dump to_upload_files to sql, do not pickle them
            globals.actors_to_upload = dump_table(globals.actors_to_upload, 'actors')
            globals.movies_to_upload = dump_table(globals.movies_to_upload, 'movies')
            globals.relations_to_upload = dump_table(globals.relations_to_upload, 'relations')

            print(f'Results saved to {fold_name}')
            if str(e).split('_')[0] == 'sleep':
                print(f'{datetime.datetime.now().strftime("%H:%M:%S")} - sleeping...')
                wait_time = 500
                time.sleep(wait_time)
                past_time = start_time - time.time()
                if past_time < 400 and wait_time < 1000:
                    wait_time += 50
                    time.sleep(150)
                elif past_time < 400 and wait_time >= 1000:
                    print("Timeout's too long. Shutting down!")
                    return None
                restart_scraping('Cache\\' + name_time + mode, start_time)

        else:
            print('Results not saved')

        return None


def restart_scraping(dest_to_folder, start_time=time.time()):
    mode = dest_to_folder[-1]

    print('restarting in mode: ', mode)

    with open(dest_to_folder + '\\actor_visited', 'rb') as f:
        globals.actor_visited = pickle.load(f)

    with open(dest_to_folder + '\\movie_visited', 'rb') as f:
        globals.movie_visited = pickle.load(f)

    with open(dest_to_folder + '\\actor_q', 'rb') as f:
        actor_q = pickle.load(f)

    with open(dest_to_folder + '\\movie_q', 'rb') as f:
        movie_q = pickle.load(f)

    return bfs_separated(actor_q, movie_q, mode=mode, cache=True, start_time=start_time)
