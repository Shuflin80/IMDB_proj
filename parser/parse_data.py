import re
from bs4 import BeautifulSoup
import globals


def get_actors_by_movie_soup(cast_page_soup: BeautifulSoup, url: str, index: int) -> tuple:
    try:
        actor_list = []
        # global Errors_list
        try:
            mv_id = cast_page_soup.find('h3').find('a')['href'].split('/')[-2]
        except (AttributeError, TypeError, KeyboardInterrupt):
            globals.Errors_list['Movies'].append(cast_page_soup)
            if cast_page_soup.find('meta', attrs={'name': True}):
                if cast_page_soup.find('meta')['name'] == 'MSSmartTagsPreventParsing':
                    print('oops')
                    raise globals.ParsingError(f'sleep_{index}')
            else:
                raise KeyboardInterrupt(f'_{index}')

        try:
            act_res_set = cast_page_soup.find('div', attrs={'id': 'fullcredits_content'}) \
                .find('table', attrs={'class': 'cast_list'}) \
                .find_all('tr', attrs={'class': ['even', 'odd']})

        #         movie_load1 = movie_soup.find('h1').text
        #         movie_load2 = movie_soup.find('div', attrs={'data-testid': 'storyline-plot-summary'}) \
        #                                 .find('div', attrs={'class': None}).text.strip()

        except AttributeError as e:
            globals.Errors_list['Movies'].append(mv_id)
            print(e)
            print('ERRORMOV')
            act_res_set = 0

        for actor in range(len(act_res_set)):
            try:
                actor_details = act_res_set[actor].find('td', attrs={'class': 'primary_photo'}).findNext('td')
                act_link = actor_details.find('a')
                actor_list.append('https://imdb.com' + act_link['href'])

            except (AttributeError, TypeError) as e:
                print(e)
                print('attr, type error for act', url)
                globals.Errors_list['Actors'].append(actor)
                print('ERRORACT')


    #         if len(movies_to_upload) > 1000:
    #             dump_movies(movies_to_upload)
    #             print('movies uploaded!!!')
    except KeyboardInterrupt:
        raise KeyboardInterrupt(f'{index}')

    return actor_list, mv_id


def get_movies_by_actor_soup(actor_page_soup: BeautifulSoup, index: int) -> tuple:
    movies_list = []
    rels = []
    acts = []
    try:
        try:
            act_id = actor_page_soup.find('link', attrs={'rel': 'canonical'})['href'].split('/')[-2]
            act_name = actor_page_soup.find('h1').find('span', attrs={'class': 'itemprop'}).text
        except (AttributeError, TypeError, KeyboardInterrupt) as a:
            print(a)
            print("ERROR HERE")
            # print(actor_page_soup)
            if actor_page_soup.find('meta', attrs={'name': True}):
                if actor_page_soup.find('meta')['name'] == 'MSSmartTagsPreventParsing':
                    print('oops')
                    raise globals.ParsingError(f'sleep_{index}')
            else:
                raise KeyboardInterrupt(f'_{index}')

        movies_to_omit = ['(\s|.)*TV Series(\s|.)*', 'Short', 'Video Game', 'Video short', 'Video', 'TV Movie',
                          '(\s|.)*TV Mini-Series(\s|.)*',
                          'TV Special', 'TV Short', '(\s|.)*TV Mini Series(\s|.)*', 'announced',
                          'script', 'TV Mini Series', 'Video documentary short']

        pattern_main = re.compile('\\(' + '\\)|\\('.join(movies_to_omit) + '\\)')

        try:
            search_result = actor_page_soup.find_all('div', attrs={'class': 'filmo-row',
                                                                   'id': re.compile('(actor)|(actress)-.+')})
            search_range = len(search_result)
        except AttributeError:
            # global Errors_list
            globals.Errors_list['Actors'].append(act_id)
            return ()

        for i in search_result:
            if not any(re.match(pattern_main, line) for line in i.text.split('\n')):
                m_title = i.find('a').text.strip()
                role = re.match(re.compile('\A[\w\s\.]*(?!\\()'), i.text.split('\n')[6])
                # print(role)

                if role:
                    role = role.group().strip()
                else:
                    role = '\\N'

                m_id = i.find('a')['href'].split('/')[2]

                movies_list.append('https://imdb.com' + i.find('a')['href'])

                rels.append((m_id, act_id, role))
                # global relations_to_upload
                # relations_to_upload.add((m_id, act_id, role))

        #         if len(relations_to_upload) > 10000:
        #             dump_relations(relations_to_upload)
        #             print('relations uploaded!!!')
        if act_id not in globals.actor_visited:
            acts.append((act_id, act_name))
            globals.actor_visited.add(act_id)
        # global actors_to_upload
        # if act_id not in actor_visited:
        # actors_to_upload.add((act_id, act_name))
        else:
            print('Actor duplicate, check out: ', act_id)

    #         if len(actors_to_upload) > 1000:
    #             dump_actors(actors_to_upload)
    #             print('actors uploaded!!!')
    except KeyboardInterrupt:
        raise KeyboardInterrupt(f'{index}')

    return movies_list, acts, rels
