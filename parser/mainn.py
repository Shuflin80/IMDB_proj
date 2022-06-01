import globals
from parse_data import get_movies_by_actor_soup
import requests
from bs4 import BeautifulSoup


if __name__ == "__main__":

    print('5')
    import cycles

    resp = requests.get('https://imdb.com/name/nm0430107/')
    soup = BeautifulSoup(resp.text, features="html.parser")

    print(get_movies_by_actor_soup(soup, 0))

    print(globals.actor_visited)
    print(globals.actors_to_upload)
    print(globals.Errors_list)


    print(globals.arrr)
