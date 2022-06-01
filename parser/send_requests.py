import aiohttp
import asyncio
import sys
import globals


if sys.version_info[0] == 3 and sys.version_info[1] >= 7 and sys.platform.startswith('win'):
    policy = asyncio.WindowsSelectorEventLoopPolicy()
    asyncio.set_event_loop_policy(policy)


async def fetch_sem(session, url, sem):
    async with sem:
        async with session.get(url) as response:
            await asyncio.sleep(0)
            return await response.text()


async def get_soups(actor_links, sem):
    connector = aiohttp.TCPConnector(limit=45)
    sesh = aiohttp.ClientSession(headers=globals.headers, connector=connector)
    async with sesh as session:
        coroutines = [fetch_sem(session, url, sem) for url in actor_links]
        await asyncio.sleep(0)
        return await asyncio.gather(*coroutines)
