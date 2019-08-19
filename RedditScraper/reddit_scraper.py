import html
import json
import urllib.request, urllib.error
import typing
import time

def load_from_string(json_data: str) -> typing.Any:
    data: json = json.loads(json_data)
    return data


def load_web_page(url: str) -> typing.Any:
    url = urllib.request.Request(url, headers={
        'User-Agent': 'Hackathon Project Reddit Comment Scraper'
    })
    with urllib.request.urlopen(url) as processed_url:
        return load_from_string(processed_url.read().decode())
    return None


def load_file(file_name: str):
    with open(file_name, 'r') as myfile:
        data = myfile.read()
        if data is not None:
            return load_from_string(data)
    return None


def write_to_file(file_name: str, data: str):
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def search_reddit_posts_single_run(subreddit: str, question: str, limit: int, after=None):
    url: str = "https://www.reddit.com/r/{subreddit}/search.json?" \
               "restrict_sr=1" \
               "&sort=top" \
               "&limit={limit}" \
               "&q={question}".format(subreddit=subreddit, question=question, limit=limit)
    if after is not None:
        url = url + "&after={after}".format(after=after)
    response = load_web_page(url=url)
    if response is None:
        print("Error")
        return False, None, None
    print (response)
    # response = load_file('input.json')
    # Get Number of Returned Elements
    number_of_posts: int = int(response["data"]["dist"])
    if number_of_posts == 0:
        return True, None, None
    posts = response["data"]["children"]

    infos = []

    last_post_id = None

    for post in posts:
        post_data = post["data"]

        last_post_id = post_data["name"]

        title = str(post_data["title"])
        text = str(post_data["selftext"])
        # Replace HTML formatted stuff to text
        text = html.unescape(text)
        title = html.unescape(title)
        # Now we know that the data is split by \n\n
        # In order to simplify it for backend, I'll split by \n\n to different strings
        # And store it in array
        text = text.split('\n\n')
        info = dict(text=text, title=title)
        infos.append(info)

    # As count <= limit, job done
    if number_of_posts < limit:
        return True, infos, last_post_id

    return False, infos, last_post_id


def search_reddit_posts(subreddit: str, question: str, limit: int):
    infos_combined = []
    last_post_id = None
    done = 0

    increment = 100

    while done < limit:
        # Sleep per transaction
        if last_post_id is not None:
            print ("In wait state now")
            time.sleep(30)
        done, infos, last_post_id_temp = search_reddit_posts_single_run(subreddit=subreddit, question=question,
                                                                        limit=increment, after=last_post_id)
        if done:
            break
        if last_post_id_temp is None:
            print ("Infos was None. Waiting for API to Respond")
            continue
        infos_combined = infos_combined + infos
        last_post_id = last_post_id_temp

        done = done + increment

    return infos_combined

def main():
    subreddit = "MechanicAdvice"
    question = "hey"
    limit=200
    dump = search_reddit_posts(subreddit=subreddit, question=question, limit=limit)
    print(str(dump))
    write_to_file("json.output.txt", str(dump))


main()
