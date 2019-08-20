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


def file_load(file_name: str):
    with open(file_name, 'r') as myfile:
        data = myfile.read()
        if data is not None:
            return load_from_string(data)
    return None


def file_write(file_name: str, data: typing.Any):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def scrape_reddit_results(subreddit: str, question: str, limit: int, after=None):
    url: str = "https://www.reddit.com/r/{subreddit}/search.json?" \
               "restrict_sr=1" \
               "&sort=top" \
               "&limit={limit}" \
               "&q={question}".format(subreddit=subreddit, question=question, limit=limit)
    if after is not None:
        url = url + "&after={after}".format(after=after)
    response = load_web_page(url=url)
    return response


def get_reddit_results_list(subreddit: str, question: str, limit: int, after=None):
    response = scrape_reddit_results(subreddit=subreddit, question=question, limit=limit, after=after)

    if response is None:
        print("Error")
        return None, None
    print(response)
    # Get Number of Returned Elements
    number_of_posts: int = int(response["data"]["dist"])
    posts = response["data"]["children"]

    return number_of_posts, posts


def search_reddit_posts_batch(subreddit: str, question: str, limit: int, after=None):

    number_of_posts, posts = get_reddit_results_list(subreddit=subreddit, question=question, limit=limit, after=after)

    if posts is None:
        return False, None, None

    if number_of_posts == 0:
        return True, None, None
    posts_batch = []
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
        text = text.split('\n')
        # Replace all empty elements in array
        text = list(filter(None, text))

        post = dict(text=text, title=title)
        posts_batch.append(post)

    # As count <= limit, job done
    if number_of_posts < limit:
        return True, posts_batch, last_post_id

    return False, posts_batch, last_post_id


def search_reddit_posts(subreddit: str, question: str, limit: int):
    posts = []
    last_post_id = None
    done = 0

    wait_time = 30# 30 Seconds halt time
    increment = 100

    while done < limit:
        # Sleep per transaction
        if last_post_id is not None:
            print ("In wait state now for", wait_time, "seconds")
            time.sleep(wait_time)
        finished, posts_batch, last_post_id_temp = search_reddit_posts_batch(subreddit=subreddit, question=question,
                                                                       limit=increment, after=last_post_id)
        if finished:
            break
        if last_post_id_temp is None:
            print ("Infos was None. Waiting for API to Respond")
            continue
        posts = posts + posts_batch
        last_post_id = last_post_id_temp

        # This Batch done
        done = done + increment

        if done + increment > limit:
            increment = limit - done

    return posts


def main():
    subreddit = "MechanicAdvice"
    question = "hey"
    limit = 120
    posts = search_reddit_posts(subreddit=subreddit, question=question, limit=limit)
    # This is JSON
    print(str(json.dumps(posts)))
    file_write("output.json", posts)


main()
