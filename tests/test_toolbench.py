import requests


def text_sentiment_analysis(
    text: str,
    toolbench_rapidapi_key: str = '088440d910mshef857391f2fc461p17ae9ejsnaebc918926ff',
):
    """
    "Analyse the Sentiment of the given text context"

    """
    url = 'https://text-sentiment-analysis4.p.rapidapi.com/sentiment'
    querystring = {
        'text': text,
    }

    headers = {
        'X-RapidAPI-Key': toolbench_rapidapi_key,
        'X-RapidAPI-Host': 'text-sentiment-analysis4.p.rapidapi.com',
    }

    response = requests.get(url, headers=headers, params=querystring)
    try:
        observation = response.json()
    except:  # noqa: E722
        observation = response.text
    print(f'Observation: {observation}')
    return observation


if __name__ == '__main__':
    text_sentiment_analysis('')
