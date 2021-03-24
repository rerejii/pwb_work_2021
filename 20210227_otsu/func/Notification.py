import requests
import sys
import slackweb
import os

os.environ["http_proxy"] = "http://proxy.noc.kochi-tech.ac.jp:3128"
os.environ["https_proxy"] = "http://proxy.noc.kochi-tech.ac.jp:3128"

lien_url = "https://notify-api.line.me/api/notify"
lien_token = 'WTHikUIiNEsVKZIP1pSkgCUusHuOYOjEIig5ydR2TML'
line_headers = {"Authorization": "Bearer " + lien_token}

slack_url = 'https://hooks.slack.com/services/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'


def notice(message):
    line_notice(message)
    slack_notice(message)
    # payload = {"message": message}
    # _ = requests.post(lien_url, headers=line_headers, params=payload)
    return

def slack_notice(message):
    slack = slackweb.Slack(url=slack_url)
    slack.notify(text=message)
    return
    # payload = {"text": message}
    # requests.post(slack_url, data=payload, headers={"Content-Type": "application/json"})
    # print('aaa')

def line_notice(message):
    payload = {"message": message}
    _ = requests.post(lien_url, headers=line_headers, params=payload)
    return