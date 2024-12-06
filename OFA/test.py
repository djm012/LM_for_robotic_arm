
# -*- coding: utf-8 -*-

# This code shows an example of text translation from English to Simplified-Chinese.
# This code runs on Python 2.7.x and Python 3.x.
# You may install `requests` to run this code: pip install requests
# Please refer to `https://api.fanyi.baidu.com/doc/21` for complete api document

import requests
import random
import json

def zh_to_en(zh_str):
    token = '24.3c5c278934cf7d503809ae3cb6b0182f.2592000.1693546593.282335-37062527'
    url = 'https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=' + token

    # For list of language codes, please refer to `https://ai.baidu.com/ai-doc/MT/4kqryjku9#语种列表`
    from_lang = 'zh' # example: en
    to_lang = 'en' # example: zh
    term_ids = '' # 术语库id，多个逗号隔开

    # Build request
    headers = {'Content-Type': 'application/json'}
    payload = {'q': zh_str, 'from': from_lang, 'to': to_lang, 'termIds' : term_ids}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    en_str = result["result"]["trans_result"][0]["dst"]

    return en_str


if __name__=="__main__":
    zh_str = '左侧的红色沙发'
    print(zh_to_en(zh_str))