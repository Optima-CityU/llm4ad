import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm4ad.tools.llm.llm_api_qwen import QwenAPI
from llm4ad.tools.llm.llm_api_zhipu import ZhipuAPI
from llm4ad.tools.llm.llm_api_volcengine import VolcengineAPI
from llm4ad.tools.llm.llm_api_baiduqianfan import BaiduQianfanAPI
from llm4ad.tools.llm.llm_api_tencentcloud import TencentCloudAPI


def main():
    # Qwen
    llm = QwenAPI(
        key='your-api-key',
        model='qwen-plus',
        timeout=120
    )

    # Zhipu AI
    # llm = ZhipuAPI(
    #     key='your-api-key',
    #     model='GLM-5.2',
    #     timeout=120
    # )

    # Volcengine Ark
    # llm = VolcengineAPI(
    #     key='your-api-key',
    #     model='doubao-seed-character-260628',
    #     timeout=120
    # )

    # Baidu Qianfan
    # llm = BaiduQianfanAPI(
    #     key='your-api-key',
    #     model='ernie-5.1',
    #     timeout=120
    # )

    # Tencent Cloud
    # llm = TencentCloudAPI(
    #     key='your-api-key',
    #     model='hy3-preview',
    #     timeout=120
    # )

    print(llm.draw_sample('hello'))


if __name__ == '__main__':
    main()
