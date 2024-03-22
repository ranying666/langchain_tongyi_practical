# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html

from http import HTTPStatus
import dashscope

def sample_call_streaming():
    prompt_text = '用萝卜、土豆、茄子做饭，给我个菜谱。'
    response_generator = dashscope.Generation.call(
        model='qwen-turbo',
        prompt=prompt_text,
        stream=True,
        top_p=0.8)
    # When stream=True, the return is Generator,
    # need to get results through iteration
    # for response in response_generator:
    #     # The response status_code is HTTPStatus.OK indicate success,
    #     # otherwise indicate request is failed, you can get error code
    #     # and message from code and message.
    #     if response.status_code == HTTPStatus.OK:
    #         print(response.output)  # The output text
    #         print(response.usage)  # The usage information
    #     else:
    #         print(response.code)  # The error code.
    #         print(response.message)  # The error message.
    head_idx = 0
    for resp in response_generator:
        paragraph = resp.output['text']
        print("\r%s" % paragraph[head_idx:len(paragraph)], end='')
        if(paragraph.rfind('\n') != -1):
            head_idx = paragraph.rfind('\n') + 1

sample_call_streaming()