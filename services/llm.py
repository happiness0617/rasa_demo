from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, GPT4All
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    PromptTemplate
)
from langchain.chains import ConversationChain, LLMChain

import os

os.environ['OPENAI_API_KEY'] = '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'
"""
此类功能如下：
1. 需要处理某个intent的回复：
    由此需要根据intent生成相关的prompt：
        prompt的生成需要读取数据库（处理准则，处理流程）以及第三方传递过来的数据（订单信息，物流信息）
        这里就需要redis作为消息中间件，存储消息历史
        另外要结合相关intent的知识库
2. 切换不同的LLM（ChatGPT，ChatGLM等），以测试不同的LLM的效果

"""


def generate_message(params: dict, first_intent=True, role="system"):
    """
    根据意图以及相关的参数生成相对应的prompt
    :param params: 所需的参数
    :return:
    """
    intent = params["intent"]
    if first_intent:
        base_prompt = """你作为一名羽绒服的电商客服，根据以下已知的信息回答用户的问题：
{info}
回复要求:
说话风格要口语化，每次不要超过30个字
不要重复回复相似的内容，根据用户的反馈来回答
不要回答与你工作无关的任何问题
"""
    else:
        base_prompt = """根据system提供的内容回答用户的问题：
{info}
        """
    info = ""
    if intent == "售后问题_运费问题":
        # 如果用户询问运费问题，那么参考内容要包含运费险的知识
        reference_content = params["reference_content"]
        reference_content = "亲亲，如果购买的商品有运费险，退换货时您填写正确的退货地址，上传正确的退货物流快递单号。您先垫付运费，我们确认收货后，系统自动会发起理赔的哦～（售后上门取件服务：运费险直接抵扣首重，快递员与寄件人线下收取除首重运费外的其他费用；寄件人支付运费后，页面上会展示上门取件实付运费及运费险抵扣的运费）"
        info = reference_content

    elif intent == "售后问题_退货":
        # 1. 获取订单状态，如果是未付款，那么就是咨询退货政策的，直接回复即可
        # order_info = params.get("order_info", [])
        # order_status = order_info[0]["order_status"]  # TODO 先处理单订单的状态，如果有多个订单，后面再处理
        # logistics_info = params.get("logistics_info")
        # logistics_status = logistics_info[0]["logistics_status"]

        order_status = "已付款"
        logistics_status = "已发货"

        base_info = ""
        policy_info = ""
        conversion_purpose = ""
        if order_status == "未付款":
            base_info = "用户尚未付款"
            policy_info = "在订单详情里面申请退货退款，退货原因建议选择其他/不合适/七天无理由都可以的呢，系统会自动通过您的申请"
            conversion_purpose = "专业，耐心的解决用户问题"
        # 2. 如果是已付款，未发货，那么就是催快递的，安抚客户情绪，实在不行可以直接退款
        if order_status == "待发货":
            base_info = "用户已经2023-6-9 15:00付款,现在时间为2023-6-10 13:00"
            policy_info = "如果用户确定要退货，在订单详情里面申请退货退款，退货原因建议选择其他/不合适/七天无理由都可以的呢，系统会自动通过您的申请"
            conversion_purpose = "挽留订单，并介绍衣服的款式，销量等，如果用户执意退货，给出相关的解决方案"
        # 3. 判断物流状态，如果是已发货,未签收，那么要告知退货流程，比较麻烦
        if logistics_status == "已发货":
            base_info = "我们已于2023-6-9 15:00发货，物流已经到xxx地方，"
            policy_info = "因为商品还在物流中，这个时候退货，需要拦截快递，如果快递拦截失败，需要让用户拒收快递，等仓库这边确认收到商品后，再给用户退款"
            conversion_purpose = "通过退货麻烦，推销商品等方式来挽留订单，如果用户执意退货，给出相关的解决方案"
        # 4. 如果是已签收，则正常处理退货流程
        if logistics_status == "已签收":
            base_info = "商品已于2023-6-9 15:00被签收，现在时间为2023-6-10 13:00"
            policy_info = "如果商品签收时间超过7天，婉转告诉用户超过售后时间。商品退货，需要用户提供吊牌照片以及商品完整的照片，进行核实，" \
                          "如果商品没有被穿洗过，不影响二次销售，则同意退货退款。并告知用户，等仓库确认收货后，会第一时间退款给用户"
            conversion_purpose = "了解用户退货原因，通过补偿或者换货来挽留订单，如果用户执意退货，给出相关的解决方案"

        info = f"{base_info} 意图相关知识:{policy_info}\n对话的目的:{conversion_purpose}"

    elif intent == "商品问题_款式问题":
        reference_content = params["reference_content"]
        reference_content = ""
        info = reference_content
    elif intent == "商品问题_尺码问题":
        reference_content = params["reference_content"]
        reference_content = "如果尺码较小，给用户换货的建议；如果尺码偏大，可以给解释一下冬天是需要穿衣服的"
        info = reference_content
    elif intent == "商品问题_颜色问题":
        reference_content = params["reference_content"]
        reference_content = "先推销当前衣服的颜色，再提供其他颜色，给用户换货的建议"
        info = reference_content

    prompt_str = base_prompt.format(info=info)

    if role == "system":
        message = SystemMessagePromptTemplate.from_template(prompt_str)
    elif role == "human":
        message = HumanMessagePromptTemplate.from_template(prompt_str)
    else:
        message = AIMessagePromptTemplate.from_template(prompt_str)

    return message


class LLMChat:

    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens
        self.intent = ""
        self.info_dict = {}
        self.prompt = PromptTemplate.from_template("")
        self.messages_list = []
        self.prompt_str = ""
        self.conversion_history_list = []
        self.type = "chat_model"
        self.verbose = True
        if self.type == "chat_model":
            self.model = ChatOpenAI()
        elif self.type == "model":
            self.model = OpenAI()
        self.chain = LLMChain(prompt=self.prompt, llm=self.model, verbose=self.verbose)

    # def refresh_conversion_history(self, user_input, output):
    #     self.conversion_history_list.append(HumanMessagePromptTemplate.from_template(user_input))
    #     self.conversion_history_list.append(AIMessagePromptTemplate.from_template(output))

    def add_messages(self, message):
        self.conversion_history_list.append(message)

    def run(self, user_input):
        self.conversion_history_list.append(HumanMessagePromptTemplate.from_template("{user_input}"))
        self.chain.prompt = ChatPromptTemplate.from_messages(messages=self.conversion_history_list)

        output = self.chain.run({"user_input": user_input})

        # 移除对话历史中的最后一个human模版，换成内容输出
        self.conversion_history_list.pop()
        self.conversion_history_list.append(HumanMessagePromptTemplate.from_template(user_input))
        self.conversion_history_list.append(AIMessagePromptTemplate.from_template(output))
        return output
