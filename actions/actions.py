from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import json

from services.llm import *

llm_chat = LLMChat()
intents_history_list = []


class ActionHandleConversation(Action):

    def name(self) -> Text:
        return "action_handle_conversation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print(json.dumps(tracker.latest_message, indent=2, ensure_ascii=False))
        # 获取用户的输入以及对应的意图
        user_input = tracker.latest_message.get("text", "")
        intent = tracker.latest_message.get('intent', {}).get('name',"")
        intents_history_list.append(intent)

        # TODO 根据intent，通过embedding，mapping确定需要封装的已知信息,先写死为运费相关知识
        reference_content = "亲亲，如果购买的商品有运费险，退换货时您填写正确的退货地址，上传正确的退货物流快递单号。您先垫付运费，我们确认收货后，系统自动会发起理赔的哦～（售后上门取件服务：运费险直接抵扣首重，快递员与寄件人线下收取除首重运费外的其他费用；寄件人支付运费后，页面上会展示上门取件实付运费及运费险抵扣的运费）"
        # 获取相关的订单信息，物流信息
        metadata = tracker.latest_message.get('metadata', {})

        params = {}
        params["intent"] = intent
        params["order_info"] = metadata.get("order_info", [])  # 获取订单信息列表
        params["logistics_info"] = metadata.get("logistics_info", [])  # 获取物流信息
        params["reference_content"] = reference_content

        # 根据订单信息，物流信息生成相关的回答
        system_message = generate_message(params, first_intent=len(intents_history_list) ==1,role="system")
        llm_chat.add_messages(system_message)
        output = llm_chat.run(user_input)

        dispatcher.utter_message(text=output)

        return []


