import asyncio
import os
import re
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from city import *

# 加载 .env 文件
load_dotenv()

def translate_city(city: str) -> str:
    """将中文城市名称转换为英文名称"""
    # 尝试直接匹配
    if city in CITY_MAP:
        return CITY_MAP[city]
    
    # 尝试部分匹配（如"北京市" -> "北京"）
    for cn_city, en_city in CITY_MAP.items():
        if city.startswith(cn_city) or cn_city in city:
            return en_city
    
    # 尝试拼音转换（简单实现，实际应用中可能需要更完善的拼音库）
    try:
        import pypinyin
        pinyin = pypinyin.lazy_pinyin(city)
        return ''.join(pinyin).capitalize()
    except ImportError:
        return city

async def is_weather_query(query: str) -> bool:
    """判断查询是否为天气相关"""
    weather_keywords = ["天气", "温度", "气温", "湿度", "下雨", "下雪", "晴天", "雨天", "多云", "风力"]
    for keyword in weather_keywords:
        if keyword in query:
            return True
    return False

class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.exit_stack = AsyncExitStack()


        self.openai_api_key = "aa9c4ff8c5b1cf10900985c20de2e9edbdbc1e7e"
        self.base_url = "https://aistudio.baidu.com/llm/lmapi/v3"
        self.model = "ernie-4.0-turbo-8k"

        
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.agent_client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器并列出可用工具"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具:", [tool.name for tool in tools])

    async def should_call_tool(self, query: str) -> bool:
        """使用Agent判断是否需要调用工具"""
        system_prompt = """
        你是一个智能助手，需要根据用户查询的意图决定是否需要调用外部工具。
        以下是一些判断原则：
        1. 如果查询明显是询问天气（包含"天气"、"温度"、"湿度"、"下雨"等词），且指定了城市，则应该调用天气查询工具
        2. 如果查询是闲聊、一般性问题或不需要外部数据的请求，则不需要调用工具
        3. 如果查询格式不符合标准天气查询格式（如"北京现在天气如何"），但意图明显是查询天气，仍应调用工具
        4. 优先使用工具获取实时数据，而不是依赖内置知识
        
        请根据以上原则，判断以下查询是否需要调用工具（回答"是"或"否"）：
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            response = self.agent_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=5,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip().lower()
            return content in ["是", "需要", "调用工具", "yes", "y"]
        except Exception as e:
            print(f"\n⚠️ Agent决策时发生错误: {str(e)}")
            return False

    async def extract_city(self, query: str) -> Optional[str]:
        """从查询中提取城市名称"""
        # 首先检查是否是天气查询
        if not await is_weather_query(query):
            return None
            
        # 尝试匹配标准格式
        city_match = re.search(r"(.+?)(?:现在|今天|的)天气", query)
        if city_match:
            return city_match.group(1).strip()
        
        # 尝试匹配更灵活的格式
        city_match = re.search(r"((?:北京|上海|广州|深圳|天津|重庆|成都|杭州|武汉|西安|苏州|郑州|南京|青岛|沈阳|大连|厦门|福州|长沙|哈尔滨|济南|长春|石家庄|合肥|太原|南昌|南宁|昆明|贵阳|海口|乌鲁木齐|呼和浩特|银川|西宁|兰州|拉萨|宁波|温州|无锡|常州|南通|徐州|扬州|盐城|泰州|镇江|淮安|连云港|宿迁|嘉兴|湖州|绍兴|金华|衢州|舟山|台州|丽水|芜湖|马鞍山|安庆|滁州|阜阳|宿州|六安|亳州|池州|宣城|漳州|南平|三明|龙岩|宁德|景德镇|萍乡|九江|新余|鹰潭|赣州|吉安|宜春|抚州|上饶|淄博|枣庄|东营|烟台|潍坊|济宁|泰安|威海|日照|临沂|德州|聊城|滨州|菏泽|开封|洛阳|平顶山|安阳|鹤壁|新乡|焦作|濮阳|许昌|漯河|三门峡|南阳|商丘|信阳|周口|驻马店|黄石|十堰|宜昌|襄阳|鄂州|荆门|孝感|荆州|黄冈|咸宁|随州|恩施|株洲|湘潭|衡阳|邵阳|岳阳|常德|张家界|益阳|郴州|永州|怀化|娄底|湘西|韶关|珠海|汕头|佛山|江门|湛江|茂名|肇庆|惠州|梅州|汕尾|河源|阳江|清远|东莞|中山|潮州|揭阳|云浮|柳州|桂林|梧州|北海|防城港|钦州|贵港|玉林|百色|贺州|河池|来宾|崇左|儋州|自贡|攀枝花|泸州|德阳|绵阳|广元|遂宁|内江|乐山|南充|眉山|宜宾|广安|达州|雅安|巴中|资阳|阿坝|甘孜|凉山|六盘水|遵义|安顺|毕节|铜仁|黔东南|黔南|黔西南|曲靖|玉溪|保山|昭通|丽江|普洱|临沧|楚雄|红河|文山|西双版纳|大理|德宏|怒江|迪庆|昌都|林芝|山南|那曲|阿里|铜川|宝鸡|咸阳|渭南|延安|汉中|榆林|安康|商洛|嘉峪关|金昌|白银|天水|武威|张掖|平凉|酒泉|庆阳|定西|陇南|临夏|甘南|海东|海北|黄南|海南|果洛|玉树|海西|石嘴山|吴忠|固原|中卫|克拉玛依|吐鲁番|哈密|昌吉|博尔塔拉|巴音郭楞|阿克苏|克孜勒苏|喀什|和田|伊犁|塔城|阿勒泰|石河子|阿拉尔|图木舒克|五家渠|北屯|铁门关|双河|可克达拉|昆玉|胡杨河|新星)[市省]?)", query)
        if city_match:
            return city_match.group(1).strip()
        
        # 尝试提取省级行政区
        province_match = re.search(r"((?:北京|天津|上海|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)[省市自治区]?)", query)
        if province_match:
            return province_match.group(1).strip()
        
        return None

    async def process_query(self, query: str) -> str:
        try:
            # 首先判断是否是天气查询
            is_weather = await is_weather_query(query)
            
            if is_weather:
                # 让Agent再次确认是否需要调用工具
                call_tool = await self.should_call_tool(query)
                
                if call_tool:
                    city = await self.extract_city(query)
                    if not city:
                        return "⚠️ 错误：无法识别城市名称，请明确指定城市（如：北京现在天气如何）"

                    try:
                        translated_city = translate_city(city)
                        print(f"Translated city: {translated_city}")

                        # 构造工具调用参数
                        tool_args = {"city": translated_city}

                        messages = [{"role": "user", "content": query}]

                        # 列出可用工具
                        response = await self.session.list_tools()
                        available_tools = [{
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.inputSchema
                            }
                        } for tool in response.tools]

                        # 调用 OpenAI API
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=available_tools,
                            tool_choice="auto"
                        )

                        # 处理工具调用
                        content = response.choices[0]
                        if content.finish_reason == "tool_calls":
                            tool_call = content.message.tool_calls[0]
                            tool_name = tool_call.function.name

                            # 调用工具获取天气信息
                            result = await self.session.call_tool(tool_name, tool_args)
                            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")

                            # 返回工具响应
                            return result.content[0].text

                        return content.message.content
                    except Exception as e:
                        print(f"\n⚠️ 城市转换或工具调用时发生错误: {str(e)}")
                        return f"⚠️ 错误：处理天气查询时发生问题 - {str(e)}"
                else:
                    # 如果不需要调用工具，直接返回天气信息
                    messages = [{"role": "user", "content": query}]
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages
                    )
                    return response.choices[0].message.content
            else:
                # 如果不是天气查询，直接使用原始逻辑处理
                messages = [{"role": "user", "content": query}]
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
        except json.JSONDecodeError as e:
            return f"⚠️ 错误：解析响应时发生JSON解码错误 - {str(e)}"
        except UnicodeEncodeError as e:
            return f"⚠️ 错误：编码问题 - {str(e)}"
        except Exception as e:
            return f"⚠️ 发生错误: {str(e)}"

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n🤖 MCP 客户端已启动！输入 'quit' 退出")

        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print(f"\n🤖 OpenAI: {response}")

            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())