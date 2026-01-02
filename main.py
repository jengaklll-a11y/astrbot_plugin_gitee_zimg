import os
import time
import base64
import re
import uuid
import asyncio 
import aiohttp
import json
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_gitee_zimg"

@register(PLUGIN_NAME, "jengaklll-a11y", "接入 Gitee AI（模力方舟）z-image-turbo模型（文生图），支持多key轮询，自动撤回", "1.0.5")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 基础配置
        self.base_url = "https://ai.gitee.com/v1"
        self.model_2d = "z-image-turbo"
        self.steps = int(config.get("num_inference_steps", 9))
        
        # 2. 分辨率解析
        raw_size_config = config.get("size", "1:1 (2048×2048)")
        size_match = re.search(r"\((\d+)[x×](\d+)\)", raw_size_config)
        if size_match:
            self.default_size = f"{size_match.group(1)}x{size_match.group(2)}"
        else:
            self.default_size = "2048x2048"

        self.ratio_map = {
            "1:1": "2048x2048", "3:4": "1536x2048", "4:3": "2048x1536",
            "2:3": "1360x2048", "3:2": "2048x1360", "9:16": "1152x2048",
            "16:9": "2048x1152"
        }
        self.valid_sizes = list(self.ratio_map.values())

        # API Key
        self.api_key = ""
        raw_key = config.get("api_key")
        if isinstance(raw_key, list) and raw_key:
            self.api_key = str(raw_key[0])
        elif isinstance(raw_key, str) and raw_key:
            self.api_key = raw_key
            
        if not self.api_key:
            logger.error(f"[{PLUGIN_NAME}] 未配置 API Key，插件无法工作")

        self.retention_hours = float(config.get("retention_hours", 1.0))
        self.last_cleanup_time = 0
        self.auto_recall = int(config.get("auto_recall", 0))

    # =========================================================
    # 辅助工具：稳健获取 Bot 实例
    # =========================================================
    def _get_bot(self, event: AstrMessageEvent):
        if hasattr(event, "bot") and event.bot:
            return event.bot
        if hasattr(event, "message_obj") and hasattr(event.message_obj, "bot"):
            return event.message_obj.bot
        try:
            return self.context.get_bot()
        except:
            pass
        return None

    # =========================================================
    # 自动清理模块
    # =========================================================
    def _cleanup_temp_files(self):
        if self.retention_hours <= 0: return
        
        interval = max(300, min(3600, int(self.retention_hours * 1800)))
        now = time.time()
        if now - self.last_cleanup_time < interval: return

        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        if not save_dir.exists(): return

        retention_seconds = self.retention_hours * 3600
        deleted_count = 0

        try:
            files = os.listdir(save_dir)
            for filename in files:
                file_path = save_dir / filename
                if file_path.is_file():
                    if now - file_path.stat().st_mtime > retention_seconds:
                        try:
                            ext = file_path.suffix.lower()
                            if ext in ['.jpg', '.png', '.jpeg', '.webp']:
                                os.remove(file_path)
                                deleted_count += 1
                        except Exception: pass
            self.last_cleanup_time = now
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 自动清理流程异常: {e}")

    async def _download_and_save(self, url: str, suffix: str = ".jpg") -> str:
        url = url.strip()
        headers = {"User-Agent": "Mozilla/5.0"}
        
        async with aiohttp.ClientSession() as session:
            if url.startswith("data:image"):
                header, encoded = url.split(",", 1)
                data = base64.b64decode(encoded)
            else:
                async with session.get(url, headers=headers, timeout=60) as resp:
                    if resp.status != 200: raise Exception(f"下载失败 HTTP {resp.status}")
                    data = await resp.read()
        
        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
        path = save_dir / file_name
        with open(path, "wb") as f: f.write(data)
        return str(path)

    async def _generate_2d_core(self, prompt: str, size: str = None):
        self._cleanup_temp_files()
        target_size = size if size else self.default_size
        if target_size not in self.valid_sizes: target_size = "2048x2048"

        url = f"{self.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
        payload = {
            "model": self.model_2d,
            "prompt": prompt,
            "size": target_size,
            "num_inference_steps": self.steps
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=180) as resp:
                    if resp.status != 200:
                        text_resp = await resp.text()
                        if "Forbidden" in text_resp: raise Exception("API Key无效或被风控(403)")
                        raise Exception(f"API错误 {resp.status}")
                    
                    resp_json = await resp.json()
                    if "data" in resp_json and len(resp_json["data"]) > 0:
                        data_item = resp_json["data"][0]
                        if "url" in data_item:
                            return await self._download_and_save(data_item["url"])
                        elif "b64_json" in data_item:
                            save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
                            save_dir.mkdir(parents=True, exist_ok=True)
                            path = save_dir / f"{int(time.time())}_b64.jpg"
                            with open(path, "wb") as f: f.write(base64.b64decode(data_item["b64_json"]))
                            return str(path)
                    raise Exception(f"API返回数据异常")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 生图失败: {e}")
            raise e

    # =========================================================
    # 撤回逻辑
    # =========================================================
    async def _recall_later(self, bot, raw_result):
        if self.auto_recall <= 0 or not raw_result: return
        await asyncio.sleep(self.auto_recall)
        
        def _find_id(data):
            if isinstance(data, dict):
                if "message_id" in data: return data["message_id"]
                if "data" in data: return _find_id(data["data"])
            return None

        msg_id = _find_id(raw_result)
        
        if msg_id:
            try:
                msg_id_int = int(msg_id)
                logger.info(f"[{PLUGIN_NAME}] 正在撤回消息 (ID: {msg_id_int})...")
                await bot.api.call_action("delete_msg", message_id=msg_id_int)
            except Exception as e:
                logger.error(f"[{PLUGIN_NAME}] 撤回请求失败: {e}")
        else:
            logger.warning(f"[{PLUGIN_NAME}] 撤回失效: 无法获取 ID")

    @filter.command("zimg")
    async def cmd_draw(self, event: AstrMessageEvent, prompt: str = ""): 
        """
        Gitee AI 文生图
        使用方法: /zimg <提示词> [比例]
        """
        if event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, (Image, Reply)):
                     pass 
        
        full_text = ""
        if event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, Plain):
                    full_text += component.text
        if not full_text: full_text = prompt

        idx = full_text.lower().find("/zimg")
        if idx != -1:
            real_prompt = full_text[idx + 5:].strip()
        else:
            real_prompt = full_text.strip()
            
        if not real_prompt:
            yield event.plain_result("请提供提示词。")
            return

        target_size = None
        ratio_msg = ""
        pattern = r"(\d+[:：]\d+)"
        match = re.search(pattern, real_prompt)
        
        if match:
            raw_ratio = match.group(1)
            ratio_key = raw_ratio.replace("：", ":")
            if ratio_key in self.ratio_map:
                target_size = self.ratio_map[ratio_key]
                ratio_msg = f" (比例 {ratio_key})"
                real_prompt = real_prompt.replace(raw_ratio, " ")
        
        real_prompt = real_prompt.strip(" ,")
        yield event.plain_result(f"正在绘图{ratio_msg}...")
        
        try:
            img_path = await self._generate_2d_core(real_prompt, size=target_size)
            
            # =========================================================
            # 直接调用 Bot API 发送
            # =========================================================
            bot = self._get_bot(event)
            if not bot:
                yield event.chain_result([Image.fromFileSystem(img_path)])
                return

            segments = []
            
            # 1. 引用 (Reply)
            if event.message_obj.message_id:
                segments.append({
                    "type": "reply",
                    "data": {"id": str(event.message_obj.message_id)}
                })
            
            # 2. 图片 (Image) - 放在文字之前
            abs_path = os.path.abspath(img_path)
            if os.name == 'nt':
                abs_path = abs_path.replace("\\", "/")
            
            segments.append({
                "type": "image",
                "data": {"file": f"file:///{abs_path}"}
            })

            # 3. 文本 (Text) - 放在图片之后
            # 构造文案
            text_content = "绘图成功"
            if self.auto_recall > 0:
                text_content += f"，{self.auto_recall}秒后自动撤回..."

            segments.append({
                "type": "text",
                "data": {"text": text_content}
            })

            # 4. 构建 Payload
            payload = {"message": segments}
            msg_obj = event.message_obj
            
            group_id = getattr(msg_obj, "group_id", None)
            user_id = None
            if hasattr(msg_obj, "sender") and hasattr(msg_obj.sender, "user_id"):
                user_id = msg_obj.sender.user_id
            
            if group_id:
                payload["group_id"] = group_id
                action = "send_group_msg"
            elif user_id:
                payload["user_id"] = user_id
                action = "send_private_msg"
            else:
                logger.error(f"[{PLUGIN_NAME}] 无法识别发送目标")
                return

            # 5. 发送
            result = await bot.api.call_action(action, **payload)
            
            # 6. 撤回
            if self.auto_recall > 0:
                asyncio.create_task(self._recall_later(bot, result))
                
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 执行失败: {e}")
            yield event.plain_result(f"执行失败: {e}")

