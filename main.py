import os
import time
import base64
import re
import uuid
import asyncio 
import aiohttp
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger

# 定义插件名称常量，确保全局一致
PLUGIN_NAME = "astrbot_plugin_gitee_zimg"

@register(PLUGIN_NAME, "jengaklll-a11y", "接入 Gitee AI（模力方舟）z-image-turbo模型（文生图），支持多key轮询", "1.0.3")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 基础配置
        self.base_url = "https://ai.gitee.com/v1"
        self.model_2d = "z-image-turbo"
        
        # 修正: 强制类型转换
        self.steps = int(config.get("num_inference_steps", 9))
        
        # 2. 分辨率解析逻辑
        raw_size_config = config.get("size", "1:1 (2048×2048)")
        size_match = re.search(r"\((\d+)[x×](\d+)\)", raw_size_config)
        
        if size_match:
            self.default_size = f"{size_match.group(1)}x{size_match.group(2)}"
        else:
            logger.warning(f"[{PLUGIN_NAME}] 分辨率配置格式异常: {raw_size_config}，已重置为 2048x2048")
            self.default_size = "2048x2048"

        self.ratio_map = {
            "1:1": "2048x2048", "3:4": "1536x2048", "4:3": "2048x1536",
            "2:3": "1360x2048", "3:2": "2048x1360", "9:16": "1152x2048",
            "16:9": "2048x1152"
        }
        self.valid_sizes = list(self.ratio_map.values())

        # 修正: 优化 API Key 解析逻辑
        self.api_key = ""
        raw_key = config.get("api_key")
        if isinstance(raw_key, list) and raw_key:
            self.api_key = str(raw_key[0])
        elif isinstance(raw_key, str) and raw_key:
            self.api_key = raw_key
            
        if not self.api_key:
            logger.error(f"[{PLUGIN_NAME}] 未配置 API Key，插件无法工作")

        self.retention_hours = float(config.get("retention_hours", 1.0))
        # 新增: 记录上次清理时间
        self.last_cleanup_time = 0

    # =========================================================
    # 自动清理模块
    # =========================================================
    def _cleanup_temp_files(self):
        if self.retention_hours <= 0:
            return
            
        # 动态冷却时间
        interval = max(300, min(3600, int(self.retention_hours * 1800)))
        
        now = time.time()
        if now - self.last_cleanup_time < interval:
            return

        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        if not save_dir.exists():
            return

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
                        except Exception as del_err:
                            logger.warning(f"[{PLUGIN_NAME}] 删除文件失败 {filename}: {del_err}")
            
            if deleted_count > 0:
                logger.info(f"[{PLUGIN_NAME}] 清理完成，共释放 {deleted_count} 张图片")
            
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
                    if resp.status != 200:
                        raise Exception(f"下载失败 HTTP {resp.status}")
                    data = await resp.read()
        
        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
        path = save_dir / file_name
        
        with open(path, "wb") as f:
            f.write(data)
            
        return str(path)

    async def _generate_2d_core(self, prompt: str, size: str = None):
        self._cleanup_temp_files()

        target_size = size if size else self.default_size
        if target_size not in self.valid_sizes:
            logger.warning(f"[{PLUGIN_NAME}] 分辨率 {target_size} 不在支持列表中，自动修正为 2048x2048")
            target_size = "2048x2048"

        url = f"{self.base_url}/images/generations"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://ai.gitee.com/",
            "Origin": "https://ai.gitee.com",
            "Accept": "application/json"
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
                        if "Forbidden" in text_resp:
                            raise Exception(f"请求被拒绝 (403)，可能是API Key无效或被风控。")
                        raise Exception(f"API错误 {resp.status}: {text_resp[:100]}...")
                    
                    resp_json = await resp.json()
                    
                    if "data" in resp_json and len(resp_json["data"]) > 0:
                        data_item = resp_json["data"][0]
                        if "url" in data_item:
                            return await self._download_and_save(data_item["url"])
                        elif "b64_json" in data_item:
                            save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
                            save_dir.mkdir(parents=True, exist_ok=True)
                            path = save_dir / f"{int(time.time())}_b64.jpg"
                            with open(path, "wb") as f: 
                                f.write(base64.b64decode(data_item["b64_json"]))
                            return str(path)
                    
                    raise Exception(f"API返回格式异常: {str(resp_json)[:100]}")

        except asyncio.TimeoutError:
            logger.error(f"[{PLUGIN_NAME}] API请求超时")
            raise Exception("API请求超时，模型可能正在冷启动，请稍后重试。")
        
        except aiohttp.ClientConnectorError as e:
            logger.error(f"[{PLUGIN_NAME}] 网络连接失败: {e}")
            raise Exception(f"无法连接到 Gitee API: {e}")

        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 生图未知出错 ({type(e).__name__}): {e}")
            if not str(e):
                raise Exception(f"未知错误 ({type(e).__name__})，请查看后台日志")
            raise e

    # 已移除 @filter.llm_tool 装饰的 draw 方法

    @filter.command("zimg")
    async def cmd_draw(self, event: AstrMessageEvent, prompt: str = ""): 
        """
        Gitee AI 文生图
        使用方法: /zimg <提示词> [比例]
        """
        # ==========================================
        # 1. 严格的图生图/引用拦截逻辑
        # ==========================================
        if event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, Image):
                    yield event.plain_result("⚠️ 本插件仅支持文生图，不支持发送图片进行生成（图生图）。")
                    return
                if isinstance(component, Reply):
                    yield event.plain_result("⚠️ 本插件仅支持文生图，不支持引用消息或图片进行生成。")
                    return
        
        full_text = ""
        if event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, Plain):
                    full_text += component.text
        
        if not full_text:
            full_text = prompt

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
        
        real_prompt = real_prompt.replace("，", ",")
        real_prompt = re.sub(r'\s+', ' ', real_prompt)
        real_prompt = re.sub(r',+', ',', real_prompt)
        real_prompt = re.sub(r',\s*,', ',', real_prompt)
        real_prompt = real_prompt.strip(" ,")
        
        yield event.plain_result(f"正在绘图{ratio_msg}...")
        
        try:
            img_path = await self._generate_2d_core(real_prompt, size=target_size)
            yield event.chain_result([
                Reply(id=event.message_obj.message_id), 
                Image.fromFileSystem(img_path)
            ])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")
