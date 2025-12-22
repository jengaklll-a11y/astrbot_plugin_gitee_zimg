import os
import time
import base64
import re
import json
import uuid
import aiohttp
from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger

# 定义插件名称常量，确保全局一致
PLUGIN_NAME = "astrbot_plugin_gitee_zimg"

@register(PLUGIN_NAME, "jengaklll-a11y", "接入 Gitee AI（模力方舟）z-image-turbo模型（文生图），支持多key轮询", "1.0.0")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 基础配置
        self.base_url = "https://ai.gitee.com/v1"
        self.model_2d = "z-image-turbo"
        
        # 修正: 强制类型转换，防止配置项被存为字符串
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
        # 新增: 记录上次清理时间，避免每次请求都扫描磁盘
        self.last_cleanup_time = 0

    # =========================================================
    # 自动清理模块 (优化 IO)
    # =========================================================
    def _cleanup_temp_files(self):
        if self.retention_hours <= 0:
            return
            
        # 优化: 增加冷却时间，每 3600 秒(1小时)最多检查一次
        now = time.time()
        if now - self.last_cleanup_time < 3600:
            return

        # 修正: 使用正确的插件目录名
        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        if not save_dir.exists():
            return

        retention_seconds = self.retention_hours * 3600
        deleted_count = 0

        try:
            for filename in os.listdir(save_dir):
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
            
            # 更新上次清理时间
            self.last_cleanup_time = now
            
        except Exception as e:
            logger.warning(f"[{PLUGIN_NAME}] 自动清理流程异常: {e}")

    async def _download_and_save(self, url: str, suffix: str = ".jpg") -> str:
        # 触发清理检查
        self._cleanup_temp_files() 
        
        url = url.strip()
        headers = {"User-Agent": "Mozilla/5.0"}
        
        # 修正: 移除宽泛的 try...except
        async with aiohttp.ClientSession() as session:
            if url.startswith("data:image"):
                header, encoded = url.split(",", 1)
                data = base64.b64decode(encoded)
            else:
                async with session.get(url, headers=headers, timeout=60) as resp:
                    if resp.status != 200:
                        raise Exception(f"下载失败 HTTP {resp.status}")
                    data = await resp.read()
        
        # 修正: 使用正确的插件目录名
        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 修正: 使用 uuid 防止文件名冲突
        file_name = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
        path = save_dir / file_name
        
        # 修正: 换行书写
        with open(path, "wb") as f:
            f.write(data)
            
        return str(path)

    async def _generate_2d_core(self, prompt: str, size: str = None):
        target_size = size if size else self.default_size
        
        if target_size not in self.valid_sizes:
            logger.warning(f"[{PLUGIN_NAME}] 分辨率 {target_size} 不在支持列表中，自动修正为 2048x2048")
            target_size = "2048x2048"

        # FIX: 弃用 OpenAI SDK，改用原生 aiohttp 请求以完全控制 Header
        url = f"{self.base_url}/images/generations"
        
        # 伪装成浏览器的完整 Header
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
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
                async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
                    if resp.status != 200:
                        # 尝试解析错误信息
                        text_resp = await resp.text()
                        if "Forbidden" in text_resp:
                            raise Exception(f"请求被拒绝 (403)，可能是API Key无效或被风控。")
                        raise Exception(f"API错误 {resp.status}: {text_resp[:100]}...")
                    
                    resp_json = await resp.json()
                    
                    # 解析 OpenAI 格式的返回
                    if "data" in resp_json and len(resp_json["data"]) > 0:
                        data_item = resp_json["data"][0]
                        if "url" in data_item:
                            return await self._download_and_save(data_item["url"])
                        elif "b64_json" in data_item:
                            # 修正: 使用正确的插件目录名
                            save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
                            save_dir.mkdir(parents=True, exist_ok=True)
                            path = save_dir / f"{int(time.time())}_b64.jpg"
                            with open(path, "wb") as f: 
                                f.write(base64.b64decode(data_item["b64_json"]))
                            return str(path)
                    
                    raise Exception(f"API返回格式异常: {str(resp_json)[:100]}")

        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 生图出错: {e}")
            raise e

    @filter.llm_tool(name="draw_image")
    async def draw(self, event: AstrMessageEvent, prompt: str):
        """生成一张图片"""
        yield event.plain_result("正在绘图...")
        try:
            img_path = await self._generate_2d_core(prompt)
            yield event.chain_result([Image.fromFileSystem(img_path)])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")

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
        
        # 修正: 移除冗余的 parsing，直接使用 framework 提供的 prompt
        real_prompt = prompt.strip()
        
        if not real_prompt:
            yield event.plain_result("请提供提示词。")
            return

        # ==========================================
        # 2. 比例参数提取
        # ==========================================
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
                # 从提示词中移除比例部分
                real_prompt = real_prompt.replace(raw_ratio, " ")
        
        # 清理多余空格和标点
        real_prompt = re.sub(r'\s+', ' ', real_prompt)
        real_prompt = re.sub(r'\s*([,，])\s*[,，]\s*', ', ', real_prompt)
        real_prompt = real_prompt.strip(" ,，")
        
        yield event.plain_result(f"正在绘图{ratio_msg}...")
        
        try:
            img_path = await self._generate_2d_core(real_prompt, size=target_size)
            yield event.chain_result([
                Reply(id=event.message_obj.message_id), 
                Image.fromFileSystem(img_path)
            ])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")
