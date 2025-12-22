from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger
from openai import AsyncOpenAI
import os
import time
import base64
import aiohttp
import re

@register("astrbot_plugin_gitee_zimg", "jengaklll-a11y", "接入 Gitee AI（模力方舟）z-image-turbo模型（文生图），支持多key轮询", "1.0.0")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 基础配置 (内置)
        self.base_url = "https://ai.gitee.com/v1"
        self.model_2d = "z-image-turbo"
        # 直接读取步数，配置文件的 slider 只是前端交互，传回来依然是数字
        self.steps = config.get("num_inference_steps", 9)
        
        # 2. 分辨率解析逻辑
        # 选项格式示例: "1:1 (2048×2048)"
        raw_size_config = config.get("size", "1:1 (2048×2048)")
        
        # 正则匹配：寻找括号 ( ) 内部，格式为 数字 + x或× + 数字
        size_match = re.search(r"\((\d+)[x×](\d+)\)", raw_size_config)
        
        if size_match:
            # 拼接为 API 需要的格式 "2048x2048"
            self.default_size = f"{size_match.group(1)}x{size_match.group(2)}"
        else:
            # 兜底
            logger.warning(f"[GiteeAI] 分辨率配置格式异常: {raw_size_config}，已重置为 2048x2048")
            self.default_size = "2048x2048"

        # Gitee AI 支持的分辨率列表
        self.ratio_map = {
            "1:1": "2048x2048",
            "3:4": "1536x2048",
            "4:3": "2048x1536",
            "2:3": "1360x2048",
            "3:2": "2048x1360",
            "9:16": "1152x2048",
            "16:9": "2048x1152"
        }
        self.valid_sizes = list(self.ratio_map.values())

        # Key 解析
        raw_key = config.get("api_key", "")
        self.api_key = str(raw_key[0]) if isinstance(raw_key, list) and raw_key else (raw_key if isinstance(raw_key, str) and raw_key else "")
        if not self.api_key: logger.error("[GiteeAI] 未配置 API Key")

        # 3. 清理配置
        self.retention_hours = config.get("retention_hours", 1.0)

        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    # =========================================================
    # 自动清理模块
    # =========================================================
    def _cleanup_temp_files(self):
        if self.retention_hours <= 0: return
        save_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg") / "images"
        if not save_dir.exists(): return

        now = time.time()
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
                            logger.warning(f"[GiteeAI] 删除文件失败 {filename}: {del_err}")
            if deleted_count > 0:
                logger.info(f"[GiteeAI] 清理完成，共释放 {deleted_count} 张图片")
        except Exception as e:
            logger.warning(f"[GiteeAI] 自动清理流程异常: {e}")

    async def _download_and_save(self, url: str, suffix: str = ".jpg") -> str:
        self._cleanup_temp_files() 
        url = url.strip()
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession() as session:
            try:
                if url.startswith("data:image"):
                    header, encoded = url.split(",", 1)
                    data = base64.b64decode(encoded)
                else:
                    async with session.get(url, headers=headers, timeout=60) as resp:
                        if resp.status != 200: raise Exception(f"下载失败: {resp.status}")
                        data = await resp.read()
            except Exception as e:
                raise e
        
        save_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg") / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{int(time.time())}_{os.urandom(2).hex()}{suffix}"
        with open(path, "wb") as f: f.write(data)
        return str(path)

    async def _generate_2d_core(self, prompt: str, size: str = None):
        target_size = size if size else self.default_size
        
        if target_size not in self.valid_sizes:
            logger.warning(f"[GiteeAI] 分辨率 {target_size} 不在支持列表中，自动修正为 2048x2048")
            target_size = "2048x2048"

        try:
            resp = await self.client.images.generate(
                prompt=prompt, model=self.model_2d, size=target_size,
                extra_body={"num_inference_steps": self.steps}
            )
            data = resp.data[0]
            if data.url: return await self._download_and_save(data.url)
            elif data.b64_json:
                save_dir = StarTools.get_data_dir("astrbot_plugin_gitee_aiimg") / "images"
                save_dir.mkdir(parents=True, exist_ok=True)
                path = save_dir / f"{int(time.time())}_b64.jpg"
                with open(path, "wb") as f: f.write(base64.b64decode(data.b64_json))
                return str(path)
            else: raise Exception("API未返回URL或Base64")
        except Exception as e:
            logger.error(f"生图出错: {e}")
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
        full_text = event.message_str or ""
        parts = full_text.split(None, 1)
        real_prompt = parts[1].strip() if len(parts) > 1 else (prompt if prompt else "")
        
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

