from astrbot.api.message_components import Image, Plain, Reply
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger
from openai import AsyncOpenAI
import os
import time
import base64
import aiohttp

@register("astrbot_plugin_gitee_zimg", "jengaklll-a11y", "接入 Gitee AI（模力方舟）z-image-turbo模型（文生图），支持多key轮询", "1.0.0")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 基础配置
        self.base_url = config.get("base_url", "https://ai.gitee.com/v1")
        self.model_2d = config.get("model", "z-image-turbo")
        self.size = config.get("size", "2048x2048")
        self.steps = config.get("num_inference_steps", 9)
        
        # Key 解析逻辑
        raw_key = config.get("api_key", "")
        self.api_key = str(raw_key[0]) if isinstance(raw_key, list) and raw_key else (raw_key if isinstance(raw_key, str) and raw_key else "")
        if not self.api_key: logger.error("[GiteeAI] 未配置 API Key")

        # 2. 清理配置
        self.retention_hours = config.get("retention_hours", 1.0)

        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    # =========================================================
    # 🧹 自动清理模块
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
                            # 仅清理图片文件
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

    # ---------------------------------------------------------
    # 模块：文生图核心
    # ---------------------------------------------------------
    async def _generate_2d_core(self, prompt: str):
        try:
            resp = await self.client.images.generate(
                prompt=prompt, model=self.model_2d, size=self.size,
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

    # ---------------------------------------------------------
    # 指令注册
    # ---------------------------------------------------------

    # LLM Tool 供大模型直接调用
    @filter.llm_tool(name="draw_image")
    async def draw(self, event: AstrMessageEvent, prompt: str):
        """生成一张图片"""
        yield event.plain_result("正在绘图...")
        try:
            img_path = await self._generate_2d_core(prompt)
            yield event.chain_result([Image.fromFileSystem(img_path)])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")

    # 指令调用
    @filter.command("zimg")
    async def cmd_draw(self, event: AstrMessageEvent, prompt: str = ""): 
        """
        Gitee AI 文生图
        使用方法: /zimg <提示词>
        """
        full_text = event.message_str or ""
        parts = full_text.split(None, 1)
        real_prompt = parts[1].strip() if len(parts) > 1 else (prompt if prompt else "")
        
        if not real_prompt:
            yield event.plain_result("请提供提示词。")
            return
            
        yield event.plain_result(f"正在使用 {self.model_2d} 绘图...")
        try:
            img_path = await self._generate_2d_core(real_prompt)
            yield event.chain_result([
                Reply(id=event.message_obj.message_id), 
                Image.fromFileSystem(img_path)
            ])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")




