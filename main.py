from astrbot.api.message_components import Image, Plain, File, Reply
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger
from openai import AsyncOpenAI
import os
import time
import base64
import aiohttp
import asyncio
import json
import re
import random
import shutil

@register("astrbot_plugin_gitee_zimg", "jengaklll-a11y", "Gitee AI 单模版2D+3D")
class GiteeAIImage(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 基础配置
        self.base_url = config.get("base_url", "https://ai.gitee.com/v1")
        self.model_2d = config.get("model", "z-image-turbo")
        self.model_3d = config.get("model_3d", "Hunyuan3D-2") 
        self.enable_texture = config.get("enable_texture", True)
        
        octree = config.get("octree_resolution", 400)
        if octree > 1024: octree = 400
        self.octree_resolution = octree
        
        self.steps_3d = config.get("steps_3d", 50)
        self.guidance_scale = config.get("guidance_scale", 5.0)
        self.seed = config.get("seed", -1)
        self.size = config.get("size", "1024x1024")
        self.steps = config.get("num_inference_steps", 9)
        
        raw_key = config.get("api_key", "")
        self.api_key = str(raw_key[0]) if isinstance(raw_key, list) and raw_key else (raw_key if isinstance(raw_key, str) and raw_key else "")
        if not self.api_key: logger.error("[GiteeAI] 未配置 API Key")

        # 2. 视觉优化配置
        self.enable_refine = config.get("enable_refine", False)
        
        v_url = config.get("vision_base_url", "")
        if v_url:
            v_url = v_url.rstrip("/")
            if v_url.endswith("/v1"): v_url = v_url[:-3]
            if v_url.endswith("/chat/completions"): v_url = v_url[:-17]
        self.vision_base_url = v_url or "https://api.siliconflow.cn"
        
        raw_v_key = config.get("vision_api_key", "")
        self.vision_api_key = str(raw_v_key[0]) if isinstance(raw_v_key, list) and raw_v_key else (raw_v_key if isinstance(raw_v_key, str) and raw_v_key else self.api_key)
        self.vision_model = config.get("vision_model", "google/gemini-2.0-flash-exp")
        
        # 3. 清理配置
        self.retention_hours = config.get("retention_hours", 1.0)
        
        # 硬核 Prompt (内嵌)
        self.HARDCODED_PROMPT = """
        Analyze the structure and composition of the attached image and recreate it as a 3D untextured white clay model. 
        Use a matte white material with strong ambient occlusion to highlight the geometry details. 
        The lighting should be soft studio lighting with rim lighting to distinguish the subject from the background. 
        The background must be pitch black (#000000).
        Important Constraints: Strictly remove all original colors, patterns, and surface textures. 
        Do not render any skin texture, fabric patterns, or realistic colors. 
        The final output must be completely monochromatic.
        (Output ONLY the Stable Diffusion prompt string to generate this image, do not output any other text.)
        """

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
                            os.remove(file_path)
                            deleted_count += 1
                            if ext in ['.glb', '.obj']:
                                logger.info(f"[GiteeAI] 🗑️ 已清理过期模型: {filename}")
                            elif ext in ['.jpg', '.png', '.jpeg']:
                                logger.info(f"[GiteeAI] 🧹 已清理过期图片: {filename}")
                        except Exception as del_err:
                            logger.warning(f"[GiteeAI] 删除文件失败 {filename}: {del_err}")
            if deleted_count > 0:
                logger.info(f"[GiteeAI] 清理完成，共释放 {deleted_count} 个文件")
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
    # 模块 A：视觉优化 (内嵌Prompt)
    # ---------------------------------------------------------
    async def _optimize_image_prompt(self, image_path: str) -> str:
        target_url = f"{self.vision_base_url}/v1/chat/completions"
        logger.info(f"[GiteeAI] 正在进行白模化处理 (Model: {self.vision_model})")
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
            "Authorization": f"Bearer {self.vision_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.HARDCODED_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 400,
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(target_url, headers=headers, json=payload, timeout=40) as resp:
                    resp_text = await resp.text()
                    try: result = json.loads(resp_text)
                    except: raise Exception(f"非JSON响应: {resp.status}")

                    if isinstance(result, list):
                        if len(result) > 0 and isinstance(result[0], dict):
                            if "error" in result[0]: raise Exception(f"API Error: {result[0]['error']}")
                            if "choices" in result[0]: return result[0]['choices'][0]['message']['content']
                        raise Exception(f"未知响应结构")

                    if resp.status != 200:
                        err = result.get('error', {}).get('message', resp_text)
                        raise Exception(f"HTTP {resp.status}: {err}")

                    content = result['choices'][0]['message']['content']
                    logger.info(f"[GiteeAI] 白模Prompt生成完毕")
                    return content
            except Exception as e:
                logger.error(f"[GiteeAI] 视觉模块异常: {e}")
                raise e

    # ---------------------------------------------------------
    # 模块 B：文生图
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
    # 模块 C：3D 建模 (核弹提取版)
    # ---------------------------------------------------------
    async def _generate_3d_core(self, image_path: str):
        if not self.api_key: raise Exception("未配置 API Key")
        submit_url = f"{self.base_url}/async/image-to-3d"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        current_seed = self.seed if self.seed != -1 else random.randint(1, 10000000)
        if current_seed > 10000000: current_seed %= 10000000
        if current_seed <= 0: current_seed = 1

        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('model', self.model_3d)
            data.add_field('type', 'glb') 
            data.add_field('format', 'glb')
            data.add_field('num_inference_steps', str(self.steps_3d))
            data.add_field('octree_resolution', str(self.octree_resolution))
            data.add_field('guidance_scale', str(self.guidance_scale))
            data.add_field('seed', str(current_seed))
            data.add_field('texture', 'true' if self.enable_texture else 'false')
            
            extra_params = {
                "num_inference_steps": self.steps_3d,
                "octree_resolution": self.octree_resolution,
                "guidance_scale": self.guidance_scale,
                "seed": current_seed,
                "texture": self.enable_texture
            }
            data.add_field('parameters', json.dumps(extra_params))
            data.add_field('image', open(image_path, 'rb'), filename='input.jpg')

            try:
                logger.info(f"[GiteeAI] 提交3D任务: res={self.octree_resolution}, seed={current_seed}")
                async with session.post(submit_url, headers=headers, data=data) as resp:
                    resp_json = await resp.json()
                    if resp.status != 200: raise Exception(f"提交失败({resp.status}): {resp_json.get('message')}")
                    
                    task_id = resp_json.get("id") or resp_json.get("task_id")
                    poll_url = resp_json.get("urls", {}).get("get")
                    if not task_id: raise Exception("未获取到任务ID")
                    if not poll_url: poll_url = f"{self.base_url}/task/{task_id}"
                    logger.info(f"[GiteeAI] 任务ID: {task_id}, 开始轮询...")
            except Exception as e:
                raise Exception(f"请求异常: {str(e)}")

            max_retries = 400
            for i in range(max_retries):
                await asyncio.sleep(3)
                try:
                    async with session.get(poll_url, headers=headers) as resp:
                        if resp.status != 200: continue
                        result = await resp.json()
                        status = str(result.get("status")).lower()
                        
                        if i % 10 == 0: logger.info(f"[GiteeAI] 轮询 {i}/{max_retries}: {status}")

                        is_success = False
                        if status in ["succeeded", "success", "completed", "done"]: is_success = True
                        extracted_url = self._nuclear_extract_url(result)
                        if extracted_url: is_success = True

                        if is_success:
                            logger.info("[GiteeAI] 任务成功! 准备下载...")
                            if not extracted_url:
                                logger.error(f"[GiteeAI DEBUG] 无链接! JSON: {json.dumps(result, ensure_ascii=False)}")
                                raise RuntimeError("API返回成功但未找到模型链接")
                            
                            logger.info(f"[GiteeAI] 提取到链接: {extracted_url}")
                            suffix = ".obj" if str(extracted_url).endswith(".obj") else ".glb"
                            return await self._download_and_save(extracted_url, suffix=suffix)
                        elif status == "failed":
                            err = result.get('error', '未知错误')
                            if "waiting" in str(err) or "progress" in str(err): continue
                            raise RuntimeError(f"生成失败: {err}")
                except Exception as poll_err:
                    if isinstance(poll_err, RuntimeError): raise poll_err
                    if "生成失败" in str(poll_err): raise poll_err
                    logger.warning(f"轮询暂态异常: {poll_err}")
                    continue
            raise Exception("任务等待超时")

    def _nuclear_extract_url(self, result: dict):
        def recursive_find(obj):
            if isinstance(obj, dict):
                if "glb" in obj and isinstance(obj["glb"], str) and obj["glb"].startswith("http"): return obj["glb"]
                if "url" in obj and isinstance(obj["url"], str) and obj["url"].startswith("http"): return obj["url"]
                for v in obj.values():
                    r = recursive_find(v)
                    if r: return r
            return None
        url = recursive_find(result)
        if url: return url
        json_str = json.dumps(result)
        match_glb = re.search(r'(https?://[^"]+\.glb)', json_str)
        if match_glb: return match_glb.group(1)
        match_obj = re.search(r'(https?://[^"]+\.obj)', json_str)
        if match_obj: return match_obj.group(1)
        return None

    def _scan_images_recursive(self, obj, found_urls=None, depth=0):
        if found_urls is None: found_urls = set()
        if depth > 10: return found_urls
        if hasattr(obj, "url") and isinstance(obj.url, str) and self._is_image_url(obj.url): found_urls.add(obj.url)
        if isinstance(obj, str):
            if self._is_image_url(obj): found_urls.add(obj)
            elif (obj.startswith("{") or obj.startswith("[")) and len(obj) < 5000:
                try: self._scan_images_recursive(json.loads(obj), found_urls, depth+1)
                except: pass
        elif isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, str) and self._is_image_url(v): found_urls.add(v)
                else: self._scan_images_recursive(v, found_urls, depth+1)
        elif isinstance(obj, list):
            for v in obj: self._scan_images_recursive(v, found_urls, depth+1)
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if not k.startswith("_"): self._scan_images_recursive(v, found_urls, depth+1)
        return found_urls

    def _is_image_url(self, text: str):
        if not text or not isinstance(text, str) or not text.startswith("http") or len(text)<10: return False
        text = text.lower()
        return any(e in text for e in [".jpg", ".png", "image", "qpic", "file", "download"])

    @filter.llm_tool(name="draw_image")
    async def draw(self, event: AstrMessageEvent, prompt: str):
        yield event.plain_result("正在绘图...")
        try:
            img_path = await self._generate_2d_core(prompt)
            # LLM工具通常不需要Reply，因为它本身就在对话流中
            yield event.chain_result([Image.fromFileSystem(img_path)])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")

    @filter.command("aiimg")
    async def cmd_draw(self, event: AstrMessageEvent, prompt: str = ""): 
        full_text = event.message_str or ""
        parts = full_text.split(None, 1)
        real_prompt = parts[1].strip() if len(parts) > 1 else (prompt if prompt else "")
        if not real_prompt:
            yield event.plain_result("请提供提示词。")
            return
        yield event.plain_result(f"正在使用 {self.model_2d} 绘图...")
        try:
            img_path = await self._generate_2d_core(real_prompt)
            # 修复：Reply 使用关键字参数 id
            yield event.chain_result([Reply(id=event.message_obj.message_id), Image.fromFileSystem(img_path)])
        except Exception as e:
            yield event.plain_result(f"绘图失败: {e}")

    @filter.command("img3d")
    async def cmd_img3d(self, event: AstrMessageEvent):
        target_img_url = None
        urls = set()
        self._scan_images_recursive(event, urls)
        valid = list(urls)
        if valid:
            target_img_url = valid[0]
            for u in valid:
                if ".jpg" in u or ".png" in u:
                    target_img_url = u
                    break
        
        if not target_img_url:
            yield event.plain_result("未检测到图片，请将图片和指令一起发送。")
            return

        mode_str = "带纹理" if self.enable_texture else "白模"
        yield event.plain_result(f"已捕获图片，处理中...")
        
        try:
            img_path = await self._download_and_save(target_img_url)
            
            if self.enable_refine:
                yield event.plain_result(f"✨ 正在进行【白模化】预处理 (Gemini -> White Clay)...")
                try:
                    prompt_opt = await self._optimize_image_prompt(img_path)
                    img_path_new = await self._generate_2d_core(prompt_opt)
                    
                    # 修复：Reply 使用关键字参数 id
                    yield event.chain_result([
                        Reply(id=event.message_obj.message_id),
                        Plain("✅ 白模化处理完成："),
                        Image.fromFileSystem(img_path_new)
                    ])
                    img_path = img_path_new
                except Exception as refine_err:
                    logger.error(f"优化失败: {refine_err}")
                    yield event.plain_result(f"⚠️ 白模化失败({str(refine_err)[:20]}...)，使用原图。")

            yield event.plain_result(f"正在建模 [{mode_str}] (分辨率:{self.octree_resolution})...")
            model_path = await self._generate_3d_core(img_path)
            
            yield event.plain_result("建模完成，正在发送文件...")
            file_name = os.path.basename(model_path)
            
            # 修复：Reply 使用关键字参数 id
            yield event.chain_result([
                Reply(id=event.message_obj.message_id),
                File(name=file_name, file=str(model_path))
            ])
            
        except Exception as e:
            logger.error(f"3D生成失败: {e}")
            yield event.plain_result(f"建模失败: {e}")

