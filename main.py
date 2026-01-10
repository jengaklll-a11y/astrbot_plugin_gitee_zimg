import os
import time
import base64
import uuid
import asyncio 
import aiohttp
import json
import re
from astrbot.api.message_components import Image, Plain, Reply, At
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, StarTools
from astrbot.api import logger

PLUGIN_NAME = "astrbot_plugin_gitee_zimg"

class GiteeAIUnified(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        self.base_url = "https://ai.gitee.com/v1"
        self.steps = int(config.get("num_inference_steps", 9))
        self.timeout = int(config.get("timeout_seconds", 300))
        
        self.model_t2i = "z-image-turbo"
        raw_size_config = config.get("size", "1:1 (2048×2048)")
        size_match = re.search(r"\((\d+)[x×](\d+)\)", raw_size_config)
        self.default_size = f"{size_match.group(1)}x{size_match.group(2)}" if size_match else "2048x2048"
        
        self.ratio_map = {
            "1:1": "2048x2048", "3:4": "1536x2048", "4:3": "2048x1536",
            "2:3": "1360x2048", "3:2": "2048x1360", "9:16": "1152x2048",
            "16:9": "2048x1152"
        }
        self.valid_sizes = list(self.ratio_map.values())

        self.model_i2i = "Qwen-Image-Edit-2511"
        self.qwen_guidance_scale = 4.0 

        self.api_key = ""
        raw_key = config.get("api_key")
        if isinstance(raw_key, list) and raw_key:
            self.api_key = str(raw_key[0])
        elif isinstance(raw_key, str) and raw_key:
            self.api_key = raw_key
        if not self.api_key:
            logger.error(f"[{PLUGIN_NAME}] 未配置 API Key")

        self.retention_hours = float(config.get("retention_hours", 1.0))
        self.last_cleanup_time = 0
        self.auto_recall = int(config.get("auto_recall", 0))

    def _get_bot(self, event: AstrMessageEvent):
        if hasattr(event, "bot") and event.bot: return event.bot
        try: return self.context.get_bot()
        except: return None

    def _cleanup_temp_files(self):
        if self.retention_hours <= 0: return
        interval = max(300, min(3600, int(self.retention_hours * 1800)))
        now = time.time()
        if now - self.last_cleanup_time < interval: return

        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        if not save_dir.exists(): return
        retention_seconds = self.retention_hours * 3600
        try:
            for filename in os.listdir(save_dir):
                file_path = save_dir / filename
                if file_path.is_file() and (now - file_path.stat().st_mtime > retention_seconds):
                    try: os.remove(file_path)
                    except: pass
            self.last_cleanup_time = now
        except: pass

    async def _download_bytes(self, url: str) -> tuple[bytes, str]:
        headers = {"User-Agent": "Mozilla/5.0"}
        async with aiohttp.ClientSession() as session:
            if url.startswith("data:image"):
                try:
                    header, encoded = url.split(",", 1)
                    mime = header.split(":")[1].split(";")[0]
                    return base64.b64decode(encoded), mime
                except Exception:
                    raise Exception("Base64图片解析失败")
            else:
                try:
                    dl_timeout = max(30, self.timeout // 2)
                    async with session.get(url, headers=headers, timeout=dl_timeout) as resp:
                        if resp.status != 200: 
                            raise Exception(f"下载失败 HTTP {resp.status}")
                        data = await resp.read()
                        if len(data) < 100:
                            raise Exception("下载的图片数据过小，可能无效")
                        mime = resp.headers.get("Content-Type", "image/jpeg")
                        return data, mime
                except asyncio.TimeoutError:
                    raise Exception("下载图片超时")
                except Exception as e:
                    raise Exception(f"图片下载异常: {str(e)}")

    async def _save_image(self, data: bytes) -> str:
        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "images"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{int(time.time())}_{uuid.uuid4().hex}.jpg"
        with open(path, "wb") as f: f.write(data)
        return str(path)

    async def _run_t2i(self, prompt: str, size: str):
        target_size = size if size in self.valid_sizes else self.default_size
        url = f"{self.base_url}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_t2i,
            "prompt": prompt,
            "size": target_size,
            "num_inference_steps": self.steps
        }
        
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload, timeout=self.timeout) as resp:
                        if resp.status in [502, 503, 504]:
                            logger.warning(f"[{PLUGIN_NAME}] 文生图服务端 {resp.status}，重试 ({attempt+1}/3)...")
                            await asyncio.sleep(2)
                            continue
                        
                        if resp.status != 200:
                            raise Exception(f"API错误 {resp.status}: {await resp.text()}")
                        
                        resp_json = await resp.json()
                        if "data" in resp_json and resp_json["data"]:
                            item = resp_json["data"][0]
                            if "url" in item: return item["url"]
                            if "b64_json" in item: return f"data:image/jpeg;base64,{item['b64_json']}"
                        raise Exception("API未返回有效图片数据")
            except Exception as e:
                if attempt == 2: raise e
                await asyncio.sleep(2)

    async def _run_i2i(self, prompt: str, img_urls: list):
        images_data = []
        logger.info(f"[{PLUGIN_NAME}] 正在下载 {len(img_urls)} 张图片...")
        for i, url in enumerate(img_urls):
            try:
                data, mime = await self._download_bytes(url)
                images_data.append((data, mime))
            except Exception as e:
                logger.error(f"[{PLUGIN_NAME}] 图片 {i+1} 下载失败: {e}")
                raise Exception(f"第 {i+1} 张图片下载失败，请重试")

        submit_url = f"{self.base_url}/async/images/edits"
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Failover-Enabled": "true"}
        
        task_id = None
        for attempt in range(3):
            try:
                form = aiohttp.FormData()
                form.add_field("prompt", prompt)
                form.add_field("model", self.model_i2i)
                form.add_field("num_inference_steps", str(self.steps))
                form.add_field("guidance_scale", str(self.qwen_guidance_scale)) 
                
                if len(images_data) == 1:
                    form.add_field("task_types", "style")
                elif len(images_data) == 2:
                    form.add_field("task_types", "id")
                    form.add_field("task_types", "style")
                else:
                    for _ in images_data: form.add_field("task_types", "style")

                for idx, (b_data, mime) in enumerate(images_data):
                    ext = mime.split("/")[-1] if "/" in mime else "jpg"
                    form.add_field("image", b_data, filename=f"input_{idx}.{ext}", content_type=mime)

                async with aiohttp.ClientSession() as session:
                    async with session.post(submit_url, headers=headers, data=form, timeout=60) as resp:
                        if resp.status in [502, 503, 504]:
                            logger.warning(f"[{PLUGIN_NAME}] 图生图提交 {resp.status}，重试 ({attempt+1}/3)...")
                            await asyncio.sleep(2)
                            continue
                            
                        if resp.status != 200:
                            err_text = await resp.text()
                            if "unavailable" in err_text:
                                await asyncio.sleep(2)
                                continue
                            raise Exception(f"提交失败 {resp.status}: {err_text}")
                        
                        task_id = (await resp.json()).get("task_id")
                        break 
            except Exception as e:
                if attempt == 2: raise e
                await asyncio.sleep(2)
        
        if not task_id:
            raise Exception("任务提交失败，服务端无响应")

        logger.info(f"[{PLUGIN_NAME}] 任务提交成功 ID: {task_id}，开始轮询(超时:{self.timeout}s)...")

        poll_url = f"{self.base_url}/task/{task_id}"
        poll_headers = {"Authorization": f"Bearer {self.api_key}"}
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < self.timeout: 
                await asyncio.sleep(3)
                try:
                    async with session.get(poll_url, headers=poll_headers, timeout=10) as resp:
                        if resp.status != 200: continue
                        res = await resp.json()
                        status = res.get("status")
                        if status == "success":
                            return res["output"]["file_url"]
                        if status in ["failed", "cancelled"]:
                            err = res.get('error', '未知错误')
                            if "unavailable" in str(err):
                                raise Exception("服务端繁忙 (502)，请稍后再试")
                            raise Exception(f"任务失败: {err}")
                except Exception as e:
                    if "任务失败" in str(e): raise e
                    pass
                    
        raise Exception(f"任务处理超时 (>{self.timeout}秒)，请稍后重试或在配置中调大超时时间")

    async def _extract_images(self, event: AstrMessageEvent):
        img_urls = []
        for comp in event.message_obj.message:
            if isinstance(comp, Image): img_urls.append(comp.url)
        
        for comp in event.message_obj.message:
            if isinstance(comp, At):
                target_id = getattr(comp, 'qq', None) or getattr(comp, 'id', None) or getattr(comp, 'user_id', None)
                if target_id:
                    img_urls.append(f"https://q1.qlogo.cn/g?b=qq&nk={target_id}&s=640")

        if img_urls: return img_urls

        reply_id = None
        for comp in event.message_obj.message:
            if isinstance(comp, Reply): reply_id = comp.id
        
        if reply_id:
            bot = self._get_bot(event)
            if bot:
                try:
                    resp = await bot.api.call_action("get_msg", message_id=int(reply_id))
                    if resp and "message" in resp:
                        content = resp["message"]
                        if isinstance(content, list):
                            for seg in content:
                                if isinstance(seg, dict) and seg.get("type") == "image":
                                    u = seg.get("data", {}).get("url") or seg.get("data", {}).get("file")
                                    if u and str(u).startswith("http"): img_urls.append(u)
                        elif isinstance(content, str):
                            img_urls.extend(re.findall(r'url=(http[^,\]]+)', content))
                            if not img_urls:
                                img_urls.extend(re.findall(r'file=(http[^,\]]+)', content))
                except Exception: pass
        
        return img_urls

    @filter.command("zimg")
    async def cmd_zimg(self, event: AstrMessageEvent, prompt: str = ""): 
        """
        /zimg <提示词> (比例) [图片/@用户]-> 文/图生图
        """
        self._cleanup_temp_files()
        
        # 1. 解析纯文本提示词
        full_text = ""
        for comp in event.message_obj.message:
            if isinstance(comp, Plain): full_text += comp.text
        
        # === 调试日志 1: 原始输入 ===
        logger.info(f"[{PLUGIN_NAME}] [DEBUG] 收到指令。原始文本: '{full_text}'")

        if "/zimg" in full_text:
            real_prompt = full_text.split("/zimg", 1)[1].strip()
        else:
            real_prompt = prompt.strip()
            
        # === 调试日志 2: 清洗后的提示词 ===
        logger.info(f"[{PLUGIN_NAME}] [DEBUG] 移除指令后，待匹配提示词: '{real_prompt}'")

        if not real_prompt:
            yield event.plain_result("请提供提示词。")
            return

        # 2. 尝试提取图片
        try:
            img_urls = await self._extract_images(event)
            
            # 2.1 提取比例
            target_size = None
            detected_ratio = None
            
            ratio_match = re.search(r"(\d+[:：]\d+)", real_prompt)
            if ratio_match:
                raw_ratio = ratio_match.group(1).replace("：", ":")
                if raw_ratio in self.ratio_map:
                    target_size = self.ratio_map[raw_ratio]
                    detected_ratio = raw_ratio
                    real_prompt = real_prompt.replace(ratio_match.group(1), " ").strip()
            
            # === 2.2 接入预设中心 (修复版：忽略大小写 + 调试日志) ===
            preset_name = None
            has_extra = False
            
            preset_hub = getattr(self.context, "preset_hub", None)
            matched = False

            if preset_hub and hasattr(preset_hub, "get_all_keys"):
                all_keys = preset_hub.get_all_keys()
                # === 调试日志 3: 预设列表 ===
                logger.info(f"[{PLUGIN_NAME}] [DEBUG] PresetHub已加载。当前可用预设Keys: {all_keys}")
                
                all_keys.sort(key=len, reverse=True)
                
                prompt_lower = real_prompt.lower()
                
                for key in all_keys:
                    key_lower = key.lower()
                    if prompt_lower == key_lower or prompt_lower.startswith(key_lower + " "):
                        resolved_content = preset_hub.resolve_preset(key)
                        if resolved_content:
                            preset_name = key
                            extra = real_prompt[len(key):].strip()
                            
                            if extra:
                                real_prompt = f"{resolved_content}, {extra}"
                                has_extra = True
                            else:
                                real_prompt = resolved_content
                            
                            matched = True
                            logger.info(f"[{PLUGIN_NAME}] [DEBUG] ✅ 成功命中预设: [{key}] -> 内容: {resolved_content[:20]}...")
                            break
            else:
                 logger.warning(f"[{PLUGIN_NAME}] [DEBUG] ❌ 未找到 PresetHub 实例，无法加载预设！")

            if not matched:
                logger.info(f"[{PLUGIN_NAME}] [DEBUG] ❌ 未命中任何预设，将 '{real_prompt}' 作为普通提示词。")
            # ==============================================

            # === 头像自动抓取 ===
            if preset_name and not img_urls:
                user_id = event.get_sender_id()
                if user_id:
                    logger.info(f"[{PLUGIN_NAME}] 触发预设 [{preset_name}] 且无图，自动使用用户头像 I2I")
                    avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
                    img_urls.append(avatar_url)

            # === 2.3 构建状态提示语 ===
            status_msg = f"🎨正在绘图"
            if preset_name:
                status_msg += f"「预设：{preset_name}」"
            if has_extra:
                status_msg += "(已衔接额外提示词)"
            
            if not img_urls and detected_ratio:
                status_msg += f" [{detected_ratio}]"
                
            status_msg += "..."

            if img_urls:
                logger.info(f"[{PLUGIN_NAME}] Qwen I2I Mode. Images: {len(img_urls)}")
                yield event.plain_result(status_msg)
                result_url = await self._run_i2i(real_prompt, img_urls)
                task_type = "图生图"
            else:
                logger.info(f"[{PLUGIN_NAME}] z-image T2I Mode.")
                yield event.plain_result(status_msg)
                result_url = await self._run_t2i(real_prompt, target_size)
                task_type = "文生图"

            # 3. 下载并发送结果
            img_data, _ = await self._download_bytes(result_url)
            local_path = await self._save_image(img_data)
            
            # 发送逻辑
            bot = self._get_bot(event)
            if not bot:
                yield event.chain_result([Image.fromFileSystem(local_path)])
                return

            segments = []
            if event.message_obj.message_id:
                segments.append({"type": "reply", "data": {"id": str(event.message_obj.message_id)}})
            
            abs_path = os.path.abspath(local_path).replace("\\", "/") if os.name == 'nt' else os.path.abspath(local_path)
            segments.append({"type": "image", "data": {"file": f"file:///{abs_path}"}})
            
            text_content = f"{task_type}成功"
            if self.auto_recall > 0: text_content += f"，{self.auto_recall}秒后撤回..."
            segments.append({"type": "text", "data": {"text": text_content}})

            payload = {"message": segments}
            msg_obj = event.message_obj
            
            if hasattr(msg_obj, "group_id") and msg_obj.group_id:
                payload["group_id"] = msg_obj.group_id
                action = "send_group_msg"
            elif hasattr(msg_obj, "sender") and hasattr(msg_obj.sender, "user_id"):
                payload["user_id"] = msg_obj.sender.user_id
                action = "send_private_msg"
            else:
                yield event.chain_result([Image.fromFileSystem(local_path)])
                return

            res = await bot.api.call_action(action, **payload)
            
            if self.auto_recall > 0 and res:
                await asyncio.sleep(self.auto_recall)
                try:
                    msg_id = res.get("message_id") or res.get("data", {}).get("message_id")
                    if msg_id: await bot.api.call_action("delete_msg", message_id=int(msg_id))
                except: pass

        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 失败: {e}")
            yield event.plain_result(f"执行失败: {e}")
