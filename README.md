# AstrBot Gitee AI 图像生成插件 （免费额度/日限一百次）

## 介绍

- 通过指令 `/zimg` 生成图片。
- 默认1:1输出，可在提示词指定比例，例如`/zimg 一只猫3:4一只狗`，不用担心格式问题，遇到回车空格会自动修正。

## 配置

在 AstrBot 的管理面板中配置以下参数：

- `Gitee API Key (支持多Key)`: Gitee AI API Key，请在 Gitee AI 控制台申请，见下文。
- `图片分辨率`: 默认 2048x2048。
- `迭代步数`: 默认 9。
- `临时图片保留时长/小时 (0为不删除)`:默认 1。

## Gitee AI API Key获取方法：
1.访问https://ai.gitee.com/serverless-api?model=z-image-turbo

2.<img width="2241" height="1280" alt="PixPin_2025-12-05_16-56-27" src="https://github.com/user-attachments/assets/77f9a713-e7ac-4b02-8603-4afc25991841" />

3.免费额度<img width="240" height="63" alt="PixPin_2025-12-05_16-56-49" src="https://github.com/user-attachments/assets/6efde7c4-24c6-456a-8108-e78d7613f4fb" />

4.可以涩涩，警惕违规被举报

5.好用可以给个🌟

### 图像尺寸仅支持以下格式，如果不在其中会返回报错

- "1:1 (2048×2048)"
- "3:4 (1536×2048)"
- "4:3 (2048×1536)"
- "2:3 (1360×2048)"
- "3:2 (2048×1360)"
- "9:16 (1152×2048)"
- "16:9 (2048×1152)"

## 注意事项

- 请确保您的 Gitee AI 账号有足够的额度/每天一百次免费额度。

- 生成的图片会临时保存在 `data/plugin_data/astrbot_plugin_gitee_aiimg/images` 目录下。























