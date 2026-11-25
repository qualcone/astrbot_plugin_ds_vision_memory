# astrbot_plugin_ds_vision_memory

极简版 AstrBot 识图与角色记忆插件 —— 直接调用 **DeepSeek 原生识图 + 内置联网搜索 API（`deepseek-flash` + `deepseek_web_search`）**，不再依赖 Google Cloud Vision、Gemini 或自建网页抓取。

## 为什么重写

旧版 [astrbot_plugin_vision_memory](https://github.com/qualcone/astrbot_plugin_vision_memory) 串联了 Google 以图搜图、Gemini URL Context、多厂商联网模型、网页抓取与合并逻辑，配置复杂且维护成本高。

现在 DeepSeek 官方 API 已原生支持：
- **`deepseek-flash`**：同时支持图片输入（JPEG / PNG / GIF / WebP）与服务端联网/搜图工具 **`deepseek_web_search`**；
- **服务端搜索免单次费**：自 2026-11-18 起，模型触发的 `deepseek_web_search` 与网页抓取不再单独计费，仅按 token 计费；
- **两档模式切换**：关闭思考时为快速联网/搜图；开启思考时为**深度思考搜索（Deep Thinking Search）**，自动多轮搜图、打开网页交叉核实。

## 工作流程

```
用户发送图片
  ├─ 跳过 GIF 动图；表情包/小图默认“问了才识别”
  ├─ 命中本地缓存（SHA-256 + 64 位感知哈希）→ 直接复用
  ├─ 调用 DeepSeek API（deepseek-flash + deepseek_web_search）
  │    └─ 解析四行结论（对象 / 出处 / 设定 / 依据）+ 提取 annotations 引用链接
  └─ 交给主模型回复
       ├─ 识别成功：移除原图，完整结论作为临时上下文（不占后续历史 token），历史仅留 [识别：角色 / 作品]
       ├─ 识别失败/无法确认：保留原图，由主模型直接看图
       └─ 可选：主模型核实后调用 save_character_memory 写入 LivingMemory 长期记忆
```

## 快速配置

1. 填入 **DeepSeek API Key**（必填，其余均可保持默认）。
2. 可选：在“DeepSeek 识图与联网搜索”中开启 **深度思考搜索**（更准但更慢，超时自动不低于 150 秒）。
3. 可选：如需写入 [LivingMemory](https://github.com/lxfight-s-Astrbot-Plugins/astrbot_plugin_livingmemory)，填写 `lm_api_url` 与具备 `read` + `write` 权限的 `lm_api_key`。

## 管理员测试命令

管理员在聊天中**带图发送 `/dsvtest`**（或回复一条带图消息发送 `/dsvtest`），会直接调用 DeepSeek 识图 + 联网搜索并返回：
- 耗时与 Token 用量
- 模型原始回答
- 插件解析出的四行字段
- 联网搜索引用链接（`annotations`）

## 运行要求

- AstrBot ≥ 4.24.2
- DeepSeek API Key
- LivingMemory（可选）
