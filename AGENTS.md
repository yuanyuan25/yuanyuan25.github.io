# 知行手记维护约定

- 本仓库是后训练与 Infra 笔记的主要维护位置；编辑 `notes/llm/post-training/`、`notes/llm/infra/` 中的 Markdown，再生成同名 HTML。
- `notes/llm/archive/` 是修订前原稿，连同原图和 `_reader/original-manifest.json` 保持不变。原图有错误时在当前正文解释；重画图片使用新文件名。
- 多张公式图可能构成连续推导。文字化时保留中间步骤、假设、对象与符号定义，并解释每一步为什么这样变换。正文不要堆叠文件名或新旧版本映射；必要的配图错误说明可折叠展示。公式能渲染不代表推导完整或容易理解。
- 阅读分类由 `tools/notes-reader/catalog.mjs` 维护。一级为后训练、Infra，二级为主题；无需按每个二级主题继续移动文件。
- 大模型页面属于主站「学习笔记」栏目，复用 `assets/site.css`；生成器通过 `site-layout.mjs` 读取首页的公共页头与页脚。不要另设品牌或配色。修改首页导航后重新构建笔记，并运行站点浏览器检查。
- 展示与导航以 GitHub Pages 为准，栏目使用目录地址，不在公开导航中带 `index.html`，不增加 `file://` 兼容逻辑。本地检查使用 HTTP 服务，发布后检查线上效果。
- 修改后运行 `npm --prefix tools/notes-reader run build` 与 `npm --prefix tools/notes-reader run verify`。影响导航、图片、公式排版或路径时，也运行 README 中的浏览器检查。
- `llm_reports/` 保留已有报告网址。首页与栏目入口应能双向到达新笔记；提交生成的 HTML 和本地阅读资源，但不要提交 `node_modules`。
- `.nojekyll` 用于直接提供静态文件，包括 `_reader/` 资源。构建不需要发布；仅在用户要求时提交、推送或发布。

- 两组笔记是完整独立副本，不使用指向 MyKnowledgeBase 的软链接、跳转、资源路径或生成脚本依赖，也不自动回写原目录。
