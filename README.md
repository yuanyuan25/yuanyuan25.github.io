# 知行手记 · Yuan’s Notes

记录论文阅读、学习笔记、文档总结与实践心得的个人知识库。

- **网站**：https://yuanyuan25.github.io/
- **GitHub 仓库**：https://github.com/yuanyuan25/yuanyuan25.github.io
- **本地项目目录**：`yuan-notes`

网站由原来的大模型报告项目扩展而来。`yuanyuan25.github.io` 是 GitHub 个人主站要求的仓库名称；网站品牌和本地目录使用「知行手记」及 `yuan-notes`。

## 内容结构

```text
yuan-notes/
├── index.html             # 知识库首页
├── assets/                # 公共样式与首页脚本
├── llm_reports/           # 大模型论文报告，保留原有网址
│   ├── index.html         # 报告目录
│   ├── YYYY-MM-DD.html    # 每期报告
│   ├── reports.json       # 自动生成的报告清单
│   ├── build_index.py     # 报告清单生成脚本
│   └── report_template.html
├── notes/                 # 学习笔记，按领域组织
│   ├── index.html
│   └── llm/               # 大模型笔记的主要维护位置
│       ├── index.html     # 阅读入口（自动生成）
│       ├── post-training/ # 后训练：Markdown、HTML、pic/
│       ├── infra/         # 模型与系统：Markdown、HTML、pic/
│       ├── archive/       # 修订前原稿，保持不变
│       ├── _reader/       # 本地公式、样式、搜索索引与校验清单
│       └── revision-notes.md
├── tools/notes-reader/    # Markdown 转 HTML 与校验工具
├── summaries/             # 文档总结
├── practice/              # 实践记录
└── templates/article.html # 通用文章模板
```

目前已有大模型报告，以及 16 篇大模型笔记（12 篇有正文，4 篇标为待补）。文档总结与实践记录保留栏目入口。

## 克隆与本地预览

```bash
git clone git@github.com:yuanyuan25/yuanyuan25.github.io.git yuan-notes
cd yuan-notes
python3 -m http.server 8000
```

访问 http://localhost:8000/ 。网站以 GitHub Pages 的展示与路由为准，栏目使用 `/`、`/llm_reports/`、`/notes/` 等目录地址，由服务器提供目录中的 `index.html`。本地调试使用 HTTP 服务，不提供双击 HTML 文件的 `file://` 兼容处理。

## 新增大模型报告

1. 参考 `llm_reports/report_template.html`，创建 `llm_reports/YYYY-MM-DD.html`。
2. 填写页面中的 `report-meta` JSON 元数据。
3. 在项目根目录运行 `python3 llm_reports/build_index.py`。
4. 一并提交报告和更新后的 `llm_reports/reports.json`。报告目录和首页最近三份报告会自动更新。

既有报告路径 `llm_reports/YYYY-MM-DD.html` 保持不变，原链接可继续使用。

## 维护后训练与 Infra 笔记

这两组笔记从 MyKnowledgeBase 完整复制到本仓库，包含当前 Markdown、HTML、图片、修订前原稿与生成工具。两份目录彼此独立，没有软链接、跨仓库引用或自动同步；原目录保持保留，后续内容更新主要在这里进行。

- `notes/llm/post-training/`：偏好与策略优化、预测与自蒸馏。
- `notes/llm/infra/`：模型基础、训练与并行、推理与性能；后续加入集群与调度。
- `notes/llm/archive/post-training/`、`archive/infra/`：修订前原 Markdown 与依赖，原样保留。

一级目录采用稳定的英文名称；二级主题在侧栏配置中维护，暂不继续拆分文件夹，避免笔记少时层级过深。文章标题与文件名可保持中文。图片与所属笔记一起维护于各目录的 `pic/`。

首次使用生成工具时，在仓库根目录执行：

```bash
npm --prefix tools/notes-reader ci --ignore-scripts --no-audit --no-fund
```

更新现有 Markdown 或图片后：

```bash
npm --prefix tools/notes-reader run build
npm --prefix tools/notes-reader run verify
```

新增笔记时，在对应目录创建 Markdown，然后在 `tools/notes-reader/catalog.mjs` 中添加文件路径、标题、分类与待补状态。生成器会更新 HTML、目录、搜索和统计；HTML 不手工编辑。只编辑既有笔记时不需要修改分类配置。

大模型页面是「学习笔记」的子栏目，使用网站公共 `assets/site.css`。生成器直接复用首页的品牌、主导航与页脚，并将学习笔记标为当前栏目；分类侧栏和公式阅读工具由专用样式补充。更新主站页头或导航后，重新构建笔记即可同步。卡片摘要在 `catalog.mjs` 中维护，文章正文仍完整来自 Markdown。

线上阅读入口为 [大模型笔记](https://yuanyuan25.github.io/notes/llm/)。浏览器验证方法和原稿保全规则见 [生成工具说明](tools/notes-reader/README.md)。提交时包含更新的 Markdown、HTML、图片和 `_reader/`；依赖目录不提交。

## 新增其他学习笔记、文档总结或实践记录

站点主体仍是静态 HTML。大模型笔记使用上述生成器，其他栏目可继续使用通用模板。

1. 将 `templates/article.html` 复制到对应栏目。模板按栏目下一级文件设计；更深路径需调整公共样式与导航的相对路径。
2. 替换标题、描述、日期、栏目名称、导航高亮与正文，删除模板的 `noindex` 标签。
3. 在栏目 `index.html` 添加链接；首次发布该栏目内容时更新首页相应卡片。

学习笔记按领域放入 `notes/<领域>/`，后续可与 `notes/llm/` 并列扩展。报告、总结和实践记录继续使用现有顶层栏目。

## 发布

检查本地页面后，提交并推送到 `master`。GitHub Pages 从 `master` 分支根目录自动发布。根目录 `.nojekyll` 保证 `_reader/` 等静态资源随站点提供。

```bash
git add <本次修改的文件>
git commit -m "Add learning notes"
git push origin master
```

远端仓库保留 `yuanyuan25.github.io`，主站地址仍为 https://yuanyuan25.github.io/ 。历史报告内容及其原有署名保留。
