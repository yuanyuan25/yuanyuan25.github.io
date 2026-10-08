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
├── notes/                 # 学习笔记
├── summaries/             # 文档总结
├── practice/              # 实践记录
└── templates/article.html # 通用文章模板
```

目前已有内容为大模型报告，另外三个栏目已建立入口，等待后续文章。

## 克隆与本地预览

```bash
git clone git@github.com:yuanyuan25/yuanyuan25.github.io.git yuan-notes
cd yuan-notes
python3 -m http.server 8000
```

访问 http://localhost:8000/ 。请通过 HTTP 预览，直接双击 HTML 文件可能无法读取报告 JSON 清单。

## 新增大模型报告

1. 参考 `llm_reports/report_template.html`，创建 `llm_reports/YYYY-MM-DD.html`。
2. 填写页面中的 `report-meta` JSON 元数据。
3. 在项目根目录运行 `python3 llm_reports/build_index.py`。
4. 一并提交报告和更新后的 `llm_reports/reports.json`。报告目录和首页最近三份报告会自动更新。

既有报告路径 `llm_reports/YYYY-MM-DD.html` 保持不变，原链接可继续使用。

## 新增学习笔记、文档总结或实践记录

站点沿用静态 HTML，无需安装前端依赖或构建工具。

1. 将 `templates/article.html` 复制到对应栏目，如 `notes/attention-basics.html`。模板按栏目下一级文件设计。
2. 替换标题、描述、日期、栏目名称、导航高亮和正文占位内容，并删除模板的 `noindex` 标签；共享样式位于 `assets/site.css`。
3. 在栏目 `index.html` 中，用文章链接替换首次发布时的空状态。可按以下形式添加条目：

   ```html
   <div class="recent-list">
     <a class="recent-item" href="attention-basics.html">
       <time datetime="2026-10-08">2026-10-08</time>
       <div><h3>注意力机制学习笔记</h3><p>从计算过程理解注意力机制。</p></div>
       <span class="arrow" aria-hidden="true">↗</span>
     </a>
   </div>
   ```

4. 栏目发布第一篇内容时，同步更新首页对应卡片中的「等待第一篇……」文字。

如果同时保留 Markdown 原稿，可与 HTML 放在同一栏目中；目前模板与栏目列表需手动维护，站点没有配置 Markdown 到文章页面的自动转换。

## 发布

检查本地页面后，提交并推送到 `master`。GitHub Pages 从 `master` 分支根目录自动发布。

```bash
git add <本次修改的文件>
git commit -m "Add learning notes"
git push origin master
```

远端仓库保留 `yuanyuan25.github.io`，主站地址仍为 https://yuanyuan25.github.io/ 。历史报告内容及其原有署名保留。
