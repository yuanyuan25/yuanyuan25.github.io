# 大模型学习笔记生成工具

Markdown 为编辑源，HTML 为生成结果。处理 `notes/llm/post-training`、`notes/llm/infra` 的当前笔记和修订记录；本仓库不保留修订前副本，后续只在 yuan-notes 维护。

## 生成与验证

```sh
cd tools/notes-reader
npm ci --ignore-scripts --no-audit --no-fund
npm run build
npm run verify
```

以 GitHub Pages 的 `/notes/llm/` 为阅读入口；本地调试通过 HTTP 服务访问相同目录路由，不要求 `file://` 兼容。公式由 KaTeX 在生成时渲染，页面包含 MathML 和可复制 LaTeX；运行时只使用本仓库中的样式、字体、搜索索引和脚本，不依赖 CDN。新增笔记时更新 `catalog.mjs` 的分类、标题、卡片摘要和待补状态配置。

`verify.mjs` 校验资源哈希、必要图片引用、当前图片、本地链接与字体，并运行 `check_math.py` 的关键公式数值检查；报告位于 `notes/llm/_reader/verification.json`。它不会运行 GPU/NPU 分布式训练。

## 可选浏览器检查

需要已有 Playwright 和浏览器。`PLAYWRIGHT_MODULE` 指向其 `index.mjs`，`CHROME_EXECUTABLE` 可指定本机 Chrome 路径；不设置后者则使用 Playwright 默认浏览器。

```sh
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs \
CHROME_EXECUTABLE=/path/to/chrome \
node verify-browser.mjs
```

该检查使用临时本地 HTTP 服务，阻止外部请求，验证桌面与手机宽度、图片、公式、表格及主要交互；报告位于 `_reader/browser-verification.json`，截图写到系统 `/tmp`。

将上述命令中的 `verify-browser.mjs` 换成 `verify-site.mjs`，可检查网站首页、学习笔记栏目与文章之间的跳转，并比对品牌、主导航、页脚、字体、配色与页头尺寸是否与主站一致。脚本自动启动并关闭仅监听本机的临时 HTTP 服务，检查桌面和手机页面；报告写到 `_reader/site-verification.json`。

## 资源维护约定

- `asset-manifest.json` 仅记录 97 个原始资源的大小、SHA256 和各笔记的必要图片引用，不含旧稿正文；构建不得重新覆盖该校验基线。
- 原图片原样复制；图内错误在 Markdown 中说明，不直接修改图片来掩盖差异。
- 页面仅链接当前 Markdown，不提供旧稿入口。未来重画图片请使用新文件名，并更新正文说明。
- KaTeX 的许可证随本地资源保存在 `_reader/katex/LICENSE`。

## 路径与站点集成

- `paths.mjs` 统一定义站点、笔记与资源根目录。
- `asset-manifest.json` 直接使用本仓库的当前路径，校验不读取 MyKnowledgeBase 或任何备份目录。
- 当前笔记的图片引用均检查文件是否存在；清单中原有的图片引用还检查是否保留、资源字节是否改变。
- 页面属于「学习笔记 → 大模型」，`notes/index.html` 提供领域入口，主导航高亮「学习笔记」。后训练和 Infra 是大模型下的一级分类，各有二级主题。
- `site-layout.mjs` 在构建时直接复用站点首页的 `site-header` 和 `site-footer`，将链接转换为相对于生成页面的目录地址，并设置当前栏目。线上栏目地址不带 `index.html`。首页修改品牌或导航后重新运行构建即可同步，生成页不另写一份页头。
- 所有生成页加载公共 `assets/site.css`，继承主站配色、字体、品牌和导航；`reader.css` 只补充分类侧栏、文章目录、公式与搜索等阅读布局。不要在阅读样式中重新定义主站色板或品牌。
- 栏目页与具体文章共用主站的 1120px 内容宽度，左侧分类位置保持一致。1200px 及以上视口采用「分类 / 正文 / 本文目录」三栏；右侧目录随阅读定位并高亮当前章节。较窄视口改为标题下方的折叠目录，手机将分类收为抽屉。阅读工具条固定在滚动窗口顶部，锚点偏移随实际高度计算。
- 独立成行的公式整体居中，长公式保留横向滚动，确保两端可见；行内公式保持正常文字流。
- GitHub Pages 所需的 `.nojekyll` 位于仓库根目录；阅读器静态资源包含本地 KaTeX，不依赖 CDN。
