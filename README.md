# rag_ops

这是一个面向 RAG（Retrieval-Augmented Generation，检索增强生成）应用的管理平台。
项目包含知识库、文件上传、文档Chunk、分块策略Embedding、混合检索Retrieval、重排Rerank、模型配置、任务监控、用户权限等。
支持自定义分片策略、模型以及多种外部存储集成。
！！！如果这个项目对你有所帮助，帮忙点个star吧 ！！！

项目作者与版权署名：`zhoujian`

GitHub 仓库：`https://github.com/qqzy123/rag_ops`

许可证：Apache License 2.0，详见 `LICENSE` 和 `NOTICE`。

## 项目目标

`rag_ops` 实现了一套可本地部署、可扩展的 RAG 管理基础系统，用于管理从原始文件到可检索知识资产的完整流程：

- 上传和管理原始文档。
- 将文档解析为统一的 `Document` 数据。
- 通过自定义可配置策略进行`Chunk`文本分块。
- 对分块结果写入`ElasticSearch`关键词索引、`Milvus`向量库和`Neo4j`图数据库。
- 自定义配置 Embedding、LLM、Rerank 等模型服务。
- 通过关键词`BM25`、向量`Vector`、混合`Hybrid`、图谱`Graph`等方式组合检索。
- 通过自定义召回，支持`Retrieval` + `Rerank`重排比对。
- 管理后台用户、角色、权限和异步处理任务。

## 核心功能

### 知识库与文件管理

- 创建和管理知识库。
- 上传、查看和删除文件,原始文件存储到 MinIO。
  
<img width="2527" height="1001" alt="image" src="https://github.com/user-attachments/assets/5235b83a-54c5-43bf-840e-9cc029a30934" />


### 文档解析与分块

- 支持 TXT、Markdown、CSV、JSON、PDF、DOC、DOCX 等文档类型。
 
<img width="2538" height="1312" alt="image" src="https://github.com/user-attachments/assets/f7c3b91c-41bf-4b9f-8d1e-4cb7573d4fd6" />

- 支持多种分块模式和策略配置，支持 chunk size、chunk overlap、token 计数、标题路径、页码、metadata 等分块相关配置。
  
<img width="2533" height="1306" alt="image" src="https://github.com/user-attachments/assets/6001bbcb-eeb0-4720-9288-c6c22ab02270" />

- 支持Elasticsearch、Milvus、Neo4j（可选）多种入库方式。

<img width="2540" height="1298" alt="image" src="https://github.com/user-attachments/assets/e7bda9f0-a21b-4853-b121-ab16cef1f188" />

- 分块结果可预览。

<img width="2547" height="1296" alt="image" src="https://github.com/user-attachments/assets/5f988648-13d8-4dff-862c-35f6d0392d9b" />


### 检索与索引

- 支持实现 关键词/BM25、向量/Vector、混合/Hybrid、关系图/Grap 多种检索召回。

<img width="2513" height="1307" alt="image" src="https://github.com/user-attachments/assets/fc8780fb-59c7-46d1-aac2-21193d48f664" />

- 支持召回结果重排比对。

<img width="2509" height="1289" alt="image" src="https://github.com/user-attachments/assets/0e5d14bb-5c77-4e47-b22d-390c16af972c" />


### 分片策略

- 支持自定义分片策略配置。

<img width="2519" height="941" alt="image" src="https://github.com/user-attachments/assets/05e25532-edc7-4d3c-95c8-7206bf4f732a" />


### 模型配置

- 支持配置 embedding、LLM、rerank 等模型。

<img width="2534" height="952" alt="image" src="https://github.com/user-attachments/assets/d99fd5cc-5061-4efe-8ea7-464f1755cd90" />

### 任务

- 支持分片任务查看。



## 技术栈

### 后端

- Python `>=3.14`
- FastAPI
- SQLAlchemy
- Pydantic Settings
- PyJWT
- LangChain document loaders / splitters
- MinIO Python SDK
- Elasticsearch Python client
- PyMilvus
- Neo4j Python driver
- Uvicorn

### 前端

- Vue 3 + Vite + TypeScript。
- Pinia 状态管理。
- Vue Router 路由。
- Element Plus UI 组件。
- G6 图可视化组件，用于 Neo4j 图谱预览。

### 本地服务

`docker-compose.yml` 提供以下开发依赖：

- MySQL 8
- MinIO
- Elasticsearch 8.17
- etcd
- Milvus standalone
- Attu
- Neo4j Community

## 仓库结构

```text
.
├── backend/                 # FastAPI 后端
│   ├── app/                 # 应用源码
│   │   ├── api/             # API 路由
│   │   ├── core/            # 配置、鉴权、响应、错误处理
│   │   ├── db/              # 数据库连接
│   │   ├── models/          # SQLAlchemy 模型
│   │   ├── schemas/         # Pydantic DTO
│   │   ├── services/        # 文档处理、检索、存储、模型服务
│   │   └── tasks/           # 后台任务消费与调度
│   ├── tests/               # 后端测试
│   ├── .env.example         # 后端环境变量模板
│   ├── pyproject.toml       # Python 项目配置
│   ├── requirements.txt     # Python 依赖
│   ├── schema.sql           # 数据库 schema
│   └── uv.lock              # uv 锁文件
├── frontend/                # Vue/Vite 前端
│   ├── src/                 # 前端源码
│   ├── package.json         # 前端依赖与脚本
│   ├── package-lock.json    # npm 锁文件
│   ├── vite.config.ts       # Vite 配置
│   └── nginx.conf           # 静态部署 nginx 示例配置
├── docker-compose.yml       # 本地开发服务编排
├── compose.env.example      # docker compose 环境变量模板
├── LICENSE                  # Apache-2.0 许可证全文
├── NOTICE                   # 作者与版权通知
├── SECURITY.md              # 安全报告说明
├── CONTRIBUTING.md          # 贡献说明
└── CITATION.cff             # 引用元数据
```

## 快速开始

### 1. 克隆项目

```powershell
git clone https://github.com/qqzy123/rag_ops.git
cd rag_ops
```

### 2. 启动本地依赖服务

复制 compose 环境变量模板：

```powershell
copy compose.env.example .env
```

编辑根目录 `.env`，至少修改以下值：

```env
MYSQL_ROOT_PASSWORD=change-this-mysql-root-password
MINIO_ROOT_USER=change-this-minio-user
MINIO_ROOT_PASSWORD=change-this-minio-password
NEO4J_AUTH=neo4j/change-this-neo4j-password
```

启动本地服务：

```powershell
docker compose --env-file .env up -d
```

### 3. 配置后端

```powershell
cd backend
copy .env.example .env
```

编辑 `backend/.env`，至少确认：

```env
SECRET_KEY=change-me-at-least-32-bytes-value
API_KEY_ENCRYPTION_SECRET=change-me-at-least-32-bytes-value
DATABASE_URL=mysql+pymysql://rag:change-me@mysql:3306/rag_manage?charset=utf8mb4
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=rag_minio
MINIO_SECRET_KEY=change-me
ELASTICSEARCH_URL=http://elasticsearch:9200
MILVUS_HOST=milvus
NEO4J_URI=bolt://neo4j:7687
```

安装依赖并启动后端：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```


### 4. 启动前端

```powershell
cd frontend
npm install
npm run dev
```

Vite 开发服务器默认运行在：

```text
http://localhost:5173
```


## API 模块

后端主要 API 模块位于 `backend/app/api/v1`：

- `/auth`：登录、登出、token 相关接口。
- `/users`：用户管理、密码修改、密码重置。
- `/knowledge-bases`：知识库管理。
- `/files`：文件上传、文件列表、文件删除、文件预览。
- `/chunking`：文档加载、分块预览、分块执行。
- `/chunk-strategies`：分块策略管理。
- `/models`：模型配置管理。
- `/retrieval`：检索接口。
- `/rerank`：重排接口。
- `/tasks`：后台任务状态查询。


## 数据与存储流向

典型文件处理流程：

```text
用户上传文件
  -> 后端鉴权与大小校验
  -> 原始文件写入 MinIO
  -> 文件记录写入 MySQL
  -> 文档解析为 Document
  -> 按策略分块
  -> chunk 元数据写入 MySQL
  -> 关键词索引写入 Elasticsearch
  -> 向量写入 Milvus
  -> 实体关系写入 Neo4j
  -> 前端查看任务、预览和检索结果
```

删除文件或知识库时，需要同步清理 MySQL、MinIO、Elasticsearch、Milvus 和 Neo4j 中对应范围的数据。

## 安全说明

- 不要提交 `.env`、真实 API Key、数据库密码、JWT 密钥或对象存储密钥。
- 模型 API Key 在后端加密保存，前端不回显明文。
- `backend/.env.example` 和 `compose.env.example` 只包含可公开占位值。
- 安全问题请按 `SECURITY.md` 私下报告，不要在公开 issue 中披露漏洞细节。
- 公开部署前请启用 HTTPS、强密码、最小权限账号、网络访问控制和日志审计。

## 开源与作者权益

本项目使用 Apache License 2.0：

- 允许使用、复制、修改、分发、商业使用。
- 要求保留版权声明、许可证声明和 NOTICE。
- 包含专利授权条款。
- 不强制修改版继续开源，也不禁止闭源商用分发。

项目中的源码头部使用 SPDX 标识：

```text
SPDX-FileCopyrightText: 2026 zhoujian
SPDX-License-Identifier: Apache-2.0
```

`zhoujian` 是作者和版权署名；`qqzy123` 是 GitHub 仓库账号。二者可以同时存在，并不冲突。

如果你的目标是更强的开源传染性，尤其是不希望别人将网络服务修改版闭源商用，应在发布前评估 `AGPL-3.0-only` 或双许可证策略。当前 Apache-2.0 更适合传播、学习、二次开发和企业采用。

## 贡献说明

当前项目主要由 `zhoujian` 维护。外部贡献说明见 `CONTRIBUTING.md`。

提交贡献前请确认：

- 你有权提交相关代码或文档。
- 贡献内容可以按 Apache-2.0 授权。
- 不包含密钥、私有数据或授权不兼容的第三方代码。

## 引用

如果你在论文、文章、课程、项目或产品中引用本项目，可以使用 `CITATION.cff` 中的元数据。

## 许可证

Copyright 2026 zhoujian

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE` for details.
