## LangChain介绍

**LangChain** 是一个用于开发由语言模型驱动的应用程序的框架。我们相信，最强大和不同的应用程序不仅将通过 API 调用语言模型，还将：

- 数据感知：将语言模型与其他数据源连接在一起。
- 主动性：允许语言模型与其环境进行交互。
因此，LangChain 框架的设计目标是为了实现这些类型的应用程序。

LangChain 框架提供了两个主要的价值主张：

- 组件：LangChain 为处理语言模型所需的组件提供模块化的抽象。LangChain 还为所有这些抽象提供了实现的集合。这些组件旨在易于使用，无论您是否使用 LangChain 框架的其余部分。
- 用例特定链：链可以被看作是以特定方式组装这些组件，以便最好地完成特定用例。这旨在成为一个更高级别的接口，使人们可以轻松地开始特定的用例。这些链也旨在可定制化。

## 为什么选择LangChain

LangChain是一个对开发者很重要的工具，它可以让使用LLM构建复杂应用变得更容易。它可以让用户把LLM连接到其他数据源。通过把LLM连接到其他数据源，应用可以处理更广泛的信息。这使得应用更强大和多样化。

**LangChain还提供了以下特点：**

**灵活性**：LangChain是一个高度灵活和可扩展的框架，它允许用户轻松地更换组件和定制链条，以满足不同的需求。

**速度**：LangChain的开发团队不断地提升库的速度，确保用户能够使用最新的LLM功能。

**社区**：LangChain有一个强大而活跃的社区，用户可以在那里寻求必要的帮助。

## LangChain的结构

这个框架由七个模块组成。每个模块可以让你管理和LLM交互的不同方面。

![img](images/v2-5c47844ffb415971c778b5360a573854_720w.webp)

- **LLM** LLM是LangChain的基础组件。它是一个对大型语言模型的封装，可以让用户利用模型的功能和能力。
- **链条Chains** 有时候，要解决任务，单独调用一个LLM的API是不够的。这个模块可以让你集成其他工具。例如，你可能需要从一个特定的URL获取数据，对返回的文本进行摘要，然后用生成的摘要来回答问题。这个模块可以让你把多个工具连接起来，以解决复杂的任务。
- **提示工程Prompts** 提示是任何自然语言处理应用的核心。它是用户和模型交互的方式，试图从模型那里得到一个输出。知道如何写一个有效的提示是很重要的。LangChain提供了提示模板，可以让用户格式化输入和其他工具。
- **文档加载器和工具** LangChain的文档加载器和工具模块可以帮助你连接到数据源和计算。工具模块提供了Bash和Python解释器会话等。这些适合于那些需要直接和底层系统交互或者需要用代码片段来计算一个特定的数学量或者解决一个问题，而不是一次性地计算答案的应用。
- **代理Agents** 一个代理是一个做出决定，采取行动，并观察所做的事情，并继续这个循环直到任务完成的LLM。LangChain库提供了可以根据输入沿途采取行动，而不是一个硬编码的确定性序列的代理。
- **索引Indexes** 最好的模型通常是那些和你的一些文本数据结合在一起的模型，以便添加上下文或向模型解释一些东西。这个模块可以帮助我们做到这一点。
- **存储记忆体Memory** 这个模块可以让用户在模型的调用之间创建一个持久化的状态。能够使用一个记住过去说过什么的模型会提高我们的应用。

# 阿里灵积模型服务

DashScope灵积模型服务建立在“模型即服务”（Model-as-a-Service，MaaS）的理念基础之上，围绕AI各领域模型，通过标准化的API提供包括模型推理、模型微调训练在内的多种模型服务。

![image-20240320101034196](images/image-20240320101034196.png)

通过围绕模型为中心，DashScope灵积模型服务致力于为AI应用开发者提供品类丰富、数量众多的模型选择，并通过API接口为其提供开箱即用、能力卓越、成本经济的模型服务。各领域模型的能力均可通过DashScope统一的API和SDK来实现被不同业务系统集成，AI应用开发和模型效果调优的效率将因此得以激发，助力开发者释放灵感、创造价值。



>  思考：灵积与通义千问是什么关系？



## 什么是向量检索服务

向量检索服务DashVector基于通义实验室自研的高效向量引擎Proxima内核，提供具备水平拓展能力的云原生、全托管的向量检索服务。DashVector将其强大的向量管理、向量查询等多样化能力，通过简洁易用的SDK/API接口透出，方便被上层AI应用迅速集成，从而为包括大模型生态、多模态AI搜索、分子结构分析在内的多种应用场景，提供所需的高效向量检索能力。

## 什么是 Embedding

简单来说，Embedding是一个多维向量的表示数组，通常由一系列数字组成。Embedding可以用来表示任何数据，例如文本、音频、图片、视频等等，通过Embedding我们可以编码各种类型的非结构化数据，转化为具有语义信息的多维向量，并在这些向量上进行各种操作，例如相似度计算、聚类、分类和推荐等。


## 整体流程概述

![image.png](images\p695626.png)



- **Embedding**：通过DashScope提供的通用文本向量模型，对语料库中所有标题生成对应的embedding向量。
- **构建索引服务和查询**：
  - 通过DashVector向量检索服务对生成embedding向量构建索引。
  - 将查询文本embedding向量作为输入，通过DashVector搜索相似的标题。
