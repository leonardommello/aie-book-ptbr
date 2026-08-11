# Recursos
Durante a escrita de *AI Engineering*, passei por muitos artigos, estudos de caso, posts de blog, repositórios, ferramentas etc. O livro em si tem mais de 1.200 links de referência, e venho acompanhando [mais de 1.000 repositórios de IA generativa no GitHub](https://huyenchip.com/llama-police). Este documento reúne os recursos que achei mais úteis para entender diferentes áreas.

Se houver recursos que você achou úteis e que ainda não estão aqui, sinta-se à vontade para abrir um PR.

- [Fundamentos de teoria de ML](#fundamentos-de-teoria-de-ml)
- [Capítulo 1. Planejamento de aplicações com modelos de fundação](#capítulo-1-planejamento-de-aplicações-com-modelos-de-fundação)
- [Capítulo 2. Entendendo modelos de fundação](#capítulo-2-entendendo-modelos-de-fundação)
    - [Treinamento de modelos grandes](#treinamento-de-modelos-grandes)
    - [Amostragem](#amostragem)
    - [Comprimento de contexto e eficiência de contexto](#comprimento-de-contexto-e-eficiência-de-contexto)
- [Capítulos 3 + 4. Metodologia de avaliação](#capítulos-3--4-metodologia-de-avaliação)
- [Capítulo 5. Engenharia de prompt](#capítulo-5-engenharia-de-prompt)
    - [Guias de engenharia de prompt](#guias-de-engenharia-de-prompt)
    - [Engenharia de prompt defensiva](#engenharia-de-prompt-defensiva)
- [Capítulo 6. RAG e agentes](#capítulo-6-rag-e-agentes)
    - [RAG](#rag)
    - [Agentes](#agentes)
- [Capítulo 7. Ajuste fino](#capítulo-7-ajuste-fino)
- [Capítulo 8. Engenharia de conjuntos de dados](#capítulo-8-engenharia-de-conjuntos-de-dados)
    - [Conjuntos de dados públicos](#conjuntos-de-dados-públicos)
- [Capítulo 9. Otimização de inferência](#capítulo-9-otimização-de-inferência)
- [Capítulo 10. Arquitetura de engenharia de IA e retorno do usuário](#capítulo-10-arquitetura-de-engenharia-de-ia-e-retorno-do-usuário)
- [Bônus: blogs de engenharia de organizações](#bônus-blogs-de-engenharia-de-organizações)

## Fundamentos de teoria de ML
Você não precisa de formação em ML para começar a construir com modelos de fundação, mas um entendimento aproximado de como a IA funciona por baixo do capô ajuda a evitar mau uso. Familiaridade com a teoria de ML torna você muito mais eficaz.

1. [Notas de aula] [Stanford CS 321N](https://cs231n.github.io/): um curso introdutório sobre redes neurais, favorito de longa data.
    
    - [Vídeos] Recomendo assistir às aulas 1 a 7 das [gravações em vídeo](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) do curso de 2017. Elas cobrem os fundamentos que não mudaram.
    - [Vídeos] A série [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ), de Andrej Karpathy, é mais mão na massa: ele mostra como implementar vários modelos do zero.
2. [Livro] [Machine Learning: A Probabilistic Perspective](https://probml.github.io/pml-book/book1.html) (Kevin P Murphy, 2012)
    
    Fundamental e abrangente, ainda que um pouco intenso. Era o livro de cabeceira de muitos amigos meus na preparação para entrevistas teóricas de vagas de pesquisa.
3. [Aman's Math Primers](https://aman.ai/primers/math/)
    
    Uma boa nota que cobre conceitos básicos de cálculo diferencial e de probabilidade.
4. Também montei uma lista de recursos de MLOps, que inclui uma seção sobre [ML + engineering fundamentals](https://huyenchip.com/mlops/#ml_engineering_fundamentals).
5. Escrevi uma [nota breve de 1.500 palavras](https://github.com/chiphuyen/dmls-book/blob/main/basic-ml-review.md) sobre como um modelo de ML aprende e sobre conceitos como função objetivo e procedimento de aprendizado.
6. *AI Engineering* também cobre os conceitos importantes diretamente relevantes para a discussão:
    
    - Arquitetura transformer (Capítulo 2)
    - Incorporações (Capítulo 3)
    - Retropropagação e parâmetros treináveis (Capítulo 7)

## Capítulo 1. Planejamento de aplicações com modelos de fundação

1. [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models](https://arxiv.org/abs/2303.10130) (OpenAI, 2023) 
    
    A OpenAI (2023) tem uma pesquisa excelente sobre o quanto diferentes ocupações estão expostas à IA. Eles definiram uma tarefa como exposta se a IA e softwares movidos a IA reduzem em pelo menos 50% o tempo necessário para completá-la. Uma ocupação com 80% de exposição significa que 80% das tarefas dessa ocupação são consideradas expostas. Segundo o estudo, ocupações com exposição de 100% ou próxima disso incluem intérpretes e tradutores, preparadores de declaração de imposto, web designers e escritores. Algumas delas aparecem na Figura 1-5. Sem grande surpresa, ocupações sem exposição à IA incluem cozinheiros, pedreiros e atletas. O estudo dá uma boa ideia dos casos de uso para os quais a IA é boa.
1. [Applied LLMs](https://applied-llms.org/) (Yan et al., 2024)
    
    Eugene Yan e companhia compartilham o que aprenderam em um ano implantando aplicações de LLM. Muitas dicas úteis!
1. [Musings on Building a Generative AI Product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product) (Juan Pablo Bottaro, com coautoria de Karthik Ramgopal, LinkedIn, 2024) 
    
    Um dos melhores relatos que li sobre implantar aplicações de LLM: o que funcionou e o que não funcionou. Discutem saídas estruturadas, os compromissos entre latência e vazão, os desafios da avaliação (passaram a maior parte do tempo criando diretrizes de anotação) e o desafio da última milha ao construir aplicações de IA generativa.
1. [Apple's human interface guideline](https://developer.apple.com/design/human-interface-guidelines/machine-learning), para projetar aplicações de ML
    
    Descreve como pensar o papel da IA e do humano na sua aplicação, o que influencia as decisões de interface.
1. [LocalLlama subreddit](https://www.reddit.com/r/LocalLLaMA/): vale checar de tempos em tempos para ver o que as pessoas andam fazendo.
1. [State of AI Report](https://www.stateof.ai/) (atualizado anualmente): muito abrangente. Vale passar os olhos para ver o que você perdeu.
1. [16 Changes to the Way Enterprises Are Building and Buying Generative AI](https://a16z.com/generative-ai-enterprise-2024/) (Andreessen Horowitz, 2024)
1. ["Like Having a Really Bad PA": The Gulf between User Expectation and Experience of Conversational Agents](https://dl.acm.org/doi/abs/10.1145/2858036.2858288) (Luger and Sellen, 2016)
    
    Um artigo sólido e à frente do seu tempo sobre a experiência do usuário com agentes conversacionais. Defende muito bem o valor das interfaces de diálogo e o que é preciso para torná-las úteis, com entrevistas em profundidade com 14 usuários. "*Já se argumentou que o verdadeiro valor dos sistemas de interface de diálogo sobre a manipulação direta (GUI) está onde a complexidade da tarefa é maior.*"
1. [Stanford Webinar - How AI is Changing Coding and Education, Andrew Ng & Mehran Sahami](https://www.youtube.com/watch?v=J91_npj0Nfw&ab_channel=StanfordOnline) (2024) 
    
    Uma ótima discussão, que mostra como o departamento de ciência da computação de Stanford pensa o futuro do ensino da área. Minha citação favorita: "Ciência da computação é sobre pensamento sistemático, não sobre escrever código."
1. [Professional artists: how much has AI art affected your career? - 1 year later : r/ArtistLounge](https://www.reddit.com/r/ArtistLounge/comments/1ap0cm3/professional_artists_how_much_has_ai_art_affected/) 
    
    Muita gente compartilha como a IA impactou seu trabalho. Por exemplo:

    *"De vez em quando, estou em reuniões em que gestores sonham em substituir programadores, escritores e artistas visuais por IA. Odeio essas reuniões e tento evitá-las, mas ainda acabo envolvido de vez em quando. A vida toda amei programação e arte. Mas, hoje em dia, sinto com frequência uma tristeza estranha no coração."*

## Capítulo 2. Entendendo modelos de fundação

### Treinamento de modelos grandes

Artigos que detalham o processo de treinamento de modelos importantes são minas de ouro. Recomendo ler todos. Mas, se você só puder escolher 3, recomendo Gopher, InstructGPT e Llama 3.

1. [GPT-2] [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (OpenAI, 2019) 
2. [GPT-3] [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (OpenAI, 2020) 
3. [Gopher] [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) (DeepMind, 2021) 
4. [InstructGPT] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (OpenAI, 2022)
5. [Chinchilla] [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (DeepMind, 2022)
6. [Qwen technical report](https://arxiv.org/abs/2309.16609) (Alibaba, 2022)
7. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) (Alibaba, 2024)
8. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Anthropic, 2022)
9. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Meta, 2023) 
10. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Meta, 2023)
11. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta, 2024)
    
    Este artigo é muito bom. A seção sobre geração e verificação de dados sintéticos é especialmente importante.
12. [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652) (01.AI, 2024)

**Leis de escalonamento**

1. [From bare metal to high performance training: Infrastructure scripts and best practices - imbue](https://imbue.com/research/70b-infrastructure/)
    
    Discute como escalar computação para treinar modelos grandes. Usa 4.092 GPUs H100 distribuídas por 511 computadores, 8 GPUs por computador
2. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (OpenAI, 2020)
    
    Lei de escalonamento anterior. Só até 1B de parâmetros fora das incorporações e 1B de tokens.
3. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffman et al., 2022)
    
    Conhecida como lei de escalonamento Chinchilla, esta talvez seja a publicação mais famosa sobre leis de escalonamento.
4. [Scaling Data-Constrained Language Models](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html) (Muennighoff et al., 2023) 
    
    *"Constatamos que, com dados restritos e um orçamento de computação fixo, treinar com até 4 épocas de dados repetidos gera mudanças desprezíveis na perda em comparação com dados únicos. Com mais repetição, porém, o valor de acrescentar computação acaba decaindo a zero."*
5. [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Chung et al., 2022)
    
    Um artigo muito bom, que fala da importância da diversidade dos dados de instrução.
6. [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) (Sardana et al., 2023) 
7. [AI models are devouring energy. Tools to reduce consumption are here, if data centers will adopt](https://www.ll.mit.edu/news/ai-models-are-devouring-energy-tools-reduce-consumption-are-here-if-data-centers-will-adopt) ( MIT Lincoln Laboratory, 2023)
8. [Will we run out of data? Limits of LLM scaling based on human-generated data](https://arxiv.org/abs/2211.04325) (Villalobos et al., 2022)

**Coisas divertidas**

1. [Evaluating feature steering: A case study in mitigating social biases](https://www.anthropic.com/research/evaluating-feature-steering) (Anthropic, 2024)
    
    Essa área de pesquisa é fantástica. Eles focaram em 29 [características relacionadas a vieses sociais](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#safety-relevant-bias) e descobriram que o direcionamento de características influencia vieses sociais específicos, mas também pode produzir 'efeitos fora do alvo' inesperados.
2. [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) (Anthropic, 2024)
3. [GitHub - ianand/spreadsheets-are-all-you-need](https://github.com/ianand/spreadsheets-are-all-you-need)

    *"Implementa a passagem direta do GPT2 (um ancestral do ChatGPT) inteiramente no Excel, usando funções padrão de planilha."*
4. [BertViz: Visualize Attention in NLP Models (BERT, GPT2, BART, etc.)](https://github.com/jessevig/bertviz)
    
    Uma visualização útil da atenção multicabeça em ação, desenvolvida para mostrar como o BERT funciona. 

### Amostragem

1. [A Guide to Structured Generation Using Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/) (Aidan Cooper, 2024)

    Um tutorial aprofundado e detalhado sobre gerar saídas estruturadas.
2. [Fast JSON Decoding for Local LLMs with Compressed Finite State Machine](https://lmsys.org/blog/2024-02-05-compressed-fsm/) (LMSYS, 2024)
3. [How fast can grammar-structured generation be?](https://blog.dottxt.co/how-fast-cfg.html) (Brandon T. Willard, 2024)

Também escrevi um post sobre [amostragem para geração de texto](https://huyenchip.com/2024/01/16/sampling.html) (2024).

### Comprimento de contexto e eficiência de contexto

1. [Everything About Long Context Fine-tuning](https://huggingface.co/blog/wenbopan/long-context-fine-tuning) (Wenbo Pan, 2024)
2. [Data Engineering for Scaling Language Models to 128K Context](https://arxiv.org/abs/2402.10171v1) (Yu et al., 2024)
3. [The Secret Sauce behind 100K context window in LLMs: all tricks in one place](https://blog.gopenai.com/how-to-speed-up-llms-and-use-100k-context-window-all-tricks-in-one-place-ffd40577b4c) (Galina Alperovich, 2023)
4. [Extending Context is Hard…but not Impossible](https://kaiokendev.github.io/context) (kaioken, 2023)
5. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
    
    Apresenta o RoPE, uma técnica para lidar com incorporações posicionais que permite a modelos baseados em transformer trabalhar com contextos mais longos.

## Capítulos 3 + 4. Metodologia de avaliação

1. [Challenges in evaluating AI systems](https://www.anthropic.com/news/evaluating-ai-systems) (Anthropic, 2023)
    
   Discute as limitações dos benchmarks comuns de IA para mostrar por que a avaliação é tão difícil.
2. [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) (Liang et al., Stanford 2022)
3. [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615) (Google, 2022) 
4. [Open-LLM performances are plateauing, let's make the leaderboard steep again](https://huggingface.co/spaces/open-llm-leaderboard/blog) (Hugging Face, 2024)
    
    Explicação útil sobre por que a Hugging Face escolheu certos benchmarks para o seu placar, uma referência valiosa na hora de selecionar benchmarks para o seu placar pessoal. 
5. [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023)
6. [LLM Task-Specific Evals that Do & Don't Work](https://eugeneyan.com/writing/evals/) (Eugene Yan, 2024) 
7. [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) (Hamel Hussain, 2024) 
8. [Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks](https://arxiv.org/abs/2305.10160) (Google & AI2, May 2023)
9. [alopatenko/LLMEvaluation](https://github.com/alopatenko/LLMEvaluation) (Andrei Lopatenko)
    
    Uma grande coleção de recursos de avaliação. Os [slides](https://github.com/alopatenko/LLMEvaluation/blob/main/LLMEvaluation.pdf) sobre avaliação também trazem muitas indicações.
10. [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251) (Perez et al., 2022)
    
    Um artigo divertido, que usa IA para descobrir novos comportamentos de IA. Eles usam métodos com graus variados de automação para gerar conjuntos de avaliação de 154 comportamentos diversos.
11. [Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models](https://arxiv.org/abs/2309.01219) (Zhang et al., 2023)
12. O [LLM Rankings](https://openrouter.ai/rankings) da OpenRouter mostra os principais modelos de código aberto na plataforma, ordenados por uso (volume de tokens). Isso ajuda a avaliar modelos de código aberto por popularidade. Queria que mais serviços de inferência publicassem estatísticas assim.

## Capítulo 5. Engenharia de prompt

### Guias de engenharia de prompt

1. [Anthropic's Prompt Engineering Interactive Tutorial](https://docs.google.com/spreadsheets/d/19jzLgRruG9kjUQNKtCg1ZjdD6l6weA6qRXG5zLIAhC8/edit#gid=1733615301)
    
    Prático, abrangente e divertido. Os exercícios interativos baseados em Google Sheets facilitam experimentar prompts diferentes e ver na hora o que funciona e o que não funciona. Me surpreende que outros provedores de modelos não tenham guias interativos parecidos.
2. [Brex's prompt engineering guide](https://github.com/brexhq/prompt-engineering)
    
    Traz uma lista de prompts de exemplo que a Brex usa internamente.
3. [Meta's prompt engineering guide](https://llama.meta.com/docs/how-to-guides/prompting/)
4. [Google's Gemini prompt engineering guide](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf)
5. [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) 
6. Coleções de exemplos de prompt da [OpenAI](https://platform.openai.com/examples), da [Anthropic](https://docs.anthropic.com/en/prompt-library/library) e do [Google](https://console.cloud.google.com/vertex-ai/generative/prompt-gallery).
7. [Larger language models do in-context learning differently
](https://arxiv.org/abs/2303.03846) (Wei et al., 2023)
8. [How I think about LLM prompt engineering](https://fchollet.substack.com/p/how-i-think-about-llm-prompt-engineering) (Francois Chollet, 2023) 

### Engenharia de prompt defensiva

1. [Offensive ML Playbook](https://wiki.offsecml.com/Welcome+to+the+Offensive+ML+Playbook)
    
    Tem muitos recursos sobre ML adversarial e sobre como defender seus sistemas de ML contra ataques, tanto de texto quanto de imagem
2. [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208) (OpenAI, 2024)
    
    Um bom artigo sobre como a OpenAI treinou um modelo para incorporar uma hierarquia de prompts e assim protegê-lo de jailbreak. 
3. [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) (Greshake et al., 2023) 
    
    Tem uma ótima lista de exemplos de injeção indireta de prompt no apêndice.
4. [Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks](https://arxiv.org/abs/2302.05733) (Kang et al., 2023)
5. [Scalable Extraction of Training Data from (Production) Language Models](https://arxiv.org/abs/2311.17035) (Nasr et al., 2023)
6. [How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs](https://arxiv.org/abs/2401.06373) (Zeng et al., 2024)
7. [LLM Security](https://llmsecurity.net/): uma coleção de artigos sobre segurança de LLMs.
8. Ferramentas que ajudam a automatizar sondagens de segurança incluem [PyRIT](https://github.com/Azure/PyRIT), [Garak](https://github.com/leondz/garak/), [persuasive_jailbreaker](https://github.com/CHATS-lab/persuasive_jailbreaker), [GPTFUZZER](https://arxiv.org/abs/2309.10253) e [MasterKey](https://arxiv.org/abs/2307.08715).
9. [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674) (Meta, 2023)
10. [AI Security Overview](https://owaspai.org/docs/ai_security_overview/#threat-model) (AI Exchange)

## Capítulo 6. RAG e agentes

### RAG

1. [Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051) (Chen et al., 2017)
    
    Introduz o padrão RAG para ajudar em tarefas intensivas em conhecimento, como responder perguntas.
2. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) 
3. [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) (Gao et al., 2023)
4. [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (Anthropic, 2024)
    
    Um tema importante e pouco discutido é como preparar dados para um sistema RAG. Este post apresenta várias técnicas de preparação de dados para RAG e pontos bem práticos sobre quando usar RAG e quando usar contexto longo.
5. Tutoriais de fatiamento da [Pinecone](https://www.pinecone.io/learn/chunking-strategies/) e da [Langchain](https://js.langchain.com/v0.1/docs/modules/data_connection/document_transformers/)
6. [The 5 Levels Of Text Splitting For Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4) (Greg Kamradt, 2024)
7. [GPT-4 + Streaming Data = Real-Time Generative AI](https://www.confluent.io/blog/chatgpt-and-streaming-data-for-real-time-generative-ai/) (Confluent, 2023)
    
    Um ótimo post detalhando o padrão de recuperar dados em tempo real em aplicações de RAG.
8. [Everything You Need to Know about Vector Index Basics](https://zilliz.com/learn/vector-index) (Zilliz, 2023)
    
    Uma excelente série sobre busca vetorial e bancos de dados vetoriais.
9. [A deep dive into the world's smartest email AI](https://www.shortwave.com/blog/deep-dive-into-worlds-smartest-email-ai/) (Hiranya Jayathilaka, 2023)
    
    Se você conseguir ignorar o título, o post é um estudo de caso detalhado sobre usar o padrão RAG para construir um assistente de e-mail.
10. [Livro] [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/information-retrieval-book.html) (Manning, Raghavan e Schütze, 2008)
    
    A recuperação de informação é a espinha dorsal do RAG. Este livro é para quem quer mergulhar muito, muito fundo nas diferentes técnicas de organizar e consultar dados textuais.

### Agentes

1. [[2304.09842] Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models](https://arxiv.org/abs/2304.09842) (Lu et al., 2023)
    
    Meu estudo favorito sobre planejadores baseados em LLM, como eles usam ferramentas e seus modos de falha. Uma descoberta interessante é que LLMs diferentes têm preferências de ferramenta diferentes. 
2. [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (Park et al., 2023)
3. [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (Schick et al., 2023)
4. [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) e o artigo [Gorilla: Large Language Model Connected with Massive APIs](https://arxiv.org/abs/2305.15334) (Patil et al., 2023)
    
    A lista dos 4 erros comuns de chamada de função cometidos pelo ChatGPT é interessante.
5. [THUDM/AgentBench: A Benchmark to Evaluate LLMs as Agents](https://github.com/THUDM/AgentBench)  (ICLR'24) 
6. [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (Nakano et al., 2021)
7. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
8. [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
9. [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) (Wang et al., 2023)
10. [Livro] [Artificial Intelligence: A Modern Approach](https://www.amazon.com/Artificial-Intelligence-A-Modern-Approach/dp/0134610997) (Russell e Norvig, a 4ª edição é de 2020)
    
    Planejamento tem relação estreita com busca, e este livro clássico traz vários capítulos aprofundados sobre busca.

## Capítulo 7. Ajuste fino

1. [Best practices for fine-tuning GPT-3 to classify text](https://docs.google.com/document/d/1rqj7dkuvl7Byd5KQPUJRxc19BJt8wo0yHNwK84KfU3Q/edit) (OpenAI) 
    
    Um rascunho da OpenAI. Embora o guia foque no GPT-3, muitas técnicas se aplicam ao ajuste fino completo em geral. Ele explica como funciona o ajuste fino do GPT-3, como preparar os dados de treinamento, como avaliar seu modelo e quais são os erros comuns
2. [Easily Train a Specialized LLM: PEFT, LoRA, QLoRA, LLaMA-Adapter, and More](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft) (Cameron R. Wolfe, 2023)
    
    Para o ajuste fino com eficiência de parâmetros de forma mais geral, um artigo bem pesquisado, de 7.000 palavras, sobre a evolução do ajuste fino baseado em adaptadores, por que o LoRA é tão popular e por que ele funciona
3. [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://arxiv.org/abs/2312.05934) (Ovadia et al., 2024) 
    
    Resultados interessantes para ajudar a responder à pergunta: ajuste fino ou RAG?
4. [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) (Houlsby et al., 2019)
    
    O artigo que introduz o conceito de PEFT.
5. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
    
    Leitura obrigatória.
6. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
7. [Direct Preference Optimization with Synthetic Data on Anyscale](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data) (2024)
8. [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) (kipply, 2022)
9. [Transformer Math 101](https://blog.eleuther.ai/transformer-math/) (EleutherAI, 2023): cálculo do consumo de memória, com mais foco em treinamento.
10. [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.15647) (Lialin et al., 2023)
    
    Um estudo abrangente de diferentes métodos de ajuste fino. Nem todas as técnicas, porém, são relevantes hoje.
11. [My experience on starting with fine tuning LLMs with custom data : r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/14vnfh2/my_experience_on_starting_with_fine_tuning_llms/) (2023)
12. [Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) (NVIDIA Docs)

## Capítulo 8. Engenharia de conjuntos de dados
1. [Annotation Best Practices for Building High-Quality Datasets](https://www.grammarly.com/blog/engineering/annotation-best-practices/) (Grammarly, 2022) 
2. [Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) (Chung et al., 2022) 
3. [The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/abs/2305.17493) (Shumailov et al., 2023)
4. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta, 2024)
    
    O artigo inteiro é bom, mas a seção sobre geração e verificação de dados sintéticos é especialmente importante.
5. [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277) (Peng et al., 2023)
    
    Usa o GPT-4 para gerar dados de seguimento de instruções para o ajuste fino de LLMs.
6. [Best Practices and Lessons Learned on Synthetic Data for Language Models](https://arxiv.org/abs/2404.07503) (Liu et al., DeepMind 2024)
7. [UltraChat] [Enhancing Chat Language Models by Scaling High-quality Instructional Conversations](https://arxiv.org/abs/2305.14233) (Ding et al., 2023)
8. [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) (Lee et al., 2021)
9. [Can LLMs learn from a single example?](https://www.fast.ai/posts/2023-09-04-learning-jumps/) (Jeremy Howard and Jonathan Whitaker, 2023)
    
    Experimento divertido que mostra ser possível ver melhora do modelo com apenas um exemplo de treinamento.
10. [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023)

### Conjuntos de dados públicos

Estes são alguns lugares onde procurar conjuntos de dados disponíveis publicamente. Aproveite os dados disponíveis, mas nunca confie neles por completo. Dados precisam ser inspecionados e validados a fundo.

Sempre verifique a licença de um conjunto de dados antes de usá-lo. Faça o possível para entender de onde vêm os dados. Mesmo que o conjunto tenha uma licença que permita uso comercial, é possível que parte dele venha de uma fonte que não permite.

1. [Hugging Face](https://huggingface.co/datasets) e [Kaggle](https://www.kaggle.com/datasets?fileType=csv) hospedam, cada um, centenas de milhares de conjuntos de dados.
2. O Google tem um [Dataset Search](https://datasetsearch.research.google.com/) maravilhoso e subestimado.
3. Governos costumam ser ótimos provedores de dados abertos. O [Data.gov](https://data.gov) hospeda algo em torno de centenas de milhares de conjuntos de dados, e o [data.gov.in](https://data.gov.in), dezenas de milhares. 
4. O [Institute for Social Research](https://www.icpsr.umich.edu/web/pages/ICPSR/index.html) ICPSR, da University of Michigan, tem dados de dezenas de milhares de estudos sociais.
5. O [UC Irvine's Machine Learning Repository](https://archive.ics.uci.edu/datasets) e o [OpenML](https://www.openml.org/search?type=data&sort=runs&status=active) são dois repositórios mais antigos, cada um com vários milhares de conjuntos de dados.
6. A [Open Data Network](https://www.opendatanetwork.com/) permite buscar entre dezenas de milhares de conjuntos de dados.
7. Provedores de serviços em nuvem costumam hospedar uma pequena coleção de conjuntos de dados abertos; o mais notável é o [AWS's Open Data](https://registry.opendata.aws/).
8. Frameworks de ML costumam trazer pequenos conjuntos de dados prontos, que você carrega enquanto usa o framework, como os [TensorFlow datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).
9. Algumas ferramentas de arcabouço de avaliação hospedam conjuntos de benchmark grandes o bastante para ajuste fino com PEFT. O [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) da Eleuther AI, por exemplo, hospeda mais de 400 conjuntos de benchmark, com média de mais de 2.000 exemplos cada.
10. A [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/) é um ótimo repositório de conjuntos de dados de grafos.


## Capítulo 9. Otimização de inferência

1. [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) (NVIDIA Technical Blog, 2023)
    
    Uma visão geral muito boa de diferentes técnicas de otimização.
2. [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/) (Pytorch, 2023)
    
    Um bom estudo de caso, com a melhoria de desempenho obtida por diferentes técnicas.
3. [Efficiently Scaling Transformer Inference](https://arxiv.org/pdf/2211.05102) (Pope et al., 2022)
    
    Um artigo muito técnico, mas muito bom, sobre inferência, vindo do time de Jeff Dean. Minha parte favorita é a seção que discute em que focar diante de diferentes compromissos (por exemplo, latência versus custo).
4. [Optimizing AI Inference at Character.AI](https://research.character.ai/optimizing-inference/) (Character.AI, 2024)
    
    Menos um artigo técnico e mais um artigo do tipo "olha só, eu consigo fazer isso". É bem impressionante o que o time técnico da Character.AI conseguiu. O post discute o projeto da atenção, a otimização de cache e o treinamento em int8.
5. [Vídeo] [GPU optimization workshop with OpenAI, NVIDIA, PyTorch, and Voltron Data](https://www.youtube.com/watch?v=v_q2JTIqE20&ab_channel=MLOpsLearners) 
6. [Vídeo] [Essence VC Q1 Virtual Conference: LLM Inference](https://www.youtube.com/watch?v=XPArX12gXVE) (com vLLM, TVM e Modal Labs)
7. [Techniques for KV Cache Optimization in Large Language Models](https://www.omrimallis.com/posts/techniques-for-kv-cache-optimization/) (Omri Mallis, 2024)
    
    Um excelente post explicando a otimização do KV cache, uma das partes que mais consomem memória na inferência com transformers.
    
    [João Lages](https://medium.com/@joaolages/kv-caching-explained-276520203249) tem uma excelente visualização do KV cache.

8. [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (DeepMind, 2023)
9. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) (Zhong et al., 2024) 
10. [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) (Tim Dettmers, 2023) 
    
    Stas Bekman também tem [ótimas notas](https://github.com/stas00/ml-engineering/tree/master/compute/accelerator) sobre avaliação de aceleradores. 
11. [Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads](https://www.usenix.org/system/files/atc19-jeon.pdf) (Jeon et al., 2019)
    
    Um estudo detalhado de clusters de GPU usados para treinar redes neurais profundas (DNNs) em ambiente multiusuário. Os autores analisam um rastro de dois meses de um cluster de GPU da Microsoft, com foco em três questões-chave que afetam a utilização do cluster: escalonamento em grupo e restrições de localidade, utilização das GPUs e falhas de jobs.
12. [AI Datacenter Energy Dilemma - Race for AI Datacenter Space](https://www.semianalysis.com/p/ai-datacenter-energy-dilemma-race) (SemiAnalysis, 2024)
    
    Ótima análise sobre o negócio dos data centers e seus gargalos.

Também tenho um post mais antigo, [A friendly introduction to machine learning compilers and optimizers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html) (Chip Huyen, 2018)


## Capítulo 10. Arquitetura de engenharia de IA e retorno do usuário

1. [Chapter 4: Monitoring](https://sre.google/workbook/monitoring/), do Google SRE Book
1. [Guidelines for Human-AI Interaction](https://www.microsoft.com/en-us/research/publication/guidelines-for-human-ai-interaction/) (Microsoft Research)
    
    A Microsoft propôs 18 diretrizes de design para a interação humano-IA, cobrindo decisões antes do desenvolvimento, durante o desenvolvimento, quando algo dá errado e ao longo do tempo.
1. [Peering Through Preferences: Unraveling Feedback Acquisition for Aligning Large Language Models](https://arxiv.org/abs/2308.15812v3) (Bansal et al., 2023)
    
    Um estudo sobre como o protocolo de retorno influencia o desempenho de treinamento de um modelo.
1. [Feedback-Based Self-Learning in Large-Scale Conversational AI Agents](https://arxiv.org/abs/1911.02557) (Ponnusamy et al., Amazon 2019)
1. [A scalable framework for learning from implicit user feedback to improve natural language understanding in large-scale conversational AI systems](https://arxiv.org/abs/2010.12251) (Park et al., Amazon 2020)

O design do retorno do usuário para IA conversacional é uma área pouco pesquisada, então ainda não há muitos recursos, mas espero que isso mude logo.


## Bônus: blogs de engenharia de organizações

Gosto de ler bons blogs técnicos. Estes são alguns dos blogs de engenharia que costumo acompanhar.

1. [LinkedIn Engineering Blog](https://www.linkedin.com/blog/engineering)   
2. [Engineering Blog - DoorDash](https://careersatdoordash.com/engineering-blog/) 
3. [Engineering | Uber Blog](https://www.uber.com/en-US/blog/engineering/)
4. [The Unofficial Google Data Science Blog](https://www.unofficialgoogledatascience.com/)  
5. [Pinterest Engineering Blog – Medium](https://medium.com/pinterest-engineering)
6. [Netflix TechBlog](https://netflixtechblog.com/)
7. [Blog | LMSYS Org](https://lmsys.org/blog/) 
8. [Blog | Anyscale](https://www.anyscale.com/blog)
9. [Data Science and ML | Databricks Blog](https://www.databricks.com/blog/category/engineering/data-science-machine-learning)  
10. [Together Blog](https://www.together.ai/blog) 
11. [Duolingo Engineering](https://blog.duolingo.com/hub/engineering/) 
