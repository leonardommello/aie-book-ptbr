# Sumário

_[Versão em PDF](ToC.pdf)_
|                                                                                      |  |
|---------------------------------------------------------------------------------------------|-----:|
| **Prefácio**                                                                                |    ix |
| **1. Introdução à construção de aplicações de IA com modelos de fundação**                  |     1 |
| A ascensão da engenharia de IA                                                              |     2 |
| - De modelos de linguagem a modelos de linguagem de grande porte                              |     2 |
| - De modelos de linguagem de grande porte a modelos de fundação                               |     8 |
| - De modelos de fundação à engenharia de IA                                                  |    12 |
| Casos de uso de modelos de fundação                                                         |    16 |
| - Programação                                                                               |    20 |
| - Produção de imagem e vídeo                                                                |    22 |
| - Escrita                                                                                   |    22 |
| - Educação                                                                                  |    24 |
| - Bots conversacionais                                                                      |    26 |
| - Agregação de informação                                                                   |    26 |
| - Organização de dados                                                                      |    27 |
| - Automação de fluxos de trabalho                                                           |    28 |
| Planejamento de aplicações de IA                                                            |    28 |
| - Avaliação do caso de uso                                                                  |    29 |
| - Definição de expectativas                                                                 |    32 |
| - Planejamento de marcos                                                                    |    33 |
| - Manutenção                                                                                |    34 |
| A pilha da engenharia de IA                                                                 |    35 |
| - As três camadas da pilha de IA                                                            |    37 |
| - Engenharia de IA versus engenharia de ML                                                 |    39 |
| - Engenharia de IA versus engenharia full-stack                                            |    46 |
| Resumo                                                                                      |    47 |
| **2. Entendendo modelos de fundação**                                                      |    49 |
| Dados de treinamento                                                                        |    50 |
| - Modelos multilíngues                                                                      |    51 |
| - Modelos específicos de domínio                                                            |    56 |
| Modelagem                                                                                   |    58 |
| - Arquitetura do modelo                                                                     |    58 |
| - Tamanho do modelo                                                                         |    67 |
| Pós-treinamento                                                                             |    78 |
| - Ajuste fino supervisionado                                                                |    80 |
| - Ajuste fino por preferência                                                               |    83 |
| Amostragem                                                                                  |    88 |
| - Fundamentos da amostragem                                                                 |    88 |
| - Estratégias de amostragem                                                                 |    90 |
| - Computação em tempo de teste                                                              |    96 |
| - Saídas estruturadas                                                                       |    99 |
| - A natureza probabilística da IA                                                          |   105 |
| Resumo                                                                                      |   111 |
| **3. Metodologia de avaliação**                                                            |   113 |
| Desafios de avaliar modelos de fundação                                                    |   114 |
| Entendendo as métricas de modelagem de linguagem                                           |   118 |
| - Entropia                                                                                  |   119 |
| - Entropia cruzada                                                                          |   120 |
| - Bits por caractere e bits por byte                                                       |   121 |
| - Perplexidade                                                                              |   121 |
| - Interpretação e casos de uso da perplexidade                                             |   122 |
| Avaliação exata                                                                            |   125 |
| - Correção funcional                                                                        |   126 |
| - Medidas de similaridade em relação a dados de referência                                 |   127 |
| - Introdução às incorporações                                                              |   134 |
| IA como juiz                                                                               |   136 |
| - Por que IA como juiz?                                                                     |   137 |
| - Como usar IA como juiz                                                                    |   138 |
| - Limitações da IA como juiz                                                                |   141 |
| - Que modelos podem atuar como juízes?                                                     |   145 |
| Ranqueamento de modelos com avaliação comparativa                                          |   148 |
| - Desafios da avaliação comparativa                                                        |   152 |
| - O futuro da avaliação comparativa                                                        |   155 |
| Resumo                                                                                      |   156 |
| **4. Avaliar sistemas de IA**                                                              |   159 |
| Critérios de avaliação                                                                      |   160 |
| - Capacidade específica de domínio                                                          |   161 |
| - Capacidade de geração                                                                     |   163 |
| - Capacidade de seguir instruções                                                          |   172 |
| - Custo e latência                                                                          |   177 |
| Seleção de modelos                                                                         |   179 |
| - Fluxo de trabalho de seleção de modelos                                                   |   179 |
| - Construir ou comprar o modelo                                                             |   181 |
| - Navegar pelos benchmarks públicos                                                        |   191 |
| Projete seu pipeline de avaliação                                                          |   200 |
| - Passo 1. Avaliar todos os componentes de um sistema                                      |   200 |
| - Passo 2. Criar um guia de avaliação                                                      |   202 |
| - Passo 3. Definir métodos e dados de avaliação                                            |   204 |
| Resumo                                                                                      |   208 |
| **5. Engenharia de prompt**                                                                |   211 |
| Introdução ao uso de prompts                                                               |   212 |
| - Aprendizado em contexto: zero-shot e few-shot                                            |   213 |
| - Prompt de sistema e prompt de usuário                                                    |   215 |
| - Comprimento de contexto e eficiência de contexto                                         |   218 |
| Boas práticas de engenharia de prompt                                                      |   220 |
| - Escreva instruções claras e explícitas                                                   |   220 |
| - Forneça contexto suficiente                                                              |   223 |
| - Divida tarefas complexas em subtarefas mais simples                                      |   224 |
| - Dê tempo ao modelo para pensar                                                           |   227 |
| - Itere sobre seus prompts                                                                 |   229 |
| - Avalie as ferramentas de engenharia de prompt                                            |   230 |
| - Organize e versione os prompts                                                           |   233 |
| Engenharia de prompt defensiva                                                             |   235 |
| - Prompts proprietários e engenharia reversa de prompt                                     |   236 |
| - Jailbreak e injeção de prompt                                                            |   238 |
| - Extração de informação                                                                    |   243 |
| - Defesas contra ataques por prompt                                                        |   248 |
| Resumo                                                                                      |   251 |
| **6. RAG e agentes**                                                                       |   253 |
| RAG                                                                                         |   253 |
| - Arquitetura de RAG                                                                       |   256 |
| - Algoritmos de recuperação                                                                |   257 |
| - Otimização da recuperação                                                                |   268 |
| - RAG além de textos                                                                       |   273 |
| Agentes                                                                                    |   275 |
| - Visão geral de agentes                                                                   |   276 |
| - Ferramentas                                                                              |   278 |
| - Planejamento                                                                             |   281 |
| - Modos de falha e avaliação de agentes                                                    |   298 |
| Memória                                                                                    |   300 |
| Resumo                                                                                      |   305 |
| **7. Ajuste fino**                                                                         |   307 |
| Visão geral do ajuste fino                                                                 |   308 |
| Quando fazer ajuste fino                                                                 |   311 |
| - Razões para fazer ajuste fino                                                            |   311 |
| - Razões para não fazer ajuste fino                                                        |   312 |
| - Ajuste fino e RAG                                                                        |   316 |
| Gargalos de memória                                                                        |   319 |
| - Retropropagação e parâmetros treináveis                                                  |   320 |
| - Contas de memória                                                                        |   322 |
| - Representações numéricas                                                                 |   325 |
| - Quantização                                                                              |   328 |
| Técnicas de ajuste fino                                                                    |   332 |
| - Ajuste fino com eficiência de parâmetros                                                 |   333 |
| - Fusão de modelos e ajuste fino multitarefa                                               |   347 |
| - Táticas de ajuste fino                                                                   |   357 |
| Resumo                                                                                      |   361 |
| **8. Engenharia de conjuntos de dados**                                                    |   363 |
| Curadoria de dados                                                                         |   365 |
| - Qualidade dos dados                                                                      |   368 |
| - Cobertura dos dados                                                                      |   370 |
| - Quantidade de dados                                                                      |   372 |
| - Aquisição e anotação de dados                                                            |   377 |
| Aumento e síntese de dados                                                                 |   380 |
| - Por que sintetizar dados                                                                 |   381 |
| - Técnicas tradicionais de síntese de dados                                               |   383 |
| - Síntese de dados com IA                                                                  |   386 |
| - Destilação de modelos                                                                    |   395 |
| Processamento de dados                                                                     |   396 |
| - Inspecionar os dados                                                                     |   397 |
| - Remover duplicatas                                                                       |   399 |
| - Limpar e filtrar os dados                                                                |   401 |
| - Formatar os dados                                                                          |   401 |
| Resumo                                                                                      |   403 |
| **9. Otimização de inferência**                                                            |   405 |
| Entendendo a otimização de inferência                                                      |   406 |
| - Visão geral da inferência                                                                |   406 |
| - Métricas de desempenho de inferência                                                     |   412 |
| - Aceleradores de IA                                                                       |   419 |
| Otimização de inferência                                                                   |   426 |
| - Otimização do modelo                                                                     |   426 |
| - Otimização do serviço de inferência                                                      |   440 |
| Resumo                                                                                      |   447 |
| **10. Arquitetura de engenharia de IA e retorno do usuário**                               |   449 |
| Arquitetura de engenharia de IA                                                            |   449 |
| - Passo 1. Enriquecer o contexto                                                           |   450 |
| - Passo 2. Colocar salvaguardas                                                            |   451 |
| - Passo 3. Acrescentar roteador e gateway de modelos                                      |   456 |
| - Passo 4. Reduzir a latência com caches                                                  |   460 |
| - Passo 5. Acrescentar padrões de agentes                                                  |   463 |
| - Monitoramento e observabilidade                                                          |   465 |
| - Orquestração de pipelines de IA                                                          |   472 |
| Retorno do usuário                                                                         |   474 |
| - Extração do retorno conversacional                                                      |   475 |
| - Design do retorno                                                                        |   480 |
| - Limitações do retorno                                                                    |   490 |
| Resumo                                                                                      |   492 |
| **Epílogo**                                                                                |   495 |
| **Índice remissivo**                                                                       |   497 |
