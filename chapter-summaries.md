# Resumos dos capítulos

Estes são os resumos de cada capítulo, extraídos do livro. Alguns deles podem não fazer muito sentido para quem ainda não leu os capítulos correspondentes, mas espero que deem uma ideia do que o livro trata.

* [Capítulo 1. Introdução à construção de aplicações de IA com modelos de fundação](#capítulo-1-introdução-à-construção-de-aplicações-de-ia-com-modelos-de-fundação)
* [Capítulo 2. Entendendo modelos de fundação](#capítulo-2-entendendo-modelos-de-fundação)
* [Capítulo 3. Metodologia de avaliação](#capítulo-3-metodologia-de-avaliação)
* [Capítulo 4. Avaliar sistemas de IA](#capítulo-4-avaliar-sistemas-de-ia)
* [Capítulo 5. Engenharia de prompt](#capítulo-5-engenharia-de-prompt)
* [Capítulo 6. RAG e agentes](#capítulo-6-rag-e-agentes)
* [Capítulo 7. Ajuste fino](#capítulo-7-ajuste-fino)
* [Capítulo 8. Engenharia de conjuntos de dados](#capítulo-8-engenharia-de-conjuntos-de-dados)
* [Capítulo 9. Otimização de inferência](#capítulo-9-otimização-de-inferência)
* [Capítulo 10. Arquitetura de engenharia de IA e retorno do usuário](#capítulo-10-arquitetura-de-engenharia-de-ia-e-retorno-do-usuário)

## Capítulo 1. Introdução à construção de aplicações de IA com modelos de fundação
Tabela 1-3. Casos de uso comuns de IA generativa em aplicações de consumo e corporativas.


| Categoria               | Exemplos de casos de uso de consumo          | Exemplos de casos de uso corporativos           |
|------------------------|-----------------------------------------|-------------------------------------------|
| Programação                | Programação                                  | Programação                                    |
| Produção de imagem e vídeo | Edição de foto e vídeo<br>Design      | Apresentações<br>Geração de anúncios             |
| Escrita               | E-mail<br>Redes sociais e posts de blog    | Redação publicitária<br>SEO<br>Relatórios, memorandos, documentos de projeto |
| Educação             | Tutoria<br>Correção de redações               | Integração de novos funcionários<br>Treinamento de requalificação |
| Bots conversacionais   | Chatbot de uso geral<br>Companhia de IA         | Suporte ao cliente<br>Copilotos de produto      |
| Agregação de informação | Sumarização<br>Conversar com seus documentos     | Sumarização<br>Pesquisa de mercado          |
| Organização de dados     | Busca de imagens<br>Memex                   | Gestão do conhecimento<br>Processamento de documentos |
| Automação de fluxos de trabalho   | Planejamento de viagens<br>Planejamento de eventos       | Extração, entrada e anotação de dados<br>Geração de leads |
<br>

Este capítulo foi pensado para cumprir dois propósitos. O primeiro é explicar o surgimento da engenharia de IA como disciplina, graças à disponibilidade dos modelos de fundação. O segundo é dar uma visão geral do processo necessário para construir aplicações sobre esses modelos. Espero que o capítulo tenha alcançado esse objetivo. Por ser um capítulo panorâmico, ele tocou apenas de leve em muitos conceitos, que serão aprofundados no restante do livro.

O capítulo discutiu a rápida evolução da IA nos últimos anos. Percorreu algumas das transformações mais notáveis, começando pela transição de modelos de linguagem para modelos de linguagem de grande porte, graças a uma abordagem de treinamento chamada autossupervisão. Em seguida, traçou como os modelos de linguagem incorporaram outras modalidades de dados até se tornarem modelos de fundação e como os modelos de fundação deram origem à engenharia de IA.

O crescimento acelerado da engenharia de IA é movido pelas muitas aplicações que as capacidades emergentes dos modelos de fundação viabilizam. Este capítulo discutiu alguns dos padrões de aplicação mais bem-sucedidos, tanto para consumidores quanto para empresas. Apesar do número impressionante de aplicações de IA já em produção, ainda estamos nos estágios iniciais da engenharia de IA, com incontáveis inovações por construir.

Antes de construir uma aplicação, uma pergunta importante e muitas vezes negligenciada é se você deveria construí-la. Este capítulo tratou dessa pergunta junto com as principais considerações para construir aplicações de IA.

Embora engenharia de IA seja um termo novo, ele evoluiu da engenharia de ML, a disciplina mais ampla envolvida na construção de aplicações com todo tipo de modelo de ML. Muitos princípios da engenharia de ML continuam válidos para a engenharia de IA. No entanto, a engenharia de IA também traz consigo novos desafios e novas soluções. A última seção do capítulo discute a pilha da engenharia de IA, incluindo como ela mudou em relação à engenharia de ML.

Um aspecto da engenharia de IA especialmente difícil de capturar por escrito é a quantidade incrível de energia coletiva, criatividade e talento de engenharia que a comunidade traz. Esse entusiasmo coletivo muitas vezes é avassalador, já que é impossível se manter em dia com as novas técnicas, descobertas e proezas de engenharia que parecem acontecer o tempo todo.

Um consolo é que, como a IA é ótima em agregar informação, ela pode nos ajudar a agregar e resumir todas essas novidades. Mas as ferramentas só ajudam até certo ponto. Quanto mais avassalador é um campo, mais importante é ter um arcabouço que ajude a navegá-lo. Este livro pretende oferecer esse arcabouço.

O restante do livro explora esse arcabouço passo a passo, começando pelo bloco fundamental da engenharia de IA: os modelos de fundação que tornam possíveis tantas aplicações incríveis.

## Capítulo 2. Entendendo modelos de fundação
<center><img src="assets/rlhf.png" width="800"><br>
<i>O fluxo geral de treinamento com pré-treinamento, SFT e RLHF. Imagem originalmente do meu <a href="https://huyenchip.com/2023/05/02/rlhf.html">post sobre RLHF</a> (maio de 2023)</i>
</center>
<br>

Este capítulo discutiu as principais decisões de projeto na construção de um modelo de fundação. Como a maioria das pessoas vai usar modelos de fundação prontos em vez de treinar um do zero, pulei os detalhes minuciosos do treinamento em favor dos fatores de modelagem que ajudam você a determinar quais modelos usar e como usá-los.

Um fator crucial que afeta o desempenho de um modelo são seus dados de treinamento. Modelos grandes exigem uma grande quantidade de dados de treinamento, que podem ser caros e demorados de obter. Por isso, os provedores de modelos costumam aproveitar quaisquer dados disponíveis. Isso leva a modelos que vão bem nas muitas tarefas presentes nos dados de treinamento, que podem não incluir a tarefa específica que você quer. Este capítulo mostrou por que muitas vezes é necessário fazer curadoria dos dados de treinamento para desenvolver modelos voltados a idiomas específicos, sobretudo os de poucos recursos, e a domínios específicos.

Depois de obter os dados, o desenvolvimento do modelo pode começar. Embora o treinamento costume dominar as manchetes, um passo importante antes dele é arquitetar o modelo. O capítulo examinou escolhas de modelagem, como a arquitetura e o tamanho do modelo. A arquitetura dominante para modelos de fundação baseados em linguagem é o transformer. Este capítulo explorou os problemas que a arquitetura transformer foi projetada para resolver, bem como suas limitações.

A escala de um modelo pode ser medida por três números-chave: o número de parâmetros, o número de tokens de treinamento e o número de FLOPs necessários para o treinamento. Dois aspectos que influenciam a quantidade de computação necessária para treinar um modelo são o tamanho do modelo e o tamanho dos dados. A lei de escalonamento ajuda a determinar o número ótimo de parâmetros e de tokens dado um orçamento de computação. Este capítulo também examinou os gargalos de escalonamento. Até agora, aumentar a escala de um modelo em geral o torna melhor. Mas por quanto tempo isso continuará valendo?

Devido à baixa qualidade dos dados de treinamento e à autossupervisão durante o pré-treinamento, o modelo resultante pode produzir saídas que não se alinham ao que os usuários querem. Isso é tratado no pós-treinamento, que consiste em dois passos: ajuste fino supervisionado e ajuste fino por preferência. A preferência humana é diversa e impossível de capturar em uma única fórmula matemática, então as soluções existentes estão longe de ser infalíveis.

Este capítulo também cobriu um dos meus temas favoritos: a amostragem, o processo pelo qual um modelo gera tokens de saída. A amostragem torna os modelos de IA probabilísticos. Essa natureza probabilística é o que faz modelos como ChatGPT e Gemini serem ótimos para tarefas criativas e divertidos de conversar. Ela também causa, porém, inconsistência e alucinações.

Trabalhar com modelos de IA exige construir seus fluxos de trabalho em torno dessa natureza probabilística. O restante deste livro explora como tornar a engenharia de IA, se não determinística, ao menos sistemática. O primeiro passo rumo a uma engenharia de IA sistemática é estabelecer um pipeline de avaliação sólido, que ajude a detectar falhas e mudanças inesperadas. A avaliação de modelos de fundação é tão crucial que dediquei dois capítulos a ela, a começar pelo próximo.

## Capítulo 3. Metodologia de avaliação
<center><img src="assets/ai-judge.png" width="600"><br>
<i>Figura 3-8. Exemplo de um juiz de IA que avalia a qualidade de uma resposta dada uma pergunta.</i>
</center>
<br>

Quanto mais fortes os modelos de IA ficam, maior o potencial de falhas catastróficas, o que torna a avaliação ainda mais importante. Ao mesmo tempo, avaliar modelos poderosos, capazes de gerar respostas abertas, é difícil. Esses desafios levam muitos times à avaliação humana. Ter humanos no ciclo para verificações de sanidade sempre ajuda e, em muitos casos, a avaliação humana é essencial. Este capítulo, porém, focou em diferentes abordagens de avaliação automática.

O capítulo começa com uma discussão sobre por que modelos de fundação são mais difíceis de avaliar do que modelos de ML tradicionais. Embora muitas técnicas novas de avaliação estejam sendo desenvolvidas, os investimentos em avaliação ainda ficam atrás dos investimentos em desenvolvimento de modelos e de aplicações.

Como muitos modelos de fundação têm um componente de modelo de linguagem, aproximamos o foco nas métricas de modelagem de linguagem, entre elas a perplexidade e a entropia cruzada. Muita gente com quem conversei acha essas métricas confusas, então incluí uma seção sobre como interpretá-las e como aproveitá-las na avaliação e no processamento de dados.

Em seguida, o capítulo deslocou o foco para as diferentes abordagens de avaliação de respostas abertas, incluindo correção funcional, escores de similaridade e IA como juiz. As duas primeiras abordagens são exatas, enquanto a avaliação por IA como juiz é subjetiva.

Diferentemente da avaliação exata, as métricas subjetivas dependem muito do juiz. Seus escores precisam ser interpretados no contexto de quais juízes estão sendo usados. Escores que pretendem medir a mesma qualidade, mas vêm de juízes de IA diferentes, podem não ser comparáveis. Juízes de IA, como toda aplicação de IA, passam por iterações, o que significa que seus julgamentos mudam. Isso os torna pouco confiáveis como benchmarks para acompanhar as mudanças de uma aplicação ao longo do tempo. Embora promissores, os juízes de IA devem ser complementados por avaliação exata, avaliação humana ou ambas.

Ao avaliar modelos, você pode avaliar cada um de forma independente e depois ordená-los pelos escores. Como alternativa, pode ordená-los usando sinais comparativos: qual dos dois modelos é melhor? A avaliação comparativa é comum nos esportes, sobretudo no xadrez, e vem ganhando espaço na avaliação de IA. Tanto a avaliação comparativa quanto o processo de alinhamento no pós-treinamento precisam de sinais de preferência, que são caros de coletar. Isso motivou o desenvolvimento dos modelos de preferência: juízes de IA especializados que preveem qual resposta os usuários preferem.

Enquanto as métricas de modelagem de linguagem e as medidas de similaridade desenhadas à mão existem há algum tempo, a IA como juiz e a avaliação comparativa só ganharam adoção com o surgimento dos modelos de fundação. Muitos times estão descobrindo como incorporá-las a seus pipelines de avaliação. Como construir um pipeline de avaliação confiável para aplicações de resposta aberta é o tema do próximo capítulo.

## Capítulo 4. Avaliar sistemas de IA
<center><img src="assets/evaluation-process.png" width="600"><br>
<i>Figura 4-5. Visão geral do fluxo de trabalho para avaliar modelos para a sua aplicação.</i>
</center>
<br>

Este é um dos temas mais difíceis, mas acredito que também um dos mais importantes, sobre os quais escrevi a respeito de IA. Não ter um pipeline de avaliação confiável é um dos maiores obstáculos à adoção de IA. A avaliação toma tempo, mas um pipeline confiável permite reduzir riscos, descobrir oportunidades de melhorar o desempenho e comparar progressos, o que poupa tempo e dor de cabeça mais adiante.

Diante do número crescente de modelos de fundação prontamente disponíveis, para a maioria de quem desenvolve aplicações o desafio já não está em desenvolver modelos, e sim em selecionar os modelos certos para a sua aplicação. Este capítulo discutiu uma lista de critérios usados com frequência para avaliar modelos para aplicações e como esses critérios são avaliados. Tratou de como avaliar tanto capacidades específicas de domínio quanto capacidades de geração, incluindo consistência factual e segurança. Muitos critérios para avaliar modelos de fundação evoluíram do processamento de linguagem natural (NLP) tradicional, entre eles fluência, coerência e fidelidade.

Para ajudar a responder se convém hospedar um modelo ou usar uma API de modelo, este capítulo delineou os prós e os contras de cada abordagem em sete eixos, entre eles privacidade dos dados, linhagem dos dados, desempenho, funcionalidade, controle e custo. Essa decisão, como toda decisão de construir ou comprar, é única para cada time e depende não só do que o time precisa, mas também do que o time quer.

Este capítulo também explorou os milhares de benchmarks públicos disponíveis. Benchmarks públicos ajudam a descartar modelos ruins, mas não ajudam a encontrar os melhores modelos para as suas aplicações. Eles também estão provavelmente contaminados, já que seus dados entram nos dados de treinamento de muitos modelos. Existem placares públicos que agregam múltiplos benchmarks para ordenar modelos, mas como esses benchmarks são escolhidos e agregados não é um processo claro. As lições aprendidas com os placares públicos ajudam na seleção de modelos, pois selecionar modelos é parecido com criar um placar particular que os ordena segundo as suas necessidades.

O capítulo termina mostrando como usar todas as técnicas e todos os critérios de avaliação discutidos no capítulo anterior e como criar um pipeline de avaliação para a sua aplicação. Não existe método de avaliação perfeito. É impossível capturar a capacidade de um sistema de alta dimensionalidade com escores de uma ou de poucas dimensões. Há muitas limitações e vieses associados à avaliação de sistemas modernos de IA. Isso, porém, não significa que não devamos avaliar. Combinar métodos e abordagens diferentes ajuda a mitigar boa parte desses desafios.

Ainda que as discussões dedicadas à avaliação terminem aqui, a avaliação voltará a aparecer diversas vezes, não só ao longo do livro como ao longo do seu processo de desenvolvimento de aplicações. O Capítulo 6 explora a avaliação de sistemas de recuperação e de sistemas agênticos, enquanto os Capítulos 7 e 9 focam no cálculo do uso de memória, da latência e dos custos de um modelo. A verificação da qualidade dos dados é tratada no Capítulo 8, e o uso do retorno do usuário para avaliar aplicações em produção, no Capítulo 10.

Com isso, vamos ao processo de adaptação de modelos propriamente dito, começando por um tema que muita gente associa à engenharia de IA: a engenharia de prompt.

## Capítulo 5. Engenharia de prompt
<center><img src="assets/prompt-anatomy.png" width="600"><br>
<i>Figura 5-1. Um exemplo simples que mostra a anatomia de um prompt.</i>
</center>
<br>

Modelos de fundação fazem muitas coisas, mas você precisa dizer exatamente o que quer. O processo de elaborar uma instrução para que o modelo faça o que você quer chama-se engenharia de prompt. Quanta elaboração é necessária depende de quão sensível o modelo é a prompts. Se uma pequena mudança provoca uma grande mudança na resposta do modelo, mais elaboração será necessária.

Você pode pensar na engenharia de prompt como comunicação entre humano e IA. Qualquer um se comunica, mas nem todos se comunicam bem. Engenharia de prompt é fácil de começar, o que leva muita gente a achar, erroneamente, que também é fácil fazê-la bem.

A primeira parte deste capítulo discute a anatomia de um prompt, por que o aprendizado em contexto funciona e as boas práticas de engenharia de prompt. Seja para se comunicar com uma IA, seja com outras pessoas, instruções claras, com exemplos e informação relevante, são essenciais. Truques simples, como pedir ao modelo que vá devagar e pense passo a passo, rendem melhorias surpreendentes. Como os humanos, os modelos de IA têm suas manias e seus vieses, que precisam ser levados em conta para uma relação produtiva com eles.

Modelos de fundação são úteis porque seguem instruções. Essa habilidade, porém, também os expõe a ataques por prompt, em que agentes mal-intencionados fazem os modelos seguirem instruções maliciosas. Este capítulo discute diferentes abordagens de ataque e possíveis defesas contra elas. Como a segurança é um jogo de gato e rato em constante evolução, nenhuma medida será infalível. Os riscos de segurança continuarão sendo um obstáculo relevante à adoção de IA em ambientes de alto risco.

Este capítulo também discute técnicas para escrever instruções melhores e conseguir que os modelos façam o que você quer. Contudo, para cumprir uma tarefa, um modelo precisa não só de instruções, mas também de contexto relevante. Como fornecer informação relevante a um modelo é o assunto do próximo capítulo.

## Capítulo 6. RAG e agentes
<center><img src="assets/rag-architecture.png" width="700"><br>
<i>Figura 6-3. Visão de alto nível de como funciona um recuperador baseado em incorporações, ou semântico.</i>
</center>
<br>

Dada a popularidade do RAG e o potencial dos agentes, os leitores iniciais disseram que este é o capítulo que mais os empolga.

Este capítulo começou pelo RAG, o padrão que surgiu primeiro entre os dois. Muitas tarefas exigem um conhecimento de fundo extenso, que costuma exceder a janela de contexto do modelo. Copilotos de código, por exemplo, podem precisar de acesso a bases de código inteiras, e assistentes de pesquisa podem precisar analisar vários livros. Desenvolvido originalmente para superar as limitações de contexto de um modelo, o RAG também permite um uso mais eficiente da informação, melhorando a qualidade das respostas e reduzindo custos. Desde os primeiros dias dos modelos de fundação, ficou claro que o padrão RAG seria imensamente valioso para uma ampla gama de aplicações, e desde então ele foi adotado rapidamente tanto em casos de uso de consumo quanto corporativos.

O RAG emprega um processo de dois passos. Primeiro recupera informação relevante de uma memória externa e depois usa essa informação para gerar respostas mais precisas. O sucesso de um sistema RAG depende da qualidade do seu recuperador. Recuperadores baseados em termos, como Elasticsearch e BM25, são muito mais leves de implementar e oferecem linhas de base fortes. Recuperadores baseados em incorporações exigem mais computação, mas têm o potencial de superar os algoritmos baseados em termos.

A recuperação baseada em incorporações é movida pela busca vetorial, que também é a espinha dorsal de muitas aplicações centrais da internet, como busca e sistemas de recomendação. Muitos algoritmos de busca vetorial desenvolvidos para essas aplicações podem ser usados em RAG.

O padrão RAG pode ser visto como um caso particular de agente, em que o recuperador é uma ferramenta que o modelo pode usar. Os dois padrões permitem que um modelo contorne sua limitação de contexto e se mantenha mais atualizado, mas o padrão agêntico faz ainda mais do que isso. Um agente é definido pelo seu ambiente e pelas ferramentas a que tem acesso. Em um agente movido a IA, a IA é o planejador que analisa a tarefa dada, considera diferentes soluções e escolhe a mais promissora. Uma tarefa complexa pode exigir muitos passos para ser resolvida, o que requer um modelo poderoso para planejar. A capacidade de planejamento de um modelo pode ser ampliada com reflexão e com um sistema de memória que o ajude a acompanhar seu progresso.

Quanto mais ferramentas você dá a um modelo, mais capacidades ele tem, o que lhe permite resolver tarefas mais difíceis. Porém, quanto mais automatizado o agente fica, mais catastróficas são suas falhas. O uso de ferramentas expõe os agentes a muitos dos riscos de segurança discutidos no Capítulo 5. Para que agentes funcionem no mundo real, é preciso implantar mecanismos de defesa rigorosos.

Tanto o RAG quanto os agentes lidam com muita informação, que frequentemente excede o comprimento máximo de contexto do modelo subjacente. Isso exige a introdução de um sistema de memória para gerenciar e usar toda a informação que um modelo tem. O capítulo terminou com uma breve discussão sobre como é esse componente.

RAG e agentes são métodos baseados em prompt, pois influenciam a qualidade do modelo apenas por meio das entradas, sem modificar o modelo em si. Embora viabilizem muitas aplicações incríveis, modificar o modelo subjacente abre ainda mais possibilidades. Como fazer isso é o tema do próximo capítulo.

## Capítulo 7. Ajuste fino
<center><img src="assets/rag-vs-finetune.png" width="700"><br>
<i>Figura 7-3. Exemplos de fluxos de desenvolvimento de aplicações. Depois da recuperação simples (como a recuperação baseada em termos), experimentar uma recuperação mais complexa (como a busca híbrida) ou o ajuste fino depende de cada aplicação e de seus modos de falha.</i>
</center>
<br>

Fora os capítulos de avaliação, o de ajuste fino foi o mais difícil de escrever. Ele toca em uma ampla gama de conceitos, antigos (aprendizado por transferência) e novos (PEFT), fundamentais (fatoração de posto baixo) e experimentais (fusão de modelos), matemáticos (cálculo de memória) e táticos (ajuste de hiperparâmetros). Organizar todos esses aspectos em uma estrutura coerente e ao mesmo tempo acessível foi difícil.

O processo de ajuste fino em si não é difícil. Muitos frameworks de ajuste fino cuidam do treinamento por você. Eles chegam a sugerir métodos comuns de ajuste fino com hiperparâmetros padrão sensatos.

O contexto ao redor do ajuste fino, porém, é complexo. Começa por saber se você deveria mesmo ajustar um modelo. Este capítulo começou com as razões para fazer ajuste fino e as razões para não fazer. Também discutiu uma pergunta que já me fizeram muitas vezes: quando fazer ajuste fino e quando fazer RAG.

Em seus primeiros dias, o ajuste fino era parecido com o pré-treinamento: ambos envolviam atualizar todos os pesos do modelo. À medida que os modelos cresceram, porém, o ajuste fino completo tornou-se impraticável para a maioria dos praticantes. Quanto mais parâmetros a atualizar durante o ajuste fino, mais memória ele consome. A maioria dos praticantes não tem acesso a recursos suficientes (hardware, tempo e dados) para fazer ajuste fino completo com modelos de fundação.

Muitas técnicas de ajuste fino foram desenvolvidas com a mesma motivação: alcançar bom desempenho com o mínimo de memória. O PEFT, por exemplo, reduz os requisitos de memória do ajuste fino ao reduzir o número de parâmetros treináveis. O treinamento quantizado, por sua vez, mitiga esse gargalo de memória reduzindo o número de bits necessários para representar cada valor.

Depois de dar uma visão geral do PEFT, o capítulo aproximou o foco no LoRA: por que ele funciona e como funciona. O LoRA tem muitas propriedades que o tornam popular entre os praticantes. Além de ser eficiente em parâmetros e em dados, ele é modular, o que torna muito mais fácil servir e combinar vários modelos LoRA.

A ideia de combinar modelos ajustados levou o capítulo à fusão de modelos, cujo objetivo é combinar vários modelos em um só, que funcione melhor do que eles separadamente. Este capítulo discutiu os muitos casos de uso da fusão de modelos, da implantação em dispositivo ao aumento de escala de modelos, e as abordagens gerais para fundi-los.

Um comentário que ouço com frequência dos praticantes é que o ajuste fino é fácil, mas conseguir dados para ele é difícil. Obter dados anotados de alta qualidade, sobretudo dados de instrução, é desafiador. O próximo capítulo mergulha nesses desafios.

## Capítulo 8. Engenharia de conjuntos de dados
<center><img src="assets/model-perf-dataset.png" width="600"><br>
<i>Figura 8-3. A curva de ganho de desempenho com diferentes tamanhos de conjunto de dados ajuda a estimar o impacto de exemplos de treinamento adicionais no desempenho do seu modelo.</i>
</center>
<br>

Ainda que o processo real de criar dados de treinamento seja incrivelmente intrincado, os princípios de criar um conjunto de dados são surpreendentemente diretos. Para construir um conjunto de dados que treine um modelo, você começa pensando nos comportamentos que quer que o modelo aprenda e então projeta um conjunto de dados que mostre esses comportamentos. Dada a importância dos dados, os times vêm criando papéis dedicados a dados, responsáveis por obter os conjuntos apropriados garantindo privacidade e conformidade.

Os dados de que você precisa dependem não só do seu caso de uso, mas também da fase de treinamento. O pré-treinamento exige dados diferentes dos exigidos pelo ajuste fino por instrução e pelo ajuste fino por preferência. No entanto, o projeto do conjunto de dados em todas as fases compartilha os mesmos três critérios centrais: qualidade, cobertura e quantidade.

Embora a quantidade de dados com que um modelo é treinado seja o que ganha as manchetes, ter dados de alta qualidade com cobertura suficiente é igualmente importante. Uma pequena quantidade de dados de alta qualidade supera uma grande quantidade de dados ruidosos. Do mesmo modo, muitos times descobriram que aumentar a diversidade dos seus conjuntos de dados é uma chave para melhorar o desempenho dos seus modelos.

Diante da dificuldade de obter dados de alta qualidade, muitos times recorreram a dados sintéticos. Gerar dados programaticamente é um objetivo antigo, mas só quando a IA passou a criar dados realistas e complexos os dados sintéticos se tornaram uma solução prática para muito mais casos de uso. Este capítulo discutiu diferentes técnicas de síntese de dados, com um mergulho na síntese de dados de instrução para ajuste fino.

Assim como os dados reais, os dados sintéticos precisam ser avaliados para garantir sua qualidade antes de treinar modelos. Avaliar dados gerados por IA é tão complicado quanto avaliar outras saídas de IA, e as pessoas tendem mais a usar dados gerados que conseguem avaliar de forma confiável.

Dados são difíceis porque muitos passos da criação de um conjunto de dados não são facilmente automatizáveis. É difícil anotar dados, mas é ainda mais difícil criar diretrizes de anotação. É difícil automatizar a geração de dados, mas é ainda mais difícil automatizar a verificação. A síntese ajuda a gerar mais dados, mas você não automatiza pensar em quais dados quer. Não dá para automatizar facilmente as diretrizes de anotação. Não dá para automatizar prestar atenção aos detalhes.

Problemas difíceis levam a soluções criativas. Algo que me chamou a atenção ao pesquisar para este capítulo foi quanta criatividade há no projeto de conjuntos de dados. Existem tantas maneiras de as pessoas construírem e avaliarem dados. Espero que a variedade de técnicas de síntese e de verificação discutidas aqui inspire você a projetar o seu conjunto de dados.

Digamos que você tenha feito a curadoria de um conjunto de dados maravilhoso, que permite treinar um modelo incrível: como servir esse modelo? O próximo capítulo discute como otimizar a inferência para latência e custo.

## Capítulo 9. Otimização de inferência
<center><img src="assets/inference-service.png" width="500"><br>
<i>Figura 9-1. Um serviço de inferência simples.</i>
</center>
<br>

A utilidade de um modelo depende muito do seu custo e da sua latência de inferência. Uma inferência mais barata torna decisões movidas a IA mais acessíveis, enquanto uma inferência mais rápida permite integrar IA a mais aplicações. Dado o enorme impacto potencial da otimização de inferência, ela atraiu um número notável de pessoas talentosas, que não param de propor abordagens inovadoras.

Antes de começar a tornar as coisas mais eficientes, é preciso entender como a eficiência é medida. Este capítulo começou com as métricas comuns de latência, vazão e utilização. Para inferência baseada em modelos de linguagem, a latência pode ser separada em **tempo até o primeiro token** (TTFT), influenciado pela fase de preenchimento prévio, e **tempo por token de saída** (TPOT), influenciado pela fase de decodificação. As métricas de vazão estão diretamente relacionadas ao custo. Existe um compromisso entre latência e vazão. Você pode reduzir o custo se aceitar mais latência, e reduzir a latência costuma implicar aumentar o custo.

A eficiência com que um modelo roda depende do hardware em que ele é executado. Por isso, este capítulo também deu uma visão geral rápida do hardware de IA e do que é preciso para otimizar modelos em diferentes aceleradores.

O capítulo seguiu então com diferentes técnicas de otimização de inferência. Dada a disponibilidade de APIs de modelos, a maioria de quem desenvolve aplicações vai usar essas APIs com suas otimizações embutidas, em vez de implementar as técnicas por conta própria. Ainda que essas técnicas não sejam relevantes para todos, acredito que entender quais técnicas são possíveis ajuda a avaliar a eficiência das APIs de modelos.

Neste capítulo, foquei na otimização no nível do modelo e no nível do serviço de inferência. A otimização no nível do modelo costuma exigir mudar o próprio modelo, o que pode alterar seus comportamentos. Já a otimização no nível do serviço de inferência normalmente mantém o modelo intacto e muda apenas como ele é servido.

As técnicas no nível do modelo incluem técnicas agnósticas de modelo, como quantização e destilação. Arquiteturas diferentes exigem suas próprias otimizações. Como um gargalo central dos modelos transformer está no mecanismo de atenção, por exemplo, muitas técnicas de otimização envolvem tornar a atenção mais eficiente, incluindo o gerenciamento do KV cache e a escrita de kernels de atenção. Um grande gargalo de um modelo de linguagem autorregressivo está no seu processo de decodificação autorregressiva e, por consequência, muitas técnicas também foram desenvolvidas para atacá-lo.

As técnicas no nível do serviço de inferência incluem várias estratégias de agrupamento em lote e de paralelismo. Há também técnicas desenvolvidas especialmente para modelos de linguagem autorregressivos, entre elas o desacoplamento entre preenchimento prévio e decodificação e o cache de prompts.

A escolha das técnicas de otimização depende das suas cargas de trabalho. O KV cache, por exemplo, é bem mais importante para cargas com contextos longos do que para cargas com contextos curtos. Já o cache de prompts é crucial para cargas que envolvem segmentos de prompt longos e sobrepostos ou conversas de múltiplos turnos. A escolha também depende dos seus requisitos de desempenho. Se latência baixa tem prioridade maior que custo, por exemplo, você pode querer escalar o paralelismo de réplicas. Mais réplicas exigem mais máquinas, mas cada máquina atende menos requisições, o que lhe permite alocar mais recursos por requisição e, assim, melhorar o tempo de resposta.

Ainda assim, entre os diversos casos de uso, as técnicas mais impactantes costumam ser a quantização (que em geral funciona bem em vários modelos), o paralelismo de tensores (que reduz a latência e permite servir modelos maiores), o paralelismo de réplicas (relativamente simples de implementar) e a otimização do mecanismo de atenção (que acelera bastante os modelos transformer).

A otimização de inferência encerra a lista de técnicas de adaptação de modelos cobertas neste livro. O próximo capítulo explora como integrar essas técnicas em um sistema coeso.

## Capítulo 10. Arquitetura de engenharia de IA e retorno do usuário
<center><img src="assets/aie-architecture.png" width="800"><br>
<i>Figura 10-10. Uma arquitetura comum de aplicação de IA generativa.</i>
</center>
<br>
Se cada capítulo anterior focou em um aspecto específico da engenharia de IA, este olhou para o processo de construir aplicações sobre modelos de fundação como um todo.

O capítulo teve duas partes. A primeira discutiu uma arquitetura comum para aplicações de IA. Embora a arquitetura exata de uma aplicação possa variar, essa arquitetura de alto nível oferece um arcabouço para entender como os diferentes componentes se encaixam. Usei a abordagem passo a passo na construção dessa arquitetura para discutir os desafios de cada etapa e as técnicas que você pode usar para enfrentá-los.

Embora seja necessário separar componentes para manter seu sistema modular e sustentável, essa separação é fluida. Há muitas maneiras pelas quais os componentes podem se sobrepor em funcionalidades. Salvaguardas, por exemplo, podem ser implementadas no serviço de inferência, no gateway de modelos ou como componente autônomo.

Cada componente adicional pode tornar seu sistema mais capaz, mais seguro ou mais rápido, mas também aumenta a complexidade do sistema e o expõe a novos modos de falha. Uma parte integral de qualquer sistema complexo é o monitoramento e a observabilidade. Observabilidade envolve entender como seu sistema falha, projetar métricas e alertas em torno das falhas e garantir que o sistema seja projetado de forma a tornar essas falhas detectáveis e rastreáveis. Embora muitas boas práticas e ferramentas de observabilidade vindas da engenharia de software e do aprendizado de máquina tradicional se apliquem a aplicações de engenharia de IA, os modelos de fundação introduzem novos modos de falha, que exigem métricas e considerações de projeto adicionais.

Ao mesmo tempo, a interface conversacional viabiliza novos tipos de retorno do usuário, que você pode aproveitar para análises, melhoria do produto e o volante de dados. A segunda parte do capítulo discutiu várias formas de retorno conversacional e como projetar sua aplicação para coletá-lo de forma eficaz.

Tradicionalmente, o design do retorno do usuário é visto como responsabilidade de produto, e não de engenharia, e por isso costuma ser negligenciado pelos engenheiros. Contudo, como o retorno do usuário é uma fonte crucial de dados para melhorar continuamente os modelos de IA, cada vez mais engenheiros de IA se envolvem no processo para garantir que recebam os dados de que precisam. Isso reforça a ideia do Capítulo 1 de que, em comparação com a engenharia de ML tradicional, a engenharia de IA está mais perto do produto. Isso se deve tanto à importância crescente do volante de dados quanto à experiência de produto como vantagens competitivas.

Muitos desafios de IA são, no fundo, problemas de sistema. Para resolvê-los, muitas vezes é preciso dar um passo atrás e considerar o sistema como um todo. Um único problema pode ser tratado por diferentes componentes trabalhando de forma independente, ou uma solução pode exigir a colaboração de vários componentes. Um entendimento completo do sistema é essencial para resolver problemas reais, destravar novas possibilidades e garantir a segurança.
