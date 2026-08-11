# Livro AI Engineering e outros recursos
> _Este repositório será atualizado com mais recursos nas próximas semanas._

- [Sobre o livro AI Engineering](#sobre-o-livro)
    - [Sumário](ToC.md)
    - [Resumos dos capítulos](chapter-summaries.md)
    - [Notas de estudo](study-notes.md)
- [Recursos de engenharia de IA](resources.md)
- [Exemplos de prompt](prompt-examples.md)
- [Estudos de caso](case-studies.md)
- [Desalinhamento em IA](misalignment.md)
- [Apêndice](appendix.md)
- Ferramentas divertidas:
    
    - [Gerador de mapa de calor de conversas do ChatGPT e do Claude](scripts/ai-heatmap.ipynb)
- E mais...

## Sobre o livro
A disponibilidade de modelos de fundação transformou a IA de disciplina especializada em uma poderosa ferramenta de desenvolvimento que qualquer pessoa pode usar. Este livro cobre o processo de ponta a ponta de adaptar modelos de fundação para resolver problemas do mundo real, reunindo técnicas já consagradas em outras áreas da engenharia e técnicas que surgem com os modelos de fundação.

[<img src="assets/aie-cover.png" width="250">](https://amzn.to/49j1cGS)[<img src="assets/aie-cover-back.png" width="250">](https://amzn.to/49j1cGS)

O livro está disponível em:
- [Amazon](https://amzn.to/49j1cGS)
- [O'Reilly](https://oreillymedia.pxf.io/c/5719111/2146021/15173)
- [Kindle](https://amzn.to/3Vq2ryu)
- [Audible](https://www.audible.com/pd/AI-Engineering-Audiobook/B0DWJ1SP94?qid=1781010240)

e na maioria dos lugares onde se vendem livros técnicos.

Ele foi [traduzido](translations.md) para: [japonês](https://www.oreilly.co.jp/books/9784814401383/), [coreano](https://www.hanbit.co.kr/store/books/look.php?p_code=B3535685426), [chinês tradicional](https://www.gotop.com.tw/books/bookdetails.aspx?types=a&bn=A806), [chinês simplificado](https://product.dangdang.com/30013700.html), [vietnamita](https://shop.timescorporation.vn/sach-ky-thuat-ai-xay-dung-ung-dung-voi-mo-hinh-nen-tac-gia-chip-huyen), [espanhol](https://bit.ly/4piliIn), [italiano](https://www.amazon.it/-/en/Ingegneria-Dellia-Italian-Applicazioni-Fondazione/dp/B0GM5V47D7/), [francês](https://www.lisez.com/livres/ingenierie-de-lia/9782412104415), [português](https://novatec.com.br/livros/engenharia-de-ia/), [alemão](https://dpunkt.de/produkt/generative-ki-systeme-entwickeln/), [tailandês](https://www.naiin.com/product/detail/694038), [russo](https://www.flip.kz/catalog?prod=5764935) e [polonês](https://helion.pl/ksiazki/inzynieria-ai-tworzenie-aplikacji-z-wykorzystaniem-modeli-bazowych-chip-huyen,inaitw.htm#format/d).

Em breve estará disponível em grego, ucraniano e uzbeque.


_Este NÃO é um livro tutorial, portanto não traz muitos trechos de código._

## Sobre o que é este livro
Este livro oferece um arcabouço para adaptar modelos de fundação — que incluem tanto modelos de linguagem de grande porte (LLMs) quanto modelos multimodais de grande porte (LMMs) — a aplicações específicas. Ele não apenas descreve várias soluções para construir uma aplicação de IA, como também levanta perguntas que você pode fazer para avaliar a melhor solução para as suas necessidades. Estas são apenas algumas das muitas perguntas que o livro ajuda a responder:

1. Devo construir esta aplicação de IA?
1. Como avalio minha aplicação? Posso usar IA para avaliar saídas de IA?
1. O que causa alucinações? Como detectá-las e mitigá-las?
1. Quais são as boas práticas de engenharia de prompt?
1. Por que RAG funciona? Quais são as estratégias para fazer RAG?
1. O que é um agente? Como construir e avaliar um agente?
1. Quando fazer ajuste fino de um modelo? Quando não fazer?
1. De quantos dados preciso? Como valido a qualidade dos meus dados?
1. Como deixo meu modelo mais rápido, mais barato e mais seguro?
1. Como crio um ciclo de retroalimentação para melhorar minha aplicação continuamente?

O livro também ajuda você a navegar pelo cenário avassalador da IA: tipos de modelos, benchmarks de avaliação e um número aparentemente infinito de casos de uso e padrões de aplicação.

O conteúdo deste livro é ilustrado com estudos de caso reais, em muitos dos quais trabalhei diretamente, apoiados por amplas referências e revisados a fundo por especialistas das mais variadas formações. Embora o livro tenha levado dois anos para ser escrito, ele se apoia na minha experiência com modelos de linguagem e sistemas de ML da última década.

Assim como meu livro anterior, _[Designing Machine Learning Systems (DMLS)](https://amzn.to/4fXVZH2)_, este foca nos fundamentos da engenharia de IA, e não em uma ferramenta ou API específica. Ferramentas ficam obsoletas rápido; fundamentos duram mais.

### Como ler _AI Engineering_ (AIE) junto com _Designing Machine Learning Systems_ (DMLS)
AIE pode ser um complemento de DMLS. DMLS trata da construção de aplicações sobre modelos de ML tradicionais, o que envolve mais anotação de dados tabulares, engenharia de atributos e treinamento de modelos. AIE trata da construção de aplicações sobre modelos de fundação, o que envolve mais engenharia de prompt, construção de contexto e ajuste fino com eficiência de parâmetros. Os dois livros são autocontidos e modulares, então você pode ler qualquer um deles de forma independente.

Como modelos de fundação são modelos de ML, alguns conceitos valem para os dois. Se um tema é relevante para AIE mas já foi discutido em profundidade em DMLS, ele ainda aparece neste livro, porém em menor extensão, com indicações de recursos relevantes.

Muitos temas são cobertos em DMLS e não em AIE, e vice-versa. O primeiro capítulo deste livro também trata das diferenças entre a engenharia de ML tradicional e a engenharia de IA.

Um sistema do mundo real costuma envolver tanto modelos de ML tradicionais quanto modelos de fundação, então conhecer os dois é quase sempre necessário.

## Para quem é este livro

Este livro é para quem quer aproveitar modelos de fundação para resolver problemas do mundo real. É um livro técnico, então a linguagem é voltada a papéis técnicos: engenheiros de IA, engenheiros de ML, cientistas de dados, gerentes de engenharia e gerentes de produto técnicos. Ele é para você se algum dos cenários a seguir soa familiar:
* Você está construindo ou otimizando uma aplicação de IA, seja do zero, seja para tirá-la da fase de demonstração e levá-la a produção. Talvez você também enfrente problemas como alucinações, segurança, latência ou custos e precise de soluções direcionadas.
* Você quer simplificar o processo de desenvolvimento de IA do seu time, tornando-o mais sistemático, rápido e confiável.
* Você quer entender como a sua organização pode aproveitar modelos de fundação para melhorar os resultados do negócio e como montar um time para isso.

Você também pode se beneficiar do livro se pertence a um dos grupos a seguir:
* Quem desenvolve ferramentas e quer identificar áreas mal atendidas da engenharia de IA para posicionar seus produtos no ecossistema.
* Quem pesquisa e quer entender melhor os casos de uso de IA.
* Quem se candidata a vagas e busca clareza sobre as habilidades necessárias para seguir carreira como engenheiro de IA.
* Qualquer pessoa que queira entender melhor as capacidades e as limitações da IA e como ela pode afetar diferentes funções.

Gosto de ir ao fundo das coisas, então algumas seções se aprofundam mais no lado técnico. Muitos leitores iniciais gostaram do detalhamento, mas sei que ele pode não ser para todo mundo. Vou avisar antes de as coisas ficarem técnicas demais. Sinta-se à vontade para pular adiante se achar que está minucioso demais!


## Resenhas
- _"Este livro oferece um guia abrangente e bem estruturado sobre os aspectos essenciais da construção de sistemas de IA generativa. Leitura obrigatória para qualquer profissional que queira escalar a IA por toda a empresa."_ - Vittorio Cretella, former global CIO at P&G and Mars

- _"Chip Huyen entende de IA generativa. Ela é uma professora e escritora notável, cujo trabalho foi decisivo para ajudar times a levar IA para produção. Apoiado nessa profunda experiência, AI Engineering é um guia abrangente e holístico para construir aplicações de IA generativa em produção."_ - Luke Metz, co-creator of ChatGPT, ex-research manager @ OpenAI

- _"Todo engenheiro de IA que constrói aplicações reais deveria ler este livro. É um guia essencial para o projeto de sistemas de IA de ponta a ponta, do desenvolvimento e da avaliação de modelos até a implantação e a operação em larga escala."_ - Andrei Lopatenko, Director Search and AI, Neuron7

- _"Este livro serve como guia essencial para construir produtos de IA que escalam. Ao contrário de outros livros, focados em ferramentas ou em tendências do momento que mudam o tempo todo, Chip entrega conhecimento fundamental e atemporal. Seja você gerente de produto ou engenheiro, o livro fecha com eficácia a lacuna de colaboração entre times multifuncionais, o que o torna leitura obrigatória para qualquer pessoa envolvida com desenvolvimento de IA."_ - Aileen Bui, AI Product Operations Manager, Google

- _"Esta é a transição definitiva para a engenharia de IA, escrita por uma das grandes referências da engenharia de ML! Chip acompanhou projetos e carreiras bem-sucedidos em todos os estágios de uma empresa e, pela primeira vez, condensou sua experiência para quem está entrando agora na área."_ - swyx, Curator, AI.Engineer

- _"AI Engineering é um guia prático que traz as informações mais atuais sobre desenvolvimento de IA, acessível tanto para lideranças iniciantes quanto para as experientes. Este livro é um recurso essencial para quem quer construir sistemas de IA robustos e escaláveis."_ - Vicki Reyzelman, Chief AI Solutions Architect, Mave Sparks

- _"AI Engineering é um guia abrangente que funciona como referência essencial tanto para entender quanto para implementar sistemas de IA na prática."_ - Han Lee, Director - Data Science, Moody's.

- _"AI Engineering é um guia essencial para qualquer pessoa que construa software com IA generativa! Ele desmistifica a tecnologia, destaca a importância da avaliação e mostra o que fazer para alcançar qualidade antes de partir para um ajuste fino caro."_ - Rafal Kawala, Senior AI Engineering Director, 16 years of experience working in a Fortune 500 company

Veja o que estão dizendo sobre o livro no Twitter [@aisysbooks](https://twitter.com/aisysbooks/likes)!

## Agradecimentos
Este livro teria levado muito mais tempo para ser escrito e teria deixado de fora muitos temas importantes se não fosse por tantas pessoas maravilhosas que me ajudaram no processo.

Como o cronograma do projeto era apertado — dois anos para um livro de 150.000 palavras que cobre tanto terreno —, sou grata aos revisores técnicos que reservaram seu tempo precioso para revisar este livro tão rápido.

[Luke Metz](https://x.com/luke_metz) é uma caixa de ressonância incrível, que checou minhas premissas e me impediu de seguir pelo caminho errado. [Han-chung Lee](https://www.linkedin.com/in/hanchunglee/), sempre a par das novidades de IA e do que acontece na comunidade, me apontou recursos que eu tinha deixado passar. Luke e Han foram os primeiros a revisar meus rascunhos antes de eu enviá-los à rodada seguinte de revisores técnicos, e sou eternamente grata a eles por tolerarem minhas bobagens e meus erros.

Tendo liderado a inovação em IA em empresas da Fortune 500, [Vittorio Cretella](https://www.linkedin.com/in/vittorio-cretella/) e [Andrei Lopatenko](https://www.linkedin.com/in/lopatenko/) trouxeram um retorno valiosíssimo, que combinou profundo domínio técnico com visão executiva. [Vicki Reyzelman](https://www.linkedin.com/in/vickireyzelman/) me ajudou a aterrar o conteúdo e a mantê-lo relevante para quem vem da engenharia de software.

[Eugene Yan](https://eugeneyan.com/), amizade querida e figura brilhante da ciência aplicada, me deu apoio técnico e emocional. Shawn Wang ([swyx](https://x.com/swyx)) fez uma checagem de clima importante, que me deixou mais confiante em relação ao livro. [Sanyam Bhutani](https://x.com/bhutanisanyam1) é uma das pessoas que melhor aprendem e uma das almas mais humildes que conheço: não só escreveu um retorno atencioso como gravou vídeos para explicar seus comentários.

Kyle Krannen é um líder de destaque em aprendizado profundo, que entrevistou seus colegas e compartilhou comigo um relato incrível sobre o processo de ajuste fino da equipe, o que orientou o capítulo sobre ajuste fino. [Mark Saroufim](https://x.com/marksaroufim), uma mente inquisitiva sempre com o dedo no pulso dos problemas mais interessantes, me apresentou ótimos recursos sobre eficiência. O retorno de Kyle e de Mark foi decisivo para escrever os Capítulos 7 e 9.

[Kittipat "Bot" Kampa](https://www.linkedin.com/in/kittipat-bot-kampa-1b1965/), além de responder às minhas muitas perguntas, compartilhou comigo uma visualização detalhada de como ele pensa a plataforma de IA. Admiro a abordagem sistemática de [Denys Linkov](https://www.linkedin.com/in/denyslinkov/) para avaliação e desenvolvimento de plataforma. [Chetan Tekur](https://www.linkedin.com/in/chetantekur/) deu ótimos exemplos, que me ajudaram a estruturar os padrões de aplicação de IA. Também quero agradecer a [Alex (Shengzhi Li) Li](https://www.linkedin.com/in/findalexli/) e a [Hien Luu](https://www.linkedin.com/in/hienluu/) pelo retorno atencioso ao meu rascunho sobre arquitetura de IA.

[Aileen Bui](https://www.linkedin.com/in/aileenbui/) é um tesouro e compartilhou comentários e exemplos únicos sob a perspectiva de gerência de produto. Obrigada a [Todor Markov](https://www.linkedin.com/in/todor-markov-4aa38a67/) pelos conselhos acionáveis sobre o capítulo de RAG e agentes. Obrigada a [Tal Kachman](https://www.linkedin.com/in/tal-kachman/) por entrar de última hora e empurrar o capítulo de ajuste fino até a linha de chegada.

Há tanta gente maravilhosa cuja companhia e cujas conversas me deram ideias que orientaram o conteúdo deste livro. Tentei ao máximo incluir aqui o nome de todos que ajudaram, mas, pela falibilidade inerente da memória humana, sem dúvida deixei muitos de fora. Se esqueci de incluir seu nome, saiba que não foi por falta de apreço pela sua contribuição; me lembre disso, por favor, para que eu possa corrigir o quanto antes!

Andrew Francis, Anish Nag, [Anthony Galczak](https://www.linkedin.com/in/wgalczak/), [Anton Bacaj](https://x.com/abacaj), Balázs Galambosi, Charles Frye, Charles Packer, Chris Brousseau, Eric Hartford, Goku Mohandas, Hamel Husain, Harpreet Sahota, Hassan El Mghari, Huu Nguyen, Jeremy Howard, Jesse Silver, John Cook, [Juan Pablo Bottaro](https://www.linkedin.com/in/juan-pablo-bottaro/), Kyle Gallatin, Lance Martin, Lucio Dery, Matt Ross, Maxime Labonne, Miles Brundage, Nathan Lambert, Omar Khattab, [Phong Nguyen](https://www.linkedin.com/in/xphongvn/), Purnendu Mukherjee, Sam Reiswig, Sebastian Raschka, Shahul ES, Sharif Shameem, Soumith Chintala, Teknium, Tim Dettmers, Undi5, Val Andrei Fajardo, Vern Liang, Victor Sanh, Wing Lian, Xiquan Cui, Ying Sheng e Kristofer.

Quero agradecer a todos os leitores iniciais que também escreveram com comentários. Douglas Bailley é um superleitor e compartilhou muito retorno atencioso. E Nutan Sahoo, por sugerir uma forma elegante de explicar a perplexidade.

Aprendi muito com as discussões on-line com tanta gente. Obrigada a todos que já responderam às minhas perguntas, comentaram meus posts ou me mandaram um e-mail com suas ideias.

É claro que o livro não teria sido possível sem a equipe da O'Reilly, em especial meus editores de desenvolvimento (Melissa Potter, Corbin Collins, Jill Leonard) e meus editores de produção (Kristen Brown e Elizabeth Kelly). Liz Wheeler é a pessoa mais criteriosa com quem já trabalhei na edição. Nicole Butterfield é uma força e acompanhou este livro da ideia ao produto final.

Este livro, afinal, é o acúmulo de lições valiosíssimas que aprendi ao longo da carreira. Devo essas lições a colegas e ex-colegas extremamente competentes e pacientes. Cada pessoa com quem trabalhei me ensinou algo novo sobre trazer o ML para o mundo.

---

<br>
<br>

Chip Huyen, *AI Engineering*. O'Reilly Media, 2025.

    @book{aiebook2025,  
        address = {USA},  
        author = {Chip Huyen},  
        isbn = {978-1801819312},   
        publisher = {O'Reilly Media},  
        title = {{AI Engineering}},  
        year = {2025}  
    }
