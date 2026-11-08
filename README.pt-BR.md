# God of War: Ascension — port nativo para macOS (Apple Silicon)

**[Read in English](README.md)**

God of War: Ascension (PS3, BCES-01741, Europa) **recompilado estaticamente para código arm64
nativo**, rodando no macOS. Sem emulador e sem Wine:
- o código da PPU vira C e é compilado para o processador da Apple;
- os programas das SPUs, inclusive os jobs de geometria EDGE, são traduzidos por imagem;
- a GPU RSX é reimplementada sobre Vulkan, via MoltenVK (Metal).

**Estado: jogável, nativo no macOS, com gráficos corretos e sem travadas de compilação de shader.**
Jogado em sessões longas num M4 Max.

> Este repositório **não contém dados do jogo**. Você precisa da sua própria cópia, obtida
> legalmente (imagem do disco), e de decriptar o seu `EBOOT.BIN`. Tudo o que deriva do jogo (código
> recompilado, caches de shader, saves) é gerado na sua máquina e fica fora do git.

## Máquina de teste

| | |
|---|---|
| Computador | MacBook Pro 14" (Mac16,6) |
| Chip | Apple M4 Max — CPU de 14 núcleos (10 de desempenho + 4 de eficiência), GPU de 32 núcleos |
| Memória | 36 GB unificada |
| Sistema | macOS 27.0.1 |
| Gráficos | MoltenVK 1.4.2 + Vulkan loader 1.4.363, SDL2 (sdl2-compat 2.32), glslang 16.6 |
| Tela | 3024×1964 Retina; o jogo renderiza em 1280×720, ampliado (Catmull-Rom + nitidez adaptativa) |

Todos os números abaixo foram medidos nesta máquina, na tomada.

## Desempenho

fps por cena, pelo contador de quadros do próprio jogo, em janelas de 50–90 s:

| Cena | fps |
|---|---|
| Menu principal | 60 (travado); ~74 destravado |
| Início do primeiro nível | 46–54 |
| Luta com a Megera | **~60** (59,9 no A/B; 61,1 no trecho estável; 59,7 em 10 minutos) |
| Ponte / área da prisão | 42–44 |
| Luta no telhado (a cena mais pesada medida) | 22–27 |
| Cena da ilusão do harém | ~30 (era ~3,6 antes de traduzir os fragmentos EDGE que ela usa) |

A luta com a Megera foi medida de novo em 08/11/2026 (janela fixa da luta, sem janela, A/B
intercalado). As outras cenas são de builds anteriores e ainda não foram medidas depois do
trabalho abaixo.

Como a cena da Megera foi de injogável a ~60 fps:

| Etapa | fps |
|---|---|
| Primeira vez no nível (código das SPUs interpretado) | 3,5 |
| Fragmentos de código EDGE traduzidos (443 posicionamentos) | 20 |
| Float das SPUs em NEON com FPCR em arredondamento para zero + flush-to-zero | 33 |
| Nativos EDGE via SIMDe onde vencem o código traduzido | 35 |
| Auxiliares NEON para operações das SPUs, checagem de faixa do float em 3 instruções | ~41 |
| Giro da thread principal no park limitado ao próximo tique de vblank | 43,7 |
| Lanes extras do SPURS para as listas de jobs que a thread principal espera | 50,8 |
| Texturas novas sem esperar a GPU; lotes de desenho maiores; tabela densa na coerência das SPUs | ~52 |
| Espelho das reservas vivas, recuo na trava de lock-line, hash de textura adaptativo | ~56 |
| Trava de lock-line das SPUs dividida por linha | 57,6 |
| Cópias de código relocável das SPUs reconhecidas pelo conteúdo (fora do interpretador); variáveis por thread guardadas no contexto da SPU | ~60 |

Compilação de shaders (as travadas ao entrar numa área nova ou fazer algo pela primeira vez):

| | Antes | Depois |
|---|---|---|
| Pipelines criados numa sessão longa | ~8.200 (tabela cheia em ~13 min → imagem quebrada) | ~900–1.450 |
| Pior pausa quando surge um pipeline novo | 98 ms + 190 ms no desenho | 0,4 ms (compilado em threads de fundo) |
| Pré-compilação dos pipelines conhecidos na abertura | 7,2 s | 0,12 s na thread do jogo; o resto em segundo plano |
| Mesmo par de shaders em vários pipelines | recompilado pelo Metal a cada um | um módulo compartilhado (8,8 → 0,4 ms) |

O jogo é limitado pela CPU (trabalho das SPUs), não pela GPU. Na luta com a Megera o quadro mediano
fica abaixo de 16,7 ms; os ~14% de quadros acima de 20 ms são limitados pelos jobs de geometria do
EDGE, que o jogo restringe a quatro SPUs.

## O que foi feito ou corrigido para o macOS

A lista completa está em **[docs/CHANGES.md](docs/CHANGES.md)**. Em resumo:

- **Correção no arm64:**
  - escrita de 16 bytes das SPUs rasgada, que causava travamentos raros com o andador do FIFO lendo
    dado como comando;
  - ordem de memória no FIFO e no DMA das SPUs;
  - corrida na reserva das SPUs;
  - `_setjmp`, TLS e relógio.
- **Desempenho:**
  - fragmentos de código EDGE traduzidos (1.720 posicionamentos);
  - auxiliares NEON e float nativo das SPUs no arm64;
  - latência do kernel do SPURS;
  - escolha por região entre o código EDGE nativo e o traduzido.
- **Gráficos (RSX → Vulkan/MoltenVK):**
  - depth bounds test emulado (o Metal não tem);
  - cascatas e direção da comparação de sombra;
  - polygon offset (fim da acne de sombra);
  - texturas 3D (LUT de gradação de cor);
  - constantes de blend;
  - precisão e alias de registradores R/H no fragment program;
  - cache de 3.072 texturas.
- **Travadas de shader:**
  - chave de pipeline canônica;
  - catálogo de pipelines em disco;
  - módulos de shader compartilhados;
  - compilação assíncrona;
  - cache de SPIR-V.
- **Integração com o Mac:**
  - janela na thread principal;
  - DualSense e teclado pelo SDL;
  - tela cheia com ampliador HiDPI;
  - `.app` com ícone.
- **Ferramentas:**
  - captura sob demanda de todos os passes de renderização (cor, profundidade, estêncil, estado por
    desenho);
  - registro do código das SPUs que cai no interpretador;
  - benchmark rápido sem janela a partir de um save.

## Estrutura

| Pasta | O que é |
|---|---|
| `ps3recomp/` | [sp00nznet/ps3recomp](https://github.com/sp00nznet/ps3recomp) v0.17.0 + o patch do GoW:A + todas as mudanças de macOS/arm64 (MIT) |
| `gowa-port/` | [MaChInEgUn3/God-of-War-Ascension-Recompiled](https://github.com/MaChInEgUn3/God-of-War-Ascension-Recompiled) + mudanças de macOS (MIT) |
| `scripts/` | testes, benchmark e análise |
| `app/` | monta o `God of War Ascension.app` (ícone gerado a partir da arte do seu disco) |
| `data/` | só endereços: imagens SPU do EBOOT europeu e os posicionamentos de fragmentos EDGE para traduzir do seu EBOOT |
| `jogar.sh` | lançador (janela, controle, save, caches) |
| `docs/` | guia de build, lista de mudanças, notas técnicas |

## Compilar e jogar

Veja **[docs/BUILDING.md](docs/BUILDING.md)**. Em resumo:
1. dependências do Homebrew;
2. montar a sua ISO;
3. decriptar o seu EBOOT;
4. traduzir o código da PPU e das SPUs;
5. compilar com CMake/Ninja;
6. jogar com `./jogar.sh` ou montar o app com `app/fazer_app.sh`.

Controles: DualSense ou outro controle pelo SDL, ou teclado:
- setas = direcional;
- Z = ✕, X = ○, A = □, S = △;
- Q/W = L1/R1, 1/2 = L2/R2;
- Enter ou Esc = START, Tab = SELECT;
- ⌘Q sai.

## Problemas conhecidos

- "Réplicas" de personagens desenhadas em algumas paredes; reflexo da água ainda errado.
- O fundo desfocado (profundidade de campo) pode parecer quadriculado.
- Áreas novas ainda podem cair de fps até seus fragmentos EDGE serem traduzidos. Veja
  `docs/BUILDING.md`, seção "queda de fps numa área nova".
- A compilação de shader nunca congela o jogo, mas um objeto pode aparecer alguns quadros atrasado
  na primeira vez.

## Contribuir

Forks e pull requests são bem-vindos, principalmente para:
- os defeitos gráficos em aberto;
- desempenho (o gargalo está nas SPUs);
- outros jogos de PS3 na mesma base.

As notas em [docs/TECHNICAL-NOTES.md](docs/TECHNICAL-NOTES.md) explicam como cada problema foi
achado.

## Créditos e licença

- **ps3recomp**, de Ned Heller / sp00nznet — MIT.
- **God of War: Ascension Recompiled**, de MaChInEgUn3 e colaboradores — MIT. É a base deste port:
  patch do jogo, extração dos job modules, nativos EDGE.
- Ports de referência para Windows: arnaldolima1/gow-ascension-pc e otavix010/gow-ascension-pc.
  Deste último vieram correções do fragment program e das SPUs.
- Port para macOS: Pedro Henrique ([@Pedrohds1](https://github.com/Pedrohds1)), desenvolvido com o
  Claude (Anthropic).

Os nossos arquivos são MIT ([LICENSE](LICENSE)); `ps3recomp/` e `gowa-port/` mantêm as licenças
deles.

God of War é marca da Sony Interactive Entertainment. Este projeto não é afiliado nem endossado pela
Sony e não contém código nem arte do jogo protegidos por direitos autorais.
