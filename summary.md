# Resumo  "Designing Machine Learning Systems" por Chip Huyen.

##  1 overview of ml systems assets


Este capítulo de abertura teve como objetivo proporcionar aos leitores uma compreensão do que é necessário para implementar com sucesso o aprendizado de máquina (ML) no mundo real. Começamos com uma exploração dos diversos casos de uso de ML atualmente em produção. 
O capítulo oferece uma visão abrangente dos sistemas de aprendizado de máquina (ML), destacando os elementos-chave que determinam a viabilidade de uma solução de ML para um problema específico, além de enfatizar os fatores que podem tornar o ML uma abordagem especialmente eficaz. Além disso, ele diferencia a aplicação de ML em pesquisas e em produção, enquanto também destaca os desafios inerentes à aplicação de práticas tradicionais de engenharia de software a sistemas de ML. 

### Quando Usar e Quando Evitar o ML:**

  - Características do Problema para Soluções de ML: Uma solução de ML requer dados disponíveis, padrões complexos a serem aprendidos, a capacidade de enquadrar o problema como uma questão de previsão e a presença de padrões similares entre os dados de treinamento e os dados não vistos.

 - Fatores que tornam o ML Especialmente Útil: Tarefas repetitivas, baixo custo associado a previsões erradas, escalabilidade das tarefas e a presença de padrões que estão em constante mudança.

### Casos de Uso Típicos de ML**

  - Aplicações para Consumidores vs. Empresariais: Enquanto os aplicativos voltados para consumidores priorizam a latência, a precisão é fundamental em aplicativos empresariais, onde o ML é utilizado para reduzir custos, prever a rotatividade de clientes, monitorar a marca, entre outros.

### ML na Pesquisa e na Produção**

- Diferenças nos Requisitos, Prioridades Computacionais, Dinâmica dos Dados, Equidade e Interoperabilidade entre sistemas de ML utilizados em pesquisas e em ambientes de produção.


### Desafios em Sistemas de ML em Comparação com Software Tradicional:**

- Os sistemas de ML envolvem o gerenciamento não apenas do código, mas também dos dados e dos artefatos do modelo, tornando essencial a aplicação de estratégias de versionamento, teste, implantação e monitoramento específicas.

- Os desafios incluem o versionamento de dados, a avaliação da qualidade dos dados, a implantação de modelos que demandam grandes recursos computacionais e o monitoramento e depuração de modelos complexos implantados.


- Compreender as características e desafios únicos dos sistemas de ML é fundamental para o sucesso na implantação e manutenção de soluções de ML em cenários do mundo real.


# 2 - Objetivos do Projeto, Requisitos e Enquadramento
Este capítulo mostra que é  essencial entender os requisitos que o sistema precisa atender para ser considerado bem-sucedido. Esses requisitos variam dependendo do caso de uso, mas neste capítulo, o autor foca em quatro requisitos gerais: confiabilidade, escalabilidade, manutenabilidade e adaptabilidade. 

Após determinar que uma solução de ML é viável para o seu problema , é possível alinhar os objetivos de negócios com os objetivos de ML e definir os requisitos operacionais que o sistema precisa satisfazer.

Além disso, o capítulo também aborda como o enquadramento do problema pode afetar a facilidade ou dificuldade de construir e manter sua solução.

## A Relação entre Objetivos de Negócios e de ML

1. **Objetivos de Negócios e Métricas de ML:** 
   - Empresas não se importam com métricas de ML sofisticadas como `acurácia`, `precisão`, `revocação`, `F1`, etc. Projetos de ML nos quais os cientistas de dados se concentram demais em hackear métricas de ML sem prestar atenção às métricas de negócios tendem a falhar.
   - Mapear métricas de negócios para métricas de ML é mais fácil para algumas aplicações de ML do que para outras. Por exemplo, o impacto de sistemas de detecção de fraudes em aplicações financeiras é muito claro e fácil de medir.
   - Muitas empresas criam suas próprias métricas para mapear métricas de negócios para métricas de ML, como a Netflix, que mede o desempenho de seu sistema de recomendação usando a taxa de sucesso: o número de reproduções de qualidade dividido pelo número de recomendações que um usuário vê.

## Requisitos para Sistemas de ML

### Confiabilidade

- O sistema deve continuar a desempenhar a função correta no nível desejado mesmo diante de adversidades.
- Determinar a "correção" em sistemas de ML é mais difícil do que em sistemas de software, pois os sistemas de ML podem falhar silenciosamente, continuando a produzir previsões incorretas.

### Escalabilidade

- Considere os diferentes eixos nos quais um sistema de ML precisa escalar.
- Os sistemas de ML podem crescer em tamanho do modelo (por exemplo, usando mais parâmetros), o que significa que seu hardware precisa de mais RAM para funcionar.
- Os sistemas de ML podem crescer em volume de tráfego que atendem. Seu sistema precisa ser capaz de acompanhar.
- Os sistemas de ML podem crescer no número de modelos para um caso de uso específico.

### Manutenibilidade

- Pessoas de diferentes origens trabalham em um único sistema de ML (engenheiros de ML, engenheiros de DevOps, SMEs, etc.).
- É importante estruturar o fluxo de trabalho de forma que cada grupo possa trabalhar com as ferramentas com as quais estão confortáveis, em vez de um grupo impor um conjunto de ferramentas para todos.
- O código deve ser documentado, os dados e artefatos devem ser versionados.

### Adaptabilidade

- As distribuições de dados e os requisitos de negócios mudam rapidamente. Seu sistema precisa ser capaz de se adaptar a essas mudanças naturais.

## Enquadramento de Problemas de ML

### Tipos de Tarefas Supervisionadas de ML

#### Classificação vs. Regressão:**
   - Um modelo de regressão pode ser facilmente enquadrado como um modelo de classificação e vice-versa.

#### **Problemas de Classificação Multiclasse:**
   - Problemas de alta cardinalidade são difíceis. Verifique se há outras alternativas antes de se comprometer com um modelo multiclasse.
   - A coleta de dados para problemas de alta cardinalidade é desafiadora.
   - A classificação hierárquica pode ser útil para problemas com muitas classes.

#### **Problemas de Classificação Multirrótulo:**
   - Os problemas de classificação multirrótulo são difíceis porque cada observação pode ter uma quantidade diferente de rótulos.
   - Existem duas abordagens principais para a classificação multirrótulo.


## Enquadramento da Função Objetivo em Aplicações com Múltiplos Objetivos

- Problemas multiobjetivos, como classificar itens por qualidade e engajamento, exigem consideração cuidadosa das funções objetivas.
- O desacoplamento de objetivos e o treinamento de modelos separados para cada objetivo podem simplificar o desenvolvimento e a manutenção do modelo.

Em resumo, alinhar objetivos de negócios com objetivos de ML, entender os requisitos e enquadrar problemas de forma eficaz são etapas essenciais no desenvolvimento de sistemas de ML bem-sucedidos.



## 3 - Fundamentos de Engenharia de Dados


Este capítulo destaca a importância de escolher o formato adequado para armazenar dados em sistemas de ML, discutindo formatos de dados, modelos de dados e motores de armazenamento. Aborda também três modos de transferência de dados entre processos e explora a diferença entre processamento em lote e em fluxo. Esses conceitos são cruciais para o desenvolvimento eficaz de sistemas de ML, preparando o terreno para a coleta de dados e criação de conjuntos de treinamento.

### Qualidade do algoritmo versus qualidade e quantidade dos dados

- Há um grande debate sobre se, no futuro, a qualidade da ML será impulsionada pela qualidade e quantidade dos dados (como tem sido até agora) ou pela qualidade dos algoritmos de treinamento. Alguns acreditam que, com o aumento do poder computacional, algoritmos mais inteligentes e poderosos compensarão a qualidade e quantidade inferiores de dados.

- O debate continua. No entanto, ninguém pode negar que, por enquanto, a qualidade e quantidade dos dados são essenciais. Por isso, as necessidades mais fundamentais para o aprendizado de máquina estão todas relacionadas aos dados, não ao treinamento do modelo.



### Fontes de Dados

Os dados para alimentar sistemas de ML geralmente vêm de diferentes fontes, categorizadas como:

 #### Dados de entrada do usuário:** 
 
 - variados em formato, como texto, imagens, vídeos e arquivos. Dados malformados são comuns.

 #### Dados gerados pelo sistema:** logs e saídas do sistema, como uso de memória, CPU e metadados sobre o comportamento do usuário.

  - Menos propensos a serem malformados do que os dados do usuário.
  - É comum registrar tudo o que for possível devido à complexidade da depuração em sistemas de ML.
  - O grande volume de dados gerados pelo sistema apresenta desafios de sinal e armazenamento.

 #### Bancos de dados internos: 
 - usados pelos serviços de software para executar os negócios.

#### Dados de segunda parte: 
- coletados por outra empresa sobre seus clientes e disponibilizados para você.

#### Dados de terceiros: 
- coletados por empresas sobre o público que não são seus clientes.

### Formatos de Dados

A escolha do formato de dados impacta na legibilidade, velocidade de recuperação, transmissão e custo de armazenamento. Alguns formatos populares incluem:

  | Formato   | Binário/Texto  | Legível por humanos | Exemplos de uso             |
  | --------- | -------------- | ------------------- | --------------------------- |
  | JSON      | Texto          | Sim                 | Em todo lugar              |
  | CSV       | Texto          | Sim                 | Em todo lugar              |
  | Parquet   | Binário        | Não                 | Hadoop, Redshift           |
  | Avro      | Binário        | Não                 | Hadoop                      |
  | Protobuf  | Binário        | Não                 | Google, TensorFlow         |
  | Pickle    | Binário        | Não                 | Python, PyTorch            |

  *Tabela: Exemplos de formatos de dados populares*

#### JSON

- Extremamente popular, mas pode causar dificuldades devido à sua ubiquidade.
- Boa legibilidade e flexibilidade.
- Compromete-se implicitamente a um esquema que o leitor precisará assumir.
- Mudar o esquema retroativamente é doloroso.

#### Texto versus Binário

- Texto é legível, mas ocupa mais espaço.
- Binário é compacto, mas não é legível.

#### Linha Principal versus Coluna Principal

- **Linha Principal:** 
  - Melhor para muitas gravações, mas difícil para muitas leituras.
- **Coluna Principal:**
  - Melhor para muitas leituras, mas difícil para muitas gravações.

### Modelos de Dados

Os modelos de dados incluem:

- **Modelos de dados relacionais**: o esquema é determinado antecipadamente.
- **Modelos de dados NoSQL**:
  - Nenhum esquema é determinado antecipadamente. A responsabilidade de assumir um esquema é transferida para as aplicações que leem os dados.
    - Modelo de Documento
    - Modelos de Grafo


### Motores de Banco de Dados

Os bancos de dados são otimizados para processamento transacional ou analítico.

### Processamento de Dados

Extração, transformação e carregamento (ETL) de dados de diferentes fontes para destino desejado.

### Modos de Fluxo de Dados

Como passar dados entre processos diferentes que não compartilham memória?


