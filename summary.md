# Resumo  "Designing Machine Learning Systems" por Chip Huyen.

## Capítulo 1


Este capítulo de abertura teve como objetivo proporcionar aos leitores uma compreensão do que é necessário para implementar com sucesso o aprendizado de máquina (ML) no mundo real. Começamos com uma exploração dos diversos casos de uso de ML atualmente em produção. 
O capítulo oferece uma visão abrangente dos sistemas de aprendizado de máquina (ML), destacando os elementos-chave que determinam a viabilidade de uma solução de ML para um problema específico, além de enfatizar os fatores que podem tornar o ML uma abordagem especialmente eficaz. Além disso, ele diferencia a aplicação de ML em pesquisas e em produção, enquanto também destaca os desafios inerentes à aplicação de práticas tradicionais de engenharia de software a sistemas de ML. 

**Quando Usar e Quando Evitar o ML:**

 Características do Problema para Soluções de ML: Uma solução de ML requer dados disponíveis, padrões complexos a serem aprendidos, a capacidade de enquadrar o problema como uma questão de previsão e a presença de padrões similares entre os dados de treinamento e os dados não vistos.

Fatores que tornam o ML Especialmente Útil: Tarefas repetitivas, baixo custo associado a previsões erradas, escalabilidade das tarefas e a presença de padrões que estão em constante mudança.

**Casos de Uso Típicos de ML**

 Aplicações para Consumidores vs. Empresariais: Enquanto os aplicativos voltados para consumidores priorizam a latência, a precisão é fundamental em aplicativos empresariais, onde o ML é utilizado para reduzir custos, prever a rotatividade de clientes, monitorar a marca, entre outros.

**ML na Pesquisa e na Produção**

Diferenças nos Requisitos, Prioridades Computacionais, Dinâmica dos Dados, Equidade e Interoperabilidade entre sistemas de ML utilizados em pesquisas e em ambientes de produção.


**Desafios em Sistemas de ML em Comparação com Software Tradicional:**

Os sistemas de ML envolvem o gerenciamento não apenas do código, mas também dos dados e dos artefatos do modelo, tornando essencial a aplicação de estratégias de versionamento, teste, implantação e monitoramento específicas.

Os desafios incluem o versionamento de dados, a avaliação da qualidade dos dados, a implantação de modelos que demandam grandes recursos computacionais e o monitoramento e depuração de modelos complexos implantados.


Compreender as características e desafios únicos dos sistemas de ML é fundamental para o sucesso na implantação e manutenção de soluções de ML em cenários do mundo real.


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

1. **Classificação vs. Regressão:**
   - Um modelo de regressão pode ser facilmente enquadrado como um modelo de classificação e vice-versa.

2. **Problemas de Classificação Multiclasse:**
   - Problemas de alta cardinalidade são difíceis. Verifique se há outras alternativas antes de se comprometer com um modelo multiclasse.
   - A coleta de dados para problemas de alta cardinalidade é desafiadora.
   - A classificação hierárquica pode ser útil para problemas com muitas classes.

3. **Problemas de Classificação Multirrótulo:**
   - Os problemas de classificação multirrótulo são difíceis porque cada observação pode ter uma quantidade diferente de rótulos.
   - Existem duas abordagens principais para a classificação multirrótulo.


## Enquadramento da Função Objetivo em Aplicações com Múltiplos Objetivos

- Problemas multiobjetivos, como classificar itens por qualidade e engajamento, exigem consideração cuidadosa das funções objetivas.
- O desacoplamento de objetivos e o treinamento de modelos separados para cada objetivo podem simplificar o desenvolvimento e a manutenção do modelo.

Em resumo, alinhar objetivos de negócios com objetivos de ML, entender os requisitos e enquadrar problemas de forma eficaz são etapas essenciais no desenvolvimento de sistemas de ML bem-sucedidos.




