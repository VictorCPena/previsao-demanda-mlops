# Model card e validação

## Objetivo e fonte

Prever unidades demandadas por produto para o próximo dia. Dados padrão: sintéticos, seis produtos,
730 dias desde 2024-01-01, semente 42. Gerador com tendência, sazonalidade semanal/anual e promoções.
Não há vendas reais, ruptura de estoque, censura de demanda ou efeitos de campanhas externas.

## Features disponíveis no instante da previsão

Calendário e promoção planejada para D; lags de 1/7/14/28 dias; médias móveis de 7/28 dias calculadas
**após shift(1)**. A demanda de D não participa das features de D. A implementação valida dias contínuos.
Identificador numérico do produto é tratado como feature nesta pequena simulação; dados reais com muitos
produtos podem exigir encoding/modelos hierárquicos diferentes.

## Protocolo temporal

Holdout: últimos 90 dias. Desenvolvimento: dias anteriores. Três folds expanding-window, cada validação
com 42 dias. Divisão por data global: todos os produtos da data ficam no mesmo lado. Campeão selecionado
pela média MAE de CV; o holdout não escolhe o modelo. Baseline repete D-7. Métricas: MAE, RMSE e WAPE.

A avaliação é rolling de um dia: para cada D, a demanda real de D-1 já é conhecida. Isso é válido para
operação diária e diferente de prever todos os 90 dias num único instante. O endpoint recursivo usa suas
próprias previsões como histórico futuro e é apenas funcional; precisa de backtesting por horizonte.

Faixa empírica: quantil 90% do erro absoluto nos folds de validação, por modelo. A cobertura observada
no holdout é registrada. Não é promessa de cobertura, especialmente com drift ou séries dependentes.

## Operação local

MLflow em SQLite com artefatos locais. Cada candidato possui run, métricas, hash de dados e versão no registry.
Alias champion aponta ao escolhido na CV. API usa `deployment.json` como fonte de verdade e carrega
joblib da versão ativa. A mudança do alias e do arquivo não é uma transação distribuída; o arquivo
é trocado atomicamente. Em falha entre as etapas, refaça a promoção para reconciliar os dois estados.

Nenhum treinamento automático é disparado pelo monitor. Drift é triagem: PSI>0,2 é heurística, KS
é descritivo devido à dependência temporal. Monitor também compara MAE de janelas após os valores reais.
Uma mudança nos atributos pode ser sazonalidade; não prova degradação. Retreinar exige investigar,
validar um candidato e decidir a promoção. Endpoints não expõem administração de modelos.

## Limites

Uso local educacional, sem autenticação ou teste de carga; bind em 127.0.0.1. Não é serviço público de produção.
Modelo não vê produtos novos. Não estima causalidade nem elasticidade de preço. Resultados de melhoria
de erro só descrevem esta simulação. Versões e caminhos locais são recriados em cada instalação.
