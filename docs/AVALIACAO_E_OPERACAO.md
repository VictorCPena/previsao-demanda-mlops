# Avaliação de modelo e operação local

## O que o holdout permite concluir

- **Um dia à frente (rolling)**: a previsão D usa 28 dias passados e o valor real de D−1. Esses 28 dias são efetivamente observados no protocolo.
- **Seleção do modelo**: média de MAE nos folds temporais, com holdout separado. O holdout reporta resultado, mas não serve para escolher o campeão.
- **Análise por produto**: `make evaluate` lê `work/evaluation/previsoes_holdout.csv` e grava `avaliacao_por_produto.json` com MAE, baseline, viés, WAPE e produtos que perderam para o baseline. É uma investigação **pós-seleção**, não outra etapa de tuning.
- **WAPE**: usa soma dos erros absolutos dividida pela soma da demanda real. É `null` quando toda a demanda do grupo é zero.

## Decisões recomendadas

1. Rode `make train` e `make evaluate`.
2. Inspecione `products_worse_than_baseline` e a direção de `bias_units`, sem alterar o campeão em função de uma única fatia pós-hoc.
3. Rode `make monitor`. PSI e KS são indicadores de investigação, não testes independentes calibrados em séries temporais.
4. Só exponha `make api` em loopback (`127.0.0.1`). Sem autenticação, limite de taxa e observabilidade, a API **não** está pronta para exposição pública.
5. Faça rollback apenas para artefatos locais confiáveis. `joblib` e MLflow podem desserializar código; nunca carregue modelos de terceiros não verificados.

## Limitações e próximos testes

- A API oferece forecast recursivo; **não reportar sua acurácia como se fosse a métrica rolling de um dia**.
- Para validar horizonte 7/14 dias, é necessário backtesting específico com previsões geradas recursivamente sem usar vendas futuras observadas.
- Para usar vendas reais, definir fuso, corte diário, atrasos de ingestão, política de produtos novos e relação entre promoção prevista e promoção efetivamente realizada.
- Para produção, incluir rastreabilidade de requisições, SLO, autenticação, monitoramento de latência e eventual re-treino com aprovação humana.
