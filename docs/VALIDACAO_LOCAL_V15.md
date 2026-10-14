# Evidências locais de verificação — v1.5

**Escopo:** execuções independentes deste pacote, com dataset sintético gerado por `generate()` (seed 42) e modelo `HistGradientBoostingRegressor` ajustado diretamente sobre a parcela de **desenvolvimento**. A seleção por CV do workflow completo *não foi repetida nesta execução de verificação*; portanto os números abaixo são uma checagem do algoritmo de rollout, não os resultados finais do MLflow registrado. Não representam demanda real.

| Passo (exatamente D+h) | Observações de origem/produto | MAE modelo | MAE baseline sazonal recursivo | WAPE modelo |
| --- | ---: | ---: | ---: | ---: |
| D+1 | 66 | 5,31 | 12,12 | 11,39% |
| D+7 | 66 | 5,58 | 12,29 | 12,27% |
| D+14 | 66 | 6,05 | 12,62 | 12,86% |

Detalhes: 6 produtos, 730 dias sintéticos por produto, holdout a partir de 2025-10-02, origem semanal, 11 origens por produto e 14 dias de rollout para cada uma. Baseline: `history[-7]` aplicada recursivamente. **Origens sobrepostas são dependentes.** O cálculo exato de cada linha está no arquivo CSV produzido por `make backtest` após o treino. O fluxo oficial da CI usa o campeão escolhido por CV, e pode produzir métricas próximas mas distintas.

## Como reproduzir o fluxo oficial

```bash
make setup
make test
make train
make evaluate
make backtest
make load-study
```

O JSON local `work/evaluation/backtest_recursivo.json` registra a versão efetiva do modelo, hash dos dados e métricas por passo/produto. **Não adicionar esse JSON ao Git**: o README apresenta metodologia e exemplos com suas limitações, enquanto a execução gera os artefatos operacionais.

**Validação de carga integrada local:** também foi iniciado um processo Uvicorn novo com um modelo sazonal de teste e uma requisição de exemplo: prontidão ~1,03 s, primeira inferência ~811 ms, 0 falhas em 8 requisições distribuídas entre os níveis 1 e 2. Ambiente do executor de desenvolvimento, sem padronização de hardware; números destinados somente a demonstrar a instrumentação, não a evidenciar capacidade de produção.
