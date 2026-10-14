# Validação de treinamento, backtest e carga — Mac local, 14/10/2026

## Proveniência

Resultados de comandos executados no Mac do autor, ambiente `macOS 13.7.8 x86_64`, Python `3.13.10`, dados sintéticos do repositório. O CI usa Python 3.12 e deverá ser validado separadamente após o push. Estas métricas **não representam uso de clientes**.

```bash
make test        # 29/29 aprovados; -W error::DeprecationWarning
make train
make backtest
make calibrate
make load-smoke
```

## Treinamento / validação temporal

- Campeão: `hist_gradient_boosting`, três versões criadas no registry MLflow `demanda-diaria`.
- Holdout: MAE `5.9483`, RMSE `7.6061`, WAPE `0.11345` (11,34%), melhoria relatada ante baseline `55.6094%`.
- Aviso MLflow: formatos pickle/cloudpickle podem executar código ao desserializar; carregar apenas artefatos próprios e confiáveis.

## Backtest recursivo em origens móveis

66 pares origem-produto por passo. Comparação emparelhada com baseline sazonal, promoções futuras planejadas assumidas conhecidas na origem.

| Passo | MAE modelo | MAE baseline | WAPE modelo | WAPE baseline | Viés modelo |
| --- | ---: | ---: | ---: | ---: | ---: |
| D+1 | 5,2407 | 12,1212 | 11,23% | 25,97% | −1,4912 |
| D+7 | 5,5795 | 12,2879 | 12,27% | 27,02% | −1,3223 |
| D+14 | 6,1426 | 12,6212 | 13,05% | 26,82% | −2,6955 |

As origens se sobrepõem e as medidas não são estatisticamente independentes. O backtest não atualiza pesos durante o holdout. `src/backtest.py` teve a operação de data inicial corrigida. A execução com `PYTHONWARNINGS=error::DeprecationWarning` foi concluída sem aviso, e os 29 testes passaram com o mesmo rigor.

## Calibração por horizonte

`make calibrate`: 14 passos com blocos de calibração/avaliação separados e embargo temporal. D+1: `calibration_n=36`, `evaluation_n=18`, quantil de erro absoluto `12.7009`, cobertura empírica `1.0` (18/18), largura média `25.4018`, MAE de avaliação `4.5969`.

A cobertura 100% é **descritiva nesta amostra pequena e específica**, não garantia de 100% ou de nível nominal em população futura. Cobertura dos demais passos deve ser consultada nas saídas locais produzidas por `make calibrate`; nenhuma conclusão sobre D+14 deve ser inferida só do resumo D+1.

## Carga local Uvicorn

Um worker, IP loopback `127.0.0.1`, uma carga JSON, warmup de 3 requisições não incluídas, 8 requisições por configuração.

| Medida | Concorrência 1 | Concorrência 2 |
| --- | ---: | ---: |
| Sucessos | 8/8 | 8/8 |
| Latência p50 | 8,94 ms | 23,93 ms |
| Latência p95 | 9,62 ms | 29,31 ms |
| Latência p99 | 9,75 ms | 30,19 ms |
| Vazão medida | 110,31 req/s | 85,45 req/s |
| Erros observados | 0 | 0 |

Prontidão do processo: 1,457 s; primeira inferência: 1.413,67 ms. São fases diferentes do cold start e não devem ser somadas automaticamente a uma única latência HTTP. P95/P99 com oito casos são **estatísticas instáveis**. Não houve TLS, roteador, balanceamento, múltiplos hosts nem Docker testado neste log.

## Próximos critérios

- Repetir carga com pelo menos centenas de chamadas por nível, 3–5 execuções, concorrências 1/2/4/8, perfis de CPU/memória e tolerância de erro previamente declarada.
- Executar `make container` e smoke do modo multiworker, se Docker disponível, e comparar os mesmos cenários com ambiente de um worker.
- Registrar distribuição de largura e cobertura de **todos os 14 passos**, por produto e sazonalidade. Evitar afirmar cobertura formal sem premissas adequadas.
