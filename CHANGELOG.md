## 1.6.0 — Estudos técnicos adicionais

- `src/uncertainty.py`: cenários de promoções desconhecidas e avaliação temporal dos intervalos recursivos D+1..14.
- Nova rota `/forecast/scenarios` e `Dockerfile`/`compose.yaml` locais com 2 workers.
- CI com avaliação de calibração temporal.

## v1.5.0 — experimentos de robustez e avaliação reproduzíveis

- Backtesting rolling-origin recursivo para 1/7/14 dias e baseline recursiva sazonal.
- Cold start e carga local com p50/p95/p99/throughput, segurança somente loopback.
- Cache do modelo invalidado por versão/mtime, endpoint recursivo compartilha rollout.

# Histórico de evolução — previsao-demanda-mlops

## 1.4.0 — estudo aprofundado

- README específico com pergunta de negócio, evidências e limitações.
- Nova capacidade analítica distinta das outras duas aplicações, com testes dedicados.
- Documentação operacional e workflow ajustado para executar a nova verificação.
- Mantido o escopo local com dados sintéticos e sem serviço pago.

## 1.2.x — base consolidada

- Repositório independente, geração de dados, testes, documentação e execução local.
- Correção do CI do MLOps no commit `40362b3` preservada neste pacote.
