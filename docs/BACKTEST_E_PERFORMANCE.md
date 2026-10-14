# Plano de avaliação — forecasting recursivo e carga

## Pergunta e separação temporal

A CV temporal escolhe o modelo e o holdout é usado uma única vez para análise. O exercício de backtesting **não seleciona um novo campeão pelo holdout**: avalia o campeão previamente congelado em origens semanais da janela de teste. Para seleção da estratégia multi-horizonte em pesquisa futura, seriam necessárias origens em validação dentro da janela de desenvolvimento. Reportar essa diferença é parte da qualidade metodológica.

## Previsões sem vazamento

- O instante de origem é o primeiro dia previsto. Recebemos somente as **28 observações anteriores** daquele produto e o planejamento de promoções futuras.
- `D+1`: `next_features` é calculado sobre histórico real disponível.
- `D+2...D+14`: `next_features` é recalculado sobre o histórico estendido por **previsões anteriores**; jamais sobre a demanda futura verdadeira.
- Na origem seguinte, os dias anteriores à nova origem agora são considerados observados — isso corresponde a replanejamento periódico, não a previsão de 14 dias feita no passado.
- A sazonal ingênua D−7 também se autorealimenta, não recebe valores futuros reais.

## Métricas

- MAE e RMSE por passo `h`, e acumuladas de D+1 até D+h.
- WAPE calculado pela soma dos erros absolutos dividida pela soma das demandas observadas, `null` quando o denominador é zero.
- Viés, volume de origens, recorte por produto e comparativo com baseline.
- Origens sobrepostas geram dependência; não inferir cobertura de IC de maneira ingênua.
- A `q90` da previsão de 1 dia não foi recalibrada para horizonte longo.

## Carga e cold start

- Processo Uvicorn novo: medir startup até `GET /health` retornar `model_ready=true`.
- Primeira previsão: `POST /predict` com artefato serializado no disco. É uma medida distinta do startup e pode pagar a desserialização inicial.
- Warmup: três requisições fora da estatística. Níveis de 1,4,8 threads com requisições repetidas e payload idêntico.
- Medir latência de chamadas completas do cliente e wall time para throughput. P95/P99 amostrais não são percentis confiáveis com poucos pedidos.
- Host e porta são 127.0.0.1 com porta efêmera, não usa serviços externos, não mede a rede real, nem CPU/RAM do servidor separadamente.
- Limites explícitos evitam cargas acidentais. Nenhum SLO foi inventado antes de medir.

## Metas de investigação posteriores (não reivindicadas)

- Avaliar intervalos por horizonte com calibração apenas em dados de validação e cobertura em holdout.
- Simular promoções desconhecidas ou mudança de mix; observar propagação de erro.
- Validar drift sazonal por produto e disparar alerta condicionado a erro observado.
- Medir uso de RAM/CPU com ferramentas do host e comportamento multiworker após configurar autenticação/rate limiting (fora desta demo local).
