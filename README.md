# Previsão de demanda com MLOps local

Prever a demanda diária de produtos e disponibilizar o modelo por API, com rastreabilidade,
comparação com baseline e promoção reversível. A entrega é um modelo registrado no MLflow e
um serviço FastAPI; as métricas são diagnósticos do treino e da operação.

## Treinar e validar

Requer Python 3.12. Na raiz deste repositório:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/train.py
.venv/bin/python src/monitor.py
.venv/bin/python src/smoke.py
.venv/bin/python -m pytest tests -q
```

Windows: use `.venv\Scripts\python`. Atalhos opcionais: `make train`, `make monitor`,
`make smoke`, `make test`. Cada etapa tem um comando próprio; monitor e smoke exigem treino prévio.
Os dados padrão são sintéticos: seis produtos, 730 dias, semente 42.

## Servir previsões

```bash
.venv/bin/python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000
```

Em outro terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -H 'Content-Type: application/json' --data-binary @work/evaluation/exemplo_requisicao.json http://127.0.0.1:8000/predict
```

O treino gera uma requisição compatível com os produtos e datas usados. `/predict` exige 28 dias
contínuos até D-1. `/forecast` aceita horizonte 1–14 e `planned_promotions` por dia, com a primeira
promoção igual a `promo`. A documentação interativa da API fica em http://127.0.0.1:8000/docs.

## Ciclo do modelo

Baseline D-7, Ridge e boosting disputam três folds temporais. A média MAE de validação escolhe
o campeão antes da avaliação nos últimos 90 dias. Features usam somente informação passada.
O holdout é rolling de **um dia**; não valida a precisão do endpoint recursivo de vários dias.

- `work/mlflow.db` e `work/mlartifacts/`: experimentos, métricas e registry locais.
- `work/deployment.json`: versão ativa e caminhos usados pela API.
- `work/evaluation/`: previsões de holdout, métricas, entrada de exemplo e diagnósticos regeneráveis.

Esses arquivos nascem ao executar e não são versionados. `make mlflow` inicia a interface local
em http://127.0.0.1:5000. Não é necessário criar conta ou contratar hospedagem.

`src/promote.py NOME_DO_MODELO` troca a versão ativa. Use os nomes registrados em
`work/evaluation/treinamento.json`; `src/smoke.py` exercita promoção, restauração do campeão e
paridade de carregamento pelo MLflow. Alias e arquivo local não formam transação distribuída;
refaça a promoção se falhar entre as etapas. A API usa o arquivo como fonte de verdade.

Monitoramento usa PSI, KS descritivo e erros observados. Alertas pedem investigação, não
iniciam retreino automático. Consulte `docs/MODEL_CARD.md` para hipóteses e limitações.

## Dados próprios

`src/train.py --input /caminho/demanda.csv` aceita `day,product_id,units,promo`, com uma linha
por produto/dia e sem lacunas. Promoções precisam ser conhecidas no instante da previsão.
Documente origem, ruptura de estoque e significado de demanda antes de interpretar métricas.
Modelos joblib/cloudpickle devem vir de fontes confiáveis. Serviço local, sem autenticação ou teste de carga.

O workflow GitHub é manual e opcional; todo o fluxo pode ser executado no computador.
