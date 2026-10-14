# Previsão de demanda com MLOps local

Laboratório de previsão diária de seis produtos com dois anos de dados sintéticos.
Inclui baseline sazonal, Ridge, boosting, validação temporal, MLflow local, API e monitoramento.

## Executar independentemente

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/train.py
.venv/bin/python src/monitor.py
.venv/bin/python src/smoke.py
.venv/bin/python -m pytest tests -q
```

Windows: substitua `.venv/bin/python` por `.venv\Scripts\python`.
No pacote completo: `python executar.py mlops` usando o ambiente instalado.

## API

```bash
.venv/bin/python -m uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000
```

Abra http://127.0.0.1:8000/docs. Envie `resultados/exemplo_requisicao.json` para `/predict`.
Para `/forecast`, acrescente `horizon` (1–14) e `planned_promotions` (uma entrada 0/1 por dia).
`promo` deve coincidir com a primeira promoção planejada.

Modelo e metadados são lidos da versão ativa. `/health` informa se o modelo está pronto. A API rejeita
produto não treinado, histórico com lacuna/duplicata ou histórico que não termine em D-1.

## MLflow

O pacote completo tem `python executar.py mlflow`. Para este repositório isolado, use um caminho absoluto
para seu banco ao executar `python -m mlflow server --backend-store-uri sqlite:////CAMINHO/ABSOLUTO/work/mlflow.db
--host 127.0.0.1 --port 5000`. No Windows, ajuste para o formato de URI SQLite correspondente ao disco.
O banco e os artefatos são locais. O arquivo `work/deployment.json` registra o URI correto utilizado.

## Promoção e rollback

```bash
.venv/bin/python src/promote.py seasonal_naive
# Faça uma previsão e observe a versão.
.venv/bin/python src/promote.py hist_gradient_boosting
```

O segundo comando é um exemplo: use o nome do campeão de `resultados/treinamento.json`.
`src/smoke.py` testa promoção, volta ao campeão e carregamento do artefato registrado no MLflow.

## Dados próprios

CSV com `day,product_id,units,promo`; uma linha por produto/dia, sem lacunas e com produto no formato P00.
Promoção deve ser conhecida no momento da previsão. Não confundir ausência de registro com demanda zero.

```bash
.venv/bin/python src/train.py --input /caminho/demanda.csv
```

Entradas externas têm procedência não verificada até você documentar origem e semântica.
Os hiperparâmetros e folds desta versão são um ponto de partida; dados reais podem exigir outro desenho.

Leia `docs/MODEL_CARD.md`: as métricas são de rolling **um dia**. O endpoint recursivo não tem
validação multihorizonte. Modelos foram treinados apenas até o início do holdout e não retreinados com ele.
Artefatos joblib/cloudpickle são gerados localmente pelo código; carregue somente artefatos de confiança.

Os HTML e gráficos incluídos são snapshots da execução entregue. Para atualizá-los, use `gerar_relatorios.py` na raiz do pacote completo após executar os fluxos. CSV/JSON são atualizados pelos scripts deste projeto.

## Versão portfólio · 1.1.0

Compatibilidade com NumPy/Pandas no Python 3.13 melhorada nas comparações de datas. Os testes
usam HTTPX/ASGI sem `fastapi.testclient` legado. `/` explica as rotas; `/health` exige que
`work/deployment.json` aponte para um artefato existente (não valida a desserialização).
O relatório atualizado está em `resultados/RESUMO_ATUAL.html`.

```bash
make setup
make test
make run
make report
make api
```

Em outro Terminal, com a API rodando:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -H 'Content-Type: application/json' \
  --data-binary @resultados/exemplo_requisicao.json \
  http://127.0.0.1:8000/predict
```

Use `make mlflow` para abrir o servidor local em http://127.0.0.1:5000 após o treino.
O workflow `.github/workflows/ci.yml` roda testes e demonstração completa na nuvem do GitHub
sob limites do plano aplicável; a aplicação e MLflow não precisam de deploy pago.

**Próxima melhoria autoral:** validar previsões recursivas de 7 dias num holdout multihorizonte
antes de atribuir métricas de 1 dia ao endpoint `/forecast`.
