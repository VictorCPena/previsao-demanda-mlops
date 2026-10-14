# Promoções desconhecidas, calibração multihorizonte e execução containerizada

## Contexto
O `/forecast` original exige plano de promoções futuro conhecido, que nem sempre existe na data da decisão. Uma previsão de 14 dias acumula erros gerados por suas próprias previsões anteriores. E o p90 residual de um único dia não constitui um intervalo válido para 14 passos recursivos.

## Cenários de promoções não observadas
`POST /forecast/scenarios` aceita apenas `product_id`, `day`, `history`, `horizon` e (opcionalmente) cenários com `name`, `weight`, `promotions` de 0/1. Se omitidos, usa três cenários **hipotéticos**, não probabilidades aprendidas: sem promoção (0,50), promoção apenas no primeiro dia (0,25) e promoção todos os dias (0,25). Retorna previsões individuais, média ponderada e mínimo/máximo por dia.

O envelope min/máx compara **decisões diferentes sobre promoções**, não é intervalo de 90% nem representa toda a incerteza de demanda. A API não precisa de uma agenda verdadeira do futuro nem de dados reais futuros.

## Intervalos multihorizonte por erro recursivo

```bash
make train
make backtest
make calibrate
```

`backtest.py` gera `work/evaluation/backtest_recursivo_linhas.csv` com previsões recursivas e `uncertainty.py` estima quantis absolutos **por passo**. Origens antigas calibram; origens posteriores avaliam, com embargo de ao menos 14 dias. O quantil finito usa `ceil((n+1)*(1-alpha))`, registrando largura média e cobertura empírica de D+1 a D+14. Sem dados suficientes, o comando falha explicitamente; não inventa certeza. Saída: `work/evaluation/intervalos_multihorizonte.json`.

**Cuidado metodológico:** origens e produtos apresentam dependência serial; logo esses quantis não asseguram cobertura distribution-free nem são equivalentes a um sistema conformal com hipóteses atendidas. Não usar intervalos como garantia de estoque, SLA ou produção sem calibração adequada e monitoramento de drift. Cenários futuros ainda precisam entrar como fonte independente de incerteza.

## Containerização com custo local zero

```bash
make train
docker compose up --build
```

`Dockerfile` usa Python 3.12, usuário não privilegiado e **2 workers Uvicorn**. `compose.yaml` monta `work/` somente leitura, filesystem da imagem read-only, tmpfs pequena, porta vinculada apenas em `127.0.0.1`. `/live` checa processo; `/health` checa se há um artefato ativo no caminho autorizado. Não é necessário serviço de nuvem pago. `/docs` estará disponível em `http://127.0.0.1:8000/docs`.

**Escalabilidade aqui demonstrada:** múltiplos workers em um único host local. Não demonstra autoscaling multi-host, ingress, TLS, tracing distribuído, disponibilidade global, rate limiting, autenticação ou deployment público. `joblib` só deve carregar artefatos produzidos localmente de fontes confiáveis.

## Testes de contrato
`tests/test_uncertainty.py` verifica pesos e cenários inválidos, recursão sem modificar a entrada, split temporal e embargo, cobertura entre 0 e 1, além da rota real via cliente ASGI. 
