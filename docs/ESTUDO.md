# Estudar os projetos sem decorar ferramentas

## Ordem sugerida

1. Execute o Data Hub e leia `resultados/execucao.json`. Veja como o mesmo pedido aparece em versões diferentes.
2. Faça uma cópia de um lote e altere um produto para um identificador inexistente. Execute em outra pasta
   `work/custom`. Observe o quality gate e confirme que o banco não ficou parcialmente atualizado.
3. Abra `src/pipeline.py` e siga o caminho: contrato → Bronze → transação → ledger → estado → fato → gate → commit → Parquet.
4. No MLOps, leia `validacao_temporal.csv`: o fim do treino deve ser anterior ao início da validação.
5. Abra o MLflow e compare baseline, Ridge e boosting. O campeão é escolhido em CV, não no holdout.
6. Envie o exemplo JSON para `/predict`. Retire um dia do histórico e observe a rejeição.
7. Promova o baseline e volte ao campeão. Veja versão e previsão mudarem sem alterar código da API.
8. No experimento, leia o plano antes do resultado. Diferencie efeito, incerteza, significância e utilidade prática.
9. Compare a simulação de análise única com testes repetidos; entenda por que olhar diariamente e parar em p<0,05 muda o erro.
10. Faça uma mudança em cada projeto, rode os testes e documente o que você decidiu e por quê.

## Perguntas que você deve conseguir responder

### Data Hub

- Por que a deduplicação usa event_id **e** verifica hash de conteúdo?
- Qual diferença entre evento repetido, evento antigo e correção histórica?
- Por que a cidade do cliente deve ser a cidade válida quando o pedido ocorreu?
- O que uma transação protege? Por que o Parquet pode ser reconstruído após commit?
- Como distinguir soft delete de cancelamento de pedido?
- Qual parte do pipeline é incremental? Qual parte ainda lê o ledger inteiro?
- O que o benchmark mediu e o que ele não permite concluir?

### MLOps

- Por que dividir aleatoriamente uma série temporal pode vazar informação?
- Qual demanda histórica está disponível ao prever D? Como muda ao prever D+7?
- Por que escolher o melhor modelo no holdout invalida sua função de teste final?
- Por que drift de atributos não prova perda de qualidade do modelo?
- Quando os valores reais chegam e permitem monitorar o erro?
- O que é run, versão registrada, alias e artefato de implantação?
- Como reverter a versão se a API tiver problemas?

### Experimentação

- O que é MDE? É a mesma coisa que um efeito observado significativo?
- Por que usuários duplicados e SRM comprometem o experimento?
- Como interpretar um aumento de 2 pontos percentuais versus aumento de 20% relativo?
- Qual papel do intervalo de confiança no guardrail?
- Por que uma covariável medida após a intervenção não pode entrar no ajuste préteste?
- Por que o teste não prova que a campanha funcionará com clientes reais?

## Exercícios de evolução

- Data Hub: trocar um full scan do ledger por lookup só dos IDs recebidos; comparar custo sem perder invariantes.
- MLOps: avaliar de verdade os horizontes 7 e 14 dias com backtesting recursivo; calcular métricas por horizonte.
- Experimento: implementar desenho sequencial válido e comparar com peeking ingênuo; registrar hipóteses antes de executar.

São extensões possíveis, não funcionalidades alegadas como concluídas nesta versão.
