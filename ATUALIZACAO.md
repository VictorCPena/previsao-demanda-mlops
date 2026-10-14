# Três projetos independentes · versão 1.2

- `datahub-vendas-confiavel`: pipeline incremental, warehouse DuckDB e tabelas Parquet.
- `previsao-demanda-mlops`: modelo, registry local, API e monitoramento.
- `experimentacao-ab-marketing`: notebook de análise e decisão de experimento.

Cada pasta é a raiz de um repositório, com ambiente e comandos próprios. Leia o README de cada um.
Não há site, relatório HTML ou resultados pré-calculados. Saídas operacionais são criadas localmente,
em `work/`, e não entram no Git. As bases incluídas são sintéticas. Não há serviço pago necessário.

## Atualizar os repositórios que você já publicou

Faça uma cópia de segurança e confira `git status` antes de alterar seu checkout.
Copie o conteúdo da pasta correspondente para a raiz do repositório existente, incluindo os arquivos
ocultos `.gitignore` e `.github`. Preserve sua pasta `.git`, dados próprios e alterações pessoais.
Se você modificou arquivos após o ZIP anterior, compare e incorpore essas mudanças antes de substituir.

Arquivos da versão anterior que devem ser removidos em cada repositório (copiar por cima não os remove):

```bash
git rm -r --ignore-unmatch resultados
git rm --ignore-unmatch src/summary_html.py docs/ESTUDO.md
```

Não remova `work/` para atualizar: ela contém dados/modelos locais. A versão MLOps passa a usar
`work/evaluation/`; execute o treino para gerar métricas e exemplo de requisição nesse caminho.
No Data Hub, a carga normal usa `work/warehouse/`; a demonstração isolada mantém `work/demo/`.

Execute os comandos do README e confira `git diff --stat` e `git status`. Depois de revisar:

```bash
git add .
git diff --cached --stat
git commit -m "Reorganiza projeto por finalidade e remove apresentações prontas"
git push
```

Revise o que será incluído antes do commit, principalmente se houver dados próprios.
Os workflows foram mantidos manuais: nenhum CI de nuvem é disparado automaticamente ao fazer push.
Os testes locais não dependem deles. Esta entrega não altera os repositórios remotos por você.

## Validação desta revisão

Python 3.12: 36 testes aprovados (13 Data Hub, 11 MLOps e 12 experimento). Executados carga
incremental, auditoria banco/Parquet, demonstração de recuperação, treino, monitoramento,
smoke ASGI da API e carregamento MLflow, análise A/B e exportação opcional do parecer.
O esquema do notebook foi validado e todas as células de código executadas em sequência.
A interface e o kernel Jupyter não foram validados: o ambiente de revisão bloqueia sockets.
Os arquivos produzidos na validação não estão nos ZIPs.
