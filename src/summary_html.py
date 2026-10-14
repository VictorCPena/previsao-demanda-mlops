"""Gera RESUMO_ATUAL.html a partir dos JSONs da última execução, sem dependências extras."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados"
KIND = "mlops"


def read(name):
    path = RESULTS / name
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path}. Execute o projeto antes de gerar este relatório.")
    return json.loads(path.read_text())


def fmt(value):
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def render():
    if KIND == "datahub":
        data = read("execucao.json")
        extra = read("auditoria.json")
        s = data["summary"]
        title, subtitle = "Data Hub • Auditoria", "DuckDB + Parquet | Eventos sintéticos"
        metrics = [("Eventos no ledger", s["ledger"]), ("Pedidos", s["order_state"]),
                   ("Receita simulada (R$)", f"{s['revenue_cents']/100:,.2f}"),
                   ("Auditoria Parquet", extra["status"])]
        details = {"Qualidade do warehouse": data["quality"],
                   "Auditoria bidirecional": extra,
                   "Reprodução / rollback": {"rollback_verified": data["rollback_verified"],
                                              "replay": data["replay"]}}
        conclusion = "Os dados são sintéticos. Auditoria compara os exports com o banco, não com uma fonte externa."
    elif KIND == "mlops":
        data = read("treinamento.json")
        extra = read("monitoramento.json")
        title, subtitle = "MLOps • Previsão diária", "Validação temporal | MLflow local | Dados sintéticos"
        winner = data["winner_selected_by_cv"]
        holdout = data["holdout"][winner]
        metrics = [("Modelo escolhido", winner),
                   ("MAE holdout", fmt(holdout["mae"])),
                   ("Melhoria vs baseline MAE", f"{data['improvement_vs_baseline_mae_pct']:.1f}%"),
                   ("Horizonte avaliado", "1 dia")]
        details = {"Holdout sem seleção posterior": holdout,
                   "Versões de modelos": data["versions"],
                   "Monitoramento": extra}
        conclusion = "Métricas são apenas para previsão rolling de 1 dia; o endpoint recursivo não foi validado no holdout para múltiplos dias."
    else:
        data = read("analise.json")
        title, subtitle = "Experimento A/B • Decisão", "Plano prévio | SRM | Guardrail | Dados sintéticos"
        primary = data["primary_conversion"]
        metrics = [("A — Conversão", f"{100*primary['rate_A']:.2f}%"),
                   ("B — Conversão", f"{100*primary['rate_B']:.2f}%"),
                   ("Lift relativo", f"{100*primary['relative_lift']:.2f}%"),
                   ("Amostra total", data["n_A"]+data["n_B"])]
        details = {"Decisão": data["decision"], "Validade": data["validity"],
                   "Intervalo de confiança e p-valor": primary,
                   "Guardrail de descadastros": data["guardrail_unsubscribed"]}
        conclusion = "Rejeitar adoção automática quando segurança do guardrail não foi demonstrada, mesmo com ganho primário."
    h = html.escape
    cards = "".join(f'<div class="metric"><span>{h(str(label))}</span><strong>{h(fmt(value))}</strong></div>' for label, value in metrics)
    sections = "".join(f'<details open><summary>{h(name)}</summary><pre>{h(json.dumps(value, indent=2, ensure_ascii=False))}</pre></details>'
                       for name, value in details.items())
    template = """<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;background:#0b1421;color:#f4f7fb;line-height:1.55}
main{max-width:960px;margin:auto;padding:50px 22px 90px}.eyebrow{font-size:12px;color:#69d8bd;text-transform:uppercase;letter-spacing:2px}h1{font-size:clamp(30px,5vw,52px);letter-spacing:-1.5px;margin:6px 0 3px}p{color:#b6c5d9}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:13px;margin:35px 0}.metric,details{background:#132235;border:1px solid #284058;border-radius:14px;padding:19px}.metric span{display:block;color:#b6c5d9;font-size:13px}.metric strong{display:block;color:#fff;font-size:24px;margin-top:8px;overflow-wrap:anywhere}
details{margin:12px 0}summary{font-weight:650;cursor:pointer}pre{color:#c0e0e7;font-size:13px;overflow:auto;white-space:pre-wrap;word-break:break-word}.note{border-left:3px solid #69d8bd;padding:10px 18px;background:#102535;border-radius:4px}footer{font-size:12px;margin-top:30px;color:#91a5be}</style></head>
<body><main><div class="eyebrow">Portfólio de Dados · Execução Local</div><h1>__TITLE__</h1><p>__SUBTITLE__</p><div class="metrics">__CARDS__</div><div class="note">__CONCLUSION__</div><h2>Evidências da execução</h2>__SECTIONS__<footer>Relatório gerado diretamente a partir dos JSONs atuais. Não são métricas de clientes reais.</footer></main></body></html>"""
    values = {"__TITLE__":h(title), "__SUBTITLE__":h(subtitle), "__CARDS__":cards,
              "__CONCLUSION__":h(conclusion),"__SECTIONS__":sections}
    for marker, replacement in values.items():
        template = template.replace(marker, replacement)
    destination = RESULTS / "RESUMO_ATUAL.html"
    destination.write_text(template, encoding="utf-8")
    print("Relatório atualizado:", destination)
    return destination


if __name__ == "__main__":
    render()
