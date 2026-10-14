"""Estudo local de cold start, latência, concorrência e throughput da API.

Execução intencionalmente limitada ao loopback; nunca mede capacidade de produção.
Inicia UM processo novo para o cold start. Depois usa o mesmo processo com
concorrências definidas e mesma requisição sintética (sem efeitos colaterais).
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from data import ROOT


def describe(times_ms):
    if not times_ms:
        raise ValueError('Amostra vazia')
    x = np.array(times_ms, dtype=float)
    return {'count': int(x.size), 'p50_ms': float(np.percentile(x,50)),
            'p95_ms': float(np.percentile(x,95)), 'p99_ms': float(np.percentile(x,99)),
            'mean_ms': float(x.mean()), 'max_ms': float(x.max())}


def local_base(url):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1','localhost','::1') or not parsed.port or parsed.username or parsed.password:
        raise ValueError('Benchmark permitido somente em HTTP loopback com porta explícita')
    return url.rstrip('/')


def request_json(url, payload, timeout=15):
    started = time.perf_counter()
    request = Request(url, json.dumps(payload).encode(), {'Content-Type':'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())
            return {'ok': response.status==200 and 'prediction' in data,
                    'status': response.status, 'latency_ms': 1000*(time.perf_counter()-started)}
    except (HTTPError,URLError,TimeoutError,OSError,ValueError) as exc:
        return {'ok': False, 'error': str(exc), 'latency_ms':1000*(time.perf_counter()-started)}


def measure_level(url, payload, concurrency, requests):
    if not 1 <= concurrency <= 16 or not 1 <= requests <= 500:
        raise ValueError('Limites de segurança: 1..16 workers, 1..500 requisições')
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: request_json(url+'/predict',payload), range(requests)))
    elapsed = time.perf_counter()-start
    ok = [r for r in results if r['ok']]
    return {'concurrency':concurrency, 'requests':requests, 'success':len(ok),
            'errors':len(results)-len(ok), 'error_rate':1-len(ok)/len(results),
            'wall_seconds':elapsed, 'requests_per_second':len(ok)/elapsed if elapsed else None,
            'latency_success':describe([r['latency_ms'] for r in ok]) if ok else None,
            'sample_errors':[r.get('error',f"status={r.get('status')}") for r in results if not r['ok']][:5]}


def wait_ready(url, process, seconds=25):
    deadline = time.perf_counter()+seconds
    while time.perf_counter()<deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Servidor encerrou antes da prontidão (rc={process.returncode})')
        try:
            with urlopen(url+'/health',timeout=1) as response:
                health=json.loads(response.read())
                if response.status==200 and health.get('model_ready'):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(.10)
    raise TimeoutError('Servidor não ficou pronto em 25s')


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        return s.getsockname()[1]


def run(requests=60, levels=(1,4,8), output=None, workspace=None):
    if not 1 <= requests <= 500 or not levels or max(levels)>16 or min(levels)<1:
        raise ValueError('Valores fora dos limites de benchmark local')
    workspace=Path(workspace or ROOT/'work')
    payload=json.loads((workspace/'evaluation/exemplo_requisicao.json').read_text())
    port=free_port()
    url=local_base(f'http://127.0.0.1:{port}')
    env=os.environ.copy()
    env['DEMAND_WORKSPACE']=str(workspace)
    env.update({'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
    started=time.perf_counter()
    proc=subprocess.Popen([sys.executable,'-m','uvicorn','serve:app','--app-dir',str(ROOT/'src'),
                           '--host','127.0.0.1','--port',str(port),'--no-access-log'],
                          cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        wait_ready(url,proc)
        ready_seconds=time.perf_counter()-started
        cold=request_json(url+'/predict',payload)
        if not cold['ok']:
            raise RuntimeError(f'Primeira inferência falhou: {cold}')
        # Warmup is excluded from measured steady-state latency.
        warmup=[request_json(url+'/predict',payload) for _ in range(3)]
        if not all(r['ok'] for r in warmup):
            raise RuntimeError('Aquecimento falhou')
        levels_results=[measure_level(url,payload,int(level),requests) for level in levels]
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait(timeout=5)
    report={'scope':'127.0.0.1: single uvicorn worker, one local machine, one JSON payload',
            'system':{'platform':platform.platform(),'python':platform.python_version(),
                      'processor':platform.machine()},
            'startup_to_health_seconds':ready_seconds,'first_inference_ms':cold['latency_ms'],
            'warmup_requests_excluded':len(warmup),'levels':levels_results,
            'notes':['Cold start = boot do processo + primeira inferência, medidos separadamente.',
                     'Sem rede externa, TLS, balanceador, serviços pagos ou infraestrutura real.',
                     'Concorrência é de clientes locais; inferência CPU e serialização podem limitar escala.',
                     'Não comparável entre computadores sem controle do hardware e ruído.',
                     'P95/P99 amostrais: com poucos pedidos são muito instáveis.']}
    output=Path(output or ROOT/'work/evaluation/load_study.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests',type=int,default=60)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,4,8])
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    print(json.dumps(run(args.requests,args.concurrency,args.output),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
