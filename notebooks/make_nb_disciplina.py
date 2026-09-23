"""Gera o notebook da disciplina: tres familias, um desfecho por execucao, em PT-BR.

Mesmo esquema dos doze do projeto: clona o repositorio publico, constroi a
matriz congelada pelo registry e reporta pelo pns_modelkit. A diferenca e que
aqui as tres familias rodam lado a lado no mesmo notebook.
"""
import json
import sys

REPO = "https://github.com/isasaade-23/pns2013-lab-exams-en.git"


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").split("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n").split("\n")}


C = []

C.append(md("""
# Detectando doenca cronica nao diagnosticada na PNS 2013

**Disciplina de Machine Learning** · Isabela Venancio da Silva (FSP/USP)

A Pesquisa Nacional de Saude de 2013 mediu pressao arterial e coletou sangue de
uma subamostra de respondentes. Da para comparar o que a pessoa **responde** ter
sido diagnosticada com o que a **medida** diz. Entre 45% e 83% das pessoas cuja
medida cruza o limiar clinico respondem "nao" a pergunta de diagnostico.

Este notebook treina tres familias de modelos para recuperar esse rotulo a
partir do questionario, em quem nunca recebeu o diagnostico:

| Familia | Por que esta aqui |
|---|---|
| Regressao logistica | Modelo linear com splines na idade. Coeficiente se le direto |
| Random forest | Nao linearidade e interacao sem precisar declarar nenhuma |
| TabPFN | Modelo fundacional tabular: pre-treinado, faz uma passagem para frente em vez de um ajuste |

**Um desfecho por execucao.** A celula de configuracao escolhe entre pressao
arterial, hemoglobina glicada e colesterol total. As tres familias rodam sobre a
mesma matriz e a mesma particao, entao a comparacao entre elas e limpa.

## O que este notebook nao decide

O pre-processamento e congelado e vem de fora: uma planilha declara cada
variavel e o que se faz com ela, e o script de build le essa planilha. Sao 117
variaveis declaradas e 91 que entram na matriz. **Para mudar uma variavel,
muda-se a planilha, nao o codigo.**

A construcao roda em duas camadas. A primeira e deterministica e nao olha o
desfecho: carrega, filtra linhas, constroi o rotulo, corrige codigos, resolve os
faltantes que o proprio fluxo do questionario explica, reconstroi os contadores e
congela. A segunda, que tem imputacao, splines, codificacao e padronizacao, e
ajustada **dentro de cada dobra** da validacao cruzada e nunca ve as linhas de
teste. Essa ordem e a garantia contra vazamento.

## Vazamento, que aqui e o problema central

Tres cuidados que nao sao obvios e que ja custaram caro neste projeto:

1. **Resultado de exame nunca entra como preditor.** A premissa e um rastreio por
   questionario, para quem nunca coletou sangue.
2. **O que identifica a propria condicao sai, e isso depende do desfecho.** Ter
   diagnostico de diabetes e um preditor legitimo de pressao alta e ao mesmo
   tempo e o criterio de entrada do modelo de diabetes. Uma coluna so de
   "vazamento" nao expressa isso; a planilha carrega uma por desfecho.
3. **Contador e soma do bloco.** O numero de medicamentos em uso e a soma das
   colunas de medicamento. Se uma dessas colunas sai e o contador nao e
   refeito, a variavel removida volta pela soma. Isso ja produziu AUC de 0,86
   com sensibilidade 1,000 antes de ser percebido.
"""))

C.append(md("""
---
# 1 - Preparacao
"""))

C.append(md("""
## 1.1 - Instalacao

`xgboost` nao e usado aqui, mas `optuna` faz a busca de hiperparametros e
`tabpfn-client` fala com a API do modelo fundacional. `openpyxl` le a planilha.
"""))
C.append(code("!pip install -q optuna tabpfn-client openpyxl"))

C.append(md(f"""
## 1.2 - Dados e codigo compartilhado

Clona o repositorio publico, que tem o arquivo da pesquisa, os dois dicionarios,
o pre-processamento congelado (`pns_preprocess.py` e a planilha) e o auxiliar de
modelagem (`pns_modelkit.py`).

O Drive fica **desligado por padrao**, porque montar pede permissao toda sessao.
Assim a saida vai para o disco da sessao e e compactada no fim. Com
`MOUNT_DRIVE = True` a saida vai para a pasta compartilhada.
"""))
C.append(code(f"""
import os, subprocess, sys

REPO_URL = "{REPO}"
REPO     = "/content/pns2013-lab-exams-en"

# Clona, ou atualiza um clone que a sessao ja tenha: uma sessao iniciada antes
# do ultimo commit continuaria rodando o codigo antigo.
if os.path.exists(REPO):
    subprocess.run(["git", "-C", REPO, "fetch", "-q", "--depth", "1", "origin", "main"])
    subprocess.run(["git", "-C", REPO, "reset", "-q", "--hard", "origin/main"])
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
sys.path.insert(0, os.path.join(REPO, "pipeline"))

DATA     = os.path.join(REPO, "data", "pns2013_lab_exams.xlsx")
REGISTRY = os.path.join(REPO, "pipeline", "PNS_preprocessing_registry_v1.xlsx")

MOUNT_DRIVE = False
DRIVE = "/content/drive/MyDrive/01_USP/Disciplina_Ml"

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive nao montado:", e)

print("codigo em", subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                                  capture_output=True, text=True).stdout.strip())
print("dados   ", os.path.exists(DATA))
print("planilha", os.path.exists(REGISTRY))
"""))

C.append(md("""
## 1.3 - Configuracao

| Chave | O que faz |
|---|---|
| `DESFECHO` | `"hypertension"`, `"diabetes"` ou `"cholesterol"` |
| `LIMIAR` | `None` usa o limiar de diretriz. As analises de sensibilidade pre-especificadas estao na tabela abaixo |
| `N_TENTATIVAS` | orcamento da busca. 20 cabe numa sessao gratuita; `COMPLETO = True` sobe para 100 |
| `RODAR_TABPFN` | desliga a terceira familia se a chave da API nao estiver disponivel |

| Desfecho | Definicao | Autoridade do limiar | Sensibilidade |
|---|---|---|---|
| `hypertension` | PAS >= 140 **ou** PAD >= 90 mmHg | OMS 2021 e DBHA 2020, identicas | `(130, 80)`, ACC/AHA 2017 |
| `diabetes` | HbA1c >= 6,5% | SBD e OMS. A PNS 2013 nao tem glicemia de jejum | `(6.0,)` OMS/IEC, `(5.7,)` ADA |
| `cholesterol` | Colesterol total >= 200 mg/dL | Corte convencional de rastreio | `(190,)` |

O nome do primeiro desfecho e **pressao arterial elevada medida**, nao
hipertensao: as diretrizes exigem duas ocasioes fora de medicacao e a PNS mediu
uma vez.
"""))
C.append(code("""
import importlib
import numpy as np, pandas as pd
import pns_preprocess as pp
import pns_modelkit as mk
importlib.reload(pp); importlib.reload(mk)

DESFECHO     = "hypertension"     # "hypertension" | "diabetes" | "cholesterol"
LIMIAR       = None               # None = limiar de diretriz
COORTE       = "undiagnosed"      # quem nunca recebeu o diagnostico
COMPLETO     = False              # True -> 100 tentativas, a rodada final
N_TENTATIVAS = 100 if COMPLETO else 20
RODAR_TABPFN = True
RS           = mk.RS              # 42, fixo, para a particao ser a mesma

BASE   = f"{DRIVE}" if os.path.isdir(DRIVE) else "/content/trabalho"
SAIDA  = os.path.join(BASE, "resultados")
BUILD  = os.path.join(BASE, "matrizes")
os.makedirs(SAIDA, exist_ok=True); os.makedirs(BUILD, exist_ok=True)
print("escrevendo em", SAIDA)
"""))

C.append(md("""
---
# 2 - A matriz congelada

Constroi a matriz pelo registry, ou reaproveita uma construida antes cujos
hashes de dados e planilha ainda batem. Em seguida confere os totais do projeto:
**6.329** linhas a 15,2% para pressao, **6.812** a 3,8% para diabetes e **5.927**
a 32,1% para colesterol.

**O que olhar.** Se a conferencia falhar, pare. Quer dizer que a planilha ou o
arquivo de dados mudaram, e nenhum resultado daqui e comparavel ate entender o
porque.
"""))
C.append(code("""
bundle = mk.load_or_build(DESFECHO, data=DATA, registry=REGISTRY,
                          outdir=BUILD, threshold=LIMIAR, cohort=COORTE)
mk.check_frozen(bundle)

X, y = bundle["X"], bundle["y"]
print(f"\\n{X.shape[0]} linhas x {X.shape[1]} preditores, "
      f"prevalencia {y.mean():.1%}")
print("autoridade do limiar:", bundle["threshold_authority"])
print("sha dos dados", bundle["data_sha"], "| sha da planilha", bundle["registry_sha"])

atr = mk.attrition(bundle, BUILD)
display(atr) if atr is not None else None
"""))

C.append(md("""
**Fluxo de participantes e papeis das colunas.** A tabela acima e a contagem
linha a linha, que vira o fluxograma do artigo. Abaixo, por onde cada coluna
entra na segunda camada e quantos preditores cada bloco do questionario tem.
"""))
C.append(code("""
for k, v in bundle["roles"].items():
    print(f"{k:<10} {len(v):>3}  {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")

blocos = pd.Series({c: bundle["spec"][c]["block"]
                    for c in X.columns if c in bundle["spec"]}).value_counts()
print("\\npreditores por bloco\\n", blocos.to_string())
"""))

C.append(md("""
---
# 3 - Particao e pre-processador

Particao 80/20 estratificada com semente fixa: as tres familias treinam e sao
avaliadas exatamente nas mesmas linhas.

O pre-processador e passado **para dentro** do pipeline, nunca ajustado aqui.
Cada familia recebe a versao que lhe convem: a logistica com padronizacao e
spline na idade, as arvores sem nenhuma das duas, porque arvore divide a idade
direto e a padronizacao nao muda divisao.
"""))
C.append(code("""
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

Xtr, Xte, ytr, yte = mk.split(bundle)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

def prep(linear):
    \"\"\"linear=True: spline na idade e padronizacao. Arvore nao precisa de nenhuma.\"\"\"
    return mk.preprocessor(bundle, spline=linear, scale=linear)

print(f"{X.shape[1]} preditores -> {prep(True).fit_transform(Xtr).shape[1]} colunas codificadas")
print(f"positivos: treino {int(ytr.sum())}, teste {int(yte.sum())}")
"""))

C.append(md("""
---
# 4 - As tres familias

Cada uma e reportada duas vezes, no padrao da biblioteca e ajustada, porque a
comparacao entre as duas e a forma honesta de dizer se a busca comprou alguma
coisa. No projeto ela vale cerca de +0,006 de AUC na logistica.

O desbalanceamento e tratado por peso de classe, nao por reamostragem. Importa
sobretudo em diabetes, com 3,8% de positivos.
"""))

C.append(md("""
## 4.1 - Regressao logistica

Penalizacao L1 ou L2 escolhida na busca. A idade entra como spline cubica
natural de cinco nos, porque o risco nao e linear no log das chances.
"""))
C.append(code("""
import optuna
from sklearn.linear_model import LogisticRegression
optuna.logging.set_verbosity(optuna.logging.WARNING)

resultados, ajustados = [], {}

def avalia_familia(nome, faz_estimador, espaco, linear):
    \"\"\"Padrao e ajustado, lado a lado. Devolve as duas linhas de metrica.\"\"\"
    pipe = Pipeline([("prep", prep(linear)), ("clf", faz_estimador({}))])
    cv_padrao = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
    pipe.fit(Xtr, ytr)
    linha_p, p_padrao = mk.evaluate(pipe, Xte, yte, f"{nome} (padrao)")
    linha_p["CV_AUC"] = cv_padrao.mean()
    print(f"{nome} padrao: CV {cv_padrao.mean():.3f}, teste {linha_p['AUC_test']:.3f}")

    def objetivo(trial):
        params, scores = espaco(trial), []
        for k, (itr, iva) in enumerate(cv.split(Xtr, ytr)):
            p = Pipeline([("prep", prep(linear)), ("clf", faz_estimador(params))])
            p.fit(Xtr.iloc[itr], ytr.iloc[itr])
            scores.append(roc_auc_score(ytr.iloc[iva],
                                        p.predict_proba(Xtr.iloc[iva])[:, 1]))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RS),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    estudo.optimize(objetivo, n_trials=N_TENTATIVAS, show_progress_bar=True)

    melhor = estudo.best_params
    pipe_a = Pipeline([("prep", prep(linear)),
                       ("clf", faz_estimador(espaco(optuna.trial.FixedTrial(melhor))))])
    pipe_a.fit(Xtr, ytr)
    linha_a, p_ajust = mk.evaluate(pipe_a, Xte, yte, f"{nome} (ajustado)")
    linha_a["CV_AUC"] = estudo.best_value
    print(f"{nome} ajustado: CV {estudo.best_value:.3f}, teste {linha_a['AUC_test']:.3f}")
    print(" ", melhor)

    resultados.extend([linha_p, linha_a])
    ajustados[nome] = dict(pipe=pipe_a, prob=p_ajust, params=melhor)
    return linha_p, linha_a


def espaco_logreg(t):
    return {"C": t.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": t.suggest_categorical("penalty", ["l1", "l2"])}

def logreg(params):
    return LogisticRegression(max_iter=3000, class_weight="balanced",
                              solver="liblinear", random_state=RS, **params)

avalia_familia("Regressao logistica", logreg, espaco_logreg, linear=True)
"""))

C.append(md("""
## 4.2 - Random forest

Celula lenta: sao cinco ajustes por tentativa. Com 20 tentativas leva algo como
15 minutos; com 100, mais de uma hora.
"""))
C.append(code("""
from sklearn.ensemble import RandomForestClassifier

_PROF = {"nenhuma": None, "6": 6, "10": 10, "16": 16, "24": 24}
_FEAT = {"sqrt": "sqrt", "log2": "log2", "0.3": 0.3, "0.5": 0.5}

def espaco_rf(t):
    return {"n_estimators": t.suggest_int("n_estimators", 250, 600),
            "max_depth": _PROF[t.suggest_categorical("max_depth", list(_PROF))],
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 25),
            "max_features": _FEAT[t.suggest_categorical("max_features", list(_FEAT))]}

def rf(params):
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                  random_state=RS, **params)

avalia_familia("Random forest", rf, espaco_rf, linear=False)
"""))

C.append(md("""
## 4.3 - TabPFN

Modelo fundacional tabular. E pre-treinado: nao ha busca de hiperparametro, e o
"ajuste" e o custo de enviar as linhas de treino. Por isso ele sai do formato
padrao-versus-ajustado e aparece uma vez so.

**Dois caminhos.** `BACKEND = "api"` manda a matriz para a Prior Labs e precisa
de chave, guardada nas saved keys do Colab como `TABPFN_TOKEN` e gerada em
[platform.priorlabs.ai/account/api-keys](https://platform.priorlabs.ai/account/api-keys).
`BACKEND = "local"` baixa os pesos abertos e roda na sessao, sem chave e sem
nada sair da maquina; peca uma GPU em Ambiente de execucao → Alterar o tipo.

**O que sai da sessao.** No caminho da API, a matriz codificada. A PNS 2013 e
microdado publico do IBGE, entao aqui e aceitavel; com dado identificavel nao
seria.
"""))
C.append(code("""
BACKEND = "api"            # "api" ou "local"
LOGIN_NAVEGADOR = False    # True -> login pelo navegador em vez de chave

if RODAR_TABPFN:
    if BACKEND == "api":
        chave = os.environ.get("TABPFN_TOKEN", "")
        try:
            from google.colab import userdata
            chave = userdata.get("TABPFN_TOKEN") or chave
        except Exception as e:
            print("nenhuma saved key lida:", e)
        chave = (chave or "").strip()
        os.environ["TABPFN_TOKEN"] = chave
        print(f"chave: {len(chave)} caracteres, terminando em "
              f"{chave[-4:] if chave else '(vazia)'}")

        import tabpfn_client
        if LOGIN_NAVEGADOR:
            tabpfn_client.interactive_login()
        elif chave:
            tabpfn_client.set_access_token(chave)

        try:                                  # uma chamada barata, para falhar rapido
            from tabpfn_client import UserDataClient
            UserDataClient.get_data_summary()
            print("chave aceita")
        except Exception as e:
            print(f"a API recusou a chave ({type(e).__name__}: {e})")
            print("saidas: gerar chave em platform.priorlabs.ai/account/api-keys, "
                  "ou LOGIN_NAVEGADOR = True, ou BACKEND = 'local'")
            RODAR_TABPFN = False
        if RODAR_TABPFN:
            from tabpfn_client import TabPFNClassifier
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True)
        from tabpfn import TabPFNClassifier
        try:
            import torch
            print("TabPFN local, dispositivo:",
                  "cuda" if torch.cuda.is_available() else "cpu (lento, peca GPU)")
        except Exception:
            print("TabPFN local pronto")
"""))
C.append(code("""
if RODAR_TABPFN:
    def tabpfn(**kw):
        \"\"\"balance_probabilities importa com 3,8% de positivos. Versoes diferem
        nos argumentos aceitos, entao cada um e descartado em vez de quebrar.\"\"\"
        if BACKEND == "local":
            kw.setdefault("device", "auto")
        else:
            kw.setdefault("model_path", "auto")
        for tentativa in ({"balance_probabilities": True, "random_state": RS, **kw},
                          {"random_state": RS, **kw}, kw):
            try:
                return TabPFNClassifier(**tentativa)
            except TypeError:
                continue
        return TabPFNClassifier()

    pipe_tab = Pipeline([("prep", prep(False)), ("clf", tabpfn())])
    pipe_tab.fit(Xtr, ytr)
    linha_tab, p_tab = mk.evaluate(pipe_tab, Xte, yte, "TabPFN")

    if BACKEND == "local":        # cinco idas a API custariam caro; local nao
        cv_tab = cross_val_score(Pipeline([("prep", prep(False)), ("clf", tabpfn())]),
                                 Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
        linha_tab["CV_AUC"] = cv_tab.mean()

    resultados.append(linha_tab)
    ajustados["TabPFN"] = dict(pipe=pipe_tab, prob=p_tab, params={"backend": BACKEND})
    print(f"TabPFN: teste {linha_tab['AUC_test']:.3f} "
          f"[{linha_tab['AUC_lo']:.3f}, {linha_tab['AUC_hi']:.3f}]")
else:
    print("TabPFN desligado")
"""))

C.append(md("""
---
# 5 - Resultados

Tudo medido na metade de teste, na qual nada acima foi ajustado.
"""))

C.append(md("""
## 5.1 - Tabela de metricas

`AUC_lo` e `AUC_hi` sao um intervalo bootstrap estratificado de 1.000 reamostras
sobre a AUC de teste. Dois pontos de corte sao reportados: o de Youden, e um
ponto de rastreio que segura a sensibilidade em torno de 90%, que e onde um
instrumento de triagem seria de fato operado. E onde o valor preditivo positivo
mostra o preco da prevalencia: com 15% de positivos, a maioria dos sinalizados
nao tem o desfecho.
"""))
C.append(code("""
tabela = mk.metrics_frame(resultados)
display(tabela)
"""))

C.append(md("""
## 5.2 - Curvas ROC

Uma curva por familia, so as versoes ajustadas. Curvas muito proximas com
familias tao diferentes sao a assinatura de um sinal que e essencialmente
aditivo: a arvore nao encontra interacao que a logistica nao ja capture.
"""))
C.append(code("""
curvas = {nome: (yte, d["prob"]) for nome, d in ajustados.items()}
fig_roc = mk.plot_roc(curvas, f"{DESFECHO} - tres familias")
"""))

C.append(md("""
## 5.3 - Calibracao

Discriminacao e ordenamento; calibracao e se a probabilidade prevista
corresponde a frequencia observada. Um modelo pode ordenar bem e continuar
mentindo sobre o risco absoluto, o que importa se a saida for usada para
decidir quem chamar para exame.
"""))
C.append(code("""
melhor_nome = tabela.sort_values("AUC_test", ascending=False)["model"].iloc[0]
familia = melhor_nome.split(" (")[0]
fig_cal = mk.plot_calibration(yte, ajustados[familia]["prob"],
                              f"{DESFECHO} - calibracao, {familia}")
print("melhor pela AUC de teste:", melhor_nome)
"""))

C.append(md("""
## 5.4 - Importancia por permutacao

Embaralha uma variavel por vez na metade de teste e mede quanto a AUC cai, e
soma por bloco do questionario. Permutar e nao reajustar: a pergunta e quanto o
modelo treinado depende daquela coluna.

**O que olhar.** No projeto, o bloco de acesso a servico contribui exatamente
0,000 para pressao arterial, e o de sono menos ainda. Sao resultados nulos que
devem ser reportados, nao apagados. Bloco que de repente pesa demais costuma ser
vazamento, e a primeira coisa a conferir e se algum contador deixou de ser
reconstruido.
"""))
C.append(code("""
imp, blocos_imp = mk.permutation_report(ajustados[familia]["pipe"], Xte, yte, bundle,
                                        n_repeats=5 if familia != "TabPFN" else 3,
                                        top=None if familia != "TabPFN" else 20)
display(imp.head(20))
display(blocos_imp)
fig_imp = mk.plot_importance(imp, f"{DESFECHO} - importancia por permutacao")
"""))

C.append(md("""
---
# 6 - Exportacao

Uma pasta por familia, com a linha de metrica, as tabelas de importancia, os
parametros escolhidos, as figuras e um relatorio de texto com os hashes da
construcao. No fim, um zip com tudo.
"""))
C.append(code("""
import shutil

for nome, d in ajustados.items():
    curto = {"Regressao logistica": "logreg", "Random forest": "rf",
             "TabPFN": "tabpfn"}[nome]
    linhas = tabela[tabela["model"].str.startswith(nome)]
    mk.export(SAIDA, DESFECHO, curto, bundle, linhas,
              importance=imp if nome == familia else None,
              blocks=blocos_imp if nome == familia else None,
              best_params=d["params"],
              figures=[("roc", fig_roc)] + ([("calibracao", fig_cal),
                                             ("importancia", fig_imp)]
                                            if nome == familia else []))

tabela.to_csv(os.path.join(SAIDA, f"tabela_{DESFECHO}.csv"), index=False)
z = shutil.make_archive(os.path.join(BASE, f"disciplina_{DESFECHO}"), "zip", SAIDA)
print("zip em", z)
"""))

nb = {"cells": C,
      "metadata": {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    src = c["source"]
    c["source"] = [l + "\n" for l in src[:-1]] + [src[-1]]

out = sys.argv[1] if len(sys.argv) > 1 else "pns2013_tres_modelos.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("escrito", out, len(C), "celulas")
