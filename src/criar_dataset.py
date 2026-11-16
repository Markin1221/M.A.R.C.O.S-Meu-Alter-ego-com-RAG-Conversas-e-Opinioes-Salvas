"""Monta o dataset de fine-tuning: exemplos do que te disseram e do que você respondeu."""
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

PASTA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Marcos-dados")
ENTRADA = PASTA / "mensagens_limpas.jsonl"
PASTA_SAIDA = PASTA / "finetune"

EU = "Markin🔱"
SISTEMA = "Você é o Markin conversando no WhatsApp. Responda do jeito que ele responderia."
MAX_MINUTOS = 30             # pausa maior que isso separa as conversas
MAX_TURNOS = 6               # quantas falas anteriores entram como contexto
MAX_CARACTERES_TURNO = 400
PARTE_VALIDACAO = 0.05       # conversas guardadas para medir se o treino está funcionando

APAGADAS = re.compile(r"mensagem (foi )?apagada|você apagou esta mensagem", re.IGNORECASE)
LIMPEZAS = [
    (re.compile(r"https?://\S+"), "[link]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), "[cpf]"),
    (re.compile(r"(\+?55\s?)?\(?\b\d{2}\)?\s?9\d{4}[-\s]?\d{4}\b"), "[telefone]"),
    (re.compile(r"\s*<?Mensagem editada>?", re.IGNORECASE), ""),
]


def data_hora(msg):
    return datetime.strptime(f"{msg['data']} {msg['hora']}", "%d/%m/%Y %H:%M:%S")


def limpar(texto):
    for padrao, troca in LIMPEZAS:
        texto = padrao.sub(troca, texto)
    return texto.strip()


def ler_conversas():
    """Separa as mensagens em conversas: muda o contato ou passa de 30 minutos, começa outra."""
    with open(ENTRADA, encoding="utf-8") as f:
        mensagens = [json.loads(linha) for linha in f]

    conversas, atual = [], []
    for msg in mensagens:
        if APAGADAS.search(msg["texto"]):
            continue
        msg["texto"] = limpar(msg["texto"])
        if not msg["texto"]:
            continue
        if atual and (
            msg["conversa"] != atual[-1]["conversa"]
            or (data_hora(msg) - data_hora(atual[-1])).total_seconds() > MAX_MINUTOS * 60
        ):
            conversas.append(atual)
            atual = []
        atual.append(msg)
    if atual:
        conversas.append(atual)
    return conversas


def em_turnos(conversa):
    """Junta mensagens seguidas da mesma pessoa num turno só."""
    turnos = []
    for msg in conversa:
        papel = "assistant" if msg["autor"] == EU else "user"
        if turnos and turnos[-1]["role"] == papel:
            turnos[-1]["content"] += "\n" + msg["texto"]
        else:
            turnos.append({"role": papel, "content": msg["texto"]})
    for turno in turnos:
        turno["content"] = turno["content"][:MAX_CARACTERES_TURNO]
    return turnos


def exemplos(turnos):
    """Um exemplo para cada resposta sua, com as falas anteriores como contexto."""
    for i, turno in enumerate(turnos):
        if turno["role"] != "assistant" or i == 0:
            continue
        contexto = turnos[max(0, i - MAX_TURNOS):i]
        while contexto and contexto[0]["role"] == "assistant":
            contexto = contexto[1:]   # o contexto precisa começar com a outra pessoa
        if contexto:
            yield {"messages": [{"role": "system", "content": SISTEMA}, *contexto, turno]}


def salvar(nome, lista):
    with open(PASTA_SAIDA / nome, "w", encoding="utf-8") as f:
        for exemplo in lista:
            f.write(json.dumps(exemplo, ensure_ascii=False) + "\n")


def main():
    PASTA_SAIDA.mkdir(exist_ok=True)
    conversas = ler_conversas()

    # Separa por conversa, não por exemplo, para a validação não repetir o treino.
    random.seed(42)
    random.shuffle(conversas)
    corte = int(len(conversas) * PARTE_VALIDACAO)

    treino = [e for c in conversas[corte:] for e in exemplos(em_turnos(c))]
    validacao = [e for c in conversas[:corte] for e in exemplos(em_turnos(c))]
    random.shuffle(treino)
    salvar("treino.jsonl", treino)
    salvar("validacao.jsonl", validacao)

    caracteres = sum(len(m["content"]) for e in treino for m in e["messages"])
    print(f"{len(conversas)} conversas -> {len(treino)} exemplos de treino e {len(validacao)} de validação")
    print(f"Cerca de {caracteres / 3.5 / 1e6:.1f} milhões de tokens no treino")


if __name__ == "__main__":
    main()
