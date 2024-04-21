from pc import ProcessadorConsultas
from gli import GeradorListaInvertida
from index import Indexador
from busca import Buscador
import logging
import datetime


def run():
    logging.basicConfig(filename=r".\result\LOG.log", encoding='utf-8', level=logging.DEBUG)

    logging.info("Início das operações - Datetime: " + str(datetime.datetime.now()))

    try:
        ProcessadorConsultas()
    except Exception as e:
        logging.error("Erro ao processar consultas: " + str(e))
        exit()
    try:
        GeradorListaInvertida()
    except Exception as e:
        logging.error("Erro ao gerar lista invertida: " + str(e))
        exit()
    try:
        Indexador()
    except Exception as e:
        logging.error("Erro ao gerar modelo: " + str(e))
        exit()
    try:
        Buscador()
    except Exception as e:
        logging.error("Erro ao realizar consultas: " + str(e))
        exit()


if __name__ == "__main__":
    run()
