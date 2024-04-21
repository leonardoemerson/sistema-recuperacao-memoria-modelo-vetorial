import time
from lxml import etree
import logging
import pandas as pd
from splitcode import text_treatment
from configparser import ConfigParser

class ProcessadorConsultas:
    def __init__(self):
        logging.info("Processador de Consultas inicializado.")
        self.start_time = time.time()

#1 - O processador de consultas deverá ler um arquivo de configuração

        config = ConfigParser()
        config.read(r'.\config\PC.CFG')

        self.LEIA = config.get("pc", 'LEIA')
        self.CONSULTAS = config.get("pc", 'CONSULTAS')
        self.ESPERADOS = config.get("pc", 'ESPERADOS')

        logging.info("Processador de Consultas - Leitura de arquivos de configuração finalizada.")

        self.df_consultas = pd.DataFrame([])
        self.df_esperados = pd.DataFrame([])

        self.processa_consultas()
        self.output()

        logging.info("Processador de Consultas - Consultas processadas com sucesso em "
                     + str(time.time() - self.time_inicio) + "s")

#2 - O processador de consultas deverá ler um arquivo em formato XML
#3 - O Processador de Consultas deverá gerar dois arquivos

    @staticmethod
    def get_votes_score(score: str) -> int:
        # Retorna a pontuação para um determinado score
        return sum([1 for i in score if i != "0"])

    def processa_consultas(self):

        root = etree.parse(self.LEIA)

        query_numbers_consultas = []
        query_numbers_esperados = []
        doc_numbers = []
        doc_votes = []
        query_text_consultas = []

        for query_elemento in root.xpath('//QUERY'):
            query_number = query_elemento.findtext('QueryNumber')
            query_text = query_elemento.findtext("QueryText")

            query_numbers_consultas.append(query_number)
            query_text_consultas.append(query_text)

            records_elemento = query_elemento.find('Records')
            if records_elemento is not None:
                for item_elem in records_elemento.findall('Item'):
                    doc_number = item_elem.text
                    score = item_elem.get('score')
                    doc_vote = self.get_votes_score(score)

                    query_numbers_esperados.append(query_number)
                    doc_numbers.append(doc_number)
                    doc_votes.append(doc_vote)

        df_consultas = pd.DataFrame({"QueryNumber": query_numbers_consultas,
                                     "QueryText": query_text_consultas})

        logging.info("Processador de Consultas - Leitura dos dados realizada em "
                     + str(time.time() - self.time_inicio) + "s")

        df_consultas["QueryNumber"] = df_consultas["QueryNumber"].str.replace(';', '')
        df_consultas["QueryNumber"] = df_consultas["QueryNumber"].astype(int)
        df_consultas["QueryText"] = text_treatment(df_consultas["QueryText"])

        df_esperados = pd.DataFrame({'QueryNumber': query_numbers_esperados,
                                     'DocNumber': doc_numbers,
                                     'DocVote': doc_votes})

        df_esperados["QueryNumber"] = df_esperados["QueryNumber"].astype(int)

        self.df_consultas = df_consultas
        self.df_esperados = df_esperados

        logging.info("Processador de Consultas - " + str(df_consultas.shape[0]+1) + " consultas foram lidas.")

    def output(self):
        self.df_consultas.to_csv(self.CONSULTAS, index=False, sep=";")
        self.df_esperados.to_csv(self.ESPERADOS, index=False, sep=";")

if __name__ == "__main__":
    ProcessadorConsultas()
