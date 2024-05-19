import pandas as pd
from lxml import etree
from configparser import RawConfigParser, ConfigParser
from splitcode import text_treatment, MultiODict, count_in_list, stem
import logging
import time


class GeradorListaInvertida:
    def __init__(self):
        logging.info("Gerador de Lista Invertida inicializado.")
        self.time_inicio = time.time()
#1 - O Gerador Lista Invertida deverá ler um arquivo de configuração
        config = RawConfigParser(dict_type=MultiOrderedDict, strict=False)
        config.read(r'.\config\GLI.CFG')
        self.LEIA = config.get("gli", 'LEIA').split("\n")
        self.ESCREVA_LI = config.get("gli", 'ESCREVA_LI')
        self.ESCREVA_N_D_T_MAX = config.get("gli", 'ESCREVA_N_D_T_MAX')
        self.ESCREVA_RECORDS = config.get("gli", 'ESCREVA_RECORDS')

        config = ConfigParser()
        config.read(r'.\config\STEM.CFG')
        self.STEM = config.get("stem", 'STEM')

        logging.info("Gerador de Lista Invertida - Leitura de arquivos de configuração finalizada.")

        self.df_records = pd.DataFrame([])
        self.df_li = pd.DataFrame([])
        self.df_n_d_t_max = pd.DataFrame([])

        self.gera_lista_invertida()
        self.output()

        logging.info("Gerador de Lista Invertida - Listas Invertidas processadas em "
                     + str(time.time() - self.time_inicio) + "s")

    def gera_lista_invertida(self):
        #2 - O Gerador Lista Invertida deverá ler um conjunto de arquivos em formato XML
        list_df = []
        for arquivo in self.LEIA:
            root = etree.parse(arquivo)
            record_nums = []
            abstracts = []
#3 - Só serão usados os campos RECORDNUM, que contém identificador do texto e ABSTRACT, que contém o texto a ser classificado
            for query_elem in root.xpath('//RECORD'):
                record_num = query_elem.findtext('RECORDNUM')
                abstract = query_elem.findtext('ABSTRACT')
                if abstract is None:
                    abstract = query_elem.findtext('EXTRACT')

                record_nums.append(record_num)
                abstracts.append(abstract)


            df_local = pd.DataFrame({'RecordNum': record_nums, 'Abstract': abstracts})
            list_df.append(df_local)


        df_records = pd.concat(list_df)

        logging.info("Gerador de Lista Invertida - Dados lidos em "
                     + str(time.time() - self.time_inicio) + "s")

        df_records["Abstract"] = tratar_texto(df_records["Abstract"])
        df_records.rename({"Abstract": "Termo"}, axis=1, inplace=True)

        df_records['RecordNum'] = df_records['RecordNum'].str.strip()
        df_records['RecordNum'] = df_records['RecordNum'].astype(int)

        df_records_exploded = df_records.copy()
        df_records_exploded["Termo"] = df_records_exploded["Termo"].str.split()
        df_records_exploded = df_records_exploded.explode("Termo")

        df_records_exploded = df_records_exploded[df_records_exploded["Termo"].str.len() >= 2]

        if self.STEM == "STEMMER":
            df_records_exploded["Termo"] = stem(df_records_exploded["Termo"])

        df_n_d_t_max = df_records_exploded.groupby(["RecordNum", "Termo"], as_index=False).size() \
            .groupby(["RecordNum"], as_index=False).max("size")
        df_n_d_t_max.rename({"size": "n_d_t_max"}, axis=1, inplace=True)

        df_li = df_records_exploded.groupby(["Termo"], as_index=False).agg({"RecordNum": lambda x: x.tolist()})
        df_li["RecordNum"] = df_li["RecordNum"].apply(count_in_list)

        df_li.rename({"RecordNum": "RecordList"}, axis=1, inplace=True)

        df_li["n_d"] = df_li["RecordList"].apply(lambda x: len(x))

        self.df_records = df_records
        self.df_li = df_li
        self.df_n_d_t_max = df_n_d_t_max

        logging.info("Gerador de Lista Invertida - Foram lidos " + str(df_records.shape[0]+1) + " records")
#4 - O Gerador Lista Invertida deverá gerar um arquivo
    def output(self):

        self.df_records.to_csv(self.ESCREVA_RECORDS, index=False, sep=";")
        self.df_n_d_t_max.to_csv(self.ESCREVA_N_D_T_MAX, index=False, sep=";")
        self.df_li.to_csv(self.ESCREVA_LI, index=False, sep=";")


if __name__ == "__main__":
    GeradorListaInvertida()
