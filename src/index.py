import pandas as pd
from configparser import ConfigParser
import numpy as np
from splitcode import tf, idf
import logging
import time


class Indexador:
    def __init__(self):
        logging.info("Indexador inicializado.")
        self.time_inicio = time.time()
#1 - O indexador será configurado por um arquivo INDEX.CFG
        config = ConfigParser()
        config.read(r'.\config\INDEX.CFG')

        self.LEIA_LI = config.get("index", 'LEIA_LI')
        self.LEIA_N_D_T_MAX = config.get("index", 'LEIA_N_D_T_MAX')
        self.ESCREVA_MODELO = config.get("index", 'ESCREVA_MODELO')
        self.ESCREVA_W = config.get("index", 'ESCREVA_W')

        logging.info("Indexador - Arquivos de configuração lidos")

        self.df_n_d_t_max = pd.read_csv(self.LEIA_N_D_T_MAX, sep=";", index_col=0)
        self.N_d = self.df_n_d_t_max.shape[0]

        self.df_modelo = pd.DataFrame([])
        self.df_W = pd.DataFrame([])

        self.indexa()
        self.output()

        logging.info("Indexador - Modelo processado com sucesso em "
                     + str(time.time() - self.time_inicio) + "s")
#2 - O Indexador deverá implementar um indexador segundo o Modelo Vetorial
    def tf_idf(self, row: pd.Series) -> dict:
        if type(row["RecordList"]) == str:
            record_list = eval(row["RecordList"])
        else:
            record_list = row["RecordList"]

        w = dict()
        for record_num, n_d_t in record_list.items():
            n_d_t_max = self.df_n_d_t_max.loc[record_num, "n_d_t_max"]
            w[record_num] = tf(n_d_t, n_d_t_max) * row["idf"]

        return w

    def indexa(self):
        df_modelo = pd.read_csv(self.LEIA_LI, sep=";")

        logging.info("Indexador - Dados lidos em "
                     + str(time.time() - self.time_inicio) + "s")

        df_modelo["idf"] = df_modelo["n_d"].apply(lambda n_d: idf(n_d, self.N_d))
        df_modelo.drop("n_d", axis=1, inplace=True)

        df_modelo["w_d_t"] = df_modelo.apply(lambda row: self.tf_idf(row), axis=1)
        df_modelo.drop("RecordList", axis=1, inplace=True)

        dict_W = dict.fromkeys(self.df_n_d_t_max.index.values, 0)
        for idx, row in df_modelo.iterrows():
            for key in row["w_d_t"]:
                dict_W[key] += row["w_d_t"][key] ** 2

        df_W = pd.DataFrame({"RecordNum": dict_W.keys(), "W": dict_W.values()})
        df_W["W"] = np.sqrt(df_W["W"])

        self.df_modelo = df_modelo
        self.df_W = df_W

        logging.info("Indexador - Foram lidos " + str(df_modelo.shape[0]+1) + " records")
#3 -  O sistema deverá salvar toda essa estrutura do Modelo Vetorial para utilização posterior
    def output(self):
        self.df_modelo.to_csv(self.ESCREVA_MODELO, index=False, sep=";")
        self.df_W.to_csv(self.ESCREVA_W, index=False, sep=";")


if __name__ == "__main__":
    Indexador()
