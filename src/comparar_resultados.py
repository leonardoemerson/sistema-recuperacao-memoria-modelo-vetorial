import matplotlib.pyplot
import pandas as pd
from configparser import ConfigParser

from nltk.metrics.scores import precision, recall, f_measure
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

config = ConfigParser()
config.read(r'.\config\AVALIA.CFG')

LEIA_DIR = config.get("avalia", 'LEIA_DIR')
LEIA_ESPERADOS = config.get("avalia", 'LEIA_ESPERADOS')

df_stem = pd.read_csv(LEIA_DIR + r"\RESULTADOS-STEMMER.csv", sep=";")
df_no_stem = pd.read_csv(LEIA_DIR + r"\RESULTADOS-NOSTEMMER.csv", sep=";")
df_esperados = pd.read_csv(LEIA_ESPERADOS, sep=";")
df_esperados = df_esperados.sort_values(["QueryNumber", "DocVote"], ascending=False)


def precision_recall_11_point(df_resultados, df_esperados):
    ls_df_scores = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]
    
        set_esperados = set(df_esperados_local["DocNumber"].values)
    
        ls_precisao = []
        ls_revocacao = []
        set_records = set()
        last_recall = 0
        for idx, row in df_resultados_local.iterrows():
            set_records.add(row["RecordNum"])
    
            recall_local = recall(set_esperados, set_records)
            if recall_local == last_recall:
                continue
    
            precision_local = precision(set_esperados, set_records)
    
            ls_precisao.append(precision_local)
            ls_revocacao.append(recall_local)
    
            if recall_local == 1:
                break
    
            last_recall = recall_local
    
        df_scores_local = pd.DataFrame({"Precisão": ls_precisao, "Revocação": ls_revocacao})
    
        ls_precisao = []
        ls_revocacao = []
        for revocacao in (np.array(range(11)) / 10):
            ls_revocacao.append(revocacao)
            ls_precisao.append(df_scores_local.loc[(df_scores_local["Revocação"] >= revocacao), "Precisão"].max())
    
        df_scores_local = pd.DataFrame({"Precisão": ls_precisao, "Revocação": ls_revocacao})
        df_scores_local["QueryNumber"] = query_number
        ls_df_scores.append(df_scores_local)

    df_scores = pd.concat(ls_df_scores)

    df_scores = df_scores.fillna(0)

    return df_scores


df_onze_pontos_nostem = precision_recall_11_point(df_no_stem, df_esperados)
df_onze_pontos_stem = precision_recall_11_point(df_stem, df_esperados)

df_onze_pontos_nostem_agrupado = df_onze_pontos_nostem.groupby("Revocação", as_index=False)["Precisão"].mean()
df_onze_pontos_nostem_agrupado["stem"] = "no stemmer"
df_onze_pontos_stem_agrupado = df_onze_pontos_stem.groupby("Revocação", as_index=False)["Precisão"].mean()
df_onze_pontos_stem_agrupado["stem"] = "stemmer"
df_onze_pontos_agrupado = pd.concat([df_onze_pontos_nostem_agrupado, df_onze_pontos_stem_agrupado])

fig = sns.lineplot(df_onze_pontos_agrupado, x="Revocação", y="Precisão", hue="stem").set_title('11 pontos de precisão e recall')
fig.figure.savefig(r".\result\diagramas\11pontos.png")

df_onze_pontos_agrupado.to_csv(r".\result\diagramas\11pontos.csv", index=False)


def f_1(df_resultados, df_esperados):
    ls_query_number = []
    ls_f_1 = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        if df_esperados_local.shape[0] >= 10:
            set_esperados = set(df_esperados_local.iloc[0:10]["DocNumber"].values)
            set_resultados = set(df_resultados_local.iloc[0:10]["RecordNum"].values)

        else:
            set_esperados = set(df_esperados_local["DocNumber"].values)
            set_resultados = set(df_resultados_local.iloc[0:len(set_esperados)]["RecordNum"].values)

        f_1 = f_measure(set_esperados, set_resultados, alpha=0.5)

        ls_query_number.append(query_number)
        ls_f_1.append(f_1)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "f_1": ls_f_1})
    return df_score


df_f_1_nostem = f_1(df_no_stem, df_esperados)
avg_f_1_nostem = df_f_1_nostem.f_1.mean()

df_f_1_stem = f_1(df_stem, df_esperados)
avg_f_1_stem = df_f_1_stem.f_1.mean()

print("f_1 do dados sem stemming: " + str(avg_f_1_nostem))
print("f_1 do dados com stemming: " + str(avg_f_1_stem))


def precision_at_n(df_resultados, df_esperados, n):
    ls_query_number = []
    ls_precision = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        if df_esperados_local.shape[0] >= n:
            set_esperados = set(df_esperados_local.iloc[0:n]["DocNumber"].values)
            set_resultados = set(df_resultados_local.iloc[0:n]["RecordNum"].values)
        else:
            set_esperados = set(df_esperados_local["DocNumber"].values)
            set_resultados = set(df_resultados_local.iloc[0:len(set_esperados)]["RecordNum"].values)

        pres = precision(set_esperados, set_resultados)

        ls_query_number.append(query_number)
        ls_precision.append(pres)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "Precision@n": ls_precision})
    return df_score


df_p_5_nostem = precision_at_n(df_no_stem, df_esperados, 5)
avg_p_5_nostem = df_p_5_nostem["Precision@n"].mean()

df_p_5_stem = precision_at_n(df_stem, df_esperados, 5)
avg_p_5_stem = df_p_5_stem["Precision@n"].mean()

print("Precision@5 do dados sem stemming: " + str(avg_p_5_nostem))
print("Precision@5 do dados com stemming: " + str(avg_p_5_stem))

df_p_10_nostem = precision_at_n(df_no_stem, df_esperados, 10)
avg_p_10_nostem = df_p_10_nostem["Precision@n"].mean()

df_p_10_stem = precision_at_n(df_stem, df_esperados, 10)
avg_p_10_stem = df_p_10_stem["Precision@n"].mean()

print("Precision@10 do dados sem stemming: " + str(avg_p_10_nostem))
print("Precision@10 do dados com stemming: " + str(avg_p_10_stem))


# histograma r-precision
def precision_at_r(df_resultados, df_esperados):
    ls_query_number = []
    ls_precision = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        set_esperados = set(df_esperados_local["DocNumber"].values)
        set_resultados = set(df_resultados_local.iloc[0:len(set_esperados)]["RecordNum"].values)

        pres = precision(set_esperados, set_resultados)

        ls_query_number.append(query_number)
        ls_precision.append(pres)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "Precision@r": ls_precision})
    return df_score


df_p_r_nostem = precision_at_r(df_no_stem, df_esperados)
df_p_r_stem = precision_at_r(df_stem, df_esperados)

df_p_r = df_p_r_nostem.merge(df_p_r_stem, on="QueryNumber", suffixes=("_nostem", "_stem"))

df_p_r["Diferença entre Precision@r com e sem stemming"] = df_p_r["Precision@r_stem"] - df_p_r["Precision@r_nostem"]
df_p_r.sort_values("QueryNumber", inplace=True)

fig = sns.barplot(df_p_r, x="QueryNumber", y="Diferença entre Precision@r com e sem stemming").set_title('R-Precision (comparativo)')
fig.figure.savefig(r".\result\diagramas\R-Precision comparativo.png")

df_p_r.to_csv(r".\result\diagramas\R-Precision comparativo.csv", index=False)


# MAP
def average_precision(df_resultados, df_esperados):
    ls_query_number = []
    ls_average_precision = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        set_esperados = set(df_esperados_local["DocNumber"].values)

        precision_at_k = []
        n_total_recuperados = 0
        n_relevantes_recuperados = 0
        for doc_num in df_resultados_local["RecordNum"].values:
            n_total_recuperados += 1
            if doc_num in set_esperados:
                n_relevantes_recuperados += 1
                precision_at_k.append(n_relevantes_recuperados / n_total_recuperados)

        avg_pres = np.mean(precision_at_k)

        ls_average_precision.append(avg_pres)
        ls_query_number.append(query_number)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "average_precision": ls_average_precision})
    return df_score


df_map_nostem = average_precision(df_no_stem, df_esperados)
avg_map_nostem = df_map_nostem["average_precision"].mean()

df_map_stem = average_precision(df_stem, df_esperados)
avg_map_stem = df_map_stem["average_precision"].mean()

print("MAP do dados sem stemming: " + str(avg_map_nostem))
print("MAP do dados com stemming: " + str(avg_map_stem))


# MRR
def reciprocal_rank(df_resultados, df_esperados, k):
    ls_query_number = []
    ls_rr = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        set_esperados = set(df_esperados_local["DocNumber"].values)

        n_total_recuperados = 0
        rr = 0
        for doc_num in df_resultados_local["RecordNum"].values:
            n_total_recuperados += 1
            if doc_num in set_esperados:
                if n_total_recuperados > k:
                    pass
                else:
                    rr = 1/n_total_recuperados
                break

        ls_rr.append(rr)
        ls_query_number.append(query_number)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "reciprocal_rank": ls_rr})
    return df_score


df_mrr_nostem = reciprocal_rank(df_no_stem, df_esperados, 10)
avg_mrr_nostem = df_mrr_nostem["reciprocal_rank"].mean()

df_mrr_stem = reciprocal_rank(df_stem, df_esperados, 10)
avg_mrr_stem = df_mrr_stem["reciprocal_rank"].mean()

print("MRR do dados sem stemming: " + str(avg_mrr_nostem))
print("MRR do dados com stemming: " + str(avg_mrr_stem))


# Discounted Cumulative Gain
def discounted_cumulative_gain(df_resultados, df_esperados, k):
    ls_query_number = []
    ls_mdcg = []
    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        set_esperados = set(df_esperados_local["DocNumber"].values)

        ls_dcg = []
        n_total_recuperados = 0
        n_relevantes_recuperados = 0
        for doc_num in df_resultados_local["RecordNum"].values:
            n_total_recuperados += 1
            if doc_num in set_esperados:
                n_relevantes_recuperados += 1
                if n_total_recuperados == 1:
                    ls_dcg.append(df_esperados[(df_esperados["DocNumber"] == doc_num)
                                               & (df_esperados["QueryNumber"] == query_number)]["DocVote"]
                                  / n_total_recuperados)
                else:
                    ls_dcg.append(df_esperados[(df_esperados["DocNumber"] == doc_num)
                                               & (df_esperados["QueryNumber"] == query_number)]["DocVote"]
                                  / np.log2(n_total_recuperados))
            if n_relevantes_recuperados == k:
                break

        mdcg = np.mean(ls_dcg)

        ls_mdcg.append(mdcg)
        ls_query_number.append(query_number)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "dcg": ls_mdcg})
    return df_score


df_dcg_nostem = discounted_cumulative_gain(df_no_stem, df_esperados, 10)
avg_dcg_nostem = df_dcg_nostem["dcg"].mean()

df_dcg_stem = discounted_cumulative_gain(df_stem, df_esperados, 10)
avg_dcg_stem = df_dcg_stem["dcg"].mean()

print("DCG do dados sem stemming: " + str(avg_mrr_nostem))
print("DCG do dados com stemming: " + str(avg_mrr_stem))


# Normalized Discounted Cumulative Gain
def norm_discounted_cumulative_gain(df_resultados, df_esperados, k):
    ls_query_number = []
    ls_midcg = []

    for query_number in df_esperados["QueryNumber"].unique():
        df_resultados_local = df_resultados[df_resultados["QueryNumber"] == query_number]
        df_esperados_local = df_esperados[df_esperados["QueryNumber"] == query_number]

        k_local = np.min((k, df_esperados_local.shape[0]))

        idcg = 0
        for i in range(k_local):
            idcg += df_esperados_local["DocVote"].values[i] / np.log2(i + 2)

        set_esperados = set(df_esperados_local["DocNumber"].values)

        ls_idcg = []
        n_total_recuperados = 0
        n_relevantes_recuperados = 0
        for doc_num in df_resultados_local["RecordNum"].values:
            n_total_recuperados += 1
            if doc_num in set_esperados:
                n_relevantes_recuperados += 1
                ls_idcg.append((df_esperados.loc[(df_esperados["DocNumber"] == doc_num)
                                                 & (df_esperados["QueryNumber"] == query_number), "DocVote"].values[0]
                                / np.log2(n_total_recuperados+1))/idcg)
            if n_relevantes_recuperados == k_local:
                break

        midcg = np.mean(ls_idcg)

        ls_midcg.append(midcg)
        ls_query_number.append(query_number)

    df_score = pd.DataFrame({"QueryNumber": ls_query_number, "dcg": ls_midcg})
    return df_score


df_idcg_nostem = norm_discounted_cumulative_gain(df_no_stem, df_esperados, 10)
avg_idcg_nostem = df_idcg_nostem["dcg"].mean()

df_idcg_stem = norm_discounted_cumulative_gain(df_stem, df_esperados, 10)
avg_idcg_stem = df_idcg_stem["dcg"].mean()

print("NDCG do dados sem stemming: " + str(avg_mrr_nostem))
print("NDCG do dados com stemming: " + str(avg_mrr_stem))
