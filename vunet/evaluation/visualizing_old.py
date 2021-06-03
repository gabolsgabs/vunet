import itertools
from keras import backend as K
from keras.backend import tf
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot(data, y_labels, title):
    fig, ax = plt.subplots()
    x_labels = [title[:-1] + "_" + str(i) for i in range(data.shape[0] + 1)]
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title)
    im = ax.imshow(data)
    fig.colorbar(im)
    return


def createdf(data):
    data = np.array(data)
    df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
    df["values"] = df["values"].astype(np.float)
    # df['mean'] = df['mean'].astype(np.float)
    # df['std'] = df['std'].astype(np.float)
    return df


def createdfcrr(data):
    data = np.array(data)
    df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
    df["fix"] = df["fix"].astype(np.float)
    df["cc"] = df["cc"].astype(np.float)
    df["cd"] = df["cd"].astype(np.float)
    df["sd"] = df["sd"].astype(np.float)
    df["sc"] = df["sc"].astype(np.float)
    # df['mean'] = df['mean'].astype(np.float)
    # df['std'] = df['std'].astype(np.float)
    return df


def ms(data, c, t, md, rt, mt, addmt):
    mean = rt[t][md][mt][0]
    std = rt[t][md][mt][1]
    if addmt:
        data.append([c, t, md, mt, mean - std])
        data.append([c, t, md, mt, mean])
        data.append([c, t, md, mt, mean + std])
    else:
        data.append([c, t, md, mean - std])
        data.append([c, t, md, mean])
        data.append([c, t, md, mean + std])
    return data


def evolution2pandas(output):
    data = [["", "Percentage", "task", "model", "SAR", "SDR", "SIR"]]
    m = [
        "no_vocals_adapt_cnn_nfai_05_57-0.00448.h5",
        "no_vocals_adapt_cnn_nfa_1_106-0.00426.h5",
        "no_vocals_adapt_cnn_nfai_15_169-0.00422.h5",
        "no_vocals_adapt_cnn_nfai_3_349-0.00416.h5",
        "no_vocals_adapt_cnn_nfai_5_576-0.00406.h5",
    ]
    model = "nfa"
    i = 0
    for t in output:
        tmp = output[t]["3c"]["no_vocals_n_99-0.00434.json"]
        if t != "vocals":
            data.append(
                [i, "0%", t, "3cN", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
            )
            data.append(
                [i, "0%", t, "3cF", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
            )
        else:
            data.append(
                [i, "0%", t, "3c+vN", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
            )
            data.append(
                [i, "0%", t, "3c+vF", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
            )

        i += 1

    for v in m:
        for t in output:
            c = v.split("_")[-2]
            v = v.replace(".h5", ".json")
            if "nfai" in v:
                tmp = output[t][model + "i"][v]
            else:
                tmp = output[t][model][v]
            if t != "vocals":
                data.append(
                    [i, c + "0%", t, "3cN", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
                )
            else:
                data.append(
                    [
                        i,
                        c + "0%",
                        t,
                        "3c+vN",
                        tmp["sar"][0],
                        tmp["sdr"][0],
                        tmp["sir"][0],
                    ]
                )
            i += 1

    m = [
        "no_vocals_adapt_cnn_fai_05_58-0.00542.h5",
        "no_vocals_adapt_cnn_fai_1.h5",
        "no_vocals_adapt_cnn_fai_15_74-0.00539.h5",
        "no_vocals_adapt_cnn_fai_3.h5",
        "no_vocals_adapt_cnn_fai_5_475-0.00537.h5",
    ]
    model = "fai"
    for v in m:
        for t in output:
            v = v.replace(".h5", ".json")
            c = v.split("_")[5].replace(".json", "")
            tmp = output[t][model][v]
            if t != "vocals":
                data.append(
                    [i, c + "0%", t, "3cF", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
                )
            else:
                data.append(
                    [
                        i,
                        c + "0%",
                        t,
                        "3c+vF",
                        tmp["sar"][0],
                        tmp["sdr"][0],
                        tmp["sir"][0],
                    ]
                )
            i += 1

    m = [
        "vocals_prog01_05.h5",
        "vocals_prog01_1.h5",
        "vocals_prog01_15.h5",
        "vocals_prog01_3_23-0.00422.h5",
        "vocals_prog01_5_42-0.00406.h5",
    ]
    t = "vocals"
    model = "vocals_prog01"
    data.append([i, "0%", t, "Just_Vocals", 0, 0, 0])
    for v in m:
        v = v.replace(".h5", ".json")
        c = v.split("_")[2].replace(".json", "")
        tmp = output[t][model][v]
        data.append(
            [i, c + "0%", t, "Just_Vocals", tmp["sar"][0], tmp["sdr"][0], tmp["sir"][0]]
        )
        i += 1
    data = np.array(data)
    df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
    df["SAR"] = df["SAR"].astype(np.float)
    df["SIR"] = df["SIR"].astype(np.float)
    df["SDR"] = df["SDR"].astype(np.float)


def result2pandas_means(rt):
    # data = [['', 'task', 'model', 'metric', 'mean', 'std']]
    data = [["", "task", "model", "metric", "values"]]
    data_sar = [["", "task", "model", "values"]]
    data_sir = [["", "task", "model", "values"]]
    data_sdr = [["", "task", "model", "values"]]
    c = 0
    for t in rt:
        for md in rt[t]:
            for mt in rt[t][md]:
                data = ms(data, c, t, md, rt, mt, True)
                if "sdr" in mt:
                    data_sdr = ms(data_sdr, c, t, md, rt, mt, False)
                elif "sar" in mt:
                    data_sar = ms(data_sar, c, t, md, rt, mt, False)
                elif "sir" in mt:
                    data_sir = ms(data_sir, c, t, md, rt, mt, False)
                c += 1
    df = createdf(data)
    dfsar = createdf(data_sar)
    dfsir = createdf(data_sir)
    dfsdr = createdf(data_sdr)


def details2pandas_everthing(d):
    data = [["", "task", "model", "metric", "values"]]
    data_sar = [["", "task", "model", "values"]]
    data_sir = [["", "task", "model", "values"]]
    data_sdr = [["", "task", "model", "values"]]
    c = 0
    for md in d:
        for t in d[md]:
            for mt in d[md][t]:
                for v in d[md][t][mt]:
                    data.append([c, t, md, mt, v])
                    if "sdr" in mt:
                        data_sdr.append([c, t, md, v])
                    elif "sar" in mt:
                        data_sar.append([c, t, md, v])
                    elif "sir" in mt:
                        data_sir.append([c, t, md, v])
                    c += 1
    df = createdf(data)
    dfsar = createdf(data_sar)
    dfsir = createdf(data_sir)
    dfsdr = createdf(data_sdr)

    # MultiComp = MultiComparison(dfsdr['values'], dfsdr['model'])
    # print(MultiComp.tukeyhsd().summary())
    # https://pythonhealthcare.org/2018/04/13/55-statistics-multi-comparison-with-tukeys-test-and-the-holm-bonferroni-method/
    # ax = sns.catplot(col='metric', x="task", y="values", hue="model",  data=df, orient='v', linewidth=4, kind="boxen", aspect=.7,  order=["vocals", "drums", 'bass', 'rest'])

    return dfsar, dfsir, dfsdr, df


def details2pandas_corr(d):
    data = [["song", "task", "fix", "sc", "sd", "cd", "cc", "metric"]]
    data_sar = [["song", "task", "fix", "sc", "sd", "cd", "cc"]]
    data_sir = [["song", "task", "fix", "sc", "sd", "cd", "cc"]]
    data_sdr = [["song", "task", "fix", "sc", "sd", "cd", "cc"]]
    sar = []
    sir = []
    sdr = []
    o = []
    for md in d:
        for t in d[md]:
            for mt in d[md][t]:
                if "sdr" in mt:
                    o.append([t, md])
                    sdr.append(d[md][t][mt])
                if "sir" in mt:
                    sir.append(d[md][t][mt])
                if "sar" in mt:
                    sar.append(d[md][t][mt])
    sar = np.array(sar)
    sir = np.array(sir)
    sdr = np.array(sdr)
    o = np.array(o)

    for i in ["bass", "drums", "vocals", "rest"]:
        for c, j in enumerate(sar[np.where(o == i)[0]].T):
            data_sar.append([c, i, j[0], j[1], j[2], j[3], j[4]])
            data.append([c, i, j[0], j[1], j[2], j[3], j[4], "sar"])
        for c, j in enumerate(sir[np.where(o == i)[0]].T):
            data_sir.append([c, i, j[0], j[1], j[2], j[3], j[4]])
            data.append([c, i, j[0], j[1], j[2], j[3], j[4], "sir"])
        for c, j in enumerate(sdr[np.where(o == i)[0]].T):
            data_sdr.append([c, i, j[0], j[1], j[2], j[3], j[4]])
            data.append([c, i, j[0], j[1], j[2], j[3], j[4], "sdr"])

    df = createdfcrr(data)
    dfsar = createdfcrr(data_sar)
    dfsir = createdfcrr(data_sir)
    dfsdr = createdfcrr(data_sdr)

    # scipy.stats.pearsonr(df[df['task'] == t]['sc'], df[df['task'] == t]['fix'])
    # https://cmdlinetips.com/2018/02/how-to-subset-pandas-dataframe-based-on-values-of-a-column/
    # crr, p = scipy.stats.pearsonr(df['fix'], df['cc'])
    # sns.set_context("paper", font_scale=1.5)
    # p = np.format_float_scientific(p, precision=1)
    # crr = np.format_float_positional(crr,  precision=3)
    # g = sns.lmplot(x='fix', y='sc', data=df, hue='metric')
    # g = sns.lmplot(x='fix', y='sc', data=df, hue='task')
    # g = sns.pairplot(df, hue='Task', hue_order=['Vocals', 'Drums', 'Bass', 'Rest'], plot_kws = {'alpha': 0.8}, y_vars=df.columns[1], x_vars=df.columns[2:-1], height=2, aspect=2, dropna=False)
    # sns.set_context("talk")
    # g.fig.suptitle('Pearson correlation')
    # vu = 25
    # vd = -15
    # for i in range(4):
    #     g.axes[0,i].set_ylim((vd, vu))
    #     g.axes[0,i].set_xlim((vd, vu))

    return dfsar, dfsir, dfsdr, df


if __name__ == "__main__":
    conditions = [
        np.array(i).astype(np.float)
        for i in list(itertools.product([0, 1], repeat=4))
        if np.sum(i) <= 1
    ]

    labels_1 = ["Nothing", "Voice", "Rest", "Drums", "Bass"]

    path_base = "/u/anasynth/meseguerbrocal/models/source_separation/conditioned/"
    path_model = path_base + "musdb_split/sas_ts/sas_12-0.00383.h5"

    model = load_model(path_model, custom_objects={"tf": tf})
    model_cond = Model(
        inputs=model.inputs[1],
        outputs=[model.layers[10].output, model.layers[11].output],
    )

    n_gb = list(K.int_shape(model_cond.output[0]))[-1]

    gammas = np.empty((len(conditions), n_gb))
    betas = np.empty((len(conditions), n_gb))

    for i, c in enumerate(conditions):
        print(i, c)
        g, b = model_cond.predict(c.reshape(1, 1, -1))
        gammas[
            i,
        ] = np.squeeze(g)
        betas[
            i,
        ] = np.squeeze(b)
