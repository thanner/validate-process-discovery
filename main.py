import os
import threading

import pandas as pd
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.algo.evaluation.precision import evaluator as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.algo.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer import importer as pnml_importer

log_path = "resource/log/pre-processed"
petrinet_path = "resource/model"
results_path = "resource/results"

columns = ["name", "fitness", "precision", "f-score", "generalization", "simplicity"]


def import_log(log_name):
    return xes_importer.apply(os.path.join(log_path, log_name))


def import_petrinet(petrinet_name):
    return pnml_importer.apply(os.path.join(petrinet_path, petrinet_name))


def check_petrinet_approach_already_analyzed(process_name, petrinet_approach_name):
    dataframe_name = os.path.join(results_path, f"{process_name}.csv")
    create_csv_if_not_exists(dataframe_name)
    df = pd.read_csv(dataframe_name)
    if (df['name'] == petrinet_approach_name).any():
        raise Exception(f"{petrinet_approach_name} in {process_name} already analyzed")


def check_sound(petrinet_name, net, im, fm):
    is_sound = woflan.apply(net, im, fm, parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                                     woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                                     woflan.Parameters.RETURN_DIAGNOSTICS: False})
    if not is_sound:
        raise Exception(f"{petrinet_name} is not sound")


def calculate_fscore(fitness, precision):
    return 2 * (fitness * precision) / (fitness + precision)


def calculate_metrics(petrinet_approach_name, log, net, im, fm):
    fitness = replay_fitness_evaluator.apply(log, net, im, fm,
                                             variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)[
        'averageFitness']
    precision = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    generalization = generalization_evaluator.apply(log, net, im, fm)
    simplicity = simplicity_evaluator.apply(net)
    fscore = calculate_fscore(fitness, precision)
    results = {"name": petrinet_approach_name, "fitness": fitness, "precision": precision, "f-score": fscore,
               "generalization": generalization, "simplicity": simplicity}
    return results


def create_csv_if_not_exists(dataframe_name):
    if not os.path.isfile(dataframe_name):
        df = pd.DataFrame(columns=columns)
        df.to_csv(dataframe_name, index=False)


def save_results(process_name, results):
    dataframe_name = os.path.join(results_path, f"{process_name}.csv")
    create_csv_if_not_exists(dataframe_name)
    df = pd.read_csv(dataframe_name)
    df = df.append(results, ignore_index=True)
    df.to_csv(dataframe_name, columns=columns, index=False)


def analyze_petrinet_approach(process_name, log, petrinet_approach_name):
    try:
        check_petrinet_approach_already_analyzed(process_name, petrinet_approach_name)
        petrinet_name = os.path.join(petrinet_approach_name, f"{process_name}.pnml")
        net, im, fm = import_petrinet(petrinet_name)
        check_sound(petrinet_name, net, im, fm)
        print(f"Start calculate metrics for approach {petrinet_approach_name} and process {process_name}")
        results = calculate_metrics(petrinet_approach_name, log, net, im, fm)
        save_results(process_name, results)
    except Exception as e:
        print(str(e))


def handle_log_analysis(process_name, log):
    for petrinet_approach_name in sorted(os.listdir(petrinet_path)):
        threading.Thread(target=analyze_petrinet_approach, args=(process_name, log, petrinet_approach_name)).start()


def make_analysis():
    for log_name in sorted(os.listdir(log_path)):
        print(f"Start log analysis {log_name}")
        process_name = log_name.split(os.extsep)[0]
        log = import_log(log_name)
        handle_log_analysis(process_name, log)


make_analysis()

#########
# Análise
#########
# Fitness está dando igual
# Precision (e f-score) está diferente
# Está considerando mais elementos como unsound
