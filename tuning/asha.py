import heapq
import json
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint

import boto3
import math
from botocore.client import Config

from tuning import util


class PriorityEntry(object):
    def __init__(self, error, params):
        self.error = error
        self.params = params

    def __lt__(self, other):
        return self.error < other.error


def run(n_workers, min_r, max_r, reduction_factor, early_stopping_rounds, cv):
    # 10 minute timeout for the lambda
    time_out = 10 * 60
    boto_config = Config(read_timeout=time_out)
    client = boto3.client('lambda', config=boto_config)

    rungs = []
    s_max = int(math.ceil(math.log(max_r / min_r, reduction_factor))) + 1

    for i in range(s_max):
        rungs.append([])

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        def get_task():
            next_config = _get_config(rungs, s_max, reduction_factor)

            return {
                'future': executor.submit(
                    client.invoke,
                    FunctionName='ASHA-xgboost-evaluation-dev-run',
                    Payload=json.dumps({
                        'body': json.dumps({
                            'params': next_config['params'],
                            'cv': cv,
                            'seed': 33,
                            'num_boost_round': int(min_r * math.pow(reduction_factor, next_config['rung'])),
                            'early_stopping_rounds': early_stopping_rounds,
                        })
                    })
                ),
                'rung': next_config['rung'],
                'params': next_config['params']
            }

        workers = []
        for i in range(n_workers):
            workers.append(get_task())

        finished = False
        reached_top = False
        num_configs = 0
        while not finished:
            finished = reached_top

            for i in range(n_workers):
                current = workers[i]
                if current['future'] is None:
                    continue

                if current['future'].done():
                    num_configs += 1

                    response = current['future'].result()
                    payload = json.loads(response['Payload'].read())
                    result = json.loads(payload['body'])

                    print(f'Rung {current["rung"]}: ', end='')
                    pprint(result)

                    heapq.heappush(rungs[current['rung']], PriorityEntry(result['error'], current['params']))

                    if reached_top:
                        # don't add anymore tasks
                        current['future'] = None
                    else:
                        workers[i] = get_task()
                else:
                    finished = False

            if len(rungs[s_max - 1]) > 1:
                reached_top = True

        print(f'Total configs evaluated: {num_configs}')

        best_config = heapq.heappop(rungs[s_max - 1])
        best_rung = 0

        for i in range(len(rungs)):
            if len(rungs[i]) > 0:
                config = heapq.heappop(rungs[i])
                if config.error < best_config.error:
                    best_rung = i
                    best_config = config

        print(f'Best rung: {best_rung}')
        return [1 - best_config.error, best_config.params]


def _get_config(rungs, s_max, reduction_factor):
    for i in range(s_max - 2, -1, -1):
        if len(rungs[i]) >= reduction_factor:
            config = heapq.heappop(rungs[i])
            return {'params': config.params, 'rung': i + 1}

    return {'params': util.get_random_params(), 'rung': 0}
