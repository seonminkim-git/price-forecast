import pickle
import pandas as pd


class load_data():

    def __init__(self, datadir):
        with open(datadir + 'data.pkl', 'rb') as file:
            data = pickle.load(file)

        with open(datadir + 'datainfo.pkl', 'rb') as file:
            datainfo = pickle.load(file)

        # temp
        print('(temp) removed file:', datainfo[9])
        del data[9]
        del datainfo[9]

        idx = data[0].index
        for i, d in enumerate(data):
            tmp = d.loc[idx].copy()
            tmp.columns = [datainfo[i].split('.')[0] + '-' + c for c in tmp.columns]
            data[i] = tmp

        data_all = pd.concat(data, axis=1).copy()
        data_all.index = data_all.index.to_timestamp()

        self.data = data_all
        self.info = datainfo

    def select_data(self, input_type=0):
        x = self.data
        idx2 = [a1 or a2 or a3 for a1, a2, a3 in
                zip(['철강원자재 가격' in c for c in x.columns], ['철강생산량-철근' in c for c in x.columns],
                    ['품목별 수출액, 수입액-철및강-수입액' in c for c in x.columns])]
        idx3 = [a1 or a2 or a3 or a4 for a1, a2, a3, a4 in
                zip(idx2, ['철강생산량' in c for c in x.columns], ['원유 가격' in c for c in x.columns],
                    ['원달러 환율' in c for c in x.columns])]

        if input_type in [2, '공급_small']:
            self.data = x[x.columns[idx2]]
        elif input_type in [3, '공급_large']:
            self.data = x[x.columns[idx3]]
        elif input_type in [0, '전부 다', 1, '단변량']:
            pass
        else:
            print('ntype error')

        return self.data


