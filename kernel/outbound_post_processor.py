import pandas as pd
from sot_fbm_staffing.formulation.formulation import PostProcessor


class OutboundPostProcessor(PostProcessor):
    def __init__(self, data_holder):
        self.output: pd.DataFrame = None
        self.data = None
        self.end = None
        self.dh = data_holder
        self.y_sub_canalized = {}
        self.backlog_by_subcarrier = {}
        self.surplus_by_subcarrier = {}

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        self.data = data
        self.end = self.data['epoch'].max()
        return self.sub_canalize()

    def local_fill(self, cpt, stage, time, subca, y):
        self.y_sub_canalized[(cpt, stage, time, subca)] = min(y, self.backlog_by_subcarrier[(cpt, stage, time, subca)])
        self.surplus_by_subcarrier[(cpt, stage, time, subca)] = self.backlog_by_subcarrier[(cpt, stage, time, subca)] \
                                                                - self.y_sub_canalized[(cpt, stage, time, subca)]
        return y - self.y_sub_canalized[(cpt, stage, time, subca)]

    def sub_canalize(self):

        self.sub_canalize_initial_slot_or_stage()

        # stage > 0, t_slot > 0
        for j in range(1, self.dh.stages):
            for t in range(1, self.end):
                for c in self.dh.dict_cpts_for_epoch[t]:
                    y_cjt = int(self.data[(self.data['stage'] == j) & (self.data['epoch'] == t) &
                                          (self.data['cpt'] == c)]['y'].sum())

                    for sub in self.dh.subcarriers_for_cpt_and_epoch(c, t):
                        self.backlog_by_subcarrier[(c, j, t, sub)] = self.backlog_by_subcarrier.get((c, j, t, sub), 0)\
                                                                     + self.surplus_by_subcarrier.get((c, j, t - 1, sub), 0)\
                                                                     + self.y_sub_canalized[(c, j - 1, t, sub)]
                        y_cjt = self.local_fill(c, j, t, sub, y_cjt)

        self.save_sub_canalization()
        return self.output

    def sub_canalize_initial_slot_or_stage(self):

        base_cases = [(0, j) for j in range(self.dh.stages)] + [(t, 0) for t in range(1, self.end)]

        for t, j in base_cases:
            for c in self.dh.dict_cpts_for_epoch[t]:
                y_cjt = int(self.data[(self.data['cpt'] == c) & (self.data['stage'] == j) & (self.data['epoch'] == t)]['y'].sum())

                for sub in self.dh.subcarriers_for_cpt_and_epoch(c, t):
                    self.backlog_by_subcarrier[(c, j, t, sub)] = self.initial_backlogs_by_subcarrier(c, j, t, sub)
                    y_cjt = self.local_fill(c, j, t, sub, y_cjt)

    def initial_backlogs_by_subcarrier(self, c, j, t, sub):
        if j == 0 and t == 0:
            return self.dh.initial_subcarrier_backlog(sub, c, stage=0) \
                    + self.dh.subcarrier_demand(sub, c, slot=0)
        elif j == 0:  # t > 0
            return self.surplus_by_subcarrier.get((c, 0, t - 1, sub), 0) \
                    + self.dh.subcarrier_demand(sub, c, t)
        else:  # j > 0, t == 0
            return self.dh.initial_subcarrier_backlog(sub, c, j) \
                    + self.y_sub_canalized[(c, j - 1, 0, sub)]

    def save_sub_canalization(self):
        table = []
        for t in range(self.end):
            for j in range(self.dh.stages):
                for c in self.dh.dict_cpts_for_epoch[t]:
                    for sub in self.dh.subcarriers_for_cpt_and_epoch(c, t):
                        table.append({'epoch' : t,
                                      'stage' : j,
                                      'cpt' : c, 
                                      'carrier' : sub, 
                                      'y_cjt' : int(self.data[(self.data['cpt'] == c) & 
                                                              (self.data['stage'] == j) & 
                                                              (self.data['epoch'] == t)]['y'].sum()),
                                      'b_sub_ca' : self.backlog_by_subcarrier[(c, j, t, sub)],
                                      'y_sub_ca' : self.y_sub_canalized[(c, j, t, sub)],
                                      'exc_sub_ca' : self.surplus_by_subcarrier[(c, j, t, sub)]})
        self.output = pd.DataFrame.from_records(table)
