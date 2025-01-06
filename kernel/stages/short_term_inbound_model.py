import cplex

from sot_fbm_staffing.stages.short_term_model import ShortTermModel


class ShortTermInboundModel(ShortTermModel):
    def set_inner_constraints(self):
        ShortTermModel.set_inner_constraints(self)
        self.receive_in_real_time()
        self.bound_backlogs_by_above()

    def receive_in_real_time(self):
        for t in range(self.start, self.end + 1):
            cpts = [c for c in self.dh.dict_cpts_for_epoch[t] if c <= self.end and self.min_stage_for_cpt[c] == 0]
            indices_b = [self.b[cpt][0][t] for cpt in cpts]
            indices_y = [self.y[cpt][0][t] for cpt in cpts]
            indices = indices_b + indices_y

            if indices:
                values = [1] * len(indices_b) + [-1] * len(indices_y)
                self.cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=indices_b + indices_y, val=values)],
                    senses=['E'],
                    rhs=[0],
                    names=["c_receive_in_real_time_%d" % t]
                )
