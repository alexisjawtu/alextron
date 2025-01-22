from kernel.stages.short_term_model import ShortTermModel


class ShortTermOutboundModel(ShortTermModel):      
    def set_inner_constraints(self):
        ShortTermModel.set_inner_constraints(self)
        self.bound_backlogs_by_above()
        self.bound_backlogs_by_below()
