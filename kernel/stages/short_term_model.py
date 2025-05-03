import cplex
import logging

import pandas as pd

from typing import List, Tuple, Dict

import kernel.data_frames_field_names as fld_names

from kernel.data.helpers import CplexIds, ShiftHolder, TabulatedParameterHolder, DataHolder, Process
from kernel.data.shift import ShiftKind
from kernel.general_configurations import DevelopDumping, Configuration, FileNames

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ShortTermModel:
    def __init__(
            self,
            cpx: cplex.Cplex,
            dh: DataHolder,
            sh: ShiftHolder,
            process: Process,
            start: int,
            rolling_fixed_end: int,
            end: int,
            processing_max_epoch: int,
            stages: int,
            rolled_boundary_values: pd.DataFrame,
            parameters: TabulatedParameterHolder,
            previous_solution: List,
            previous_cplex_ids: CplexIds,
            stage_names: Tuple[str],
            df_transfers: pd.DataFrame,
            dc_shifts_from: Dict,
            dc_shifts_to: Dict,
            dc_workers_initial: Dict,
            shift_kinds: List,
            dc_contract_modalities_for_shift_name: Dict,
            dc_contract_types_and_modalities: Dict,
            dc_shift_names_per_contract_modality: Dict
    ) -> None:

        self.cpx = cpx
        self.dh = dh
        self.sh: ShiftHolder = sh
        self.process = process
        self.start: int = start
        self.rolling_fixed_end = rolling_fixed_end
        self.end: int = end
        self.global_max_epoch: int = processing_max_epoch
        self.stages: int = stages  # the number of stages
        self.stage_names: Tuple[str] = stage_names

        # BasicModel.previous_made_output comes into ShortTermModel.rolled_boundaries
        self.rolled_boundaries = rolled_boundary_values.processed

        self.previous_cplex_ids = previous_cplex_ids
        self.previous_cplex_solution_list = previous_solution

        self.parameters: TabulatedParameterHolder = parameters
        self.initial_backlog_cpts = [c for j, c in self.dh.initial.keys()]

        # vars
        self.v_perms_shift_assigned = {}  # stage--specialized staff at stage j and SHIFT number s
        self.xt = {}  # xt_j_s_t: stage--specialized staff at stage j, and shift s, and SLOT t

        self.v_total_perms_shift_assigned = {}  # total staff assigned at stage j and SHIFT number s (aka z per shift)
        self.zt = {}  # zt_j_s_t: total staff assigned at stage j, and shift s, and SLOT t

        # The following paragraph of vars has the 'reps evolution' model variables
        self.v_available_workers_quantity: Dict = {}  # indexed per stage and kind.name
        self.v_hired_workers_quantity: Dict = {}
        self.v_dismissed_workers_quantity: Dict = {}
        self.v_transferred_workers_quantity: Dict = {}

        self.b = {}  # b_c_j_t: backlog of cpt c at stage j and time slot t
        self.y = {}  # y_c_j_t: processed units of cpt c at stage j and time slot t
        self.stock = {}  # surplus b - y

        self.max_wrkrs_per_shift_kind = {}  # m_s: shift--ized local minmax variables

        self.df_transfers: pd.DataFrame = df_transfers
        self.dc_shifts_from: Dict = dc_shifts_from
        self.dc_shifts_to: Dict = dc_shifts_to

        self.dc_workers_initial: Dict = dc_workers_initial
        self.shift_kinds: List = shift_kinds  # an enumeration of the shift_kinds to traverse

        # TODO: check if we declared only the vars with modalities per shift_name
        self.dc_contract_modalities_for_shift_name: Dict = dc_contract_modalities_for_shift_name
        self.dc_contract_types_and_modalities: Dict = dc_contract_types_and_modalities
        self.dc_shift_names_per_contract_modality: Dict = dc_shift_names_per_contract_modality
        self.dict_initial_backlog = {}
        self.min_stage_for_cpt = self.dh.min_stages_for_cpts()

        for j in range(self.stages):
            t = 0
            cpt_list = [c for c in self.dh.dict_cpts_for_epoch[t] if c <= self.end and self.min_stage_for_cpt[c] <= j]
            for c in cpt_list:
                self.dict_initial_backlog[(c, j)] = self.dh.initial_backlogs(c, j)

    def declare_local_min_maxs(self):
        # Legacy minmax variables for current process. One per kind of shift.
        for kind in self.shift_kinds:
            m_s = self.cpx.variables.add(
                obj=[1],
                lb=[0],
                ub=[cplex.infinity],
                types=['C'],
                names=[f"m_{self.process.name[0:3]}_{kind.value}"]
            )
            self.max_wrkrs_per_shift_kind[kind] = m_s[0]

    def declare_workers(self):
        for modalities in self.dc_contract_types_and_modalities.values():
            for modality in modalities:
                self.v_perms_shift_assigned[modality] = {}
                self.v_total_perms_shift_assigned[modality] = {}  # totals

                self.xt[modality] = {}
                self.zt[modality] = {}
                perms_shift_assigned_modality = self.v_perms_shift_assigned[modality]
                total_perms_shift_assigned_modality = self.v_total_perms_shift_assigned[modality]
                xt_modality = self.xt[modality]
                zt_modality = self.zt[modality]
                for j in range(self.stages):
                    # While less performant, this is much readable.

                    # workers per shift
                    # permanent
                    perms_shift_assigned_modality[j] = {}
                    perms_shift_assigned_modality_j = perms_shift_assigned_modality[j]
                    total_perms_shift_assigned_modality[j] = {}  # totals
                    total_perms_shift_assigned_modality_j = total_perms_shift_assigned_modality[j]

                    # workers per hour
                    # permanent
                    xt_modality[j] = {}
                    xt_modality_j = xt_modality[j]
                    zt_modality[j] = {}  # totals
                    zt_modality_j = zt_modality[j]

                    for shift in self.sh.shifts_for_epoch_range(self.start, self.end):
                        max_x = self.parameters.max_workers_x(modality, j, shift)
                        max_z = self.parameters.max_workers_x(modality, j, shift)

                        xjs = self.cpx.variables.add(
                            lb=[0],
                            ub=[max_x],
                            types=['C'],
                            names=['v_perms_shift_assigned_%s_%s_%d_%d' % (self.process.name[0:3], modality, j, shift)]
                        )
                        perms_shift_assigned_modality_j[shift] = xjs[0]
                        xt_modality_j[shift] = {}
                        xt_modality_j_shift = xt_modality_j[shift]

                        zjs = self.cpx.variables.add(
                            lb=[0],
                            ub=[max_z],
                            types=['C'],
                            names=['v_total_perms_shift_assigned_%s_%s_%d_%d' % (
                                self.process.name[0:3], modality, j, shift)]
                        )
                        total_perms_shift_assigned_modality_j[shift] = zjs[0]
                        zt_modality_j[shift] = {}
                        zt_modality_j_shift = zt_modality_j[shift]

                        epochs_scope = self.sh.epochs_for_shift[shift][self.sh.epochs_for_shift[shift] <= self.end]

                        for epoch in epochs_scope:
                            xjst = self.cpx.variables.add(
                                lb=[0],
                                ub=[max_x],
                                types=['C'],
                                names=['xt_%s_%s_%d_%s_%d' % (self.process.name[0:3], modality, j, shift, epoch)]
                            )
                            xt_modality_j_shift[epoch] = xjst[0]

                            zjst = self.cpx.variables.add(
                                lb=[0],
                                ub=[max_z],
                                types=['C'],
                                names=['zt_%s_%s_%d_%s_%d' % (self.process.name[0:3], modality, j, shift, epoch)]
                            )
                            zt_modality_j_shift[epoch] = zjst[0]

    def declare_items(self):
        # Declaration of items variables
        cpts = [cpt for cpt in self.dh.cpts_for_epoch_range(self.start, self.end) if cpt <= self.end]
        for cpt in cpts:

            bc = {}  # backlog
            self.b[cpt] = bc

            yc = {}  # processed
            self.y[cpt] = yc

            for j in range(self.min_stage_for_cpt[cpt], self.stages):

                bcj = {}
                bc[j] = bcj

                ycj = {}
                yc[j] = ycj

                start_epoch = max(self.dh.min_epoch_for_cpt[cpt], self.start - 1, 0)
                end_epoch = min(cpt - 1, self.end)

                for epoch in range(start_epoch, end_epoch + 1):

                    if epoch == self.start - 1:  # Rolling horizon legacy

                        val = self.rolled_boundaries.query('epoch == @epoch and stage == @j and '
                                                           'cpt == @cpt')[['backlog', 'processed']] \
                            if not self.rolled_boundaries.empty else pd.DataFrame(columns=['backlog', 'processed'])

                        bval = int(val['backlog'].iloc[0]) if not val['backlog'].empty else 0
                        b_cjt = self.cpx.variables.add(
                            lb=[bval],
                            ub=[bval],
                            types=['C'],
                            names=[f"b_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )
                        bcj[epoch] = b_cjt[0]

                        yval = int(val['processed'].iloc[0]) if not val['processed'].empty else 0
                        y_cjt = self.cpx.variables.add(
                            lb=[yval],
                            ub=[yval],
                            types=['C'],
                            names=[f"y_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )
                        ycj[epoch] = y_cjt[0]

                    else:
                        b_cjt = self.cpx.variables.add(
                            lb=[0],
                            ub=[cplex.infinity],
                            types=['C'],
                            names=[f"b_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )
                        bcj[epoch] = b_cjt[0]

                        y_cjt = self.cpx.variables.add(
                            lb=[0],
                            ub=[cplex.infinity],
                            types=['C'],
                            names=[f"y_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )
                        ycj[epoch] = y_cjt[0]

    # declarations of transference, hiring and dismissal
    def declare_available_wrkrs_qntty(self) -> None:
        for modalities in self.dc_contract_types_and_modalities.values():
            for modality in modalities:
                self.v_available_workers_quantity[modality] = {}
                available_workers_quantity_modality = self.v_available_workers_quantity[modality]
                for stage in range(self.stages):
                    available_workers_quantity_modality[stage] = {}
                    available_workers_quantity_modality_stage = available_workers_quantity_modality[stage]
                    for sh_name in self.dc_shift_names_per_contract_modality[modality]:
                        available_workers_quantity_modality_stage[sh_name] = self.cpx.variables.add(
                            types=["C"],
                            names=[f"v_available_workers_quantity_{modality}_{self.stage_names[stage]}_{sh_name}"]
                        )[0]

    def declare_hired_wrkrs_qntty(self) -> None:
        # OBS: Alternate declaration of variables, to benchmark the dict-comprehension vs. the cycles
        self.v_hired_workers_quantity = {
            modality: {
                stage: {
                    sh_name: self.cpx.variables.add(
                        obj=[self.sh.get_cost(ShiftKind[sh_name], fld_names.HIRING_COST)],
                        types=["C"],
                        names=[f"v_hired_workers_quantity_{modality}_{self.stage_names[stage]}_{sh_name}"]
                    )[0] for sh_name in self.dc_shift_names_per_contract_modality[modality]
                } for stage in range(self.stages)
            } for modalities in self.dc_contract_types_and_modalities.values() for modality in modalities
        }

    def declare_dismissed_wrkrs_qntty(self) -> None:
        self.v_dismissed_workers_quantity = {
            modality: {
                stage: {
                    sh_name: self.cpx.variables.add(
                        obj=[self.sh.get_cost(ShiftKind[sh_name], fld_names.DISMISSAL_COST)],
                        types=["C"],
                        names=[f"v_dismissed_workers_quantity_{modality}_{self.stage_names[stage]}_{sh_name}"]
                    )[0]
                    for sh_name in self.dc_shift_names_per_contract_modality[modality]
                } for stage in range(self.stages)
            } for modalities in self.dc_contract_types_and_modalities.values() for modality in modalities
        }

    def declare_transferred_wrkrs_qntty(self) -> None:
        for modalities in self.dc_contract_types_and_modalities.values():
            for modality in modalities:
                for stage in range(self.stages):
                    for i, r in self.df_transfers[self.df_transfers[fld_names.HIRING_MODALITY] == modality].iterrows():
                        # iterrows yields pairs (index, pd.Series). Therefore we use 'r' to catch the records.
                        shift_name_from = r[fld_names.SHIFT_FROM]
                        shift_name_to = r[fld_names.SHIFT_TO]
                        self.v_transferred_workers_quantity[modality, stage, shift_name_from, shift_name_to] = \
                            self.cpx.variables.add(
                                obj=[r[fld_names.TRANSFER_COST]],
                                types=["C"],
                                names=[f"v_transferred_workers_quantity"
                                       f"_{modality}"
                                       f"_{shift_name_from}"
                                       f"_{shift_name_to}"
                                       f"_{self.stage_names[stage]}"]
                            )[0]

    def set_reps_evolution_restrictions(self) -> None:
        """
        Conceptual docstring. The names of the variables don't match.
        The following yields for each hiring modality:

            qnt_workers_available(shift_kind sk, stage j) ==

            qnt_workers_available_initial(sk,j) | (week == first week) +

            qnt_workers_hired(sk,j) -

            qnt_workers_dismissed(sk,j) +

            sum(sko, qnt_workers_transferred(sko,sk,j)) -

            sum(skd, qnt_workers_transferred(sk,skd,j)) + 

            qnt_workers_available(sk-1,j)  | (week > first week)
        """
        for j in range(self.stages):
            for shift_type, shift_names in self.sh.mapper.dc_partial_order_of_shifts.items():
                # That is: shift.TabulatedShiftMapper.dc_partial_order_of_shifts
                for n in range(len(shift_names)):
                    for modality in self.dc_contract_modalities_for_shift_name[shift_names[n]]:

                        indices = [self.v_available_workers_quantity[modality][j][shift_names[n]],
                                   self.v_hired_workers_quantity[modality][j][shift_names[n]],
                                   self.v_dismissed_workers_quantity[modality][j][shift_names[n]]]

                        values = [1, -1, 1]

                        right_hands = [0]

                        if Configuration.activate_transfer:
                            # an array of allowed destinations to reach from sn
                            tmp_dc = self.dc_shifts_from.get(modality, {})
                            shifts_from_sn = tmp_dc.get(shift_names[n], [])

                            # an array of allowed origins to reach sn
                            tmp_dc = self.dc_shifts_to.get(modality, {})
                            shifts_to_sn = tmp_dc.get(shift_names[n], [])

                            indices += [self.v_transferred_workers_quantity[modality, j, shift_names[n], dest]
                                        for dest in shifts_from_sn] + \
                                       [self.v_transferred_workers_quantity[modality, j, orig, shift_names[n]]
                                        for orig in shifts_to_sn]

                            values += [1] * len(shifts_from_sn) + [-1] * len(shifts_to_sn)

                        if n == 0:  # first week
                            right_hands = [
                                self.dc_workers_initial[modality, shift_type, j]]  # this is a constant, not a variable!

                        else:  # second week and later
                            # add the term corresponding to the previous shift_name from the same shift_type
                            indices += [self.v_available_workers_quantity[modality][j][shift_names[n - 1]]]
                            values += [-1]

                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                            senses=['E'],
                            rhs=right_hands,
                            names=[f"c_reps_evolution_{modality}_{self.stage_names[j]}_{shift_names[n]}"]
                        )

    def set_total_permanents_equalities(self):
        for modalities in self.dc_contract_types_and_modalities.values():
            for modality in modalities:
                for j in range(self.stages):
                    for shift_idx in self.sh.shifts_for_epoch_range(self.start, self.end):
                        shift_name = self.sh.get_shifts(shift_idx).kind.name
                        indices = [self.v_available_workers_quantity[modality][j][shift_name],
                                   self.v_total_perms_shift_assigned[modality][j][shift_idx]]

                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=[1, -1])],
                            senses=['E'],
                            rhs=[0],
                            names=[f"c_total_permanents_equalities"
                                   f"_{modality}"
                                   f"_{self.stage_names[j]}"
                                   f"_{shift_name}"
                                   f"_{shift_idx}"]
                        )

    def declare_and_constraint_stock_variables(self):
        cpts = [cpt for cpt in self.dh.cpts_for_epoch_range(self.start, self.end) if cpt <= self.end]
        for cpt in cpts:

            stock_dict = {}
            self.stock[cpt] = stock_dict

            for j in range(self.min_stage_for_cpt[cpt], self.stages):

                sd = {}
                stock_dict[j] = sd

                start_epoch = max(self.dh.min_epoch_for_cpt[cpt], self.start - 1, 0)
                end_epoch = min(cpt - 1, self.end)

                for epoch in range(start_epoch, end_epoch + 1):
                    if epoch == self.start - 1:  # Rolling horizon legacy
                        val = self.rolled_boundaries.query('epoch == @epoch and stage == @j and cpt == @cpt')[
                            ['backlog', 'processed']] if not self.rolled_boundaries.empty \
                            else pd.DataFrame(columns=['backlog', 'processed'])

                        bval = int(val['backlog'].iloc[0]) if not val['backlog'].empty else 0
                        yval = int(val['processed'].iloc[0]) if not val['processed'].empty else 0

                        s_cjt = self.cpx.variables.add(
                            lb=[bval - yval],
                            ub=[bval - yval],
                            types=['C'],
                            names=[f"v_stock_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )
                        sd[epoch] = s_cjt[0]
                    else:
                        s_cjt = self.cpx.variables.add(
                            obj=[0.001],
                            lb=[0],
                            ub=[cplex.infinity],
                            types=['C'],
                            names=[f"v_stock_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )
                        sd[epoch] = s_cjt[0]
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[self.stock[cpt][j][epoch],
                                                            self.b[cpt][j][epoch],
                                                            self.y[cpt][j][epoch]],
                                                       val=[1, -1, 1])],
                            senses=['E'],
                            rhs=[0],
                            names=[f"c_stock_def_{self.process.name[0:3]}_{cpt}_{j}_{epoch}"]
                        )

    def set_optimal_permanent_reps_values(self) -> None:
        # Here we set and fix the values of perm reps found in the previous run.
        # This method, and the variables involved, should be in ShortTermFormulation too.
        list_indices_fix_perms = []
        list_indices_fix_total_perms = []
        list_indices_fix_hr_perms = []
        list_indices_fix_hr_total_perms = []

        for modalities in self.dc_contract_types_and_modalities.values():
            for modality in modalities:
                for shift in self.sh.shifts_for_epoch_range(self.start, self.end):
                    epochs_scope = self.sh.epochs_for_shift[shift][self.sh.epochs_for_shift[shift] <= self.end]

                    for j in range(self.stages):

                        opt_val_x = self.previous_cplex_solution_list[
                            self.previous_cplex_ids.dc_x_fixed[modality][j][shift]]

                        opt_val_z = self.previous_cplex_solution_list[
                            self.previous_cplex_ids.dc_z_fixed[modality][j][shift]]

                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[self.v_perms_shift_assigned[modality][j][shift]], val=[1])],
                            senses=['E'],
                            rhs=[opt_val_x],
                            names=[f"c_fix_perms_{self.process.name[0:3]}_{modality}_{j}_{shift}"]
                        )
                        list_indices_fix_perms.append({
                            "process": self.process.name[0:3],
                            "modality": modality,
                            "stage": j,
                            "shift": shift
                        })

                        self.cpx.linear_constraints.add(
                            lin_expr=[
                                cplex.SparsePair(ind=[self.v_total_perms_shift_assigned[modality][j][shift]], val=[1])],
                            senses=['E'],
                            rhs=[opt_val_z],
                            names=[f"c_fix_total_perms_{self.process.name[0:3]}_{modality}_{j}_{shift}"]
                        )
                        list_indices_fix_total_perms.append({
                            "process": self.process.name[0:3],
                            "modality": modality,
                            "stage": j,
                            "shift": shift
                        })

                        for epoch in epochs_scope:
                            opt_val_x = self.previous_cplex_solution_list[
                                self.previous_cplex_ids.dc_xt_fixed[modality][j][shift][epoch]]

                            opt_val_z = self.previous_cplex_solution_list[
                                self.previous_cplex_ids.dc_zt_fixed[modality][j][shift][epoch]]

                            self.cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(ind=[self.xt[modality][j][shift][epoch]], val=[1])],
                                senses=['E'],
                                rhs=[opt_val_x],
                                names=[f"c_fix_hr_perms_{self.process.name[0:3]}_{modality}_{j}_{shift}_{epoch}"]
                            )

                            list_indices_fix_hr_perms.append({
                                "process": self.process.name[0:3],
                                "modality": modality,
                                "stage": j,
                                "shift": shift,
                                "epoch": epoch
                            })

                            self.cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(ind=[self.zt[modality][j][shift][epoch]], val=[1])],
                                senses=['E'],
                                rhs=[opt_val_z],
                                names=[f"c_fix_hr_total_perms_{self.process.name[0:3]}_{modality}_{j}_{shift}_{epoch}"]
                            )

                            list_indices_fix_hr_total_perms.append({
                                "process": self.process.name[0:3],
                                "modality": modality,
                                "stage": j,
                                "shift": shift,
                                "epoch": epoch
                            })

    def set_stock_objective_constraints(self):
        # This is a space to experiment with objective functions to anticipate backlog
        pass

    def set_available_reps_natural_constraint(self):
        for modalities in self.dc_contract_types_and_modalities.values():
            for modality in modalities:
                for stage in range(self.stages):
                    for shift in self.sh.shifts_for_epoch_range(self.start, self.end):
                        kind_name = self.sh.get_shifts(shift).kind.name
                        indices = [self.v_perms_shift_assigned[modality][stage][shift],
                                   self.v_available_workers_quantity[modality][stage][kind_name]]
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=[1, -1])],
                            senses=['L'],
                            rhs=[0],
                            names=[f"c_available_reps_natural_constraint_"
                                   f"{self.process.name[0:3]}_"
                                   f"{modality}_"
                                   f"{self.stage_names[stage]}_"
                                   f"{shift}"]
                        )

    def set_inner_constraints(self):
        # TODO optimize this function
        # unified backlog definition, time t >= 0, stage j >= 0
        t_range = range(max(0, self.start - 1), self.end + 1)
        for j in range(self.stages):
            for t in t_range:
                # cpt_list == [ cpts alive in t,j ]
                try:
                    cpt_list = [c for c in self.dh.dict_cpts_for_epoch[t]
                                if c <= self.end and self.min_stage_for_cpt[c] <= j]

                    for c in cpt_list:
                        if t == 0 and j == 0:
    
                            # b_c_0_0 == demand_{t=0, c} + initial_demand_{j=0, c}
    
                            indices = [self.b[c][j][t]]
                            values = [1]
                            right_hand = self.dh.demand.get((t, c), 0) + self.dict_initial_backlog[(c, j)]
    
                        elif t == 0 and j > 0:
    
                            # b_c_j_0 - y_c_{j-1}_0 = initial_demand_{c, j}
    
                            if self.y[c].get(j - 1):
                                indices = [self.b[c][j][t], self.y[c][j - 1][0]]
                                values = [1, -1]
    
                            else:
                                indices = [self.b[c][j][t]]
                                values = [1]
    
                            right_hand = self.dict_initial_backlog[(c, j)]
    
                        elif t > 0 and j == 0:
    
                            # b_c_0_t - b_c_0_{t-1} + y_c_0_{t-1} = demand_{t, c}
    
                            if self.b[c][j].get(t - 1):
                                indices = [self.b[c][j][t], self.b[c][0][t - 1], self.y[c][0][t - 1]]
                                values = [1, -1, 1]
    
                            else:
                                indices = [self.b[c][0][t]]
                                values = [1]
    
                            right_hand = self.dh.demand.get((t, c), 0)
    
                        else:  # t > 0 and j > 0
    
                            # b_c_j_t - b_c_j_{t-1} + y_c_j_{t-1} - y_c_{j-1}_t = 0
    
                            if self.b[c][j].get(t - 1):
                                indices = [self.b[c][j][t], self.b[c][j][t - 1], self.y[c][j][t - 1]]
                                values = [1, -1, 1]
    
                            else:
                                indices = [self.b[c][j][t]]
                                values = [1]
    
                            if self.min_stage_for_cpt[c] <= j - 1:
                                indices += [self.y[c][j - 1][t]]
                                values += [-1]
    
                            right_hand = 0
    
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                            senses=['E'],
                            rhs=[right_hand],
                            names=[f"c_backlog_{self.process.name[0:3]}_{int(c)}_{self.stage_names[j]}_{int(t)}"]
                        )

                except KeyError:
                    logger.error("Could not set backlog constraints. Check the range of epochs.\n")

        ####################################################################################################
        # TODO: following workaround should be removed once the issue is solved
        # OBS: as a solution for the ghosts, instead of the present workaround, try putting this attribute in the
        #      constructor:
        #
        #               self.initial_backlog_stage_cpts = self.dh.initial.keys()
        #
        # to distinguish by stage.
        ####################################################################################################

        # Workaround: fix ghost processing for stages > 0 limiting the processing by the input backlog
        cpts = [cpt for cpt in self.dh.cpts_for_epoch_range(self.start, self.end) if 1 == self.min_stage_for_cpt[cpt]]
        for cpt in cpts:
            for j in range(1, self.stages):
                start_epoch = max(self.dh.min_epoch_for_cpt[cpt], self.start - 1, 0)
                end_epoch = min(cpt - 1, self.end)

                p_backlog_initial = self.dh.initial_backlogs(cpt, j)
                p_backlog_planned = sum(self.dh.demand.get((t, cpt), 0) for t in (start_epoch, end_epoch + 1))

                p_backlog = p_backlog_initial + p_backlog_planned

                indices = [self.y[cpt][j][t] for t in range(start_epoch, end_epoch + 1)]

                if len(indices):
                    values = [1] * len(indices)

                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                        senses=['L'],
                        rhs=[p_backlog],
                        names=['c_max_processing_%s_%d_0_%d' % (self.process.name[0:3], cpt, j)]
                    )

        # produced cannot exceed backlogs
        for t in range(self.start, self.end + 1):
            cpts = [c for c in self.dh.dict_cpts_for_epoch[t] if c <= self.end]
            for j in range(self.stages):
                for cpt in [c for c in cpts if j >= self.min_stage_for_cpt[c]]:
                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[self.y[cpt][j][t], self.b[cpt][j][t]], val=[1, -1])],
                        senses=['L'],
                        rhs=[0],
                        names=[f'c_ylim_{self.process.name[0:3]}_{int(cpt)}_{j}_{t}']
                    )

        for j in range(self.stages):
            for s in self.sh.shifts_for_epoch_range(self.start, self.end):
                epochs_scope = self.sh.epochs_for_shift[s][self.sh.epochs_for_shift[s] <= self.end]

                for t in epochs_scope:
                    # TODO: documentr restr
                    for m in self.dc_contract_types_and_modalities:
                        for modality in self.dc_contract_types_and_modalities[m]:
                            self.cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(
                                    ind=[self.xt[modality][j][s][t], self.v_perms_shift_assigned[modality][j][s]],
                                    val=[1, -self.parameters.presence_rate(modality, t, s) *
                                         self.parameters.absence_rate(modality, t, s)])],
                                senses=['L'],
                                rhs=[0],
                                names=[f"c_perms_presenteeism_{self.process.name[0:3]}_{modality}_{j}_{s}_{t}"]
                            )
                            # TODO: documentr restr
                            self.cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(
                                    ind=[self.zt[modality][j][s][t], self.v_total_perms_shift_assigned[modality][j][s]],
                                    val=[1, -1])],
                                senses=['E'],
                                rhs=[0],
                                names=[f"c_z_totals_per_hour_{self.process.name[0:3]}_{modality}_{j}_{s}_{t}"]
                            )

        # surplus equal to zero, to conform to CPTs
        for j in range(self.stages):
            for c in filter(lambda _: _ <= self.global_max_epoch + 1 and self.min_stage_for_cpt[_] <= j,
                            self.b.keys()):
                # TODO: Document restr
                self.cpx.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[self.b[c][j][c - 1], self.y[c][j][c - 1]], val=[1, -1])],
                    senses=['E'],
                    rhs=[0],
                    names=[f"c_zero_surplus_{self.process.name[0:3]}_{c}_{j}"]
                )

    def bound_backlogs_by_above(self):
        self.bound_backlogs(self.parameters.backlogs_upper_bounds, 'L', 'c_backlog_upbound')

    def bound_backlogs_by_below(self):
        self.bound_backlogs(self.parameters.backlogs_lower_bounds, 'G', 'c_backlog_lwbound')

    def bound_backlogs(self, func, sense_char, constraint_name):
        for t in range(self.start, self.end + 1):
            for j in range(1, self.stages):
                # In case there is no shift for this epoch, get an empty list
                s = self.sh.shifts_for_epoch.get(t, [])

                for shift in s:
                    bound = func(j, shift)

                    if bound:
                        cpts = [c for c in self.dh.dict_cpts_for_epoch[t] if
                                c <= self.end and self.min_stage_for_cpt[c] <= j]

                        indices = [self.b[c][j][t] for c in cpts]
                        values = [1] * len(indices)
                        # TODO: document restriction
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                            senses=[sense_char],
                            rhs=[bound],
                            names=[f"{constraint_name}_{self.process.name[0:3]}_{j}_{shift}_{t}"]
                        )
