import cplex
import logging

import numpy as np
import pandas as pd

from typing import List, Tuple

import kernel.util.readers as readers
import kernel.data_frames_field_names as fld_names

from datetime import datetime
from kernel.data.helpers import Process
from kernel.formulation.model import BasicModel
from kernel.stages.short_term_inbound_model import ShortTermInboundModel
from kernel.stages.short_term_outbound_model import ShortTermOutboundModel
from kernel.general_configurations import DevelopDumping, Configuration, FileNames

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ShortTermFormulation(BasicModel):
    def define(self) -> None:
        self.cpx.objective.set_sense(self.cpx.objective.sense.minimize)

        self.inbound_stages = 2
        self.outbound_stages = 2

        self.set_polyvalence_parameters()
        self.set_shift_numbers()
        self.set_workers_initials()
        self.set_costs_dfs()

        self.set_contract_types_and_modalities()
        self.set_unitary_contract_modality_to_shift_names_relations()  # This is temporary!

        if Configuration.activate_transfer:
            self.set_shifts_from_to()

        STAGES_NAMES = ("receiving", "checkin")  # generalize this in an elegant way and in a nice place.
        self.process_stages[Process.INBOUND] = ShortTermInboundModel(
            self.cpx,
            self.data_holders[Process.INBOUND],
            self.shift_holder, Process.INBOUND,
            self.start, self.fixed_end, self.end,
            self.max_epoch, self.inbound_stages,
            self.previous_made_output,
            self.params[Process.INBOUND],
            self.previous_cplex_solution,
            self.previous_cplex_ids.get(Process.INBOUND),
            STAGES_NAMES,
            self.df_transfers,
            self.dc_shifts_from,
            self.dc_shifts_to,
            self.dc_workers_initial_inbound,
            self.shift_kinds,
            self.dc_contract_modalities_for_shift_name,
            self.dc_contract_types_and_modalities,
            self.dc_shift_names_per_contract_modality
        )

        STAGES_NAMES = ("picking", "packing")
        self.process_stages[Process.OUTBOUND] = ShortTermOutboundModel(
            self.cpx,
            self.data_holders[Process.OUTBOUND],
            self.shift_holder, Process.OUTBOUND,
            self.start, self.fixed_end, self.end,
            self.max_epoch, self.outbound_stages,
            self.previous_made_output,
            self.params[Process.OUTBOUND],
            self.previous_cplex_solution,
            self.previous_cplex_ids.get(Process.OUTBOUND),
            STAGES_NAMES,
            self.df_transfers,
            self.dc_shifts_from,
            self.dc_shifts_to,
            self.dc_workers_initial_outbound,
            self.shift_kinds,
            self.dc_contract_modalities_for_shift_name,
            self.dc_contract_types_and_modalities,
            self.dc_shift_names_per_contract_modality
        )

        self.build_origin_destination_for_polyvalence()

        self.list_modality = [modality for m in self.dc_contract_types_and_modalities
                              for modality in self.dc_contract_types_and_modalities[m]]

        # TODO: vectorize stuff like this cycle
        for process, stm in iter(self.process_stages.items()):
            presence_rate_p = {}
            absence_rate_p = {}
            max_workers_w_p = {}
            stage_weights_w_p = {}
            stage_weights_x_p = {}
            self.dict_max_workers_w[process] = max_workers_w_p
            self.dict_presence_rate[process] = presence_rate_p
            self.dict_absence_rate[process] = absence_rate_p
            self.dict_stage_weights_w[process] = stage_weights_w_p
            self.dict_stage_weights_x[process] = stage_weights_x_p
            for modality in self.list_modality:
                max_workers_w_m = {}
                presence_rate_m = {}
                absence_rate_m = {}
                stage_weights_w_m = {}
                stage_weights_x_m = {}
                max_workers_w_p[modality] = max_workers_w_m
                presence_rate_p[modality] = presence_rate_m
                absence_rate_p[modality] = absence_rate_m
                stage_weights_w_p[modality] = stage_weights_w_m
                stage_weights_x_p[modality] = stage_weights_x_m
                for shift in self.shift_numbers:
                    max_workers_w_ms = {}
                    max_workers_w_m[shift] = max_workers_w_ms
                    for from_to in self.orig_dest:
                        max_workers_w_ms[from_to] = self.process_stages[process]. \
                            parameters.max_workers_w(modality, from_to, shift)

                for s in self.shift_holder.shifts_for_epoch_range(self.start, self.end):
                    presence_rate_ms = {}
                    absence_rate_ms = {}
                    stage_weights_w_ms = {}
                    stage_weights_x_ms = {}
                    presence_rate_m[s] = presence_rate_ms
                    absence_rate_m[s] = absence_rate_ms
                    stage_weights_w_m[s] = stage_weights_w_ms
                    stage_weights_x_m[s] = stage_weights_x_ms
                    for from_to in self.orig_dest:
                        stage_weights_w_ms[from_to] = self.params[process].stage_weights_w(modality, from_to, s)
                    for j in range(stm.stages):
                        stage_weights_x_ms[j] = self.params[process].stage_weights_x(modality, j, s)

                    epochs_scope = self.shift_holder.epochs_for_shift[s][
                        self.shift_holder.epochs_for_shift[s] <= self.end]
                    for t in epochs_scope:
                        presence_rate_ms[t] = self.params[stm.process].presence_rate(modality, t, s)
                        absence_rate_ms[t] = self.params[stm.process].absence_rate(modality, t, s)

    def set_cplex(self, cpx: cplex.Cplex) -> None:
        self.cpx = cpx

    def set_workers_initials(self) -> None:
        self.dc_workers_initial_inbound, self.dc_workers_initial_outbound = readers.read_workers_initial()

    def set_costs_dfs(self) -> None:
        self.shift_holder.df_workers_costs = readers.read_workers_costs()

    def set_contract_modalities_to_shift_names_relations(self) -> None:
        """
            Outputs are these correspondences:

            {
                "AFT0W1": ["Permanent", "Day_Laborer"],
                "AFT0W1_T": ["Temporary"]
            }

            and

            {
                "Permanent": ["AFT0W1"],
                "Day_Laborer": ["AFT0W1"],
                "Temporary": ["AFT0W1_T"]
            }
        """
        self.dc_contract_modalities_for_shift_name, self.dc_shift_names_per_contract_modality = \
            readers.read_shift_contract_modality()

    def set_unitary_contract_modality_to_shift_names_relations(self) -> None:
        """
        This is temporary, until the operation enables "shift_contract_modality.csv"

		The outputs are the following correspondences:
		
            {
                "MOR2W3": ["permanent"],
                "AFT0W1": ["permanent"],
                "AFT0W1_T": ["permanent"],
                ...
            }

            and

            {
                "permanent" : ["MOR2W3", "AFT0W1", "AFT0W1_T", ... ]
            }
        """
        pd_contract_modalities = readers.auxiliary_standard_read("contract_modality_type.csv")
        arr_modalities = pd_contract_modalities["contract_modality"].unique()
        shift_names = [k.name for k in self.shift_kinds]

        self.dc_contract_modalities_for_shift_name = {key: arr_modalities.tolist() for key in shift_names}

        self.dc_shift_names_per_contract_modality = {key: shift_names.copy() for key in arr_modalities}

    def set_contract_types_and_modalities(self) -> None:
        """
        	Permanent may be 
        		Internal_Permanent (regular employees of the company) or
        		DHL_Permanent (permanents which are brought by DHL.)
        	
        	Day_Laborer means hired with the flexibility of a day or certain days, and similarly 
        	this category may contain
        		Internal_Day_Laborer
        		SOME_COMPANY_NAME_Day_Laborer.

        	Temporary means hired for some fixed span of time, say two months.
            	Internal_Temporary (temporary worker directly hired by the current company).
            	
            {
                "Day_Laborer": ["Internal_Day_Laborer"],
                "Permanent": ["Internal_Permanent", "DHL_Permanent"],
                "Temporary": ["Internal_Temporary"]
             }
        """
        self.dc_contract_types_and_modalities = readers.read_contract_modality_type()

    def set_shifts_from_to(self) -> None:
        """
        Example input:

            shift_name_origin, shift_name_destination, cost, contract_modality
            AFTERNOON0W1     , MORNING0W1            , 10  , Internal_Day_Laborer
            AFTERNOON0W1     , NIGHT0W1              , 8   , Internal_Permanent
            AFTERNOON0W1     , NIGHT0W1              , 8   , Internal_Day_Laborer
            AFTERNOON0W1     , NIGHT0W1              , 8   , DHL_Temporary
            NIGHT6W2         , AFTERNOON1W2          , 7   , DHL_Permanent
            MORNING0W2       , NIGHT5W2              , 6   , DHL_Temporary
            NIGHT0W1         , MORNING1W1            , 11  , DHL_Temporary

        Then, part of dc_shifts_from is:

        {
            "Internal_Day_Laborer": {
                            "AFTERNOON0W1" : ["MORNING0W1", "NIGHT0W1"],
                            ...
                        }
            ...
        }

        dc_shifts_from is the dict

            { contract_modality: { sn: [sn_1, .., sn_k] } }

        such that we make transferences sn --> sn_i, for 1 <= i <= k.

        dc_shifts_to is the dict

            { contract_modality: { sn: [sn_1, .., sn_r] } }

        such that we make transferences sn_i --> sn, for 1 <= i <= r.
        """

        self.df_transfers, self.dc_shifts_from, self.dc_shifts_to = readers.read_transfers()

    def set_shift_numbers(self) -> None:
        self.shift_numbers = self.shift_holder.shifts_for_epoch_range(self.start, self.end)

    def set_polyvalence_parameters(self) -> None:
        self.polyvalence_parameters = readers.read_polyvalence_parameters()

    def build_origin_destination_for_polyvalence(self) -> List[Tuple[int]]:
        # First we list the fulfillment center stages, paired as (<<inbound>>, <<receiving>>), etc.
        fc_stages = [(process.value, j) for process, stm in iter(self.process_stages.items())
                     for j in range(stm.stages)]
        # now we list tuples containing
        #   (process_of_origin, stage_of_origin, current_working_process, current_working_stage)
        self.orig_dest = [tuple(np.array((origin_pair, destination_pair)).flatten()) for origin_pair in fc_stages
                          for destination_pair in fc_stages if destination_pair != origin_pair]

    def declare_polys_shift_assigned(self) -> None:
        """
        Declaration of real polyvalents is such that it holds the following pseudo code indexation:

            polyvalents_shift[hiring_modality][
                              process_of_origin,
                              stage_of_origin,
                              current_working_process,
                              current_working_stage][shift]

            Example:

                polyvalents_shift[m][0,0,1,1][s]

                means Inbound receiver doing Outbound packing during shift s.

        """
        self.v_polys_shift_assigned = {
            modality: {from_to: {shift: self.cpx.variables.add(
                ub=[self.dict_max_workers_w[Process(from_to[2])][modality][shift][from_to]],
                types=["C"],
                names=[f"v_polys_shift_assigned"
                       f"_{modality}"
                       f"_{'_'.join(map(str, from_to))}"
                       f"_{shift}"]
            )[0] for shift in self.shift_numbers}
                       for from_to in self.orig_dest
                       } for modality in self.list_modality
        }

    def declare_polys_hour_assigned(self) -> None:
        for m in self.list_modality:
            self.wt[m] = {}
            for from_to in self.orig_dest:
                self.wt[m][from_to] = {}
                process_enum = Process(from_to[2])
                for s in self.shift_numbers:
                    max_w = self.dict_max_workers_w[process_enum][m][s][from_to]

                    self.wt[m][from_to][s] = {}
                    epochs_scope = self.shift_holder.epochs_for_shift[s][
                        self.shift_holder.epochs_for_shift[s] <= self.end]

                    for t in epochs_scope:
                        wjst = self.cpx.variables.add(
                            obj=[Configuration.cost_polyvalents],
                            ub=[max_w],
                            types=["C"],
                            names=["wt_%s_%s_%d_%d" % (m, "_".join(map(str, from_to)), s, t)]
                        )
                        self.wt[m][from_to][s][t] = wjst[0]

    def declare_extras_per_hour(self) -> None:
        for stm in self.process_stages.values():
            process_name = stm.process.name
            self.v_hourly_workers[stm.process.value] = {}
            for j in range(stm.stages):
                self.v_hourly_workers[stm.process.value][j] = {}
                for t in range(stm.start, stm.end + 1):
                    hpjt = self.cpx.variables.add(
                        obj=[self.params[stm.process].get_cost_hourly_workers()],
                        types=["C"],
                        names=["v_hourly_workers_%s_%d_%d" % (process_name[0:3], j, t)]
                    )
                    self.v_hourly_workers[stm.process.value][j][t] = hpjt[0]

    def set_optimal_polyvalent_values(self) -> None:
        # Here we set and fix the values of polyvalence found in the previous run.
        for modality in self.list_modality:
            for from_to in self.orig_dest:
                for shift in self.shift_numbers:
                    epochs_scope = self.shift_holder.epochs_for_shift[shift][
                        self.shift_holder.epochs_for_shift[shift] <= self.end]

                    # TODO: make self.previous_cplex_ids call cleaner when refactoring all the variables
                    # into ShortTermFormulation
                    opt_val_w = self.previous_cplex_solution[
                        self.previous_cplex_ids[list(self.previous_cplex_ids.keys())[0]].dc_w_fixed[modality][from_to][
                            shift]]

                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[self.v_polys_shift_assigned[modality][from_to][shift]],
                                                   val=[1])],
                        senses=["E"],
                        rhs=[opt_val_w],
                        names=[f"c_fix_shift_polys_{modality}_{'_'.join(map(str, from_to))}_{shift}"]
                    )

                    for epoch in epochs_scope:
                        opt_val_w = self.previous_cplex_solution[
                            self.previous_cplex_ids[list(self.previous_cplex_ids.keys())[0]].dc_wt_fixed[modality][
                                from_to][shift][epoch]]

                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[self.wt[modality][from_to][shift][epoch]], val=[1])],
                            senses=["E"],
                            rhs=[opt_val_w],
                            names=[f"c_fix_hr_polys_{modality}_{'_'.join(map(str, from_to))}_{shift}_{epoch}"]
                        )
        
    def set_optimal_hourly_constraint(self) -> None:
        # Here we set and fix the values of polyvalence found in the previous run.
        for process, stm in iter(self.process_stages.items()):
            for j in range(stm.stages):
                for t in range(stm.start, stm.end + 1):
                    opt_val_h = self.previous_cplex_solution[
                        self.previous_cplex_ids[process].dc_hourly[process.value][j][t]
                    ]

                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[self.v_hourly_workers[stm.process.value][j][t]],
                                                   val=[1])],
                        senses=["E"],
                        rhs=[opt_val_h],
                        names=[f"c_fix_hourlies_{process.name[0:3]}_{j}_{t}"]
                    )

    def declare_global_minmaxs(self) -> None:
        # minmax variables. One per kind of shift and per stage.

        # TODO: fix the name, this is no more "global"

        """
        Example:

            self.max_wrkrs_per_shift_kind_global[some_modality] =
            {

                     0:  {  # <-- process 0!

                            0: {<ShiftKind.MORNING0W1: 0>: 120642, <ShiftKind.MORNING1W1: 1>: 120643, ...},
                            1: {<ShiftKind.MORNING0W1: 0>: 120686, <ShiftKind.MORNING1W1: 1>: 120687, ...}

                         },

                     1:  {  # <-- process 1!

                            0: {<ShiftKind.MORNING0W1: 0>: 120730, <ShiftKind.MORNING1W1: 1>: 120731, ...},
                            1: {<ShiftKind.MORNING0W1: 0>: 120774, <ShiftKind.MORNING1W1: 1>: 120775, ...}

                         }
            }

        """

        for modality in self.list_modality:
            self.max_wrkrs_per_shift_kind_global[modality] = {}
            for process in self.process_stages:  # here we traverse Enums! See module helpers.
                self.max_wrkrs_per_shift_kind_global[modality][process.value] = {}
                for j in range(self.process_stages[process].stages):
                    self.max_wrkrs_per_shift_kind_global[modality][process.value][j] = {}

                    for kind in self.shift_kinds:
                        m_sk = self.cpx.variables.add(
                            obj=[self.shift_holder.get_cost(kind, fld_names.UNITARY_COST)],
                            types=["C"],
                            names=[f"m_global_{modality}_{kind.name}_{process.name[0:3]}_{j}"]
                        )
                        self.max_wrkrs_per_shift_kind_global[modality][process.value][j][kind] = m_sk[0]

    def declare_global_shift_wrkrs(self) -> None:
        # Total of workers per stage and in a given shift number.
        # TODO: fix the name of the method (this is not global) and
        #       fix the name of the variable, should be: v_wrkrs_modality_shift_stage
        for modality in self.list_modality:
            self.total_wrkrs_per_shift_and_stage[modality] = {}
            for shift in self.shift_numbers:
                self.total_wrkrs_per_shift_and_stage[modality][shift] = {}
                for process in self.process_stages:
                    self.total_wrkrs_per_shift_and_stage[modality][shift][process.value] = {}
                    for j in range(self.process_stages[process].stages):
                        # the name attribute of a ShiftKind instance
                        kind_name = self.shift_holder.get_shifts(shift).kind.name

                        w_s = self.cpx.variables.add(
                            types=["C"],
                            names=[f"total_wrkrs_per_modality_shift_stage"
                                   f"_{modality}"
                                   f"_{shift}"
                                   f"_{kind_name}"
                                   f"_{process.name[0:3]}"
                                   f"_{j}"])

                        self.total_wrkrs_per_shift_and_stage[modality][shift][process.value][j] = w_s[0]

    def set_global_minmax_constraints(self) -> None:
        # global means for the whole FC
        # TODO: fix the name "global" of the method

        for modality in self.list_modality:
            for shift in self.shift_numbers:
                for process in self.process_stages:
                    for j in range(self.process_stages[process].stages):
                        kind = self.shift_holder.get_shifts(shift).kind  # a ShiftKind instance

                        indices = [self.max_wrkrs_per_shift_kind_global[modality][process.value][j][kind],
                                   self.total_wrkrs_per_shift_and_stage[modality][shift][process.value][j]]
                        values = [1, -1]

                        # Let's include the kind of the shift in the constraint name, to know the kind
                        # of a shift with certain numer immediately in the lp.
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                            senses=["G"],
                            rhs=[0],
                            names=[f"c_global_minmax_{modality}_{shift}_{kind.name}_{process.name[0:3]}_{j}"]
                        )

    def set_linking_constraints(self) -> None:
        for modality in self.list_modality:
            for s in self.shift_numbers:
                for process, stm in iter(self.process_stages.items()):
                    for j in range(stm.stages):
                        indices = [self.total_wrkrs_per_shift_and_stage[modality][s][process.value][j],
                                   stm.v_total_perms_shift_assigned[modality][j][s]]
                        values = [-1, 1]

                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                            senses=['E'],
                            rhs=[0],
                            names=[f"c_link_{s}_{process.name[0:3]}_{modality}_{j}"]
                        )

    def set_params(self) -> None:
        # Here we set CPLEX running parameters, presumably defined by the developers.
        self.cpx.parameters.timelimit.set(Configuration.time_limit_seconds)
        self.cpx.parameters.mip.display.set(Configuration.display)
        self.cpx.parameters.barrier.display.set(Configuration.barrier_display)
        self.cpx.parameters.paramdisplay.set(Configuration.params_display)
        self.cpx.parameters.mip.tolerances.mipgap.set(Configuration.mip_gap)

    def set_work_capacity_constraint(self) -> None:
        restrictions = []
        senses = []
        rhs = []
        names = []
        
        list_indices = []
        list_indices_zero = []

        for process, stm in iter(self.process_stages.items()):
            hr_work_force = self.params[process].hourly_work_force
            value_p = stm.process.value
            for modality in self.list_modality:
                xt_m = stm.xt[modality]
                wt_m = self.wt[modality]
                stage_weights_w_m = self.dict_stage_weights_w[process][modality]
                stage_weights_x_m = self.dict_stage_weights_x[process][modality]

                for t in range(self.start, self.end + 1):
                    cpts = self.data_holders[process].dict_cpts_for_epoch[t]
                    shifts_epoch = self.shift_holder.shifts_for_epoch.get(t, [])

                    if shifts_epoch:

                        for j in range(stm.stages):
                            xt_mj = xt_m[j]

                            from_to_dest = []
                            from_to_ori = []
                            for from_to in self.orig_dest:
                                if (from_to[2] == value_p) and (from_to[3] == j):
                                    from_to_dest.append(from_to)

                                if (from_to[0] == value_p) and (from_to[1] == j):
                                    from_to_ori.append(from_to)

                            wts_mf_dest = [wt_m[from_to] for from_to in from_to_dest]
                            wts_mf_ori = [wt_m[from_to] for from_to in from_to_ori]
                            indices = []
                            values = []
                            for s in shifts_epoch:
                                xt_mjs = xt_mj[s]

                                indices += [xt_mjs[t]]
                                values += [-stage_weights_x_m[s][j]]
                                # Now, for the current work capacity, we compute all the polyvalents
                                # that arrived from other stages.
                                indices += [wt[s][t] for wt in wts_mf_dest]
                                values += [-stage_weights_w_m[s][from_to] for from_to in from_to_dest]

                            y_cpts = [stm.y[cpt][j][t] for cpt in cpts if
                                      cpt <= self.end and stm.min_stage_for_cpt[cpt] <= j]
                            indices += y_cpts
                            values += [1] * len(y_cpts)

                            if Configuration.activate_hourly_workers:
                                hourly_workers_p = self.v_hourly_workers[stm.process.value]
                                indices += [hourly_workers_p[j][t]]
                                values += [-hr_work_force]

                            restrictions += [cplex.SparsePair(ind=indices, val=values)]
                            senses += ['L']
                            rhs += [0]
                            names += [f"c_work_cap_{stm.process.name[0:3]}_{j}_{t}"]
                            
                            list_indices.append({'process': process.name[0:3],
                                             'stage': j,
                                             'epoch': t})

                    else:  # during non--business hours process zero items
                        indices = [stm.y[c][j][t] for j in range(stm.stages) for c in cpts if c <= self.end and
                                   stm.min_stage_for_cpt[c] <= j]
                        values = [1] * len(indices)

                        restrictions += [cplex.SparsePair(ind=indices, val=values)]
                        senses += ['E']
                        rhs += [0]
                        names += [f"c_work_cap_zero_{stm.process.name[0:3]}_{t}"]
                        
                        list_indices_zero.append({'process': process.name[0:3],
                                                  'epoch': t})

        self.cpx.linear_constraints.add(
            lin_expr=restrictions,
            senses=senses,
            rhs=rhs,
            names=names
        )

    def set_polyvalent_presenteeism_constraint(self) -> None:
        list_indices = []
        for stm in self.process_stages.values():
            for j in range(stm.stages):
                for s in self.list_shifts_for_epoch_range:
                    epochs_scope = self.shift_holder.epochs_for_shift[s][self.shift_holder.epochs_for_shift[s] <=
                                                                         self.end]
                    for t in epochs_scope:
                        for modality in self.list_modality:
                            for from_to in self.orig_dest:
                                # bound in terms of presenteeism
                                # check if the current place is the destination
                                if (from_to[2] == stm.process.value) & (from_to[3] == j):
                                    self.cpx.linear_constraints.add(
                                        lin_expr=[cplex.SparsePair(
                                            ind=[self.wt[modality][from_to][s][t],
                                                 self.v_polys_shift_assigned[modality][from_to][s]],
                                            val=[1,
                                                 -self.dict_presence_rate[Process(from_to[0])][modality][s][t] *
                                                 self.dict_absence_rate[stm.process][modality][s][t]])],
                                        senses=['L'],
                                        rhs=[0],
                                        names=[f"c_polys_presenteeism_"
                                               f"{stm.process.name[0:3]}_"
                                               f"{modality}_"
                                               f"{'_'.join(map(str, from_to))}_{s}_{t}"]
                                    )
                                    list_indices.append({"process": stm.process.name[0:3],
                                                         'stage_origin': from_to[1],
                                                         'process_origin': from_to[0],
                                                         'stage_destination': from_to[3],
                                                         'process_destination': from_to[1],
                                                         "modality": modality
                                                         }) 

    def set_real_totals_per_stage_constraint(self) -> None:
        """
            This constraint is to take account of the real quantity of people, regardless
            of any possible polyvalence destination at the present moment.
        """
        list_indices = []
        for process, stm in iter(self.process_stages.items()):
            this_process = process.value
            for modality in self.list_modality:
                zt_m = stm.zt[modality]
                xt_m = stm.xt[modality]
                wt_m = self.wt[modality]

                for j in range(stm.stages):
                    # Fix the pair corresponding to the current process and stage
                    # this_stage = this_process, j
                    zt_mj = zt_m[j]
                    xt_mj = xt_m[j]

                    destinations = [from_to for from_to in self.orig_dest
                                    if from_to[0:2] == (this_process, j)]

                    wts_mj_from_to = [wt_m[from_to] for from_to in destinations]

                    for s in self.shift_numbers:
                        epochs_scope = self.shift_holder.epochs_for_shift[s][
                            self.shift_holder.epochs_for_shift[s] <= self.end]
                        zt_mjs = zt_mj[s]
                        xt_mjs = xt_mj[s]
                        wts_mj_from_to_s = [wts_mjf[s] for wts_mjf in wts_mj_from_to]

                        for t in epochs_scope:
                            denom_xt = self.dict_presence_rate[process][modality][s][t] * \
                                       self.dict_absence_rate[process][modality][s][t]

                            p_perm_presence_xt = 1 / denom_xt if denom_xt else 0

                            local_indices = [zt_mjs[t], xt_mjs[t]]
                            local_indices += [wts[t] for wts in wts_mj_from_to_s]

                            local_values = [-1] + [p_perm_presence_xt]

                            for from_to in destinations:
                                denom_zt = self.dict_presence_rate[Process(from_to[0])][modality][s][t] * \
                                           self.dict_absence_rate[process][modality][s][t]

                                p_perm_presence_zt = 1 / denom_zt if denom_zt else 0
                                local_values += [p_perm_presence_zt]

                            self.cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(ind=local_indices,
                                                           val=local_values)],
                                senses=['L'],
                                rhs=[0],
                                names=[f"c_real_totals_per_stage_hour_"
                                       f"{process.name[0:3]}_{modality}_{stm.stage_names[j]}_{s}_{t}"]
                            )
                        
                            list_indices.append({"process": stm.process.name[0:3],
                                                 "modality": modality,
                                                 "stage": stm.stage_names[j],
                                                 "shift": s,
                                                 "epoch": t
                                                 })
