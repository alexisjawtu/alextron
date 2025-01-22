import cplex
import logging

import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

import kernel.data_frames_field_names as fld_names
import kernel.util.readers as readers

from datetime import datetime
from kernel.data.helpers import Process, ExtraHoursParameters
from kernel.formulation.model import BasicModel
from kernel.stages.short_term_inbound_model import ShortTermInboundModel
from kernel.stages.short_term_outbound_model import ShortTermOutboundModel
from kernel.general_configurations import (
    DevelopDumping,
    Configuration,
    FileNames,
    InputOutputPaths
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def explode_modality_all(mod_name_by_shift_name, df):
    # Function that replaces the modality "ALL" with the modalities associated with that shift
    # Requires field names HIRING_MODALITY and SHIFT_NAME in the both tables
    if not df.empty:
        df_all = df.loc[
            ((df[fld_names.HIRING_MODALITY]).str.upper() == fld_names.ALL_MODALITY)]
        df.drop(df_all.index, inplace=True)
        if not df_all.empty:
            df_all = df_all.drop(columns=fld_names.HIRING_MODALITY)
            df_all = df_all.merge(mod_name_by_shift_name, how='inner', on=fld_names.SHIFT_NAME)
            df_all = df_all.explode(fld_names.HIRING_MODALITY)
            df = df.append(df_all)
    return df


class ShortTermFormulation(BasicModel):
    def define(self) -> None:
        self.cpx.objective.set_sense(self.cpx.objective.sense.minimize)

        df_shift_contract_modality = readers.auxiliary_standard_read(FileNames.SH_NAME_TO_CONTRACT_MOD)
        mod_name_by_shift_name = df_shift_contract_modality.groupby(fld_names.SHIFT_NAME).agg(list)

        df_shift_parameters = self.shift_params.data[[fld_names.SHIFT_NAME, fld_names.SHIFT_TYPE]].drop_duplicates()

        mod_name_by_shift_type = df_shift_contract_modality.merge(df_shift_parameters)
        mod_name_by_shift_type = mod_name_by_shift_type[[fld_names.SHIFT_TYPE, fld_names.HIRING_MODALITY]]

        self.set_shift_numbers()
        self.set_workers_initials(mod_name_by_shift_type)
        self.set_costs_dfs()

        self.shift_params.df_workers_costs = explode_modality_all(mod_name_by_shift_name,
                                                                  self.shift_params.df_workers_costs)
        self.set_contract_types_and_modalities()
        self.set_contract_modalities_to_shift_names_relations()

        if Configuration.activate_transfer:
            # This are 'from -> to' edges between weeks, not stages!
            self.set_shifts_from_to()
        
        self.process_stages[Process.INBOUND] = ShortTermInboundModel(
            self.cpx,
            self.data_holders[Process.INBOUND],
            self.shift_params, 
            Process.INBOUND,
            self.start, 
            self.fixed_end, 
            self.end,
            self.max_epoch, 
            self.data_holders[Process.INBOUND].stages,
            self.previous_made_output,
            self.params[Process.INBOUND],
            self.previous_cplex_solution,
            self.previous_cplex_ids.get(Process.INBOUND),
            self.df_transfers,
            self.dc_shifts_from,
            self.dc_shifts_to,
            self.dc_workers_initial_inbound,
            self.dict_shift_kinds,
            self.dc_contract_modalities_for_shift_name,
            self.dc_contract_types_and_modalities,
            self.dc_modalities_and_contract_types,
            self.dc_shift_names_per_contract_modality
        )

        self.process_stages[Process.OUTBOUND] = ShortTermOutboundModel(
            self.cpx,
            self.data_holders[Process.OUTBOUND],
            self.shift_params, 
            Process.OUTBOUND,
            self.start, 
            self.fixed_end, 
            self.end,
            self.max_epoch, 
            self.data_holders[Process.OUTBOUND].stages,
            self.previous_made_output,
            self.params[Process.OUTBOUND],
            self.previous_cplex_solution,
            self.previous_cplex_ids.get(Process.OUTBOUND),
            self.df_transfers,
            self.dc_shifts_from,
            self.dc_shifts_to,
            self.dc_workers_initial_outbound,
            self.dict_shift_kinds,
            self.dc_contract_modalities_for_shift_name,
            self.dc_contract_types_and_modalities,
            self.dc_modalities_and_contract_types,
            self.dc_shift_names_per_contract_modality
        )

        self.build_origin_destination_for_polyvalence()

        self.list_modality = [modality for m in self.dc_contract_types_and_modalities
                              for modality in self.dc_contract_types_and_modalities[m]]

        # TODO: vectorize stuff like this cycle
        for process, stm in iter(self.process_stages.items()):
            max_workers_w_p = {}
            self.dict_max_workers_w[process] = max_workers_w_p
            for shift in self.shift_numbers:
                max_workers_w_s = {}
                max_workers_w_p[shift] = max_workers_w_s
                for modality in self.shift_params.dc_modalities_by_shift_idx[shift]:
                    max_workers_w_sm = {}
                    max_workers_w_s[modality] = max_workers_w_sm
                    for from_to in self.orig_dest:
                        max_workers_w_sm[from_to] = self.process_stages[process]. \
                            parameters.max_workers_w(modality, from_to, shift)
            self.dict_presence_rate[process] = {}
            self.dict_absence_rate[process] = {}
            self.dict_stage_weights_w[process] = {}
            self.dict_stage_weights_x[process] = {}
            for s in self.shift_params.shifts_for_epoch_range(self.start, self.end):
                presence_rate_s = {}
                self.dict_presence_rate[process][s] = presence_rate_s
                absence_rate_s = {}
                self.dict_absence_rate[process][s] = absence_rate_s
                stage_weights_w_s = {}
                self.dict_stage_weights_w[process][s] = stage_weights_w_s
                stage_weights_x_s = {}
                self.dict_stage_weights_x[process][s] = stage_weights_x_s
                for m in self.shift_params.dc_modalities_by_shift_idx[s]:
                    presence_rate_sm = {}
                    presence_rate_s[m] = presence_rate_sm
                    absence_rate_sm = {}
                    absence_rate_s[m] = absence_rate_sm
                    stage_weights_w_sm = {}
                    stage_weights_w_s[m] = stage_weights_w_sm
                    stage_weights_x_sm = {}
                    stage_weights_x_s[m] = stage_weights_x_sm
                    for from_to in self.orig_dest:
                        stage_weights_w_sm[from_to] = self.params[process].stage_weights_w(m, from_to, s)
                    for j in range(stm.stages):
                        stage_weights_x_sm[j] = self.params[process].stage_weights_x(m, j, s)

                    epochs_scope = self.shift_params.epochs_for_shift[s][
                        self.shift_params.epochs_for_shift[s] <= self.end]
                    for t in epochs_scope:
                        presence_rate_sm[t] = self.params[stm.process].presence_rate(m, t, s)
                        absence_rate_sm[t] = self.params[stm.process].absence_rate(m, t, s)

    def set_cplex(self, cpx: cplex.Cplex) -> None:
        self.cpx = cpx

    def set_workers_initials(self, mod_name_by_shift_type: dict) -> None:
        self.dc_workers_initial_inbound, self.dc_workers_initial_outbound = readers.read_workers_initial(
            mod_name_by_shift_type)

    def set_costs_dfs(self) -> None:
        self.shift_params.df_workers_costs = readers.read_workers_costs()

    def set_contract_modalities_to_shift_names_relations(self) -> None:
        """
            Outs are:

            {
                "AFT0W1": ["Meli_Perm", "Meli_Diario"],
                "AFT0W1_T": ["Meli_Temp"]
            }

            and

            {
                "Meli_Perm": ["AFT0W1"],
                "Meli_Diario": ["AFT0W1"],
                "Meli_Temp": ["AFT0W1_T"]
            }
        """

        # TODO: we should tabulate a "dc_contract_modalities_for_shift", to speed up the variables
        # for extra_hours_reps
        self.dc_contract_modalities_for_shift_name, self.dc_shift_names_per_contract_modality = \
            readers.read_shift_contract_modality()

    def set_contract_types_and_modalities(self) -> None:
        """
            {
                "Diarista": ["MELI_Diarista"],
                "Perm": ["MELI_Perm", "DHL_Perm"],
                "Temporal": ["MELI_Temporal"]
             }
        """
        self.dc_contract_types_and_modalities, self.dc_modalities_and_contract_types = readers.read_contract_modality_type()

    def set_shifts_from_to(self) -> None:
        """
        Example input:

            shift_name_origin, shift_name_destination, cost, contract_modality
            AFTERNOON0W1     , MORNING0W1            , 10  , MeLi_Day
            AFTERNOON0W1     , NIGHT0W1              , 8   , MeLi_Perm
            AFTERNOON0W1     , NIGHT0W1              , 8   , MeLi_Day
            AFTERNOON0W1     , NIGHT0W1              , 8   , DHL_Temp
            NIGHT6W2         , AFTERNOON1W2          , 7   , DHL_Perm
            MORNING0W2       , NIGHT5W2              , 6   , DHL_Temp
            NIGHT0W1         , MORNING1W1            , 11  , DHL_Temp

        Then, part of dc_shifts_from is:

        {
            "MeLi_Day": {
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
        self.shift_numbers = self.shift_params.shifts_for_epoch_range(self.start, self.end)

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
        self.v_polys_shift_assigned = {from_to: {shift: {
            modality: self.cpx.variables.add(
                ub=[self.dict_max_workers_w[Process(from_to[2])][shift][modality][from_to]],
                types=["C"],
                names=[f"v_polys_shift_assigned"
                       f"_{modality}"
                       f"_{'_'.join(map(str, from_to))}"
                       f"_{shift}"]
            )[0] for modality in self.shift_params.dc_modalities_by_shift_idx[shift]}
            for shift in self.shift_numbers}
            for from_to in self.orig_dest}

    def declare_polys_hour_assigned(self) -> None:
        for from_to in self.orig_dest:
            self.wt[from_to] = {}
            process_enum = Process(from_to[2])
            for s in self.shift_numbers:
                self.wt[from_to][s] = {}
                for m in self.shift_params.dc_modalities_by_shift_idx[s]:
                    max_w = self.dict_max_workers_w[process_enum][s][m][from_to]
                    self.wt[from_to][s][m] = {}

                    epochs_scope = self.shift_params.epochs_for_shift[s][
                        self.shift_params.epochs_for_shift[s] <= self.end]

                    for t in epochs_scope:
                        wjst = self.cpx.variables.add(
                            obj=[Configuration.cost_polyvalents],
                            ub=[max_w],
                            types=["C"],
                            names=["wt_%s_%s_%d_%d" % (m, "_".join(map(str, from_to)), s, t)]
                        )
                        self.wt[from_to][s][m][t] = wjst[0]

    def declare_extra_hours_reps(self) -> None:
        """ DOCSTRING: this method declares quantities of workers that are already in a standard shift
            and agree to work overtime to cope with amounts of backlogs. 

            conventions:

                v[0:n] == extra_hours_indices after the shift
                v[-n:] == extra_hours_indices before the shift

            matrix_proc_0[j, s, mod] := the list of cpx indices of the 2*n_extra_hours slots for shift s
            unitary_cost_extra_hours[j,s,m] := the array of the unitary costs, one (the same) for each extra hour of s

            ====================================================
            table:
            mod, s --> (day_name, shift_name, mod) --> unit_cost
            ====================================================
        """
        # Declaration of perms in extra hours
        # Important: these proc indices should match Process(proc).value; 
        # don't rely on the order of dict params.
        for proc in range(len(self.params)):
            extra_hrs_perms_p = dict()
            self.v_extra_hours_perms[proc] = extra_hrs_perms_p
            local_stages = self.stage_names[Process(proc).name]
            for j in local_stages:
                extra_hrs_perms_pj = dict()
                extra_hrs_perms_p[j] = extra_hrs_perms_pj

                for s in self.dc_modalities_by_shift_idx_with_extra_hours:
                    extra_hrs_perms_pjs = dict()
                    extra_hrs_perms_pj[s] = extra_hrs_perms_pjs
                    for mod in self.dc_modalities_by_shift_idx_with_extra_hours[s]:
                        
                        extra_hrs_perms_pjsm = dict()
                        extra_hrs_perms_pjs[mod] = extra_hrs_perms_pjsm
                        unitary_cost_extra_hours = self.unitary_costs_extra_hours[s, mod]
                        
                        for t in self.dc_extra_hours_ranges[s, mod]:

                            extra_hrs_perms_pjsm[t] = self.cpx.variables.add(
                                                obj=[unitary_cost_extra_hours],
                                                types=["C"],
                                                names=["v_extra_hours_perms_%d_%d_%d_%s_%d " % (proc, j, s, mod, t)]
                                            )[0]

        # Now the declaration of polyvalents in extra hours.
        for s in self.dc_modalities_by_shift_idx_with_extra_hours:
            extra_hrs_polys_s = dict()
            self.v_extra_hours_polys[s] = extra_hrs_polys_s
            for mod in self.dc_modalities_by_shift_idx_with_extra_hours[s]: 
                extra_hrs_polys_sm = dict()
                extra_hrs_polys_s[mod] = extra_hrs_polys_sm
                unitary_cost_extra_hours = self.unitary_costs_extra_hours[s, mod]

                for from_to in self.orig_dest:
                    extra_hrs_polys_smf = dict()
                    extra_hrs_polys_sm[from_to] = extra_hrs_polys_smf
                    # the extra epochs to expand the present shift s
                    for t in self.dc_extra_hours_ranges[s, mod]:
                        extra_hrs_polys_smf[t] = self.cpx.variables.add(
                            obj=[unitary_cost_extra_hours],
                            types=["C"],
                            names=["v_extra_hours_polys_%d_%s_%s_%d" %
                                   (s, mod, '_'.join(map(str, from_to)), t)])[0]

    def declare_hourly_workers(self) -> None:
        """ DOCSTRING: this method is to cope with infeasibilities as a near to last resource.
            An hourly worker is only hired for certain hours, and doesn't belong to any
            shift or input bound. 

            In the operation, hourly_workers are referred to as "horistas". """
        for stm in self.process_stages.values():
            process_name = stm.process.name
            self.v_hourly_workers[stm.process.value] = dict()
            for j in range(stm.stages):
                self.v_hourly_workers[stm.process.value][j] = dict()
                for t in range(stm.start, stm.end + 1):
                    hpjt = self.cpx.variables.add(
                        obj=[self.params[stm.process].get_cost_hourly_workers()],
                        types=["C"],
                        names=["v_hourly_workers_%s_%d_%d" % (process_name[0:3], j, t)]
                    )
                    self.v_hourly_workers[stm.process.value][j][t] = hpjt[0]

    def set_optimal_polyvalent_values(self) -> None:
        # Here we set and fix the values of polyvalence found in the previous run.
        for from_to in self.orig_dest:
            for shift in self.shift_numbers:
                for m in self.shift_params.dc_modalities_by_shift_idx[shift]:
                    epochs_scope = self.shift_params.epochs_for_shift[shift][
                        self.shift_params.epochs_for_shift[shift] <= self.end]

                    # TODO: make self.previous_cplex_ids call cleaner when refactoring all the variables
                    # into ShortTermFormulation
                    opt_val_w = self.previous_cplex_solution[
                        self.previous_cplex_ids[list(self.previous_cplex_ids.keys())[0]].dc_w_fixed[from_to][
                            shift][m]]

                    self.cpx.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[self.v_polys_shift_assigned[from_to][shift][m]],
                                                   val=[1])],
                        senses=["E"],
                        rhs=[opt_val_w],
                        names=[f"c_fix_shift_polys_{m}_{'_'.join(map(str, from_to))}_{shift}"]
                    )

                    for epoch in epochs_scope:
                        opt_val_w = self.previous_cplex_solution[
                            self.previous_cplex_ids[list(self.previous_cplex_ids.keys())[0]].dc_wt_fixed[
                                from_to][shift][m][epoch]]

                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=[self.wt[from_to][shift][m][epoch]], val=[1])],
                            senses=["E"],
                            rhs=[opt_val_w],
                            names=[f"c_fix_hr_polys_{m}_{'_'.join(map(str, from_to))}_{shift}_{epoch}"]
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

        for modality_type, modalities in self.dc_contract_types_and_modalities.items():
            for modality in modalities:
                self.max_wrkrs_per_shift_kind_global[modality] = {}
                for process in self.process_stages:  # here we traverse Enums! See module helpers.
                    self.max_wrkrs_per_shift_kind_global[modality][process.value] = {}
                    for j in range(self.process_stages[process].stages):
                        self.max_wrkrs_per_shift_kind_global[modality][process.value][j] = {}

                        for kind in self.dict_shift_kinds:
                            if fld_names.DAILY_MODALITY != modality_type.lower():
                                obj_coeff = self.shift_params.get_cost(kind, modality, fld_names.UNITARY_COST) * \
                                            self.shift_params.get_days_quantity_per_shift(kind)

                                m_sk = self.cpx.variables.add(
                                    obj=[obj_coeff],
                                    types=["C"],
                                    names=[f"m_global_{modality}_{kind}_{process.name[0:3]}_{j}"]
                                )
                                self.max_wrkrs_per_shift_kind_global[modality][process.value][j][kind] = m_sk[0]
                            else:
                                m_sk = self.cpx.variables.add(
                                    types=["C"],
                                    names=[f"m_global_{modality}_{kind}_{process.name[0:3]}_{j}"]
                                )
                                self.max_wrkrs_per_shift_kind_global[modality][process.value][j][kind] = m_sk[0]

    def declare_global_shift_wrkrs(self) -> None:
        # Total of workers per stage and in a given shift number.
        # TODO: fix the name of the method (this is not global) and
        #       fix the name of the variable, should be: v_wrkrs_modality_shift_stage
        for shift in self.shift_numbers:
            total_wrkrs_per_shift_and_stage_shift = {}
            self.total_wrkrs_per_shift_and_stage[shift] = total_wrkrs_per_shift_and_stage_shift
            for m in self.shift_params.dc_modalities_by_shift_idx[shift]:
                total_wrkrs_per_shift_and_stage_shift_m = {}
                total_wrkrs_per_shift_and_stage_shift[m] = total_wrkrs_per_shift_and_stage_shift_m
                for process in self.process_stages:
                    total_wrkrs_per_shift_and_stage_shift_m_process = {}
                    total_wrkrs_per_shift_and_stage_shift_m[process.value] = \
                        total_wrkrs_per_shift_and_stage_shift_m_process
                    for j in range(self.process_stages[process].stages):
                        kind_name = self.shift_params.get_shift_name(shift)

                        w_s = self.cpx.variables.add(
                            types=["C"],
                            names=[f"total_wrkrs_per_modality_shift_stage"
                                   f"_{m}"
                                   f"_{shift}"
                                   f"_{kind_name}"
                                   f"_{process.name[0:3]}"
                                   f"_{j}"])

                        total_wrkrs_per_shift_and_stage_shift_m_process[j] = w_s[0]

    def set_global_minmax_constraints(self) -> None:
        # global means for the whole FC
        # TODO: fix the name "global" of the method

        for shift in self.shift_numbers:
            for modality in self.shift_params.dc_modalities_by_shift_idx[shift]:
                for process in self.process_stages:
                    for j in range(self.process_stages[process].stages):
                        shift_name = self.shift_params.get_shift_name(shift)

                        indices = [self.max_wrkrs_per_shift_kind_global[modality][process.value][j][shift_name],
                                   self.total_wrkrs_per_shift_and_stage[shift][modality][process.value][j]]
                        values = [1, -1]

                        # Let's include the kind of the shift in the constraint name, to know the kind
                        # of a shift with certain numer immediately in the lp.
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                            senses=["G"],
                            rhs=[0],
                            names=[f"c_global_minmax_{modality}_{shift}_{shift_name}_{process.name[0:3]}_{j}"]
                        )

    def set_linking_constraints(self) -> None:
        for s in self.shift_numbers:
            for modality in self.shift_params.dc_modalities_by_shift_idx[s]:
                for process, stm in iter(self.process_stages.items()):
                    for j in range(stm.stages):
                        indices = [self.total_wrkrs_per_shift_and_stage[s][modality][process.value][j],
                                   stm.v_total_shift_assigned_per_modality[j][s][modality]]
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

        dict_shifts_epoch: Dict = self.shift_params.shifts_for_epoch

        for process, stm in iter(self.process_stages.items()):
            hr_work_force = self.params[process].hourly_work_force
            value_p = process.value
            stage_weights_w = self.dict_stage_weights_w[process]
            stage_weights_x = self.dict_stage_weights_x[process]

            xt = stm.xt
            
            proc_dh = self.data_holders[process]

            for t in range(self.start, self.end + 1):
                # TODO for performance: tabulate previously, and remove the "if" cases, filtering by:
                #     1 epochs with no shifts and no extra_shifts
                #     2 epochs with any of regular shifts or extra shifts.

                proc_cpts_t = proc_dh.dict_cpts_for_epoch[t]

                shifts_epoch = dict_shifts_epoch.get(t, [])

                if shifts_epoch:  # + self.extra_shifts_for_epoch[t]: TODO <-- CONTROLAR si son listas o np.arrs

                    for j in range(stm.stages):
                        xt_j = xt[j]

                        from_to_dest = []
                        from_to_ori = []
                        for from_to in self.orig_dest:
                            if (from_to[2] == value_p) and (from_to[3] == j):
                                from_to_dest.append(from_to)

                            if (from_to[0] == value_p) and (from_to[1] == j):
                                from_to_ori.append(from_to)

                        wts_mf_dest = [self.wt[from_to] for from_to in from_to_dest]
                        indices = []
                        values = []
                        for s in shifts_epoch:
                            xt_js = xt_j[s]
                            stg_force_x_s = stage_weights_x[s]
                            for m in self.shift_params.dc_modalities_by_shift_idx[s]:
                                indices += [xt_js[m][t]]
                                values += [-stg_force_x_s[m][j]]
                                # Now, for the current work capacity, we compute all the polyvalents
                                # that arrived from other stages.
                                indices += [wt[s][m][t] for wt in wts_mf_dest]
                                values += [-stage_weights_w[s][m][from_to] for from_to in from_to_dest]

                        for s in self.extra_shifts_for_epoch.get(t, []):

                            # TODO CONTINUE HERE: polyvalence for extra hours work capacity

                            extra_perms_pjs = self.v_extra_hours_perms[value_p][j][s]

                            # Stage force is inherited from the current stage/shift
                            stg_force_extra_hrs_s = stage_weights_x[s]
                            
                            for m in self.dc_modalities_by_shift_idx_with_extra_hours[s]:
                                
                                indices += [extra_perms_pjs[m][t]]
                                values += [-stg_force_extra_hrs_s[m][j]]

                        y_cpts = [stm.y[cpt][j][t] for cpt in proc_cpts_t if
                                  cpt <= self.end and stm.min_stage_for_cpt[cpt] <= j]
                        indices += y_cpts
                        values += [1] * len(y_cpts)

                        if Configuration.activate_hourly_workers:
                            hourly_workers_p = self.v_hourly_workers[stm.process.value]
                            indices += [hourly_workers_p[j][t]]
                            values += [-hr_work_force]

                        restrictions += [cplex.SparsePair(ind=indices, val=values)]
                        senses += ["L"]
                        rhs += [0]
                        names += [f"c_work_cap_{stm.process.name[0:3]}_{j}_{t}"]

                        list_indices.append({'process': process.name[0:3],
                                             'stage': j,
                                             'epoch': t})

                else:  
                    # during non--business hours process no items, 
                    # a.k.a if not shifts_epoch and not extra_shifts_for_epoch.

                    indices = [stm.y[c][j][t] for j in range(stm.stages) for c in proc_cpts_t if c <= self.end and
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
                    epochs_scope = self.shift_params.epochs_for_shift[s][self.shift_params.epochs_for_shift[s] <=
                                                                         self.end]
                    for t in epochs_scope:
                        for modality in self.shift_params.dc_modalities_by_shift_idx[s]:
                            for from_to in self.orig_dest:
                                # bound in terms of presenteeism
                                # check if the current place is the destination
                                if (from_to[2] == stm.process.value) & (from_to[3] == j):
                                    self.cpx.linear_constraints.add(
                                        lin_expr=[cplex.SparsePair(
                                            ind=[self.wt[from_to][s][modality][t],
                                                 self.v_polys_shift_assigned[from_to][s][modality]],
                                            val=[1,
                                                 -self.dict_presence_rate[Process(from_to[0])][s][modality][t] *
                                                 self.dict_absence_rate[stm.process][s][modality][t]])],
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

            for j in range(stm.stages):
                zt_j = stm.zt[j]
                xt_j = stm.xt[j]
                # Fix the pair corresponding to the current process and stage
                # this_stage = this_process, j

                destinations = [from_to for from_to in self.orig_dest
                                if from_to[0:2] == (this_process, j)]

                wts_from_to = [self.wt[from_to] for from_to in destinations]

                for s in self.shift_numbers:
                    zt_js = zt_j[s]
                    xt_js = xt_j[s]
                    for m in self.shift_params.dc_modalities_by_shift_idx[s]:
                        epochs_scope = self.shift_params.epochs_for_shift[s][
                            self.shift_params.epochs_for_shift[s] <= self.end]

                        zt_jsm = zt_js[m]
                        xt_jsm = xt_js[m]
                        wts_from_to_sm = [wts_ft[s][m] for wts_ft in wts_from_to]

                        for t in epochs_scope:
                            denom_xt = self.dict_presence_rate[process][s][m][t] * \
                                       self.dict_absence_rate[process][s][m][t]

                            p_perm_presence_xt = 1 / denom_xt if denom_xt else 0

                            local_indices = [zt_jsm[t], xt_jsm[t]]
                            local_indices += [wts[t] for wts in wts_from_to_sm]

                            local_values = [-1] + [p_perm_presence_xt]

                            for from_to in destinations:
                                denom_zt = self.dict_presence_rate[Process(from_to[0])][s][m][t] * \
                                           self.dict_absence_rate[process][s][m][t]

                                p_perm_presence_zt = 1 / denom_zt if denom_zt else 0
                                local_values += [p_perm_presence_zt]

                            self.cpx.linear_constraints.add(
                                lin_expr=[cplex.SparsePair(ind=local_indices,
                                                           val=local_values)],
                                senses=['L'],
                                rhs=[0],
                                names=[f"c_real_totals_per_stage_hour_"
                                       f"{process.name[0:3]}_{m}_{stm.stage_names[j]}_{s}_{t}"]
                            )

                            list_indices.append({"process": stm.process.name[0:3],
                                                 "modality": m,
                                                 "stage": stm.stage_names[j],
                                                 "shift": s,
                                                 "epoch": t
                                                 })

    def set_extra_hours_variables_and_constraints(self) -> None:
        """ Put the extra_hours -- specific constraints to the model.

        Specific means that the included productivity terms corresponding to extra hours volunteers
        are written in the standard work capacity constraint.

        Important: the absence ratios here are per shift and modality, as is the input, because we must use the
        absence ratios of the shift beeing expanded. For now we use THE FIRST EPOCH OF THE SHIFT, and all this
        should be refactored to use exactly the shift, and not function calls for every epoch and to return
        the same absence coefficient.

        Hour <--> shift availability bound:
        ==================================

            extra_hr_workers_t <= presence(m, t, s) * absence(s, m) * acceptance_rate(s, m) * 
                                    total_perms_shift_assigned[mod, j, s]

        Daily legal constraints:
        =======================

            sum_{t in he(s, m)} A / presence() <= C * D * E * total_shift[m, j, s]

        Weekly legal constraints:
        ========================

        One per each shift_kind. For example, for the shift_name AFTERNOON17W48,

            SUM_{days} 6 * ratio_faltante / p + 5/p + 4/0.6 + 3 * ratio_faltante/p  <= 

                <= 12 * max (total_shift[day]) * mean(accept[day]) * min_over_days_of_weeek(positive absences)

        [ex: 0.2,   0.3,  0 → 0.7]:
        Ojo: 0.2 y 0.3 son, respectivamente, la suma de justified y unjustified

        Monotony constraints:
        ====================

        These are to avoid isolated extra hours, and go like this

            t >= t+1

            t <= t+1
        """

        # CONTINUE HERE: VER COSAS SUELTAS
        # # * agregar un "if Configuration.activate_extra_hours:" a la construcci'on
        #   de las constr.

        def build_extra_hours_parameters():
            """ The intention of this is a mixture of time performance and
            space performance experiment """
            logger.info("Making extra hours parameters.")

            # Docstring for this local procedure:
            # step 1- gather (shift_id, modality, max_extra_hours) from df_extra_hours_expanded_table.
            # step 2- expand through shift_idxs with common (day, shift_name).
            # step 3- merge with previously built epochs ids.

            sh = self.shift_params.shifts_df[[
                "work_day_name",
                "shift_name",
                "shift_idx",
                fld_names.START_MINS,
                fld_names.END_MINS]]

            sh = sh.rename(columns={"work_day_name": "day_name"})
            sh = sh[~sh.duplicated()]

            # In this merge, beware the fact that the table of shifts has all shifts and extra_param not.
            df_extra_hours_expanded_table = self.raw_eh_parameters.merge(sh)
            del sh
            dc_acceptance_ratios: Dict = df_extra_hours_expanded_table.groupby(
                ["shift_idx", "modality"])["rate_of_extra_hours_acceptance"].agg(min).to_dict()

            # In this lines we build and add columns fld_names.MIN_SH_EPOCH and fld_names.MAX_SH_EPOCH to know 
            # directly which indices correspond to the extra hours before and after.
            df_start_end = self.shift_params.shifts_df[["shift_idx", "idx"]].rename(columns={"idx": fld_names.MIN_SH_EPOCH})
            df_start_end[fld_names.MAX_SH_EPOCH] = df_start_end[fld_names.MIN_SH_EPOCH]
            df_start_end = df_start_end.groupby("shift_idx")[[fld_names.MIN_SH_EPOCH, fld_names.MAX_SH_EPOCH]].agg(
                {fld_names.MIN_SH_EPOCH: min, fld_names.MAX_SH_EPOCH: max}).reset_index()
            df_extra_hours_expanded_table = df_extra_hours_expanded_table.merge(df_start_end)

            del df_start_end
            
            # Now, fld_names.EH_FULL_RANGE will be the list of epoch indices t that are exactly the extra 
            # hours of the shift with index s.
            # Eh[s] := extra_hours_for_shift[s] (See "Eh(s)" in the discovery document).

            # start and end of previous extra hours
            df_extra_hours_expanded_table[fld_names.MIN_EH_BEFORE_SH] = df_extra_hours_expanded_table[fld_names.MIN_SH_EPOCH] \
                - df_extra_hours_expanded_table[fld_names.MAX_EH_PER_DAY]
            df_extra_hours_expanded_table[fld_names.MAX_EH_BEFORE_SH] = df_extra_hours_expanded_table[fld_names.MIN_SH_EPOCH]

            # # start and end of extra hours after the shift
            # df_extra_hours_expanded_table[fld_names.MIN_EH_AFTER_SH] = df_extra_hours_expanded_table[fld_names.MAX_SH_EPOCH] + 1
            # df_extra_hours_expanded_table[fld_names.MAX_EH_AFTER_SH] = df_extra_hours_expanded_table[fld_names.MAX_SH_EPOCH] \
            #     + df_extra_hours_expanded_table[fld_names.MAX_EH_PER_DAY] + 1

            df_extra_hours_expanded_table[fld_names.START_RATIO] \
                = df_extra_hours_expanded_table[fld_names.START_MINS]/60

            df_extra_hours_expanded_table[fld_names.END_RATIO] \
                = df_extra_hours_expanded_table[fld_names.END_MINS]/60

            df_extra_hours_expanded_table[fld_names.COMPLEMENTARY_START_RATIO] \
                = 1 - df_extra_hours_expanded_table[fld_names.START_RATIO]

            df_extra_hours_expanded_table[fld_names.COMPLEMENTARY_END_RATIO] \
                = 1 - df_extra_hours_expanded_table[fld_names.END_RATIO]

            start_fractions_mask = (df_extra_hours_expanded_table[fld_names.START_MINS] > 0)
            end_fractions_mask = (df_extra_hours_expanded_table[fld_names.END_MINS] > 0)

            df_extra_hours_expanded_table.loc[start_fractions_mask, fld_names.MAX_EH_BEFORE_SH] += 1

            # start and end of extra hours after the shift
            df_extra_hours_expanded_table[fld_names.MIN_EH_AFTER_SH] = df_extra_hours_expanded_table[fld_names.MAX_SH_EPOCH]
            df_extra_hours_expanded_table.loc[~end_fractions_mask, fld_names.MIN_EH_AFTER_SH] += 1
            df_extra_hours_expanded_table[fld_names.MAX_EH_AFTER_SH] = df_extra_hours_expanded_table[fld_names.MAX_SH_EPOCH] \
                + df_extra_hours_expanded_table[fld_names.MAX_EH_PER_DAY]

            df_extra_hours_expanded_table[fld_names.EH_FULL_RANGE] = df_extra_hours_expanded_table.apply(lambda x: np.hstack((
                np.array(range(x[fld_names.MIN_EH_BEFORE_SH], 1 + x[fld_names.MAX_EH_BEFORE_SH])),
                np.array(range(x[fld_names.MIN_EH_AFTER_SH], 1 + x[fld_names.MAX_EH_AFTER_SH])))), axis=1)

            self.dc_extra_hours_ranges: Dict = dict(zip(zip(df_extra_hours_expanded_table["shift_idx"],
                df_extra_hours_expanded_table["modality"]), df_extra_hours_expanded_table[fld_names.EH_FULL_RANGE]))

            dc_daily_maxs: Dict = df_extra_hours_expanded_table.groupby(["shift_idx", "modality"]
                                            )[fld_names.MAX_EH_PER_DAY].agg(min).to_dict()

            df_extra_hours_expanded_table = df_extra_hours_expanded_table.merge(
                self.shift_params.shifts_df[[fld_names.SHIFT_ID,
                                             fld_names.START_MINS,
                                             fld_names.END_MINS]].drop_duplicates())

            if DevelopDumping.DEV:
                df_extra_hours_expanded_table.to_csv(f"{InputOutputPaths.BASEDIR_VAL}/{FileNames.EXTRA_HOURS_DEV_TABLE}")
            
            # Here we set two attributes bound to ShortTermFormulation, because we need them in the
            # set_work_capacity_constraint method.
            self.extra_shifts_for_epoch = df_extra_hours_expanded_table[["shift_idx",fld_names.EH_FULL_RANGE]] \
                .explode(fld_names.EH_FULL_RANGE) \
                .drop_duplicates().groupby(fld_names.EH_FULL_RANGE)["shift_idx"].agg(list).to_dict()

            # unitary_costs_extra_hours:  dict( [s, mod] --> c )
            self.unitary_costs_extra_hours: Dict = dict(zip(zip(df_extra_hours_expanded_table["shift_idx"],
                                                                df_extra_hours_expanded_table["modality"]), 
                                                df_extra_hours_expanded_table["unitary_cost_extra_hours"]))

            self.dc_modalities_by_shift_idx_with_extra_hours: Dict = df_extra_hours_expanded_table.groupby(
                                            "shift_idx")["modality"].agg(list).to_dict()

            dc_presence_ratios_for_extra_hours = {}
            dc_extra_hours_ratios = {
                s_m: {t: 1 for t in l}
                for s_m, l in iter(self.dc_extra_hours_ranges.items())
            }

            return ExtraHoursParameters(dc_acceptance_ratios,
                                        dc_daily_maxs,
                                        dc_presence_ratios_for_extra_hours,
                                        dc_extra_hours_ratios)

        formatted_extra_hrs_params: ExtraHoursParameters = build_extra_hours_parameters()
        logger.info("Making extra hours variables and restrictions.")

        self.declare_extra_hours_reps()

        # TODO: Corresponding presences and absences are in WIP
        # After building according to START_RATIO, END_RATIO, etc, and the FULL_RANGES, eliminate all the .get()
        absence_p = dict()
        presence_p = dict()
        # In this case we use this dict just for their keys!
        for s in self.dc_modalities_by_shift_idx_with_extra_hours:

            # absence_ratio_psmt
            absence_ps = absence_p.get(s, dict())
            presence_ps = presence_p.get(s, dict())

            for mod in self.shift_params.dc_modalities_by_shift_idx[s]:
                # The absence of the first epoch of the shift
                absence_psm = absence_ps.get(mod, dict())
                absence = absence_psm.get(0, 1)  # next(iter(absence_psm))

                presence_psm = presence_ps.get(mod, dict())
                acceptance = formatted_extra_hrs_params.dc_acceptance_ratios[s, mod]
                extra_hs_s_m_epoch_range = self.dc_extra_hours_ranges[s, mod]

                for proc, stm in iter(self.process_stages.items()):
                    for j in range(stm.stages):
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(
                                ind=[self.v_extra_hours_perms[proc.value][j][s][mod][t],
                                     stm.v_total_shift_assigned_per_modality[j][s][mod]],
                                val=[1, - presence_psm.get(t, 1) * absence * acceptance]) for t in extra_hs_s_m_epoch_range],
                            senses=["L"] * len(extra_hs_s_m_epoch_range),
                            rhs=[0] * len(extra_hs_s_m_epoch_range),
                            names=["c_extra_hours_workers_bounds_%s_%d_%d_%s_%d"
                                   % (proc.name, j, s, mod, t) for t in extra_hs_s_m_epoch_range])

        # Daily legal constraints:
        for proc, stm in iter(self.process_stages.items()):
            extra_perms_p = self.v_extra_hours_perms[proc.value]

            # TODO: Corresponding presences and absences are in WIP
            # After building according to START_RATIO, END_RATIO, etc, and the FULL_RANGES, eliminate all the .get()
            absence_p = self.dict_absence_rate[proc]
            presence_p = dict()  # self.dict_presence_rate[Process(0)]

            for j in range(stm.stages):
                extra_perms_pj = extra_perms_p[j]

                # In this case we use this dict just for their keys!
                for s in self.dc_modalities_by_shift_idx_with_extra_hours:
                    extra_perms_pjs = extra_perms_pj[s]
                    absence_ps = absence_p[s]
                    presence_ps = presence_p.get(s, dict())

                    for mod in self.shift_params.dc_modalities_by_shift_idx[s]:
                        extra_perms_pjsm = extra_perms_pjs[mod]
                        acceptance = formatted_extra_hrs_params.dc_acceptance_ratios[s, mod]
                        extra_hs_s_m_epoch_range = self.dc_extra_hours_ranges[s, mod]
                        extra_hs_s_m_daily_max = formatted_extra_hrs_params.dc_daily_maxs[s, mod]

                        # The absence of the first epoch of the shift
                        absence_psm = absence_ps[mod]
                        absence = absence_psm[next(iter(absence_psm))]

                        # The pieces of slots in the list of extra hours of s, in case of fractional hours.
                        # We have here the extra hours before and after the shift.
                        extra_hs_ratios_sm: Dict = formatted_extra_hrs_params.dc_extra_hours_ratios[s, mod]

                        presence_psm = presence_ps.get(mod, dict())

                        # sum over epochs in ExtraHours(s) absence_pjsm
                        self.cpx.linear_constraints.add(
                            lin_expr=[cplex.SparsePair(
                                ind=[extra_perms_pjsm[t] for t in extra_hs_s_m_epoch_range] + 
                                     [stm.v_total_shift_assigned_per_modality[j][s][mod]],
                                val=[extra_hs_ratios_sm.get(t, 1) / presence_psm.get(t, 1) for t in extra_hs_s_m_epoch_range] +
                                    [-acceptance * extra_hs_s_m_daily_max * absence])],
                            senses=["L"],
                            rhs=[0],
                            names=["c_daily_legal_constr_%s_%d_%d_%s" % (proc.name, j, s, mod)])

        # # Weekly legal constraints:
        # for every shiftKind:
            
        #     self.cpx.linear_constraints.add(
        #         lin_expr=[cplex.SparsePair(
        #             ind=[],
        #             val=[])],
        #         senses=["L"],
        #         rhs=["0"],
        #         names=["c_weekly_legal_constr_%d_%d_d_s" % (proc, j, s, mod)])

        # # Monotony constraints:
        # for x in range(10):
        #     pass
