import json

files = []

# Group 1: Layout assignment
layout_assignment = [
    ("vmi_layout_assignment_cf_branch.pto", "layout_assignment", ["vadd"], "Layout assignment with cf.br branch"),
    ("vmi_layout_assignment_expand_load.pto", "layout_assignment", ["expand_load", "vload"], "Expand load with layout assignment"),
    ("vmi_layout_assignment_group_reduce_maxf_quant.pto", "layout_assignment", ["group_reduce", "vmaxf"], "Group reduce maxf quant with layout assignment"),
    ("vmi_layout_assignment_iota_remat.pto", "layout_assignment", ["vci"], "Iota rematerialization in layout assignment"),
    ("vmi_layout_assignment_mask_remat.pto", "layout_assignment", ["mask", "create_mask"], "Mask rematerialization in layout assignment"),
    ("vmi_layout_assignment_mask_select_store.pto", "layout_assignment", ["vsel", "vstore"], "Mask select+store in layout assignment"),
    ("vmi_layout_assignment_mask_use_ensure.pto", "layout_assignment", ["mask"], "Mask use ensure in layout assignment"),
    ("vmi_layout_assignment_scf_if.pto", "layout_assignment", ["vadd"], "scf.if with vadd in layout assignment"),
    ("vmi_layout_assignment_widen.pto", "layout_assignment", ["vcvt"], "Vcvt widen in layout assignment"),
]
files.extend(layout_assignment)

# Group 2: Layout E2E
layout_e2e = [
    ("vmi_layout_e2e_fold_vload.pto", "layout_e2e", ["vload"], "Layout fold vload E2E"),
    ("vmi_layout_e2e_gate_vadd.pto", "layout_e2e", ["vadd"], "Layout gate vadd E2E"),
    ("vmi_layout_e2e_gate_vload.pto", "layout_e2e", ["vload", "vcvt"], "Layout gate vload+vcvt E2E"),
    ("vmi_layout_e2e_remat_vadd.pto", "layout_e2e", ["vadd"], "Layout remat vadd E2E"),
    ("vmi_layout_e2e_scf_for.pto", "layout_e2e", ["vadd", "vcvt"], "Layout scf.for vadd+vcvt E2E"),
    ("vmi_layout_e2e_vadd_vmul.pto", "layout_e2e", ["vadd", "vmul"], "Layout vadd+vmul full pipeline E2E"),
    ("vmi_layout_e2e_vbrc_group.pto", "layout_e2e", ["vbrc"], "Layout vbrc with num_groups E2E"),
    ("vmi_layout_e2e_vcmp_vsel.pto", "layout_e2e", ["vcmp", "vsel"], "Layout vcmp+vsel full pipeline E2E"),
    ("vmi_layout_e2e_vcvt_all_dirs.pto", "layout_e2e", ["vcvt"], "Layout vcvt fp widen+narrow+fptosi E2E"),
    ("vmi_layout_e2e_vload_dintlv.pto", "layout_e2e", ["vload", "vcvt", "vstore"], "Layout vload dintlv+vcvt+vstore E2E"),
    ("vmi_layout_e2e_vload_masked.pto", "layout_e2e", ["vload", "vadd", "vstore"], "Layout vload+vadd+vstore masked E2E"),
    ("vmi_layout_e2e_vload_vstore.pto", "layout_e2e", ["vload", "vcvt", "vstore"], "Layout vload+vcvt+vstore continuous E2E"),
]
files.extend(layout_e2e)

# Group 3: Layout gate
layout_gate = [
    ("vmi_layout_gate_bitcast_group_slots.pto", "layout_gate", ["vbitcast"], "Gate bitcast group slots"),
    ("vmi_layout_gate_bitcast_support_invalid.pto", "layout_gate", ["vbitcast"], "Gate bitcast unsupported pattern invalid"),
    ("vmi_layout_gate_valid.pto", "layout_gate", ["vadd", "vload"], "Gate valid ops (vadd, vload)"),
]
files.extend(layout_gate)

# Group 4: Layout rematerialize
layout_remat = [
    ("vmi_layout_rematerialize_data.pto", "layout_rematerialize", ["vadd"], "Data rematerialization for vadd"),
    ("vmi_layout_rematerialize_relation.pto", "layout_rematerialize", ["vadd"], "Relation rematerialization for vadd"),
]
files.extend(layout_remat)

# Group 5: Layout sink
layout_sink = [
    ("vmi_layout_sink_materialization_binary.pto", "layout_sink", ["vadd", "vmul"], "Sink materialization for binary ops"),
]
files.extend(layout_sink)

# Group 6: Legacy lowering (two-stage)
legacy = [
    ("vmi_lower_legacy_plt.pto", "legacy_lower", ["plt"], "Two-stage: plt to create_mask (dynamic)"),
    ("vmi_lower_legacy_pset_pge.pto", "legacy_lower", ["pset", "pge"], "Two-stage: pset/pge to create_mask"),
    ("vmi_lower_legacy_vadd.pto", "legacy_lower", ["vadd"], "Two-stage: vadd to addf/addi + constant + mask + select"),
    ("vmi_lower_legacy_vcvt.pto", "legacy_lower", ["vcvt"], "Two-stage: vcvt to extf/truncf/fptosi/sitofp/exti/trunci"),
    ("vmi_lower_legacy_vector_scalar.pto", "legacy_lower", ["vadds", "vmuls", "vmaxs", "vmins", "vshls", "vshrs"], "Two-stage: vector-scalar to broadcast + legacy + select"),
    ("vmi_lower_legacy_vload.pto", "legacy_lower", ["vload"], "Two-stage: vload to load/deinterleave_load"),
    ("vmi_lower_legacy_vstore.pto", "legacy_lower", ["vstore"], "Two-stage: vstore to store/masked_store/interleave_store"),
]
files.extend(legacy)

# Group 7: Unified to legacy
unified = [
    ("vmi_lower_unified_to_legacy.pto", "unified_legacy", ["vadd", "vsub", "vmul", "vdiv", "vmin", "vmax", "vneg", "vabs", "vrelu", "vsqrt", "vexp", "vln", "vand", "vor", "vxor", "vshl", "vshr", "vnot", "vci", "vbitcast", "vsel", "vbrc"], "Two-stage: all unified vmi.* ops lower to legacy vmi.* ops"),
]
files.extend(unified)

# Group 8: E2E integration
e2e = [
    ("vmi_e2e_vadd_tail.pto", "e2e_integration", ["vadd"], "E2E vadd tail via ptoas CLI --vmi-two-stage-lowering"),
    ("vmi_e2e_vcvt_widen_add_store.pto", "e2e_integration", ["vload", "vcvt", "vbrc", "vadd", "vstore"], "E2E vcvt widen + vadd + vstore chain via ptoas CLI"),
    ("vmi_e2e_vload_dintlv_multichunk.pto", "e2e_integration", ["vload"], "E2E vload dintlv multichunk (256xf32) via ptoas CLI"),
]
files.extend(e2e)

# Group 9: Lowering tests (vmi_to_vpto_*)
lowering = [
    ("vmi_to_vpto_bf16_arith.pto", "lowering", ["vadd", "vmul"], "bf16 arithmetic lowering"),
    ("vmi_to_vpto_bitcast.pto", "lowering", ["vbitcast"], "Bitcast lowering"),
    ("vmi_to_vpto_bitcast_deint_tail.pto", "lowering", ["vbitcast"], "Bitcast deint tail lowering"),
    ("vmi_to_vpto_bitcast_footprint_invalid.pto", "lowering", ["vbitcast"], "Bitcast footprint invalid test"),
    ("vmi_to_vpto_bitcast_group_slots.pto", "lowering", ["vbitcast"], "Bitcast group slots lowering"),
    ("vmi_to_vpto_bitcast_partial.pto", "lowering", ["vbitcast"], "Bitcast partial lowering"),
    ("vmi_to_vpto_bitwise.pto", "lowering", ["vand", "vor", "vxor", "vnot"], "Bitwise ops lowering"),
    ("vmi_to_vpto_cmp_select.pto", "lowering", ["vcmp", "vsel"], "cmp+select lowering"),
    ("vmi_to_vpto_constant_mask_rematerialize.pto", "lowering", ["constant_mask"], "constant_mask rematerialize in lowering"),
    ("vmi_to_vpto_create_mask_rematerialize.pto", "lowering", ["create_mask"], "create_mask rematerialize in lowering"),
    ("vmi_to_vpto_divf.pto", "lowering", ["vdiv"], "divf lowering"),
    ("vmi_to_vpto_ensure_mask_granularity.pto", "lowering", ["mask"], "ensure_mask_granularity in lowering"),
    ("vmi_to_vpto_expand_load_all_active.pto", "lowering", ["expand_load"], "expand_load all-active mask"),
    ("vmi_to_vpto_expand_load_all_active_negative_offset_invalid.pto", "lowering", ["expand_load"], "expand_load negative offset invalid"),
    ("vmi_to_vpto_expand_load_partial_mask_invalid.pto", "lowering", ["expand_load"], "expand_load partial mask invalid"),
    ("vmi_to_vpto_expand_load_runtime_mask.pto", "lowering", ["expand_load"], "expand_load runtime mask"),
    ("vmi_to_vpto_group_slot_load.pto", "lowering", ["group_slot_load"], "group_slot_load lowering"),
    ("vmi_to_vpto_integer_casts.pto", "lowering", ["vcvt"], "int casts lowering (exti, trunci)"),
    ("vmi_to_vpto_iota.pto", "lowering", ["vci"], "iota lowering"),
    ("vmi_to_vpto_iota_tail.pto", "lowering", ["vci"], "iota tail lowering"),
    ("vmi_to_vpto_math_element_type_invalid.pto", "lowering", ["vneg", "vabs", "vrelu", "vexp", "vln", "vsqrt"], "math ops invalid element type"),
    ("vmi_to_vpto_memory_space_invalid.pto", "lowering", ["vload", "vstore"], "memory space invalid"),
    ("vmi_to_vpto_memref_layout_invalid.pto", "lowering", ["vload", "vstore"], "memref layout invalid"),
    ("vmi_to_vpto_min_max.pto", "lowering", ["vmin", "vmax"], "min/max lowering"),
    ("vmi_to_vpto_negf.pto", "lowering", ["vneg"], "negf lowering"),
    ("vmi_to_vpto_plt.pto", "lowering", ["plt"], "plt lowering (ptoas CLI integration)"),
    ("vmi_to_vpto_pset_pge.pto", "lowering", ["pset", "pge"], "pset/pge lowering"),
    ("vmi_to_vpto_relu_element_type_invalid.pto", "lowering", ["vrelu"], "relu element type invalid"),
    ("vmi_to_vpto_scf_if.pto", "lowering", ["vadd", "vcvt"], "scf.if lowering with VMI ops"),
    ("vmi_to_vpto_shli.pto", "lowering", ["vshl"], "shli lowering"),
    ("vmi_to_vpto_shrui.pto", "lowering", ["vshr"], "shrui lowering"),
    ("vmi_to_vpto_trunci_i8_signed_invalid.pto", "lowering", ["vcvt"], "trunci i8 signed invalid"),
    ("vmi_to_vpto_unary_math.pto", "lowering", ["vneg", "vabs", "vrelu", "vexp", "vln", "vsqrt"], "unary math lowering"),
    ("vmi_to_vpto_vadd.pto", "lowering", ["vadd"], "vadd standalone lowering"),
    ("vmi_to_vpto_vadd_i32_tail.pto", "lowering", ["vadd"], "vadd i32 tail lowering"),
    ("vmi_to_vpto_vadd_multichunk.pto", "lowering", ["vadd"], "vadd multichunk lowering"),
    ("vmi_to_vpto_vadd_sub_mul.pto", "lowering", ["vadd", "vsub", "vmul"], "vadd/vsub/vmul unified lowering"),
    ("vmi_to_vpto_vadd_tail.pto", "lowering", ["vadd"], "vadd f32 tail lowering"),
    ("vmi_to_vpto_vadds.pto", "lowering", ["vadds"], "vadds lowering"),
    ("vmi_to_vpto_vand_vor_vxor.pto", "lowering", ["vand", "vor", "vxor"], "vand/vor/vxor lowering"),
    ("vmi_to_vpto_vaxpy.pto", "lowering", ["vaxpy"], "vaxpy (fused alpha*x+y) lowering"),
    ("vmi_to_vpto_vbrc.pto", "lowering", ["vbrc"], "vbrc lowering"),
    ("vmi_to_vpto_vbrc_contiguous.pto", "lowering", ["vbrc"], "vbrc contiguous lowering"),
    ("vmi_to_vpto_vcadd.pto", "lowering", ["vcadd"], "vcadd (full reduction) lowering"),
    ("vmi_to_vpto_vci_tail.pto", "lowering", ["vci"], "vci tail lowering"),
    ("vmi_to_vpto_vcmax.pto", "lowering", ["vcmax"], "vcmax (max reduction) lowering"),
    ("vmi_to_vpto_vcmin.pto", "lowering", ["vcmin"], "vcmin (min reduction) lowering"),
    ("vmi_to_vpto_vcmp_vsel_deint2.pto", "lowering", ["vcmp", "vsel"], "vcmp+vsel deinterleaved=2 lowering"),
    ("vmi_to_vpto_vcmps_basic.pto", "lowering", ["vcmps"], "vcmps basic lowering"),
    ("vmi_to_vpto_vcvt_fp_narrow.pto", "lowering", ["vcvt"], "vcvt fp narrow lowering"),
    ("vmi_to_vpto_vcvt_fp_narrow_rounding.pto", "lowering", ["vcvt"], "vcvt fp narrow with rounding lowering"),
    ("vmi_to_vpto_vcvt_fp_narrow_tail.pto", "lowering", ["vcvt"], "vcvt fp narrow tail lowering"),
    ("vmi_to_vpto_vcvt_fp_to_si.pto", "lowering", ["vcvt"], "vcvt fp to signed int lowering"),
    ("vmi_to_vpto_vcvt_fp_widen.pto", "lowering", ["vcvt"], "vcvt fp widen lowering"),
    ("vmi_to_vpto_vcvt_fp_widen_f8.pto", "lowering", ["vcvt"], "vcvt fp widen from f8 lowering"),
    ("vmi_to_vpto_vcvt_fp_widen_multichunk.pto", "lowering", ["vcvt"], "vcvt fp widen multichunk lowering"),
    ("vmi_to_vpto_vcvt_fptosi_sitofp.pto", "lowering", ["vcvt"], "vcvt fptosi + sitofp lowering"),
    ("vmi_to_vpto_vcvt_int_narrow.pto", "lowering", ["vcvt"], "vcvt int narrow lowering"),
    ("vmi_to_vpto_vcvt_int_narrow_trunci.pto", "lowering", ["vcvt"], "vcvt int narrow + trunci lowering"),
    ("vmi_to_vpto_vcvt_int_widen.pto", "lowering", ["vcvt"], "vcvt int widen lowering"),
    ("vmi_to_vpto_vcvt_int_widen_tail.pto", "lowering", ["vcvt"], "vcvt int widen tail lowering"),
    ("vmi_to_vpto_vcvt_si_to_fp.pto", "lowering", ["vcvt"], "vcvt signed int to fp lowering"),
    ("vmi_to_vpto_vdintlv.pto", "lowering", ["vdintlv"], "vdintlv (deinterleave) lowering"),
    ("vmi_to_vpto_vdintlv_deint2.pto", "lowering", ["vdintlv"], "vdintlv deint2 lowering"),
    ("vmi_to_vpto_vdiv_deint2.pto", "lowering", ["vdiv"], "vdiv deint2 lowering"),
    ("vmi_to_vpto_vexp_vln_vsqrt.pto", "lowering", ["vexp", "vln", "vsqrt"], "vexp/vln/vsqrt lowering"),
    ("vmi_to_vpto_vexpdif.pto", "lowering", ["vexpdif"], "vexpdif (fused exp(x-max)) lowering"),
    ("vmi_to_vpto_vgather.pto", "lowering", ["vgather"], "vgather (indexed gather) lowering"),
    ("vmi_to_vpto_vgatherb.pto", "lowering", ["vgatherb"], "vgatherb (byte gather) lowering"),
    ("vmi_to_vpto_vhist.pto", "lowering", ["vhist"], "vhist (histogram) lowering"),
    ("vmi_to_vpto_vintlv.pto", "lowering", ["vintlv"], "vintlv (interleave) lowering"),
    ("vmi_to_vpto_vintlv_deint2.pto", "lowering", ["vintlv"], "vintlv deint2 lowering"),
    ("vmi_to_vpto_vload_brc.pto", "lowering", ["vload"], "vload broadcast group lowering"),
    ("vmi_to_vpto_vload_brc_tail.pto", "lowering", ["vload"], "vload broadcast tail lowering"),
    ("vmi_to_vpto_vload_continuous.pto", "lowering", ["vload"], "vload continuous lowering"),
    ("vmi_to_vpto_vload_continuous_multichunk.pto", "lowering", ["vload"], "vload continuous multichunk lowering"),
    ("vmi_to_vpto_vload_dintlv.pto", "lowering", ["vload"], "vload deinterleave lowering"),
    ("vmi_to_vpto_vload_dintlv_multichunk.pto", "lowering", ["vload"], "vload dintlv multichunk lowering"),
    ("vmi_to_vpto_vload_safe_tail.pto", "lowering", ["vload"], "vload safe tail lowering"),
    ("vmi_to_vpto_vload_stride.pto", "lowering", ["vload"], "vload stride lowering"),
    ("vmi_to_vpto_vload_unpack.pto", "lowering", ["vload", "unpack"], "vload + unpack lowering"),
    ("vmi_to_vpto_vload_unpack_tail.pto", "lowering", ["vload", "unpack"], "vload + unpack tail lowering"),
    ("vmi_to_vpto_vlrelu.pto", "lowering", ["vlrelu"], "vlrelu (leaky ReLU) lowering"),
    ("vmi_to_vpto_vmaxs.pto", "lowering", ["vmaxs"], "vmaxs (vector-scalar max) lowering"),
    ("vmi_to_vpto_vmins.pto", "lowering", ["vmins"], "vmins (vector-scalar min) lowering"),
    ("vmi_to_vpto_vmula.pto", "lowering", ["vmula"], "vmula (fused mul-add) lowering"),
    ("vmi_to_vpto_vmull.pto", "lowering", ["vmull"], "vmull (widening multiply) lowering"),
    ("vmi_to_vpto_vmuls.pto", "lowering", ["vmuls"], "vmuls (vector-scalar mul) lowering"),
    ("vmi_to_vpto_vneg_vabs_contiguous.pto", "lowering", ["vneg", "vabs"], "vneg/vabs contiguous lowering"),
    ("vmi_to_vpto_vprelu.pto", "lowering", ["vprelu"], "vprelu (parametric ReLU) lowering"),
    ("vmi_to_vpto_vscatter.pto", "lowering", ["vscatter"], "vscatter lowering"),
    ("vmi_to_vpto_vselr.pto", "lowering", ["vselr"], "vselr (dynamic lane select) lowering"),
    ("vmi_to_vpto_vselr_dynamic.pto", "lowering", ["vselr"], "vselr dynamic lowering"),
    ("vmi_to_vpto_vshl_vshr_contiguous.pto", "lowering", ["vshl", "vshr"], "vshl/vshr contiguous lowering"),
    ("vmi_to_vpto_vshls.pto", "lowering", ["vshls"], "vshls (vector-scalar shift left) lowering"),
    ("vmi_to_vpto_vshrs.pto", "lowering", ["vshrs"], "vshrs (vector-scalar shift right) lowering"),
    ("vmi_to_vpto_vstore_continuous.pto", "lowering", ["vstore"], "vstore continuous lowering"),
    ("vmi_to_vpto_vstore_dintlv.pto", "lowering", ["vstore"], "vstore dintlv lowering"),
    ("vmi_to_vpto_vstore_masked.pto", "lowering", ["vstore"], "vstore masked lowering"),
    ("vmi_to_vpto_vstore_stride.pto", "lowering", ["vstore"], "vstore stride lowering"),
    ("vmi_to_vpto_vstore_tail.pto", "lowering", ["vstore"], "vstore tail lowering"),
    ("vmi_to_vpto_vsub_deint2.pto", "lowering", ["vsub"], "vsub deint2 lowering"),
]
files.extend(lowering)

# Group 10: Verifier tests
verifiers = [
    ("vmi_bitcast_total_bits_invalid.pto", "verifier", ["vbitcast"], "Total bits mismatch in vbitcast"),
    ("vmi_bitwise_float_invalid.pto", "verifier", ["vand", "vor", "vxor", "vnot"], "Bitwise ops on float element type"),
    ("vmi_divf_integer_invalid.pto", "verifier", ["vdiv"], "Divf on integer element type"),
    ("vmi_iota_element_type_invalid.pto", "verifier", ["vci"], "Iota with invalid element type"),
    ("vmi_iota_order_invalid.pto", "verifier", ["vci"], "Iota with invalid order"),
    ("vmi_min_max_integer_invalid.pto", "verifier", ["vmin", "vmax"], "Min/max on integer element type"),
    ("vmi_negf_integer_invalid.pto", "verifier", ["vneg"], "Negf on integer element type"),
    ("vmi_plt_type_invalid.pto", "verifier", ["plt"], "PLT with invalid type"),
    ("vmi_select_mask_granularity_invalid.pto", "verifier", ["vsel"], "Select with mask granularity mismatch"),
    ("vmi_shli_float_invalid.pto", "verifier", ["vshl"], "Shli on float type"),
    ("vmi_shrui_float_invalid.pto", "verifier", ["vshr"], "Shrui on float type"),
    ("vmi_shrui_signed_invalid.pto", "verifier", ["vshr"], "Shrui on signed integer"),
    ("vmi_unary_math_integer_invalid.pto", "verifier", ["vsqrt", "vexp", "vln", "vrelu"], "Unary math ops on integer type"),
    ("vmi_vadd_lane_mismatch_invalid.pto", "verifier", ["vadd"], "Vadd with lane count mismatch"),
    ("vmi_vaxpy_element_type_invalid.pto", "verifier", ["vaxpy"], "Vaxpy with invalid element type"),
    ("vmi_vcadd_element_type_invalid.pto", "verifier", ["vcadd"], "Vcadd with invalid element type"),
    ("vmi_vcadd_reassoc_invalid.pto", "verifier", ["vcadd"], "Vcadd with invalid reassoc mode"),
    ("vmi_vcmax_element_type_invalid.pto", "verifier", ["vcmax"], "Vcmax with invalid element type"),
    ("vmi_vcmax_group_invalid.pto", "verifier", ["vcmax"], "Vcmax with invalid group"),
    ("vmi_vcmin_element_type_invalid.pto", "verifier", ["vcmin"], "Vcmin with invalid element type"),
    ("vmi_vcmin_result_lane_invalid.pto", "verifier", ["vcmin"], "Vcmin with invalid result lane count"),
    ("vmi_vcmp_element_type_invalid.pto", "verifier", ["vcmp"], "Vcmp with invalid element type"),
    ("vmi_vcmp_predicate_unsupported_invalid.pto", "verifier", ["vcmp"], "Vcmp with unsupported predicate"),
    ("vmi_vcvt_int_widen_signless_no_sign_invalid.pto", "verifier", ["vcvt"], "Vcvt int widen without sign on signless type"),
    ("vmi_vcvt_lane_mismatch_invalid.pto", "verifier", ["vcvt"], "Vcvt with lane count mismatch for element size ratio"),
    ("vmi_vcvt_rounding_bad_value_invalid.pto", "verifier", ["vcvt"], "Vcvt with bad rounding value"),
    ("vmi_vcvt_rounding_wrong_dir_invalid.pto", "verifier", ["vcvt"], "Vcvt with rounding on non-narrowing"),
    ("vmi_vcvt_same_width_invalid.pto", "verifier", ["vcvt"], "Vcvt with same-width types"),
    ("vmi_vcvt_saturate_wrong_dir_invalid.pto", "verifier", ["vcvt"], "Vcvt saturate on non-narrowing"),
    ("vmi_vcvt_sign_wrong_dir_invalid.pto", "verifier", ["vcvt"], "Vcvt sign change on fp or non-narrowing"),
    ("vmi_vdintlv_lane_mismatch_invalid.pto", "verifier", ["vdintlv"], "Vdintlv with lane count mismatch"),
    ("vmi_vexpdif_element_type_invalid.pto", "verifier", ["vexpdif"], "Vexpdif with invalid element type"),
    ("vmi_vexpdif_max_type_invalid.pto", "verifier", ["vexpdif"], "Vexpdif with invalid max type"),
    ("vmi_vgather_elem_type_mismatch_invalid.pto", "verifier", ["vgather"], "Vgather element type mismatch"),
    ("vmi_vgather_offset_type_invalid.pto", "verifier", ["vgather"], "Vgather offset type invalid"),
    ("vmi_vgatherb_elem_type_mismatch_invalid.pto", "verifier", ["vgatherb"], "Vgatherb element type mismatch"),
    ("vmi_vgatherb_offset_type_invalid.pto", "verifier", ["vgatherb"], "Vgatherb offset type invalid"),
    ("vmi_vhist_bin_type_invalid.pto", "verifier", ["vhist"], "Vhist bin type invalid"),
    ("vmi_vhist_element_type_invalid.pto", "verifier", ["vhist"], "Vhist element type invalid"),
    ("vmi_vintlv_element_mismatch_invalid.pto", "verifier", ["vintlv"], "Vintlv element type mismatch"),
    ("vmi_vintlv_lane_mismatch_invalid.pto", "verifier", ["vintlv"], "Vintlv lane count mismatch"),
    ("vmi_vload_bad_dist_mode_invalid.pto", "verifier", ["vload"], "Vload with invalid dist_mode"),
    ("vmi_vload_bad_pmode_invalid.pto", "verifier", ["vload"], "Vload with invalid pmode"),
    ("vmi_vload_dintlv_result_count_invalid.pto", "verifier", ["vload"], "Vload dintlv with wrong result count"),
    ("vmi_vload_elem_type_mismatch_invalid.pto", "verifier", ["vload"], "Vload element type mismatch with pointer"),
    ("vmi_vload_non_dintlv_multi_result_invalid.pto", "verifier", ["vload"], "Vload non-dintlv with multiple results"),
    ("vmi_vlrelu_element_type_invalid.pto", "verifier", ["vlrelu"], "Vlrelu with invalid element type"),
    ("vmi_vmul_lane_mismatch_invalid.pto", "verifier", ["vmul"], "Vmul with lane count mismatch"),
    ("vmi_vmula_element_type_invalid.pto", "verifier", ["vmula"], "Vmula with invalid element type"),
    ("vmi_vmula_type_mismatch_invalid.pto", "verifier", ["vmula"], "Vmula with operand type mismatch"),
    ("vmi_vmull_element_type_invalid.pto", "verifier", ["vmull"], "Vmull with invalid element type"),
    ("vmi_vprelu_alpha_type_invalid.pto", "verifier", ["vprelu"], "Vprelu with invalid alpha type"),
    ("vmi_vprelu_element_type_invalid.pto", "verifier", ["vprelu"], "Vprelu with invalid element type"),
    ("vmi_vscatter_elem_type_mismatch_invalid.pto", "verifier", ["vscatter"], "Vscatter element type mismatch"),
    ("vmi_vscatter_offset_type_invalid.pto", "verifier", ["vscatter"], "Vscatter offset type invalid"),
    ("vmi_vselr_element_mismatch_invalid.pto", "verifier", ["vselr"], "Vselr element type mismatch"),
    ("vmi_vselr_lane_mismatch_invalid.pto", "verifier", ["vselr"], "Vselr lane count mismatch"),
    ("vmi_vstore_bad_pmode_invalid.pto", "verifier", ["vstore"], "Vstore with invalid pmode"),
    ("vmi_vstore_brc_invalid.pto", "verifier", ["vstore"], "Vstore broadcast group invalid"),
    ("vmi_vstore_dintlv_value_count_invalid.pto", "verifier", ["vstore"], "Vstore dintlv wrong value count"),
    ("vmi_vstore_multi_mask_invalid.pto", "verifier", ["vstore"], "Vstore with multiple masks"),
    ("vmi_vstore_multi_value_invalid.pto", "verifier", ["vstore"], "Vstore with mismatched value count"),
    ("vmi_vstore_unpack_invalid.pto", "verifier", ["vstore"], "Vstore unpack invalid"),
    ("vmi_vsub_elem_type_mismatch_invalid.pto", "verifier", ["vsub"], "Vsub with element type mismatch"),
]
files.extend(verifiers)

# Group 11: Cross-feature
cross = [
    ("vmi_vcmp_seed_governance.pto", "cross_feature", ["vcmp"], "Vcmp seed governance: seed[i]=0 implies result[i]=0"),
    ("vmi_pmode_merge_skip_lowering.pto", "cross_feature", ["vadd"], "pmode=merge ops preserved through two-stage lowering"),
]
files.extend(cross)

# Group 12: Other / Misc
misc = [
    ("vmi_op_verifier_basic.pto", "verifier_basic", ["vadd", "vsub", "vmul", "vdiv", "vmin", "vmax", "vneg", "vabs", "vrelu", "vsqrt", "vexp", "vln", "vand", "vor", "vxor", "vshl", "vshr", "vnot", "vci", "vbitcast", "vsel", "vbrc", "vload", "vstore", "vcvt", "vcmp", "vcmps", "vadds", "vmuls", "vmaxs", "vmins", "vshls", "vshrs", "vaxpy", "vcadd", "vcmax", "vcmin", "vexpdif", "vgather", "vgatherb", "vhist", "vintlv", "vdintlv", "vlrelu", "vprelu", "vscatter", "vselr", "create_mask", "constant_mask", "pset", "pge", "plt", "expand_load", "group_slot_load"], "Basic verifier smoke test for all VMI ops"),
    ("vmi_producer_boundary_valid.pto", "misc", ["addf", "vsel"], "Producer boundary valid test"),
    ("vmi_vcmp_basic.pto", "misc", ["vcmp"], "Vcmp basic standalone test"),
    ("vmi_vcmps_basic.pto", "misc", ["vcmps"], "Vcmps basic standalone test"),
]
files.extend(misc)

result = {"files": files, "total": len(files)}
print(json.dumps(result, indent=2, ensure_ascii=False))
