import json, re, sys

files_raw = open('/c/Users/53284/.claude/projects/C--Users-53284-Desktop-ptocode/f4152e2f-41d2-4fa0-9c0c-85f55a8d92e7/tool-results/b83s7l7d4.txt').read().strip().split('\n')
filenames = [f.split('/')[-1].replace('.pto','') for f in files_raw]

# Op name extraction patterns from filename
# List of (stem, op_name) mappings - longer stems first to avoid partial matches
op_map = [
    ('active_prefix_index', 'vmi.active_prefix_index'),
    ('create_group_mask', 'vmi.create_group_mask'),
    ('create_mask', 'vmi.create_mask'),
    ('constant_mask', 'vmi.constant_mask'),
    ('constant_', 'vmi.constant'),
    ('channel_merge', 'vmi.channel_merge'),
    ('channel_split', 'vmi.channel_split'),
    ('group_broadcast', 'vmi.group_broadcast'),
    ('group_load', 'vmi.group_load'),
    ('group_reduce', 'vmi.group_reduce'),
    ('group_slot', 'vmi.group_slot_load/vmi.group_slot_store'),
    ('group_store', 'vmi.group_store'),
    ('group_slots', 'vmi.group_slots'),
    ('group_ops', 'vmi.group_broadcast/vmi.group_load/vmi.group_reduce/vmi.group_store'),
    ('masked_load', 'vmi.masked_load'),
    ('masked_store', 'vmi.masked_store'),
    ('mask_logic', 'vmi.mask_and/vmi.mask_or/vmi.mask_xor/vmi.mask_not'),
    ('mask_granularity', 'vmi.ensure_layout(mask)'),
    ('mask_pred', 'vmi.constant_mask'),
    ('mask_concrete', 'vmi.constant_mask'),
    ('mask_select', 'vmi.mask_select'),
    ('mask_use_ensure', 'vmi.ensure_layout'),
    ('mask_remat', 'rematerialize(mask)'),
    ('mask', 'vmi.constant_mask/vmi.create_mask'),
    ('interleaved_memory_ops', 'vmi.load/vmi.store'),
    ('producer_boundary', 'producer_boundary_check'),
    ('stride_load', 'vmi.stride_load'),
    ('stride_store', 'vmi.stride_store'),
    ('expand_load', 'vmi.expand_load'),
    ('call_boundary', 'vmi.call'),
    ('call_argument_boundary', 'vmi.call'),
    ('multi_return', 'vmi.call(multi-return)'),
    ('cf_branch', 'vmi.cf_branch'),
    ('cf_switch', 'vmi.cf_switch'),
    ('scf_for', 'vmi.scf_for'),
    ('scf_if', 'vmi.scf_if'),
    ('scf_while', 'vmi.scf_while'),
    ('scf_execute_region', 'vmi.scf_execute_region'),
    ('scf_index_switch', 'vmi.scf_index_switch'),
    ('load_truncf', 'vmi.load/vmi.truncf'),
    ('load_deint', 'vmi.load(deint)'),
    ('load_safe_tail', 'vmi.load(tail)'),
    ('load_store', 'vmi.load/vmi.store'),
    ('load', 'vmi.load'),
    ('store_deint', 'vmi.store(deint)'),
    ('store_tail', 'vmi.store(tail)'),
    ('store_ensure', 'vmi.store/vmi.ensure_layout'),
    ('store_reduce', 'vmi.store/vmi.reduce'),
    ('store', 'vmi.store'),
    ('sub_mul', 'vmi.subf/vmi.mulf'),
    ('subf', 'vmi.subf'),
    ('mulf', 'vmi.mulf'),
    ('addf', 'vmi.addf'),
    ('addi', 'vmi.addi'),
    ('absf', 'vmi.absf'),
    ('absi', 'vmi.absi'),
    ('negf', 'vmi.negf'),
    ('divf', 'vmi.divf'),
    ('min_max', 'vmi.minf/vmi.mini/vmi.maxf/vmi.maxi'),
    ('minmaxf', 'vmi.minf/vmi.maxf'),
    ('fma', 'vmi.fma'),
    ('extf', 'vmi.extf'),
    ('truncf', 'vmi.truncf'),
    ('trunci', 'vmi.trunci'),
    ('bitcast', 'vmi.bitcast'),
    ('bitwise', 'vmi.bitwise_and/vmi.bitwise_or/vmi.bitwise_xor'),
    ('broadcast', 'vmi.broadcast'),
    ('compress', 'vmi.compress'),
    ('dhist', 'vmi.dhist'),
    ('gather', 'vmi.gather'),
    ('scatter', 'vmi.scatter'),
    ('shuffle', 'vmi.shuffle'),
    ('shli', 'vmi.shli'),
    ('shrui', 'vmi.shrui'),
    ('iota', 'vmi.iota'),
    ('ensure_layout', 'vmi.ensure_layout'),
    ('ensure_mask', 'vmi.ensure_layout(mask)'),
    ('ensure_identity', 'vmi.ensure_layout(identity)'),
    ('elementwise', 'vmi.elementwise(various)'),
    ('unary_math', 'vmi.math.unary'),
    ('unpack', 'vmi.unpack'),
    ('pack', 'vmi.pack'),
    ('quant', 'vmi.quant/vmi.dequant'),
    ('reduce', 'vmi.reduce'),
    ('select', 'vmi.select'),
    ('widen', 'vmi.widen'),
    ('cmp', 'vmi.cmpf/vmi.cmpi'),
    ('integer_cast', 'vmi.trunci/vmi.extui/vmi.extsi'),
    ('type_attr', 'type_attr'),
    ('type_arity', 'vmi.elementwise'),
    ('type_only', 'vmi.elementwise(no-value)'),
    ('type_element', 'vmi.type'),
    ('surface', 'vmi.surface'),
    ('vdintlv', 'vmi.ensure_layout(vdintlv)'),
    ('vintlv', 'vmi.ensure_layout(vintlv)'),
    ('deint', 'vmi.load/vmi.store(deint)'),
    ('deint4', 'vmi.ensure_layout(deint4)'),
    ('rematerialize', 'rematerialize'),
    ('remat', 'rematerialize'),
    ('sink_materialization', 'sink_materialization'),
    ('fold', 'layout_fold'),
    ('legalize', 'legalize'),
    ('memory_space', 'vmi.load/vmi.store(memory_space)'),
    ('memref', 'vmi.load/vmi.store(memref)'),
    ('memory', 'vmi.load/vmi.store'),
    ('chist', 'vmi.chist'),
    ('compaction', 'vmi.compaction'),
    ('construction', 'vpi.construction'),
    ('function_type', 'function'),
    ('backend', 'ptoas_backend'),
    ('private_call', 'vmi.call(private)'),
    ('public_abi', 'vmi.call(public-abi)'),
    ('cli', 'ptoas-cli'),
    ('backend_required', 'ptoas_backend'),
    ('call_inline', 'vmi.call(inline)'),
    ('ento', 'vmi.ento'),
]

def extract_ops(name):
    ops = set()
    # Remove category prefixes
    for prefix in ['vmi_to_vpto_', 'vmi_layout_assignment_', 'vmi_layout_gate_',
                   'vmi_layout_fold_', 'vmi_layout_rematerialize_', 'vmi_layout_sink_materialization_',
                   'vmi_layout_group_slots_', 'vmi_layout_factor_',
                   'vmi_ptoas_', 'vmi_legalize_', 'vmi_producer_boundary_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    else:
        if name.startswith('vmi_'):
            name = name[4:]

    # Remove _invalid, _valid suffixes
    name = re.sub(r'_(invalid|valid|unsupported|todo)$', '', name)

    # Try long matches first
    sorted_ops = sorted(op_map, key=lambda x: -len(x[0]))
    matched = set()
    for stem, opname in sorted_ops:
        if stem in name and stem not in [x[0] for x in matched if False]:
            matched.add(opname)

    if not matched:
        return ['unknown']
    return sorted(matched)

def categorize(name):
    if name.startswith('vmi_to_vpto_'):
        return 'lowering'
    if name.startswith('vmi_layout_assignment_'):
        return 'layout'
    if name.startswith('vmi_layout_gate_'):
        return 'gate'
    if name.startswith('vmi_layout_fold_'):
        return 'layout'
    if name.startswith('vmi_layout_rematerialize_'):
        return 'layout'
    if name.startswith('vmi_layout_sink_'):
        return 'layout'
    if name.startswith('vmi_layout_group_slots_'):
        return 'layout'
    if name.startswith('vmi_layout_factor_'):
        return 'layout'
    if name.startswith('vmi_ptoas_'):
        return 'other'
    if name.startswith('vmi_legalize_'):
        return 'other'
    if name.startswith('vmi_producer_boundary_'):
        return 'other'
    if name.startswith('vmi_interleaved_memory_ops'):
        return 'other'
    if name.startswith('vmi_type_attr_parse'):
        return 'other'
    if name.startswith('vmi_op_verifier_basic'):
        return 'other'
    # Everything else is verifier
    return 'verifier'

def notes_for(name, cat):
    if cat == 'lowering':
        if '_invalid' in name:
            return 'Lowering VMI to VPTO - invalid input rejection'
        return 'Lowering VMI to VPTO ops'
    elif cat == 'layout':
        if '_invalid' in name:
            return 'Layout assignment - invalid input rejection'
        return 'Layout assignment pass test'
    elif cat == 'gate':
        if '_invalid' in name:
            return 'Layout gate - invalid input rejection'
        return 'Layout gate validation test'
    elif cat == 'verifier':
        if '_invalid' in name:
            return 'Verifier rejection test (invalid input)'
        elif '_valid' in name:
            return 'Verifier acceptance test (valid input)'
        else:
            return 'Verifier test'
    elif cat == 'other':
        if 'ptoas' in name:
            return 'PTOAS CLI/integration test'
        elif 'legalize' in name:
            return 'Legalization pass test'
        elif 'producer_boundary' in name:
            return 'Producer boundary check test'
        elif 'interleaved' in name:
            return 'Interleaved memory operations test'
        else:
            return 'Other VMI test'
    return ''

results = []
for fname in filenames:
    cat = categorize(fname)
    ops = extract_ops(fname)
    notes = notes_for(fname, cat)
    results.append({
        'filename': fname + '.pto',
        'category': cat,
        'ops': ops,
        'notes': notes
    })

# Count by category
by_cat = {}
for r in results:
    by_cat[r['category']] = by_cat.get(r['category'], 0) + 1

output = {
    'files': results,
    'total': len(results),
    'byCategory': by_cat
}

with open('/c/Users/53284/Desktop/ptocode/PTOAS/test/lit/vmi_new/vmi_catalog.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Total: {len(results)}")
print(f"By category: {json.dumps(by_cat)}")
